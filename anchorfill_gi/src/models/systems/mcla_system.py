# -*- coding: utf-8 -*-
"""
中文：
    mcla_system.py
    这是 MCLA 的核心训练系统。
    它负责把：
        1. 冻结的 StyleGAN2-ADA 生成器
        2. MCLA adapter
        3. manifold / image / edge / frequency loss
    串起来。

English:
    mcla_system.py
    This is the core training system of MCLA.
    It wires together:
        1. the frozen StyleGAN2-ADA generator
        2. the MCLA adapter
        3. manifold / image / edge / frequency losses
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.transforms import functional as TF

from src.models.stylegan.stylegan2ada_wrapper import StyleGAN2ADAWrapper
from src.models.modules.mcla_adapter import MCLAAdapter
from src.models.losses.pixel_loss import MaskedL1Loss
from src.models.losses.perceptual_loss import LPIPSLoss
from src.models.losses.manifold_loss import GroupwiseManifoldLoss, sample_wplus_stats


class MCLASystem(nn.Module):
    """
    中文：
        MCLA 训练系统。
        输入：
            masked_image, mask
        输出：
            init_w_plus 以及渲染结果 pred_img

        当前阶段：
            - 不使用 teacher
            - 不训练 generator
            - 只训练 adapter

    English:
        Training system for MCLA.
        Input:
            masked_image, mask
        Output:
            init_w_plus and rendered image pred_img

        Current stage:
            - no teacher
            - generator is frozen
            - only the adapter is trained
    """

    def __init__(self, cfg: dict, device: torch.device):
        super().__init__()

        self.cfg = cfg
        self.device = device

        model_cfg = cfg["model"]
        gen_cfg = model_cfg["generator"]
        mcla_cfg = model_cfg["mcla"]
        loss_cfg = cfg["loss"]

        # ---------------------------------------------------------
        # 1. 冻结的预训练 StyleGAN2-ADA 生成器
        # 1. Frozen pretrained StyleGAN2-ADA generator
        # ---------------------------------------------------------
        self.G = StyleGAN2ADAWrapper(
            repo_root=gen_cfg["repo_root"],
            checkpoint=gen_cfg["checkpoint"],
            freeze_generator=True,
            noise_mode=gen_cfg.get("noise_mode", "const"),
            truncation_psi=gen_cfg.get("truncation_psi", 1.0),
            device=str(device),
        ).to(device)

        # ---------------------------------------------------------
        # 2. MCLA adapter
        #    它直接从 masked image 学一个 clean latent proposal
        #
        # 2. MCLA adapter
        #    It directly learns a clean latent proposal from masked input
        # ---------------------------------------------------------
        self.adapter = MCLAAdapter(
            in_channels=mcla_cfg["in_channels"],
            base_channels=mcla_cfg["base_channels"],
            max_channels=mcla_cfg["max_channels"],
            feat_dim=mcla_cfg["feat_dim"],
            num_down=mcla_cfg["num_down"],
            num_ws=mcla_cfg["num_ws"],
            w_dim=mcla_cfg["w_dim"],
            layer_splits=mcla_cfg["layer_splits"],
            delta_scale=mcla_cfg["delta_scale"],
            gate_mode=mcla_cfg["gate_mode"],
            use_edge_branch=mcla_cfg.get("use_edge_branch", True),
            use_freq_branch=mcla_cfg.get("use_freq_branch", True),
            w_avg=self.G.w_avg,
        ).to(device)

        # ---------------------------------------------------------
        # 3. 基础损失模块
        # 3. Basic loss modules
        # ---------------------------------------------------------
        self.masked_l1 = MaskedL1Loss()

        self.use_perc = bool(loss_cfg["perceptual"]["enabled"])
        if self.use_perc:
            self.perc_fn = LPIPSLoss(net=loss_cfg["perceptual"].get("net", "alex")).to(device)
        else:
            self.perc_fn = None

        # ---------------------------------------------------------
        # 4. latent manifold 正则
        #    保证 adapter 预测的 W+ 尽量停留在当前 generator 的风格流形附近
        #
        # 4. Latent manifold regularization
        #    Keep predicted W+ close to the style manifold of the current generator
        # ---------------------------------------------------------
        self.manifold_loss = GroupwiseManifoldLoss(
            num_ws=mcla_cfg["num_ws"],
            w_dim=mcla_cfg["w_dim"],
            layer_splits=mcla_cfg["layer_splits"],
        ).to(device)

        self.loss_cfg = loss_cfg
        self.image_size = cfg["data"]["image_size"]

    @torch.no_grad()
    def fit_manifold_stats(self, num_samples: int = 2048, batch_size: int = 16):
        """
        中文：
            从 generator 的 mapping 网络中采样大量 latent，
            用来估计当前 generator 的 W+ 流形统计量（均值和标准差）。

        English:
            Sample many latents from the generator mapping network
            to estimate W+ manifold statistics (mean and std).
        """
        mean_wplus, std_wplus = sample_wplus_stats(
            self.G,
            num_samples=num_samples,
            batch_size=batch_size,
            device=str(self.device),
        )
        self.manifold_loss.set_stats(mean_wplus.to(self.device), std_wplus.to(self.device))

    def forward(self, masked_image: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        中文：
            前向：调用 adapter 预测初始化 W+。

        English:
            Forward: call adapter to predict initialized W+.
        """
        return self.adapter(masked_image, mask)

    def render(self, init_w_plus: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """
        中文：
            用初始化的 W+ 渲染图像，并 resize 到训练图像大小。

        English:
            Render image from initialized W+, then resize to target training size.
        """
        pred_img = self.G(init_w_plus)
        pred_img = TF.resize(pred_img, [target_h, target_w], antialias=True)
        return pred_img

    def compose_prediction(
        self,
        pred_img: torch.Tensor,
        gt_img: torch.Tensor,
        mask: torch.Tensor,
        known_region: torch.Tensor
    ) -> torch.Tensor:
        """
        中文：
            合成最终修复图：
                洞区域来自 pred_img
                已知区域来自 gt_img

        English:
            Compose final inpainted image:
                hole region comes from pred_img
                known region comes from gt_img
        """
        return pred_img * mask + gt_img * known_region

    def compute_losses(
        self,
        init_w_plus: torch.Tensor,
        pred_img: torch.Tensor,
        gt_img: torch.Tensor,
        mask: torch.Tensor,
        known_region: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        中文：
            计算所有训练损失。

        English:
            Compute all training losses.
        """
        losses = {}

        # ---------------------------------------------------------
        # 已知区域一致性 / known-region consistency
        # ---------------------------------------------------------
        if self.loss_cfg["image_known_l1"]["enabled"]:
            losses["image_known_l1"] = self.masked_l1(pred_img, gt_img, known_region)
        else:
            losses["image_known_l1"] = gt_img.new_tensor(0.0)

        # ---------------------------------------------------------
        # 洞区域一致性 / hole-region consistency
        # ---------------------------------------------------------
        if self.loss_cfg["image_hole_l1"]["enabled"]:
            losses["image_hole_l1"] = self.masked_l1(pred_img, gt_img, mask)
        else:
            losses["image_hole_l1"] = gt_img.new_tensor(0.0)

        # ---------------------------------------------------------
        # 感知损失（当前 pilot 默认关闭）
        # perceptual loss (disabled in current pilot by default)
        # ---------------------------------------------------------
        if self.use_perc and self.perc_fn is not None:
            pred_comp = self.compose_prediction(pred_img, gt_img, mask, known_region)
            losses["perceptual"] = self.perc_fn(pred_comp, gt_img)
        else:
            losses["perceptual"] = gt_img.new_tensor(0.0)

        # ---------------------------------------------------------
        # latent manifold 正则
        # latent manifold regularization
        # ---------------------------------------------------------
        if self.loss_cfg["manifold_reg"]["enabled"]:
            losses["manifold_reg"] = self.manifold_loss(init_w_plus)
        else:
            losses["manifold_reg"] = gt_img.new_tensor(0.0)

        # ---------------------------------------------------------
        # edge / frequency consistency on hole region
        # 用预处理器重新从预测图像和 GT 中提取 edge / freq 分支，再在 hole 上对齐
        # ---------------------------------------------------------
        zero_mask = torch.zeros_like(mask)

        pred_prep = self.adapter.preprocessor(pred_img, zero_mask)
        gt_prep = self.adapter.preprocessor(gt_img, zero_mask)

        if self.loss_cfg["edge_hole_l1"]["enabled"]:
            losses["edge_hole_l1"] = self.masked_l1(pred_prep["edge"], gt_prep["edge"], mask)
        else:
            losses["edge_hole_l1"] = gt_img.new_tensor(0.0)

        if self.loss_cfg["freq_hole_l1"]["enabled"]:
            losses["freq_hole_l1"] = self.masked_l1(pred_prep["high_freq"], gt_prep["high_freq"], mask)
        else:
            losses["freq_hole_l1"] = gt_img.new_tensor(0.0)

        # ---------------------------------------------------------
        # 总损失 / total loss
        # ---------------------------------------------------------
        total = (
            float(self.loss_cfg["image_known_l1"]["weight"]) * losses["image_known_l1"] +
            float(self.loss_cfg["image_hole_l1"]["weight"]) * losses["image_hole_l1"] +
            float(self.loss_cfg["perceptual"]["weight"]) * losses["perceptual"] +
            float(self.loss_cfg["manifold_reg"]["weight"]) * losses["manifold_reg"] +
            float(self.loss_cfg["edge_hole_l1"]["weight"]) * losses["edge_hole_l1"] +
            float(self.loss_cfg["freq_hole_l1"]["weight"]) * losses["freq_hole_l1"]
        )
        losses["total"] = total
        return losses