# -*- coding: utf-8 -*-
"""
中文：
    baseline_system.py
    这是第一版 baseline 的完整系统封装。
    它负责：
        1. 构建 encoder
        2. 构建 generator wrapper
        3. 前向生成 pred_full / pred_comp
        4. 调用 loss manager 计算损失

English:
    baseline_system.py
    Full system wrapper for the first baseline.
    It is responsible for:
        1. building the encoder
        2. building the generator wrapper
        3. producing pred_full / pred_comp
        4. computing losses via the loss manager
"""

from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn as nn

from src.models.encoders.simple_encoder import SimpleEncoder
from src.models.stylegan.stylegan2_wrapper import StyleGAN2Wrapper
from src.models.losses.loss_manager import BaselineLossManager


class BaselineInversionSystem(nn.Module):
    """
    中文：
        baseline inversion 系统。
        输入 batch，输出：
            - latent_pred
            - pred_full
            - pred_comp
            - losses

    English:
        Baseline inversion system.
        Given a batch, it outputs:
            - latent_pred
            - pred_full
            - pred_comp
            - losses
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

        model_cfg = cfg["model"]
        latent_cfg = model_cfg["latent"]
        enc_cfg = model_cfg["encoder"]
        gen_cfg = model_cfg["generator"]
        loss_cfg = cfg["loss"]

        self.encoder = SimpleEncoder(
            in_channels=enc_cfg["in_channels"],
            base_channels=enc_cfg["base_channels"],
            max_channels=enc_cfg["max_channels"],
            num_down=enc_cfg["num_down"],
            num_layers=latent_cfg["num_layers"],
            latent_dim=latent_cfg["latent_dim"],
        )

        self.generator = StyleGAN2Wrapper(
            backend=gen_cfg.get("backend", "mock"),
            image_size=gen_cfg["image_size"],
            num_layers=latent_cfg["num_layers"],
            latent_dim=latent_cfg["latent_dim"],
            checkpoint=gen_cfg.get("checkpoint", None),
            freeze_generator=gen_cfg.get("freeze_generator", False),
        )

        self.loss_manager = BaselineLossManager(loss_cfg)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        中文：
            执行前向传播。

        English:
            Run the forward pass.
        """
        image = batch["image"]                  # [B,3,H,W]
        mask = batch["mask"]                    # [B,1,H,W], 1=hole
        known_region = batch["known_region"]    # [B,1,H,W], 1=known
        masked_image = batch["masked_image"]    # [B,3,H,W]

        # 输入编码器的张量：masked image + mask
        # Encoder input: masked image concatenated with mask.
        encoder_input = torch.cat([masked_image, mask], dim=1)  # [B,4,H,W]

        latent_pred = self.encoder(encoder_input)               # [B,18,512]
        pred_full = self.generator(latent_pred)                 # [B,3,H,W]

        # 组合输出：已知区域直接用 GT，缺失区域用生成结果
        # Composite output: use GT on known region and generated pixels on hole region.
        pred_comp = pred_full * mask + image * known_region

        outputs = {
            "latent_pred": latent_pred,
            "pred_full": pred_full,
            "pred_comp": pred_comp,
        }
        return outputs

    def compute_losses(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        中文：
            计算损失。

        English:
            Compute losses.
        """
        image = batch["image"]
        mask = batch["mask"]
        known_region = batch["known_region"]
        boundary_band = batch["boundary_band"]

        losses = self.loss_manager(
            pred_full=outputs["pred_full"],
            pred_comp=outputs["pred_comp"],
            gt=image,
            mask=mask,
            known_region=known_region,
            boundary_band=boundary_band,
        )
        return losses

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        中文：
            单步训练逻辑。

        English:
            Single-step training logic.
        """
        outputs = self.forward(batch)
        losses = self.compute_losses(batch, outputs)

        result = {}
        result.update(outputs)
        result.update(losses)
        return result