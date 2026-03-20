# -*- coding: utf-8 -*-
"""
中文：
    sanity_optimize_latent.py
    单图 + 固定 mask 的真实 StyleGAN2 latent optimization sanity check（升级版）。
    当前版本支持：
        1. 只优化 W+ latent
        2. 冻结真实 StyleGAN2-ADA 生成器
        3. 从配置文件读取 known/hole/perceptual 权重
        4. 可选加入 LPIPS 感知损失

English:
    sanity_optimize_latent.py
    Upgraded single-image + fixed-mask latent optimization sanity check with real StyleGAN2.
    Current features:
        1. optimize W+ latent only
        2. freeze the real StyleGAN2-ADA generator
        3. read known/hole/perceptual weights from config
        4. optionally use LPIPS perceptual loss
"""

from __future__ import annotations

import os
import sys
import argparse

import torch
from torchvision.transforms import functional as TF
from torchvision.utils import save_image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 将 torchvision / LPIPS 预训练缓存放到项目目录下
# Put torchvision / LPIPS pretrained cache inside project directory.
os.environ["TORCH_HOME"] = os.path.join(PROJECT_ROOT, ".cache", "torch")

from src.utils.config import load_exp_config, make_output_dir, save_final_config
from src.utils.seed import set_seed
from src.data.datamodule import build_datamodule
from src.models.stylegan.stylegan2ada_wrapper import StyleGAN2ADAWrapper
from src.models.losses.pixel_loss import MaskedL1Loss
from src.models.losses.perceptual_loss import LPIPSLoss


def parse_args():
    """
    中文：
        解析命令行参数。

    English:
        Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to experiment yaml.")
    parser.add_argument("--steps", type=int, default=500, help="Number of latent optimization steps.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate for W+ optimization.")
    parser.add_argument("--save_interval", type=int, default=50, help="Save visualization interval.")
    return parser.parse_args()


def denorm(x: torch.Tensor) -> torch.Tensor:
    """
    中文：
        将 [-1, 1] 的图像反归一化到 [0, 1]。

    English:
        Convert images from [-1, 1] to [0, 1].
    """
    return (x.clamp(-1, 1) + 1.0) * 0.5


def save_vis(save_path: str, masked_image, mask, pred_full, pred_comp, gt):
    """
    中文：
        保存可视化拼图。
        顺序为：
            1. masked_image
            2. mask
            3. pred_full
            4. pred_comp
            5. gt

    English:
        Save a visualization grid.
        Order:
            1. masked_image
            2. mask
            3. pred_full
            4. pred_comp
            5. gt
    """
    if mask.size(1) == 1:
        mask_vis = mask.repeat(1, 3, 1, 1)
    else:
        mask_vis = mask

    grid = torch.cat([
        denorm(masked_image),
        mask_vis.clamp(0, 1),
        denorm(pred_full),
        denorm(pred_comp),
        denorm(gt),
    ], dim=0)

    save_image(grid.cpu(), save_path, nrow=1)


def main():
    args = parse_args()

    cfg = load_exp_config(args.config)
    out_dir = make_output_dir(cfg)
    save_final_config(cfg, out_dir)

    set_seed(cfg["project"]["seed"])
    device = torch.device(cfg["system"]["device"])

    # -----------------------------
    # 1. Build single-image batch
    # -----------------------------
    datamodule = build_datamodule(cfg)
    batch = next(iter(datamodule.train_dataloader()))
    batch = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }

    image = batch["image"]                  # [1,3,H,W]
    mask = batch["mask"]                    # [1,1,H,W], 1=hole
    known_region = batch["known_region"]    # [1,1,H,W], 1=known
    masked_image = batch["masked_image"]    # [1,3,H,W]

    target_h, target_w = image.shape[-2:]

    # -----------------------------
    # 2. Build real StyleGAN2
    # -----------------------------
    gen_cfg = cfg["model"]["generator"]

    G = StyleGAN2ADAWrapper(
        repo_root=gen_cfg["repo_root"],
        checkpoint=gen_cfg["checkpoint"],
        freeze_generator=True,
        noise_mode=gen_cfg.get("noise_mode", "const"),
        truncation_psi=gen_cfg.get("truncation_psi", 1.0),
        device=str(device),
    ).to(device)

    # -----------------------------
    # 3. Build optimizable W+
    # -----------------------------
    # 用 w_avg 初始化 W+
    # Initialize W+ from w_avg.
    w_plus = G.init_wplus(batch_size=1).to(device)
    w_plus = torch.nn.Parameter(w_plus)

    optimizer = torch.optim.Adam([w_plus], lr=args.lr)

    # -----------------------------
    # 4. Loss functions
    # -----------------------------
    loss_cfg = cfg["loss"]

    known_enabled = bool(loss_cfg["known_l1"]["enabled"])
    hole_enabled = bool(loss_cfg["hole_l1"]["enabled"])
    perceptual_enabled = bool(loss_cfg["perceptual"]["enabled"])

    known_weight = float(loss_cfg["known_l1"]["weight"])
    hole_weight = float(loss_cfg["hole_l1"]["weight"])
    perceptual_weight = float(loss_cfg["perceptual"]["weight"])

    masked_l1 = MaskedL1Loss()

    if perceptual_enabled:
        perceptual_loss_fn = LPIPSLoss(net=loss_cfg["perceptual"].get("net", "alex")).to(device)
    else:
        perceptual_loss_fn = None

    # -----------------------------
    # 5. Output folders
    # -----------------------------
    sample_dir = os.path.join(out_dir, "latent_opt_samples")
    os.makedirs(sample_dir, exist_ok=True)

    log_txt = os.path.join(out_dir, "latent_opt_log.txt")

    # -----------------------------
    # 6. Optimization loop
    # -----------------------------
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)

        # 真实 StyleGAN2 FFHQ 一般输出 1024，这里 resize 回 GT 尺度
        # Real StyleGAN2 FFHQ usually outputs 1024, resize back to GT size.
        pred_full = G(w_plus)  # [1,3,1024,1024]
        pred_full = TF.resize(pred_full, [target_h, target_w], antialias=True)

        # 最终合成图：已知区域替换回 GT
        # Final composited image: replace known region with GT.
        pred_comp = pred_full * mask + image * known_region

        loss_total = image.new_tensor(0.0)

        if known_enabled:
            loss_known = masked_l1(pred_full, image, known_region)
            loss_total = loss_total + known_weight * loss_known
        else:
            loss_known = image.new_tensor(0.0)

        if hole_enabled:
            loss_hole = masked_l1(pred_full, image, mask)
            loss_total = loss_total + hole_weight * loss_hole
        else:
            loss_hole = image.new_tensor(0.0)

        # 感知损失作用在最终修复图 pred_comp 上更合理，
        # 因为它更接近真正交付给用户的结果。
        #
        # Perceptual loss is applied on pred_comp, which is closer to
        # the final inpainting result delivered to users.
        if perceptual_enabled and perceptual_loss_fn is not None:
            loss_perc = perceptual_loss_fn(pred_comp, image)
            loss_total = loss_total + perceptual_weight * loss_perc
        else:
            loss_perc = image.new_tensor(0.0)

        loss_total.backward()
        optimizer.step()

        msg = (
            f"[LatentOpt] step={step:04d} | "
            f"loss_known={loss_known.item():.6f} | "
            f"loss_hole={loss_hole.item():.6f} | "
            f"loss_perc={loss_perc.item():.6f} | "
            f"loss_total={loss_total.item():.6f}"
        )

        if step % 10 == 0:
            print(msg)

        with open(log_txt, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

        if (step % args.save_interval == 0) or (step == args.steps - 1):
            save_path = os.path.join(sample_dir, f"step_{step:04d}.png")
            save_vis(save_path, masked_image, mask, pred_full, pred_comp, image)

    # -----------------------------
    # 7. Save optimized latent
    # -----------------------------
    torch.save(
        {"w_plus": w_plus.detach().cpu()},
        os.path.join(out_dir, "optimized_wplus.pt")
    )


if __name__ == "__main__":
    main()