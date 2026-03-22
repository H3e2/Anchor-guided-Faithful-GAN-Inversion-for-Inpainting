# -*- coding: utf-8 -*-
"""
中文：
    sanity_anchor_init.py
    单图对比：
        1. w_avg init + refinement
        2. anchor init + refinement

English:
    sanity_anchor_init.py
    Single-image comparison:
        1. w_avg init + refinement
        2. anchor init + refinement
"""

from __future__ import annotations

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["TORCH_HOME"] = os.path.join(PROJECT_ROOT, ".cache", "torch")

import torch
from torchvision.transforms import functional as TF
from torchvision.utils import save_image

from src.utils.config import load_exp_config, make_output_dir, save_final_config
from src.utils.seed import set_seed
from src.data.datamodule import build_datamodule
from src.models.stylegan.stylegan2ada_wrapper import StyleGAN2ADAWrapper
from src.models.modules.anchor_initializer import AnchorInitializer
from src.models.losses.pixel_loss import MaskedL1Loss
from src.models.losses.perceptual_loss import LPIPSLoss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--initializer_ckpt", type=str, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.03)
    return parser.parse_args()


def denorm(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) * 0.5


def noise_regularize(noise_bufs: dict[str, torch.Tensor]) -> torch.Tensor:
    reg_loss = 0.0
    for _, noise in noise_bufs.items():
        while noise.dim() < 4:
            noise = noise.unsqueeze(0)

        cur = noise
        while True:
            reg_loss = reg_loss + (cur * torch.roll(cur, shifts=1, dims=3)).mean().pow(2)
            reg_loss = reg_loss + (cur * torch.roll(cur, shifts=1, dims=2)).mean().pow(2)
            if cur.shape[2] <= 8 or cur.shape[3] <= 8:
                break
            cur = TF.resize(cur, [cur.shape[2] // 2, cur.shape[3] // 2], antialias=False)

    if not torch.is_tensor(reg_loss):
        reg_loss = torch.tensor(reg_loss, dtype=torch.float32)
    return reg_loss


def normalize_noise_(noise_bufs: dict[str, torch.Tensor]):
    with torch.no_grad():
        for _, noise in noise_bufs.items():
            noise_mean = noise.mean()
            noise_std = noise.square().mean().sqrt()
            noise -= noise_mean
            noise /= (noise_std + 1e-8)


def optimize_from_init(
    G,
    init_w_plus,
    image,
    mask,
    known_region,
    cfg,
    steps: int,
    lr: float,
    fixed_noise_init: dict[str, torch.Tensor] | None = None,
):
    device = image.device
    target_h, target_w = image.shape[-2:]

    w_plus = torch.nn.Parameter(init_w_plus.clone().detach())
    if fixed_noise_init is None:
        noise_bufs = G.init_noise_buffers()
    else:
        G.set_noise_buffers(fixed_noise_init)
        noise_bufs = G.get_noise_buffers()
        for _, buf in noise_bufs.items():
            buf.requires_grad = True
    optimizer = torch.optim.Adam([w_plus] + list(noise_bufs.values()), lr=lr)

    masked_l1 = MaskedL1Loss()

    loss_cfg = cfg["loss"]
    use_perc = bool(loss_cfg["perceptual"]["enabled"])
    if use_perc:
        perc_fn = LPIPSLoss(net=loss_cfg["perceptual"].get("net", "alex")).to(device)
    else:
        perc_fn = None

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)

        pred_full = G(w_plus)
        pred_full = TF.resize(pred_full, [target_h, target_w], antialias=True)
        pred_comp = pred_full * mask + image * known_region

        loss_known = masked_l1(pred_full, image, known_region)
        loss_hole = masked_l1(pred_full, image, mask)

        loss_total = (
            float(loss_cfg["known_l1"]["weight"]) * loss_known +
            float(loss_cfg["hole_l1"]["weight"]) * loss_hole
        )

        if use_perc and perc_fn is not None:
            loss_perc = perc_fn(pred_comp, image)
            loss_total = loss_total + float(loss_cfg["perceptual"]["weight"]) * loss_perc

        if "noise_regularization" in loss_cfg and bool(loss_cfg["noise_regularization"]["enabled"]):
            noise_reg = noise_regularize(noise_bufs)
            loss_total = loss_total + float(loss_cfg["noise_regularization"]["weight"]) * noise_reg

        loss_total.backward()
        optimizer.step()
        normalize_noise_(noise_bufs)

        if step % 20 == 0:
            print(
                f"[Refine] step={step:04d} | "
                f"loss_total={loss_known.item():.6f} | "
                f"loss_known={loss_hole.item():.6f} | "
                f"loss_hole={loss_perc.item():.6f} | "
                f"loss_perc={noise_reg.item():.6f} | "
                f"loss_total={loss_total.item():.6f}"
            )

    pred_full = G(w_plus)
    pred_full = TF.resize(pred_full, [target_h, target_w], antialias=True)
    pred_comp = pred_full * mask + image * known_region

    return pred_full.detach(), pred_comp.detach()


def main():
    args = parse_args()

    cfg = load_exp_config(args.config)
    out_dir = make_output_dir(cfg)
    save_final_config(cfg, out_dir)
    set_seed(cfg["project"]["seed"])

    device = torch.device(cfg["system"]["device"])

    datamodule = build_datamodule(cfg)
    batch = next(iter(datamodule.train_dataloader()))
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    image = batch["image"]
    mask = batch["mask"]
    known_region = batch["known_region"]
    masked_image = batch["masked_image"]

    gen_cfg = cfg["model"]["generator"]
    G = StyleGAN2ADAWrapper(
        repo_root=gen_cfg["repo_root"],
        checkpoint=gen_cfg["checkpoint"],
        freeze_generator=True,
        noise_mode=gen_cfg.get("noise_mode", "const"),
        truncation_psi=gen_cfg.get("truncation_psi", 1.0),
        device=str(device),
    ).to(device)

    anchor_cfg = cfg["model"]["anchor_initializer"]
    initializer = AnchorInitializer(
        in_channels=anchor_cfg["in_channels"],
        base_channels=anchor_cfg["base_channels"],
        max_channels=anchor_cfg["max_channels"],
        num_down=anchor_cfg["num_down"],
        feat_dim=anchor_cfg["feat_dim"],
        num_ws=anchor_cfg["num_ws"],
        w_dim=anchor_cfg["w_dim"],
        layer_splits=anchor_cfg.get("layer_splits", [4, 6, 8]),
        delta_scale=anchor_cfg["delta_scale"],
        gate_mode=anchor_cfg["gate_mode"],
        encoder_type=anchor_cfg.get("encoder_type", "ms_residual"),
        w_avg=G.w_avg,
    ).to(device)

    ckpt = torch.load(args.initializer_ckpt, map_location=device)
    initializer.load_state_dict(ckpt["initializer"], strict=True)
    initializer.eval()

    with torch.no_grad():
        anchor_out = initializer(masked_image, mask)
        anchor_init_w_plus = anchor_out["init_w_plus"]

    wavg_init_w_plus = G.init_wplus(batch_size=1).to(device)

    # 为了公平比较，两次 refinement 使用同一份 noise 初始状态
    # For fair comparison, both refinements start from the same noise initialization.
    fixed_noise_init = {}
    G.init_noise_buffers()
    for name, buf in G.get_noise_buffers().items():
        fixed_noise_init[name] = buf.detach().clone()

    pred_full_wavg, pred_comp_wavg = optimize_from_init(
        G, wavg_init_w_plus, image, mask, known_region, cfg, args.steps, args.lr,
        fixed_noise_init=fixed_noise_init
    )
    pred_full_anchor, pred_comp_anchor = optimize_from_init(
        G, anchor_init_w_plus, image, mask, known_region, cfg, args.steps, args.lr,
        fixed_noise_init=fixed_noise_init
    )

    save_dir = os.path.join(out_dir, "anchor_compare")
    os.makedirs(save_dir, exist_ok=True)
    print(save_dir)
    mask_vis = mask.repeat(1, 3, 1, 1)

    grid = torch.cat([
        denorm(masked_image),
        mask_vis.clamp(0, 1),
        denorm(pred_full_wavg),
        denorm(pred_comp_wavg),
        denorm(pred_full_anchor),
        denorm(pred_comp_anchor),
        denorm(image),
    ], dim=0)

    save_image(grid.cpu(), os.path.join(save_dir, "compare_wavg_vs_anchor.png"), nrow=1)


if __name__ == "__main__":
    main()