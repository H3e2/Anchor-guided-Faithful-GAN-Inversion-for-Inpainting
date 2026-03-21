# -*- coding: utf-8 -*-
"""
中文：
    train_anchor_initializer.py
    单图 overfit 版 anchor initializer 训练脚本。
    用已经优化得到的 projector W+ 作为监督，训练 initializer 预测更好的初始 W+。

English:
    train_anchor_initializer.py
    Single-image overfit training script for the anchor initializer.
    It uses the optimized projector W+ as supervision and trains
    the initializer to predict a better starting W+.
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
import torch.nn.functional as F
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
    parser.add_argument("--target_latent", type=str, required=True, help="Path to optimized_latent_and_noise.pt")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_interval", type=int, default=100)
    return parser.parse_args()


def denorm(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) * 0.5


def save_vis(save_path: str, masked_image, mask, pred_img, teacher_img, gt):
    if mask.size(1) == 1:
        mask_vis = mask.repeat(1, 3, 1, 1)
    else:
        mask_vis = mask

    grid = torch.cat([
        denorm(masked_image),
        mask_vis.clamp(0, 1),
        denorm(pred_img),
        denorm(teacher_img),
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

    # 单图 batch
    datamodule = build_datamodule(cfg)
    batch = next(iter(datamodule.train_dataloader()))
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    image = batch["image"]
    mask = batch["mask"]
    known_region = batch["known_region"]
    masked_image = batch["masked_image"]
    target_h, target_w = image.shape[-2:]

    # 真实 StyleGAN2
    gen_cfg = cfg["model"]["generator"]
    G = StyleGAN2ADAWrapper(
        repo_root=gen_cfg["repo_root"],
        checkpoint=gen_cfg["checkpoint"],
        freeze_generator=True,
        noise_mode=gen_cfg.get("noise_mode", "const"),
        truncation_psi=gen_cfg.get("truncation_psi", 1.0),
        device=str(device),
    ).to(device)

    # 读取 target optimized w+
    target_pack = torch.load(args.target_latent, map_location=device)
    target_w_plus = target_pack["w_plus"].to(device)
    target_noise_bufs = target_pack.get("noise_buffers", None)

    # 将 teacher noise 写回 generator，
    # 保证图像监督与 teacher latent 监督使用同一套生成细节。
    #
    # Load teacher noise into the generator so that image-space supervision
    # is consistent with the teacher latent.
    if target_noise_bufs is not None:
        G.set_noise_buffers(target_noise_bufs)
    
    with torch.no_grad():
        teacher_img = G(target_w_plus)
        teacher_img = TF.resize(teacher_img, [target_h, target_w], antialias=True)

    # anchor initializer
    anchor_cfg = cfg["model"]["anchor_initializer"]
    initializer = AnchorInitializer(
        in_channels=anchor_cfg["in_channels"],
        base_channels=anchor_cfg["base_channels"],
        max_channels=anchor_cfg["max_channels"],
        num_down=anchor_cfg["num_down"],
        feat_dim=anchor_cfg["feat_dim"],
        num_ws=anchor_cfg["num_ws"],
        w_dim=anchor_cfg["w_dim"],
        delta_scale=anchor_cfg["delta_scale"],
        gate_mode=anchor_cfg["gate_mode"],
        w_avg=G.w_avg,
    ).to(device)

    optimizer = torch.optim.Adam(initializer.parameters(), lr=args.lr)
    masked_l1 = MaskedL1Loss()

    loss_cfg = cfg["loss"]
    use_perc = bool(loss_cfg["perceptual"]["enabled"])
    perc_weight = float(loss_cfg["perceptual"]["weight"])
    if use_perc:
        perc_fn = LPIPSLoss(net=loss_cfg["perceptual"].get("net", "alex")).to(device)
    else:
        perc_fn = None

    sample_dir = os.path.join(out_dir, "anchor_init_train_samples")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)

        out = initializer(masked_image, mask)
        pred_w_plus = out["init_w_plus"]

        # latent supervision
        loss_latent_l1 = F.l1_loss(pred_w_plus, target_w_plus)
        loss_latent_l2 = F.mse_loss(pred_w_plus, target_w_plus)

        # image-space supervision
        pred_img = G(pred_w_plus)
        pred_img = TF.resize(pred_img, [target_h, target_w], antialias=True)

        # 图像监督分成两层理解：
        # 1) 对齐 teacher rendering：让 initializer 学会逼近 projector 的结果
        # 2) 对齐 GT：防止 teacher 自身误差被完全继承
        #
        # We use a mixed supervision:
        # 1) align with teacher rendering
        # 2) align with GT
        loss_img_known_teacher = masked_l1(pred_img, teacher_img, known_region)
        loss_img_hole_teacher = masked_l1(pred_img, teacher_img, mask)

        loss_img_known_gt = masked_l1(pred_img, image, known_region)
        loss_img_hole_gt = masked_l1(pred_img, image, mask)

        # 已知区域更偏 teacher 对齐，洞区域 teacher + GT 共同约束
        loss_img_known = 0.7 * loss_img_known_teacher + 0.3 * loss_img_known_gt
        loss_img_hole = 0.5 * loss_img_hole_teacher + 0.5 * loss_img_hole_gt


        loss_total = (
            float(loss_cfg["latent_l1"]["weight"]) * loss_latent_l1 +
            float(loss_cfg["latent_l2"]["weight"]) * loss_latent_l2 +
            float(loss_cfg["image_known_l1"]["weight"]) * loss_img_known +
            float(loss_cfg["image_hole_l1"]["weight"]) * loss_img_hole
        )

        if use_perc and perc_fn is not None:
            pred_comp = pred_img * mask + image * known_region
            loss_perc = perc_fn(pred_comp, image)
            loss_total = loss_total + perc_weight * loss_perc
        else:
            loss_perc = image.new_tensor(0.0)

        loss_total.backward()
        optimizer.step()

        if step % 10 == 0:
            print(
                f"[AnchorInitTrain] step={step:04d} | "
                f"latent_l1={loss_latent_l1.item():.6f} | "
                f"latent_l2={loss_latent_l2.item():.6f} | "
                f"img_known={loss_img_known.item():.6f} | "
                f"img_hole={loss_img_hole.item():.6f} | "
                f"perc={loss_perc.item():.6f} | "
                f"total={loss_total.item():.6f}"
            )

        if (step % args.save_interval == 0) or (step == args.steps - 1):
            save_vis(
                os.path.join(sample_dir, f"step_{step:04d}.png"),
                masked_image,
                mask,
                pred_img,
                teacher_img,
                image,
            )

    torch.save(
        {
            "initializer": initializer.state_dict(),
            "target_latent_path": args.target_latent,
        },
        os.path.join(ckpt_dir, "anchor_initializer_single.pt")
    )


if __name__ == "__main__":
    main()