# -*- coding: utf-8 -*-
"""
中文：
    train_mcla_pilot.py
    MCLA 的 pilot 训练入口。
    当前特点：
        1. 不使用 teacher
        2. 冻结 StyleGAN2-ADA
        3. 训练 MCLA adapter
        4. 自动保存时间戳目录、日志、样例图、best/latest checkpoint

English:
    train_mcla_pilot.py
    Pilot training entry for MCLA.
    Current features:
        1. no teacher
        2. frozen StyleGAN2-ADA
        3. train the MCLA adapter
        4. automatically save timestamped dirs, logs, samples, best/latest ckpts
"""

from __future__ import annotations

import os
import sys
import yaml
import time
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 让 LPIPS / torchvision 预训练缓存写到项目目录
# Put LPIPS / torchvision pretrained cache into project directory.
os.environ["TORCH_HOME"] = os.path.join(PROJECT_ROOT, ".cache", "torch")

import torch
from torch.optim import Adam
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim_fn

from src.utils.config import load_exp_config
from src.utils.seed import set_seed
from src.data.datamodule import build_datamodule
from src.models.systems.mcla_system import MCLASystem


def get_timestamp() -> str:
    """
    中文：
        返回统一格式的时间戳。

    English:
        Return a unified timestamp string.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def prepare_exp_dir(root_dir: str, exp_name: str) -> dict[str, str]:
    """
    中文：
        创建带时间戳的实验目录。

    English:
        Create a timestamped experiment directory.
    """
    timestamp = get_timestamp()
    exp_dir = os.path.join(root_dir, exp_name, timestamp)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    sample_dir = os.path.join(exp_dir, "samples")
    log_dir = os.path.join(exp_dir, "logs")
    metric_dir = os.path.join(exp_dir, "metrics")

    for p in [exp_dir, ckpt_dir, sample_dir, log_dir, metric_dir]:
        os.makedirs(p, exist_ok=True)

    return {
        "exp_dir": exp_dir,
        "ckpt_dir": ckpt_dir,
        "sample_dir": sample_dir,
        "log_dir": log_dir,
        "metric_dir": metric_dir,
    }


def setup_logger(log_file: str):
    """
    中文：
        同时输出到文件和终端的 logger。

    English:
        Logger that writes to both file and console.
    """
    import logging
    logger = logging.getLogger(f"mcla_train_{time.time()}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


class AverageMeter:
    """
    中文：
        简单平均器。

    English:
        Simple average meter.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count


def denorm(x: torch.Tensor) -> torch.Tensor:
    """
    中文：
        将 [-1,1] 转到 [0,1]

    English:
        Convert from [-1,1] to [0,1].
    """
    return (x.clamp(-1, 1) + 1.0) * 0.5


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """
    中文：
        计算 PSNR。

    English:
        Compute PSNR.
    """
    pred_01 = denorm(pred)
    target_01 = denorm(target)
    mse = torch.mean((pred_01 - target_01) ** 2).item()
    if mse < eps:
        return 99.0
    return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    中文：
        计算 SSIM（当前按 batch 中第一张图算）。

    English:
        Compute SSIM (currently on the first image in the batch).
    """
    pred_01 = denorm(pred).detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    target_01 = denorm(target).detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    return float(ssim_fn(pred_01, target_01, channel_axis=2, data_range=1.0))


def save_vis(save_path, masked_image, mask, pred_img, pred_comp, gt_img):
    """
    中文：
        保存样例图：
            1. masked input
            2. mask
            3. rendered image
            4. composed inpainting result
            5. GT

    English:
        Save a visualization grid:
            1. masked input
            2. mask
            3. rendered image
            4. composed inpainting result
            5. GT
    """
    mask_vis = mask.repeat(1, 3, 1, 1)
    grid = torch.cat([
        denorm(masked_image[:1]),
        mask_vis[:1].clamp(0, 1),
        denorm(pred_img[:1]),
        denorm(pred_comp[:1]),
        denorm(gt_img[:1]),
    ], dim=0)
    save_image(grid.cpu(), save_path, nrow=1)


def evaluate(system, val_loader, device, logger):
    """
    中文：
        验证阶段。
        当前保存的核心指标：
            - val_total
            - val_hole_l1
            - val_known_l1
            - val_psnr
            - val_ssim

    English:
        Validation stage.
        Current core metrics:
            - val_total
            - val_hole_l1
            - val_known_l1
            - val_psnr
            - val_ssim
    """
    system.eval()

    hole_meter = AverageMeter()
    known_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    total_meter = AverageMeter()

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            image = batch["image"]
            mask = batch["mask"]
            known_region = batch["known_region"]
            masked_image = batch["masked_image"]

            out = system(masked_image, mask)
            init_w_plus = out["init_w_plus"]
            pred_img = system.render(init_w_plus, image.shape[-2], image.shape[-1])
            pred_comp = system.compose_prediction(pred_img, image, mask, known_region)
            losses = system.compute_losses(
                init_w_plus=init_w_plus,
                pred_img=pred_img,
                gt_img=image,
                mask=mask,
                known_region=known_region,
            )

            hole_meter.update(losses["image_hole_l1"].item(), image.size(0))
            known_meter.update(losses["image_known_l1"].item(), image.size(0))
            total_meter.update(losses["total"].item(), image.size(0))

            psnr_value = compute_psnr(pred_comp[:1], image[:1])
            ssim_value = compute_ssim(pred_comp[:1], image[:1])
            psnr_meter.update(psnr_value, 1)
            ssim_meter.update(ssim_value, 1)

    metrics = {
        "val_total": total_meter.avg,
        "val_hole_l1": hole_meter.avg,
        "val_known_l1": known_meter.avg,
        "val_psnr": psnr_meter.avg,
        "val_ssim": ssim_meter.avg,
    }

    logger.info(
        f"[Val] total={metrics['val_total']:.6f} | "
        f"hole_l1={metrics['val_hole_l1']:.6f} | "
        f"known_l1={metrics['val_known_l1']:.6f} | "
        f"psnr={metrics['val_psnr']:.4f} | "
        f"ssim={metrics['val_ssim']:.4f}"
    )
    return metrics

def save_eval_samples(system, loader, device, save_dir, prefix, max_items=4):
    system.eval()
    os.makedirs(save_dir, exist_ok=True)

    saved = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            image = batch["image"]
            mask = batch["mask"]
            known_region = batch["known_region"]
            masked_image = batch["masked_image"]

            out = system(masked_image, mask)
            init_w_plus = out["init_w_plus"]
            pred_img = system.render(init_w_plus, image.shape[-2], image.shape[-1])
            pred_comp = system.compose_prediction(pred_img, image, mask, known_region)

            bs = image.size(0)
            for i in range(bs):
                save_vis(
                    os.path.join(save_dir, f"{prefix}_{saved:04d}.png"),
                    masked_image[i:i+1],
                    mask[i:i+1],
                    pred_img[i:i+1],
                    pred_comp[i:i+1],
                    image[i:i+1],
                )
                saved += 1
                if saved >= max_items:
                    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_exp_config(args.config)
    set_seed(cfg["project"]["seed"])

    device = torch.device(cfg["system"]["device"])
    torch.backends.cudnn.benchmark = bool(cfg["system"].get("cudnn_benchmark", True))

    exp_paths = prepare_exp_dir(cfg["paths"]["output_root"], cfg["project"]["exp_name"])
    logger = setup_logger(os.path.join(exp_paths["log_dir"], "train.log"))

    # 保存最终配置，确保实验可复现
    # Save final merged config for reproducibility.
    with open(os.path.join(exp_paths["exp_dir"], "config_final.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    logger.info("===== Start MCLA Pilot Training =====")
    logger.info(f"Experiment dir: {exp_paths['exp_dir']}")
    logger.info(f"Device: {device}")

    datamodule = build_datamodule(cfg)
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    system = MCLASystem(cfg, device).to(device)

    # 先拟合一次 generator latent manifold 统计量
    # Fit generator latent manifold stats once before training.
    logger.info("Fitting manifold statistics ...")
    system.fit_manifold_stats(
        num_samples=cfg["pilot"]["manifold_samples"],
        batch_size=cfg["pilot"]["manifold_batch_size"],
    )
    logger.info("Manifold statistics ready.")

    train_cfg = cfg["train"]
    optimizer = Adam(
        system.adapter.parameters(),
        lr=float(train_cfg["optimizer"]["lr"]),
        weight_decay=float(train_cfg["optimizer"]["weight_decay"]),
    )

    max_epochs = int(cfg["pilot"]["max_epochs"])
    print_freq = int(cfg["logging"]["print_freq"])
    save_image_freq = int(cfg["logging"]["save_image_freq"])
    val_freq = int(cfg["logging"]["val_freq"])

    monitor_key = cfg["checkpoint"]["monitor"]
    monitor_mode = cfg["checkpoint"]["mode"]
    best_metric = float("inf") if monitor_mode == "min" else -float("inf")

    global_step = 0
    for epoch in range(max_epochs):
        system.train()
        total_meter = AverageMeter()

        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            image = batch["image"]
            mask = batch["mask"]
            known_region = batch["known_region"]
            masked_image = batch["masked_image"]

            optimizer.zero_grad(set_to_none=True)

            out = system(masked_image, mask)
            init_w_plus = out["init_w_plus"]
            pred_img = system.render(init_w_plus, image.shape[-2], image.shape[-1])
            pred_comp = system.compose_prediction(pred_img, image, mask, known_region)

            losses = system.compute_losses(
                init_w_plus=init_w_plus,
                pred_img=pred_img,
                gt_img=image,
                mask=mask,
                known_region=known_region,
            )

            losses["total"].backward()
            optimizer.step()

            total_meter.update(losses["total"].item(), image.size(0))

            if global_step % print_freq == 0:
                logger.info(
                    f"[Train] epoch={epoch:03d} step={global_step:06d} | "
                    f"total={losses['total'].item():.6f} | "
                    f"known={losses['image_known_l1'].item():.6f} | "
                    f"hole={losses['image_hole_l1'].item():.6f} | "
                    f"perc={losses['perceptual'].item():.6f} | "
                    f"manifold={losses['manifold_reg'].item():.6f} | "
                    f"edge={losses['edge_hole_l1'].item():.6f} | "
                    f"freq={losses['freq_hole_l1'].item():.6f}"
                )

            if global_step % save_image_freq == 0:
                save_vis(
                    os.path.join(exp_paths["sample_dir"], f"step_{global_step:06d}.png"),
                    masked_image, mask, pred_img, pred_comp, image
                )

            global_step += 1

        logger.info(f"[Epoch] {epoch:03d} avg_train_total={total_meter.avg:.6f}")

        # latest ckpt
        latest_ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "adapter": system.adapter.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_metric": best_metric,
        }
        torch.save(latest_ckpt, os.path.join(exp_paths["ckpt_dir"], "latest.pt"))

        # validation
        if (epoch + 1) % val_freq == 0:
            metrics = evaluate(system, val_loader, device, logger)

            save_eval_samples(
                system,
                val_loader,
                device,
                os.path.join(exp_paths["sample_dir"], f"val_epoch_{epoch:04d}"),
                prefix="val",
                max_items=int(cfg["logging"].get("save_val_image_num", 4)),
            )

            with open(os.path.join(exp_paths["metric_dir"], f"val_epoch_{epoch:04d}.yaml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(metrics, f, allow_unicode=True, sort_keys=False)

            current_metric = metrics[monitor_key]
            is_better = current_metric < best_metric if monitor_mode == "min" else current_metric > best_metric

            if is_better:
                best_metric = current_metric
                best_ckpt = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "adapter": system.adapter.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_metric": best_metric,
                    "metrics": metrics,
                }
                torch.save(best_ckpt, os.path.join(exp_paths["ckpt_dir"], "best.pt"))
                logger.info(f"Best checkpoint updated: {monitor_key}={best_metric:.6f}")
    save_eval_samples(
        system,
        test_loader,
        device,
        os.path.join(exp_paths["sample_dir"], "test_final"),
        prefix="test",
        max_items=int(cfg["logging"].get("save_test_image_num", 8)),
    )
    logger.info("===== MCLA Pilot Training Finished =====")


if __name__ == "__main__":
    main()