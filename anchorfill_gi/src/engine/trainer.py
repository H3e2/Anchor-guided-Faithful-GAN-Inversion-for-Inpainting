# -*- coding: utf-8 -*-
"""
中文：
    trainer.py
    baseline 训练器。
    当前支持：
        1. 单卡训练
        2. AMP 混合精度
        3. TensorBoard 日志
        4. checkpoint 保存
        5. 样例图保存

English:
    trainer.py
    Trainer for the baseline model.
    Current features:
        1. single-GPU training
        2. AMP mixed precision
        3. TensorBoard logging
        4. checkpoint saving
        5. sample image saving
"""

from __future__ import annotations

import os
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.utils import save_image

from src.engine.logger import build_logger, build_writer, log_scalar_dict
from src.engine.checkpoint import save_checkpoint, load_checkpoint


class Trainer:
    """
    中文：
        baseline 训练器。

    English:
        Trainer for the baseline model.
    """

    def __init__(self, cfg: Dict[str, Any], model: nn.Module, datamodule):
        self.cfg = cfg
        self.model = model
        self.datamodule = datamodule

        self.device = torch.device(cfg["system"]["device"])
        self.output_dir = os.path.join(cfg["paths"]["output_root"], cfg["project"]["exp_name"])
        self.ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        self.sample_dir = os.path.join(self.output_dir, "samples")
        self.log_dir = os.path.join(self.output_dir, "logs")

        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.logger = build_logger(self.log_dir)
        self.writer = build_writer(self.log_dir)

        self.model.to(self.device)

        train_cfg = cfg["train"]
        opt_cfg = train_cfg["optimizer"]

        self.optimizer = Adam(
            self.model.parameters(),
            lr=opt_cfg["lr"],
            betas=tuple(opt_cfg["betas"]),
            weight_decay=opt_cfg["weight_decay"],
        )

        # AMP scaler / AMP 梯度缩放器
        self.use_amp = bool(train_cfg.get("amp", True))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.epochs = int(train_cfg["epochs"])
        self.global_step = 0
        self.start_epoch = 0

        resume_path = train_cfg.get("resume", None)
        if resume_path:
            self.logger.info(f"Loading checkpoint from: {resume_path}")
            self.start_epoch, self.global_step, _ = load_checkpoint(
                resume_path,
                self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                map_location=self.device,
            )

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        中文：
            将 batch 中的张量搬到 GPU。

        English:
            Move tensor items in a batch to GPU.
        """
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    def _to_float_dict(self, loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        中文：
            将 loss 字典转成 Python float，便于日志记录。

        English:
            Convert a tensor loss dict into Python floats for logging.
        """
        result = {}
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                result[k] = float(v.detach().cpu().item())
            else:
                result[k] = float(v)
        return result

    def _save_samples(self, batch: Dict[str, Any], outputs: Dict[str, Any], epoch: int, phase: str = "train"):
        """
        中文：
            保存更直观的样例图像，便于观察模型行为。
            每个样本会展示：
                1. masked_image   遮挡输入图
                2. mask           掩码可视化
                3. boundary_band  边界带可视化
                4. pred_full      生成器直接输出整图
                5. pred_comp      最终合成修复图
                6. gt             原图

        English:
            Save more informative sample images for visual inspection.
            For each sample, we show:
                1. masked_image   masked input
                2. mask           mask visualization
                3. boundary_band  boundary-band visualization
                4. pred_full      raw generator output
                5. pred_comp      final composited inpainting result
                6. gt             ground-truth image
        """
        image = batch["image"]                  # GT / ground truth
        masked_image = batch["masked_image"]    # 输入图 / masked input
        mask = batch["mask"]                    # [B,1,H,W], 1=hole
        boundary_band = batch["boundary_band"]  # [B,1,H,W]
        pred_full = outputs["pred_full"]        # 生成器整图输出
        pred_comp = outputs["pred_comp"]        # 合成后的最终图

        num_save = min(self.cfg["logging"]["image_save_num"], image.size(0))

        def denorm(x: torch.Tensor) -> torch.Tensor:
            """
            中文：
                将 [-1, 1] 范围的图像反归一化到 [0, 1]。

            English:
                Convert images from [-1, 1] back to [0, 1].
            """
            return (x.clamp(-1, 1) + 1.0) * 0.5

        def vis_mask(x: torch.Tensor) -> torch.Tensor:
            """
            中文：
                将单通道 mask/boundary band 扩展为三通道，便于保存可视化。

            English:
                Expand a single-channel mask/boundary band into 3 channels for visualization.
            """
            if x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1)
            return x.clamp(0, 1)

        # 图像类 / image-like tensors -> 反归一化到 [0,1]
        masked_vis = denorm(masked_image[:num_save])
        pred_full_vis = denorm(pred_full[:num_save])
        pred_comp_vis = denorm(pred_comp[:num_save])
        gt_vis = denorm(image[:num_save])

        # mask 类 / mask-like tensors -> 扩成三通道
        mask_vis = vis_mask(mask[:num_save])
        boundary_vis = vis_mask(boundary_band[:num_save])

        # 按“样本维度拼接”成一个大的 grid
        # Concatenate along batch dimension to build a grid.
        grid = torch.cat([
            masked_vis,
            mask_vis,
            boundary_vis,
            pred_full_vis,
            pred_comp_vis,
            gt_vis,
        ], dim=0)

        save_path = os.path.join(self.sample_dir, f"{phase}_epoch_{epoch:04d}.png")
        save_image(grid.cpu(), save_path, nrow=num_save)

    def train_one_epoch(self, epoch: int):
        """
        中文：
            执行一个 epoch 的训练。

        English:
            Run one training epoch.
        """
        self.model.train()
        train_loader = self.datamodule.train_dataloader()

        running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            batch = self._move_batch_to_device(batch)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                result = self.model.training_step(batch)
                loss = result["loss_total"]

            self.scaler.scale(loss).backward()

            grad_cfg = self.cfg["train"]["grad_clip"]
            if grad_cfg["enabled"]:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=grad_cfg["max_norm"],
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_value = float(loss.detach().cpu().item())
            running_loss += loss_value
            self.global_step += 1

            if batch_idx % self.cfg["logging"]["log_interval"] == 0:
                loss_dict = {
                    k: v for k, v in result.items()
                    if k.startswith("loss_")
                }
                scalar_dict = self._to_float_dict(loss_dict)

                self.logger.info(
                    f"[Train] Epoch {epoch:03d} | Iter {batch_idx:04d}/{len(train_loader):04d} | "
                    + " | ".join([f"{k}={v:.4f}" for k, v in scalar_dict.items()])
                )
                log_scalar_dict(self.writer, "train", scalar_dict, self.global_step)

        avg_loss = running_loss / max(len(train_loader), 1)
        self.logger.info(f"[Train] Epoch {epoch:03d} finished. avg_loss={avg_loss:.6f}")

    @torch.no_grad()
    def validate(self, epoch: int):
        """
        中文：
            执行一个 epoch 的验证。

        English:
            Run one validation epoch.
        """
        self.model.eval()
        val_loader = self.datamodule.val_dataloader()

        total = 0.0
        count = 0
        first_batch_result = None
        first_batch = None

        for batch_idx, batch in enumerate(val_loader):
            batch = self._move_batch_to_device(batch)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model.forward(batch)
                losses = self.model.compute_losses(batch, outputs)

            total += float(losses["loss_total"].detach().cpu().item())
            count += 1

            if first_batch_result is None:
                first_batch_result = outputs
                first_batch = batch

        avg_val_loss = total / max(count, 1)

        self.logger.info(f"[Val] Epoch {epoch:03d} | loss_total={avg_val_loss:.6f}")
        self.writer.add_scalar("val/loss_total", avg_val_loss, epoch)

        if self.cfg["logging"]["save_images"] and first_batch_result is not None:
            self._save_samples(first_batch, first_batch_result, epoch, phase="val")

        return avg_val_loss

    def fit(self):
        """
        中文：
            启动完整训练流程。

        English:
            Launch the full training loop.
        """
        self.logger.info("===== Start Training =====")
        self.logger.info(f"Experiment: {self.cfg['project']['exp_name']}")
        self.logger.info(f"Device: {self.device}")

        for epoch in range(self.start_epoch, self.epochs):
            self.train_one_epoch(epoch)

            do_val = ((epoch + 1) % self.cfg["logging"]["val_interval"] == 0)
            if do_val:
                self.validate(epoch)

            do_save = ((epoch + 1) % self.cfg["logging"]["save_interval"] == 0)
            if do_save:
                ckpt_path = os.path.join(self.ckpt_dir, f"epoch_{epoch:04d}.pt")
                save_checkpoint(
                    save_path=ckpt_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    scaler=self.scaler,
                    epoch=epoch,
                    global_step=self.global_step,
                    extra={"exp_name": self.cfg["project"]["exp_name"]},
                )
                self.logger.info(f"Checkpoint saved to: {ckpt_path}")

        self.logger.info("===== Training Finished =====")