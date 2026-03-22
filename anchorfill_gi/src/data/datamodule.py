# -*- coding: utf-8 -*-
"""
中文：
    datamodule.py
    统一的数据加载入口。
    这个版本解决了 3 个问题：
        1. 支持多种图片后缀：jpg / jpeg / png
        2. 支持动态 irregular mask
        3. 输出字段和你当前训练代码保持一致：
           image / mask / known_region / masked_image / boundary_band / path

English:
    datamodule.py
    Unified data loading entry.
    This version solves 3 issues:
        1. supports multiple image suffixes: jpg / jpeg / png
        2. supports dynamic irregular masks
        3. returns fields compatible with the current training code:
           image / mask / known_region / masked_image / boundary_band / path
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import torch
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Utility functions
# =========================================================
def collect_image_paths(image_dir: str):
    """
    中文：
        收集目录下所有支持的图片路径，并排序返回。

    English:
        Collect all supported image paths under a directory and return them in sorted order.
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    valid_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    paths = [p for p in image_dir.iterdir() if p.is_file() and p.suffix in valid_exts]
    paths = sorted(paths)

    if len(paths) == 0:
        raise RuntimeError(f"No images found in: {image_dir}")

    return paths


def pil_to_tensor_rgb(img: Image.Image) -> torch.Tensor:
    """
    中文：
        PIL RGB 图像转 tensor，并归一化到 [-1, 1]

    English:
        Convert a PIL RGB image to tensor and normalize it into [-1, 1].
    """
    arr = np.array(img).astype(np.float32) / 255.0  # [H,W,3]
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3,H,W]
    tensor = tensor * 2.0 - 1.0
    return tensor


def pil_mask_to_tensor(mask_img: Image.Image) -> torch.Tensor:
    """
    中文：
        PIL mask 图像转 tensor，范围 [0,1]，1 表示 hole。

    English:
        Convert a PIL mask image to tensor in [0,1], where 1 means hole.
    """
    arr = np.array(mask_img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]
    tensor = (tensor > 0.5).float()
    return tensor


def tensor_to_pil_mask(mask_tensor: torch.Tensor) -> Image.Image:
    """
    中文：
        将 [1,H,W] 的 mask tensor 转成 PIL 灰度图。

    English:
        Convert a [1,H,W] mask tensor into a PIL grayscale image.
    """
    arr = (mask_tensor.squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def compute_boundary_band(mask: torch.Tensor, radius: int = 3) -> torch.Tensor:
    """
    中文：
        计算 mask 的边界带。
        返回 [1,H,W]，1 表示边界附近区域。

    English:
        Compute the boundary band of a mask.
        Returns [1,H,W], where 1 indicates boundary-near regions.
    """
    # 用 max pool 近似膨胀，再减去原 mask
    # Approximate dilation with max-pooling, then subtract the original mask.
    mask_4d = mask.unsqueeze(0)  # [1,1,H,W]
    dilated = torch.nn.functional.max_pool2d(mask_4d, kernel_size=2 * radius + 1, stride=1, padding=radius)
    eroded = -torch.nn.functional.max_pool2d(-mask_4d, kernel_size=2 * radius + 1, stride=1, padding=radius)
    boundary = (dilated - eroded).clamp(0.0, 1.0)
    boundary = (boundary > 0.0).float()
    return boundary.squeeze(0)  # [1,H,W]


def random_irregular_mask(height: int, width: int, hole_range=(0.1, 0.5), max_tries: int = 40) -> Image.Image:
    """
    中文：
        生成动态 irregular mask。
        目标是让洞面积比例落在 hole_range 内。

    English:
        Generate a dynamic irregular mask.
        The target hole ratio should fall into hole_range.
    """
    min_ratio, max_ratio = hole_range

    for _ in range(max_tries):
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)

        # 随机画线条 / Random strokes
        num_strokes = random.randint(8, 20)
        for _ in range(num_strokes):
            x1 = random.randint(0, width - 1)
            y1 = random.randint(0, height - 1)
            x2 = random.randint(0, width - 1)
            y2 = random.randint(0, height - 1)
            thickness = random.randint(max(4, width // 64), max(12, width // 18))
            draw.line((x1, y1, x2, y2), fill=255, width=thickness)

        # 随机画圆 / Random circles
        num_circles = random.randint(4, 10)
        for _ in range(num_circles):
            r = random.randint(max(6, width // 40), max(18, width // 10))
            cx = random.randint(0, width - 1)
            cy = random.randint(0, height - 1)
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)

        # 随机画矩形 / Random rectangles
        num_rects = random.randint(2, 6)
        for _ in range(num_rects):
            x1 = random.randint(0, width - 1)
            y1 = random.randint(0, height - 1)
            x2 = random.randint(x1, min(width - 1, x1 + random.randint(width // 16, width // 3)))
            y2 = random.randint(y1, min(height - 1, y1 + random.randint(height // 16, height // 3)))
            draw.rectangle((x1, y1, x2, y2), fill=255)

        mask_np = np.array(mask).astype(np.float32) / 255.0
        ratio = mask_np.mean()

        if min_ratio <= ratio <= max_ratio:
            return mask

    # 如果多次尝试都没命中，返回最后一次 / Fallback to the last mask
    return mask


# =========================================================
# Dataset
# =========================================================
class InpaintingImageDataset(Dataset):
    """
    中文：
        通用缺块修复数据集。
        支持 train / val / test，支持动态 irregular mask。

    English:
        General image inpainting dataset.
        Supports train / val / test and dynamic irregular masks.
    """

    def __init__(
        self,
        image_dir: str,
        image_size: int = 256,
        hole_range=(0.1, 0.5),
        fixed_mask: bool = False,
        split: str = "train",
        overfit_num_samples: int = 0,
    ):
        super().__init__()

        self.image_paths = collect_image_paths(image_dir)
        self.image_size = image_size
        self.hole_range = hole_range
        self.fixed_mask = fixed_mask
        self.split = split

        # pilot / debug 场景下可裁小数据集
        # Support dataset truncation for pilot/debug stage.
        if overfit_num_samples is not None and overfit_num_samples > 0:
            self.image_paths = self.image_paths[:overfit_num_samples]

        self._fixed_masks = {}
        if self.fixed_mask:
            for idx in range(len(self.image_paths)):
                self._fixed_masks[idx] = random_irregular_mask(
                    self.image_size,
                    self.image_size,
                    hole_range=self.hole_range,
                )

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, path: Path) -> Image.Image:
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), resample=Image.BICUBIC)
        return img

    def _get_mask(self, idx: int) -> Image.Image:
        if self.fixed_mask:
            return self._fixed_masks[idx]
        return random_irregular_mask(
            self.image_size,
            self.image_size,
            hole_range=self.hole_range,
        )

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]

        img = self._load_image(path)
        mask_img = self._get_mask(idx)

        image = pil_to_tensor_rgb(img)              # [3,H,W], [-1,1]
        mask = pil_mask_to_tensor(mask_img)         # [1,H,W], {0,1}, 1=hole
        known_region = 1.0 - mask                   # [1,H,W]
        masked_image = image * known_region         # [3,H,W]
        boundary_band = compute_boundary_band(mask) # [1,H,W]

        return {
            "image": image,
            "mask": mask,
            "known_region": known_region,
            "masked_image": masked_image,
            "boundary_band": boundary_band,
            "path": str(path),
        }


# =========================================================
# DataModule-like wrapper
# =========================================================
@dataclass
class SimpleDataModule:
    """
    中文：
        一个轻量 datamodule 风格封装，和你当前 build_datamodule 的调用习惯兼容。

    English:
        A lightweight datamodule-style wrapper compatible with the current build_datamodule usage.
    """
    cfg: dict

    def __post_init__(self):
        data_cfg = self.cfg["data"]
        debug_cfg = self.cfg.get("debug", {})

        image_size = int(data_cfg["image_size"])
        hole_range = tuple(data_cfg["mask"]["hole_range"])
        fixed_mask = bool(data_cfg["mask"].get("fixed_mask", False))

        overfit_num_samples = int(debug_cfg.get("overfit_num_samples", 0)) if bool(debug_cfg.get("enabled", False)) else 0

        self.train_dataset = InpaintingImageDataset(
            image_dir=data_cfg["train"]["image_dir"],
            image_size=image_size,
            hole_range=hole_range,
            fixed_mask=fixed_mask,
            split="train",
            overfit_num_samples=overfit_num_samples,
        )

        self.val_dataset = InpaintingImageDataset(
            image_dir=data_cfg["val"]["image_dir"],
            image_size=image_size,
            hole_range=hole_range,
            fixed_mask=fixed_mask,
            split="val",
            overfit_num_samples=overfit_num_samples,
        )

        self.test_dataset = InpaintingImageDataset(
            image_dir=data_cfg["test"]["image_dir"],
            image_size=image_size,
            hole_range=hole_range,
            fixed_mask=fixed_mask,
            split="test",
            overfit_num_samples=overfit_num_samples,
        )

        self.loader_cfg = data_cfg["loader"]
        self.data_cfg = data_cfg

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.data_cfg["train"]["batch_size"]),
            shuffle=bool(self.data_cfg["train"]["shuffle"]),
            num_workers=int(self.loader_cfg["num_workers"]),
            pin_memory=bool(self.loader_cfg["pin_memory"]),
            drop_last=bool(self.loader_cfg["drop_last"]),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.data_cfg["val"]["batch_size"]),
            shuffle=bool(self.data_cfg["val"]["shuffle"]),
            num_workers=int(self.loader_cfg["num_workers"]),
            pin_memory=bool(self.loader_cfg["pin_memory"]),
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=int(self.data_cfg["test"]["batch_size"]),
            shuffle=bool(self.data_cfg["test"]["shuffle"]),
            num_workers=int(self.loader_cfg["num_workers"]),
            pin_memory=bool(self.loader_cfg["pin_memory"]),
            drop_last=False,
        )


def build_datamodule(cfg: dict) -> SimpleDataModule:
    """
    中文：
        构建 datamodule 风格对象。

    English:
        Build a datamodule-style object.
    """
    return SimpleDataModule(cfg)