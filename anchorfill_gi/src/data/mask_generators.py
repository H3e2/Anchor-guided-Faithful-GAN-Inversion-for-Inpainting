# -*- coding: utf-8 -*-
"""
中文：
    mask_generators.py
    这个文件负责生成图像修复任务中使用的随机 mask（掩码）。
    当前先实现最常用的 irregular/free-form mask，并提供边界带（boundary band）计算函数。

English:
    mask_generators.py
    This file provides random mask generators for image inpainting.
    Currently, we implement the commonly used irregular/free-form mask generator
    and a utility to compute the boundary band of a mask.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class RandomIrregularMaskConfig:
    """
    中文：
        随机不规则掩码配置。
        用于控制笔画数、线宽、顶点数以及目标空洞面积比例。

    English:
        Configuration for random irregular masks.
        It controls stroke count, brush width, vertex count, and target hole ratio.
    """
    hole_range: Tuple[float, float] = (0.4, 0.6)
    max_strokes: int = 10
    max_vertex: int = 20
    min_brush_width: int = 12
    max_brush_width: int = 40
    max_tries: int = 20


class RandomIrregularMaskGenerator:
    """
    中文：
        随机不规则掩码生成器。
        生成的 mask 约定：
            - 1 表示缺失区域（hole / masked region）
            - 0 表示已知区域（known / visible region）

    English:
        Random irregular mask generator.
        Mask convention:
            - 1 means hole / masked region
            - 0 means known / visible region
    """

    def __init__(self, config: RandomIrregularMaskConfig):
        self.config = config

    def _draw_single_mask(self, height: int, width: int) -> np.ndarray:
        """
        中文：
            生成一张单独的不规则掩码（numpy 格式）。
            返回 shape 为 [H, W]，取值为 0/1。

        English:
            Generate one irregular mask in numpy format.
            Returns a [H, W] binary mask with values in {0, 1}.
        """
        mask = np.zeros((height, width), dtype=np.uint8)

        num_strokes = random.randint(1, self.config.max_strokes)
        avg_radius = math.sqrt(height * height + width * width) / 8.0

        for _ in range(num_strokes):
            num_vertex = random.randint(1, self.config.max_vertex)
            start_x = random.randint(0, width - 1)
            start_y = random.randint(0, height - 1)

            brush_width = random.randint(
                self.config.min_brush_width,
                self.config.max_brush_width
            )

            points = [(start_x, start_y)]
            angle = random.uniform(0, 2 * math.pi)

            for _ in range(num_vertex):
                angle += random.uniform(-math.pi / 4, math.pi / 4)
                length = max(10, int(abs(random.gauss(avg_radius, avg_radius / 2))))
                next_x = int(points[-1][0] + length * math.cos(angle))
                next_y = int(points[-1][1] + length * math.sin(angle))

                next_x = np.clip(next_x, 0, width - 1)
                next_y = np.clip(next_y, 0, height - 1)
                points.append((next_x, next_y))

            # 逐段画线 / Draw line segments
            for i in range(len(points) - 1):
                cv2.line(mask, points[i], points[i + 1], color=1, thickness=brush_width)

            # 在转折点画圆使边缘更自然 / Draw circles at vertices for smoother masks
            for p in points:
                cv2.circle(mask, p, radius=brush_width // 2, color=1, thickness=-1)

        return mask

    def generate(self, height: int, width: int) -> torch.Tensor:
        """
        中文：
            生成满足目标空洞比例范围的随机 mask。
            返回 shape 为 [1, H, W] 的 torch.float32 tensor。

        English:
            Generate a random mask whose hole ratio falls into the target range.
            Returns a torch.float32 tensor with shape [1, H, W].
        """
        min_ratio, max_ratio = self.config.hole_range

        best_mask = None
        best_diff = float("inf")

        for _ in range(self.config.max_tries):
            mask = self._draw_single_mask(height, width)
            ratio = float(mask.mean())

            # 如果落在目标范围内，直接返回
            # If the hole ratio falls in the desired range, return immediately.
            if min_ratio <= ratio <= max_ratio:
                return torch.from_numpy(mask).float().unsqueeze(0)

            # 否则记录一个最接近目标范围中心的 mask
            # Otherwise, keep the closest one as fallback.
            target_mid = 0.5 * (min_ratio + max_ratio)
            diff = abs(ratio - target_mid)
            if diff < best_diff:
                best_diff = diff
                best_mask = mask

        return torch.from_numpy(best_mask).float().unsqueeze(0)


def compute_boundary_band(mask: torch.Tensor, band_width: int = 7) -> torch.Tensor:
    """
    中文：
        根据二值 mask 计算边界带（boundary band）。
        这里的 boundary band 是通过对 mask 做膨胀和腐蚀后求差得到的。

        输入:
            mask: [1, H, W] 或 [B, 1, H, W]，其中 1 表示缺失区域
            band_width: 边界带宽度

        输出:
            与输入同 batch 维度的 boundary band，值为 0/1

    English:
        Compute the boundary band from a binary mask.
        The boundary band is obtained by subtracting eroded mask from dilated mask.

        Input:
            mask: [1, H, W] or [B, 1, H, W], where 1 denotes hole region
            band_width: width of the boundary band

        Output:
            boundary band tensor with values in {0, 1}
    """
    squeeze_back = False

    if mask.dim() == 3:
        mask = mask.unsqueeze(0)  # [1, 1, H, W]
        squeeze_back = True

    if mask.dim() != 4 or mask.size(1) != 1:
        raise ValueError(f"mask must be [1,H,W] or [B,1,H,W], but got {tuple(mask.shape)}")

    kernel_size = 2 * band_width + 1

    # 膨胀 / dilation
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=band_width)

    # 腐蚀 / erosion
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=band_width)

    band = (dilated - eroded).clamp(0.0, 1.0)

    if squeeze_back:
        band = band.squeeze(0)

    return band


def build_mask_generator(mask_cfg: Dict[str, Any]):
    """
    中文：
        根据配置构建 mask generator。
        当前先支持 irregular mask，后续可扩展 bbox / segmentation / external masks。

    English:
        Build a mask generator from config.
        Currently supports irregular masks, and can be extended later.
    """
    mask_type = mask_cfg.get("type", "irregular").lower()

    if mask_type == "irregular":
        config = RandomIrregularMaskConfig(
            hole_range=tuple(mask_cfg.get("hole_range", [0.4, 0.6])),
            max_strokes=mask_cfg.get("max_strokes", 10),
            max_vertex=mask_cfg.get("max_vertex", 20),
            min_brush_width=mask_cfg.get("min_brush_width", 12),
            max_brush_width=mask_cfg.get("max_brush_width", 40),
            max_tries=mask_cfg.get("max_tries", 20),
        )
        return RandomIrregularMaskGenerator(config)

    raise NotImplementedError(f"Unsupported mask type: {mask_type}")