# -*- coding: utf-8 -*-
"""
中文：
    visible_preprocessor.py
    为 MCLA 构造多分支输入：
        1. visible RGB
        2. edge / gradient
        3. frequency (low/high)
        4. mask

English:
    visible_preprocessor.py
    Build multi-branch inputs for MCLA:
        1. visible RGB
        2. edge / gradient
        3. frequency (low/high)
        4. mask
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisiblePreprocessor(nn.Module):
    """
    中文：
        从 masked_image 和 mask 中提取多种输入分支。
        输入：
            masked_image: [B, 3, H, W], 范围通常在 [-1, 1]
            mask:         [B, 1, H, W], 1 表示 hole
        输出：
            dict，包含 visible / edge / low / high / mask 等分支

    English:
        Extract multiple branches from masked_image and mask.
        Input:
            masked_image: [B, 3, H, W], usually in [-1, 1]
            mask:         [B, 1, H, W], 1 means hole
        Output:
            A dict of visible / edge / low / high / mask branches.
    """

    def __init__(self, lowpass_kernel: int = 9):
        super().__init__()
        self.lowpass_kernel = lowpass_kernel

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0],
             [ 0.0,  0.0,  0.0],
             [ 1.0,  2.0,  1.0]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _rgb_to_gray(self, x: torch.Tensor) -> torch.Tensor:
        """
        中文：
            RGB 转灰度。

        English:
            Convert RGB to grayscale.
        """
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray

    def _compute_edge(self, x: torch.Tensor) -> torch.Tensor:
        """
        中文：
            计算 Sobel edge magnitude。

        English:
            Compute Sobel edge magnitude.
        """
        gray = self._rgb_to_gray(x)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        edge = torch.sqrt(gx * gx + gy * gy + 1e-6)
        return edge

    def _low_high_split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        中文：
            用平均池化近似低频，再得到高频残差。

        English:
            Approximate low-frequency with average pooling,
            then obtain high-frequency residual.
        """
        k = self.lowpass_kernel
        pad = k // 2
        low = F.avg_pool2d(x, kernel_size=k, stride=1, padding=pad)
        high = x - low
        return low, high

    def forward(self, masked_image: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        中文：
            生成多分支输入。
            known_region = 1 - mask

        English:
            Generate multi-branch inputs.
            known_region = 1 - mask
        """
        known_region = 1.0 - mask
        visible_rgb = masked_image * known_region

        edge = self._compute_edge(visible_rgb) * known_region
        low, high = self._low_high_split(visible_rgb)

        # 一些方便直接拼接使用的输入
        rgb_mask = torch.cat([visible_rgb, mask], dim=1)        # [B,4,H,W]
        edge_mask = torch.cat([edge, mask], dim=1)              # [B,2,H,W]
        freq_mask = torch.cat([low, high, mask], dim=1)         # [B,7,H,W]

        return {
            "visible_rgb": visible_rgb,
            "edge": edge,
            "low_freq": low,
            "high_freq": high,
            "mask": mask,
            "known_region": known_region,
            "rgb_mask": rgb_mask,
            "edge_mask": edge_mask,
            "freq_mask": freq_mask,
        }