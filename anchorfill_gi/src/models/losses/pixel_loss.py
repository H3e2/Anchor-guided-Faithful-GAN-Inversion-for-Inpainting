# -*- coding: utf-8 -*-
"""
中文：
    pixel_loss.py
    这个文件定义像素级损失函数，主要是 masked L1 loss。

English:
    pixel_loss.py
    This file defines pixel-level losses, mainly masked L1 loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MaskedL1Loss(nn.Module):
    """
    中文：
        带掩码的 L1 损失。
        region_mask 中：
            - 1 表示参与损失计算的区域
            - 0 表示忽略区域

    English:
        Masked L1 loss.
        In region_mask:
            - 1 means active region for loss computation
            - 0 means ignored region
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, region_mask: torch.Tensor) -> torch.Tensor:
        """
        中文：
            计算给定区域内的平均 L1 损失。

        English:
            Compute the average L1 loss over the active region.
        """
        diff = torch.abs(pred - target)

        # 广播到图像通道 / Broadcast mask to image channels
        if region_mask.size(1) == 1 and pred.size(1) > 1:
            region_mask = region_mask.expand(-1, pred.size(1), -1, -1)

        diff = diff * region_mask
        denom = region_mask.sum().clamp(min=self.eps)
        return diff.sum() / denom