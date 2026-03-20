# -*- coding: utf-8 -*-
"""
中文：
    boundary_loss.py
    定义边界带损失（Boundary-Band Loss）。
    该损失只在 mask 边界附近的一圈区域上计算，
    用于增强已知区域与缺失区域之间的过渡一致性。

English:
    boundary_loss.py
    Define the boundary-band loss.
    This loss is computed only on a narrow band around the mask boundary,
    and is used to improve transition consistency between known and hole regions.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BoundaryL1Loss(nn.Module):
    """
    中文：
        边界带 L1 损失。
        boundary_band 中：
            - 1 表示边界带区域
            - 0 表示非边界带区域

    English:
        Boundary-band L1 loss.
        In boundary_band:
            - 1 means boundary-band region
            - 0 means non-boundary region
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        boundary_band: torch.Tensor,
    ) -> torch.Tensor:
        """
        中文：
            在边界带区域上计算平均 L1 损失。

        English:
            Compute average L1 loss over the boundary band.
        """
        diff = torch.abs(pred - target)

        if boundary_band.size(1) == 1 and pred.size(1) > 1:
            boundary_band = boundary_band.expand(-1, pred.size(1), -1, -1)

        diff = diff * boundary_band
        denom = boundary_band.sum().clamp(min=self.eps)
        return diff.sum() / denom