# -*- coding: utf-8 -*-
"""
中文：
    loss_manager.py
    统一管理 baseline 所有损失项。

English:
    loss_manager.py
    Unified loss manager for the baseline model.
"""

from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn as nn

from src.models.losses.pixel_loss import MaskedL1Loss
from src.models.losses.perceptual_loss import LPIPSLoss
from src.models.losses.identity_loss import IdentityLoss
from src.models.losses.boundary_loss import BoundaryL1Loss


class BaselineLossManager(nn.Module):
    """
    中文：
        baseline 模型的损失管理器。
        当前支持：
            - known region L1
            - hole region L1
            - perceptual loss
            - identity loss（当前为占位版）

    English:
        Loss manager for the baseline model.
        Currently supports:
            - known-region L1
            - hole-region L1
            - perceptual loss
            - identity loss (currently a placeholder)
    """

    def __init__(self, loss_cfg: Dict[str, Any]):
        super().__init__()
        self.loss_cfg = loss_cfg

        self.l1_loss = MaskedL1Loss()
        self.boundary_loss = BoundaryL1Loss()

        if loss_cfg["perceptual"]["enabled"]:
            self.perceptual_loss = LPIPSLoss(net=loss_cfg["perceptual"].get("net", "alex"))
        else:
            self.perceptual_loss = None

        if loss_cfg["identity"]["enabled"]:
            self.identity_loss = IdentityLoss()
        else:
            self.identity_loss = None

    def forward(
        self,
        pred_full: torch.Tensor,
        pred_comp: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor,
        known_region: torch.Tensor,
        boundary_band: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        中文：
            根据预测结果和 GT 计算所有损失项。

        English:
            Compute all losses from model predictions and GT.
        """
        losses: Dict[str, torch.Tensor] = {}

        total_loss = gt.new_tensor(0.0)

        # Known region L1 / 已知区域 L1
        if self.loss_cfg["known_l1"]["enabled"]:
            loss_known = self.l1_loss(pred_full, gt, known_region)
            weight = float(self.loss_cfg["known_l1"]["weight"])
            losses["loss_known_l1"] = loss_known
            total_loss = total_loss + weight * loss_known

        # Hole region L1 / 缺失区域 L1
        if self.loss_cfg["hole_l1"]["enabled"]:
            loss_hole = self.l1_loss(pred_full, gt, mask)
            weight = float(self.loss_cfg["hole_l1"]["weight"])
            losses["loss_hole_l1"] = loss_hole
            total_loss = total_loss + weight * loss_hole

        # Boundary-band L1 / 边界带 L1
        if self.loss_cfg.get("boundary_l1", {}).get("enabled", False):
            loss_boundary = self.boundary_loss(pred_full, gt, boundary_band)
            weight = float(self.loss_cfg["boundary_l1"]["weight"])
            losses["loss_boundary_l1"] = loss_boundary
            total_loss = total_loss + weight * loss_boundary

        # Perceptual loss / 感知损失
        if self.perceptual_loss is not None:
            loss_perc = self.perceptual_loss(pred_comp, gt)
            weight = float(self.loss_cfg["perceptual"]["weight"])
            losses["loss_perceptual"] = loss_perc
            total_loss = total_loss + weight * loss_perc

        # Identity loss / 身份损失（当前占位）
        if self.identity_loss is not None:
            loss_id = self.identity_loss(pred_comp, gt)
            weight = float(self.loss_cfg["identity"]["weight"])
            losses["loss_identity"] = loss_id
            total_loss = total_loss + weight * loss_id

        losses["loss_total"] = total_loss
        return losses