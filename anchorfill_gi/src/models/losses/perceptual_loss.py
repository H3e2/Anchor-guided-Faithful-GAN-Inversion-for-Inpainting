# -*- coding: utf-8 -*-
"""
中文：
    perceptual_loss.py
    这个文件封装 LPIPS 感知损失。

English:
    perceptual_loss.py
    This file wraps LPIPS perceptual loss.
"""

from __future__ import annotations

import lpips
import torch
import torch.nn as nn


class LPIPSLoss(nn.Module):
    """
    中文：
        LPIPS 感知损失封装。
        输入图像默认应在 [-1, 1] 范围内。

    English:
        Wrapper for LPIPS perceptual loss.
        Input images are expected to be in [-1, 1].
    """

    def __init__(self, net: str = "alex"):
        super().__init__()
        self.loss_fn = lpips.LPIPS(net=net)

        # 冻结 LPIPS 参数 / Freeze LPIPS parameters
        for p in self.loss_fn.parameters():
            p.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        中文：
            计算整图 LPIPS 损失。

        English:
            Compute LPIPS loss on full images.
        """
        loss = self.loss_fn(pred, target)
        return loss.mean()