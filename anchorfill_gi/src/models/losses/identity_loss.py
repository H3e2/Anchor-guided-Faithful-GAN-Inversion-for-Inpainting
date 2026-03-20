# -*- coding: utf-8 -*-
"""
中文：
    identity_loss.py
    当前阶段我们先提供一个占位版 identity loss。
    由于 ArcFace 还未正式接入，因此这里默认返回 0。
    这样做的目的是保证 baseline 能先跑通。

English:
    identity_loss.py
    At the current stage, we provide a placeholder identity loss.
    Since ArcFace is not integrated yet, this loss returns 0 by default.
    This is only to make the baseline runnable first.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class IdentityLoss(nn.Module):
    """
    中文：
        占位版身份损失。
        后续我们会替换为真正的 ArcFace / face recognition identity loss。

    English:
        Placeholder identity loss.
        It will be replaced later by a real ArcFace / face recognition loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        中文：
            当前阶段返回 0，作为占位符。

        English:
            Return 0 at the current stage as a placeholder.
        """
        return pred.new_tensor(0.0)