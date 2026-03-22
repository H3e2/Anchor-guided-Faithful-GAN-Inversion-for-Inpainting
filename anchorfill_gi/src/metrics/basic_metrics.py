# -*- coding: utf-8 -*-
"""
中文：
    basic_metrics.py
    Pilot 阶段的基础指标：
        - PSNR
        - SSIM
        - LPIPS（可选，在外部传入）

English:
    basic_metrics.py
    Basic metrics for the pilot stage:
        - PSNR
        - SSIM
        - LPIPS (optional, externally provided)
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from skimage.metrics import structural_similarity as ssim_fn


def denorm_to_01(x: torch.Tensor) -> torch.Tensor:
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
        计算 PSNR，输入范围假定在 [-1,1] 或 [0,1] 均可。

    English:
        Compute PSNR. Input can be in [-1,1] or [0,1].
    """
    pred_01 = denorm_to_01(pred)
    target_01 = denorm_to_01(target)

    mse = F.mse_loss(pred_01, target_01).item()
    if mse < eps:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    中文：
        计算单张图像 batch 的 SSIM，默认 batch=1。

    English:
        Compute SSIM for a single-image batch. Assumes batch=1.
    """
    pred_01 = denorm_to_01(pred).detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    target_01 = denorm_to_01(target).detach().cpu().squeeze(0).permute(1, 2, 0).numpy()

    score = ssim_fn(pred_01, target_01, channel_axis=2, data_range=1.0)
    return float(score)