# -*- coding: utf-8 -*-
"""
中文：
    manifold_loss.py
    用于约束预测的 W+ 落在当前预训练 generator 的 style manifold 附近。

English:
    manifold_loss.py
    Constrain predicted W+ to stay close to the style manifold
    of the current pretrained generator.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GroupwiseManifoldLoss(nn.Module):
    """
    中文：
        对 grouped W+ 做对角协方差近似下的流形约束。
        当前使用简单的按维标准化平方距离：
            ((w - mean) / std)^2

    English:
        Group-wise manifold regularization for W+ using a diagonal-covariance approximation.
        Current formulation uses normalized squared distance:
            ((w - mean) / std)^2
    """

    def __init__(
        self,
        num_ws: int = 18,
        w_dim: int = 512,
        layer_splits: list[int] | tuple[int, int, int] = (4, 6, 8),
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_ws = num_ws
        self.w_dim = w_dim
        self.layer_splits = list(layer_splits)
        self.eps = eps

        # 中文：W+ 的均值与标准差统计量
        # English: mean and std statistics of W+
        self.register_buffer("mean_wplus", torch.zeros(num_ws, w_dim))
        self.register_buffer("std_wplus", torch.ones(num_ws, w_dim))

    @torch.no_grad()
    def set_stats(self, mean_wplus: torch.Tensor, std_wplus: torch.Tensor):
        """
        中文：
            手动设置流形统计量。

        English:
            Manually set manifold statistics.
        """
        assert mean_wplus.shape == (self.num_ws, self.w_dim)
        assert std_wplus.shape == (self.num_ws, self.w_dim)
        self.mean_wplus.copy_(mean_wplus)
        self.std_wplus.copy_(std_wplus)

    @torch.no_grad()
    def fit_from_samples(self, wplus_samples: torch.Tensor):
        """
        中文：
            从采样得到的 W+ 样本估计均值和标准差。
            输入：
                [N, num_ws, w_dim]

        English:
            Estimate mean and std from sampled W+ codes.
            Input:
                [N, num_ws, w_dim]
        """
        assert wplus_samples.dim() == 3
        assert wplus_samples.shape[1] == self.num_ws
        assert wplus_samples.shape[2] == self.w_dim

        mean = wplus_samples.mean(dim=0)
        std = wplus_samples.std(dim=0, unbiased=False).clamp_min(self.eps)

        self.mean_wplus.copy_(mean)
        self.std_wplus.copy_(std)

    def forward(self, pred_wplus: torch.Tensor) -> torch.Tensor:
        """
        中文：
            计算 W+ 的流形正则损失。
            输入：
                pred_wplus: [B, num_ws, w_dim]

        English:
            Compute manifold regularization loss for W+.
            Input:
                pred_wplus: [B, num_ws, w_dim]
        """
        z = (pred_wplus - self.mean_wplus.unsqueeze(0)) / (self.std_wplus.unsqueeze(0) + self.eps)

        losses = []
        start = 0
        for n_layers in self.layer_splits:
            end = start + n_layers
            group_z = z[:, start:end, :]
            group_loss = (group_z ** 2).mean()
            losses.append(group_loss)
            start = end

        return sum(losses) / len(losses)


@torch.no_grad()
def sample_wplus_stats(
    generator_wrapper,
    num_samples: int = 2048,
    batch_size: int = 16,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    中文：
        从当前预训练 generator 的 mapping 网络中采样 W+ 统计量。
        返回：
            mean_wplus: [num_ws, w_dim]
            std_wplus : [num_ws, w_dim]

    English:
        Sample W+ statistics from the current pretrained generator mapping network.
        Returns:
            mean_wplus: [num_ws, w_dim]
            std_wplus : [num_ws, w_dim]
    """
    ws_all = []

    total = 0
    while total < num_samples:
        cur_bs = min(batch_size, num_samples - total)
        z = torch.randn(cur_bs, generator_wrapper.z_dim, device=device)
        ws = generator_wrapper.mapping(z)  # [B, num_ws, w_dim]
        ws_all.append(ws.detach())
        total += cur_bs

    ws_all = torch.cat(ws_all, dim=0)  # [N, num_ws, w_dim]
    mean = ws_all.mean(dim=0)
    std = ws_all.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std