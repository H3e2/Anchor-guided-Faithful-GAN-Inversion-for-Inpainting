# -*- coding: utf-8 -*-
"""
中文：
    anchor_initializer.py
    Visible-Region Anchored Initializer 的核心模块。
    它根据可见区域特征预测：
        1. delta_w：对 w_avg 的偏移
        2. gate：分层门控
    最终得到：
        w0_plus = w_avg_plus + gate * delta_w

English:
    anchor_initializer.py
    Core module of the Visible-Region Anchored Initializer.
    It predicts:
        1. delta_w: latent offset from w_avg
        2. gate: layer-wise gate
    Final initialization:
        w0_plus = w_avg_plus + gate * delta_w
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.encoders.visible_anchor_encoder import VisibleAnchorEncoder


class AnchorInitializer(nn.Module):
    """
    中文：
        基于可见区域的 W+ 初始化器。

    English:
        Visible-region-based W+ initializer.
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 64,
        max_channels: int = 512,
        num_down: int = 5,
        feat_dim: int = 1024,
        num_ws: int = 18,
        w_dim: int = 512,
        delta_scale: float = 0.5,
        gate_mode: str = "scalar",
        w_avg: torch.Tensor | None = None,
    ):
        super().__init__()

        self.num_ws = num_ws
        self.w_dim = w_dim
        self.delta_scale = delta_scale
        self.gate_mode = gate_mode

        self.encoder = VisibleAnchorEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            max_channels=max_channels,
            num_down=num_down,
            feat_dim=feat_dim,
        )

        self.delta_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(feat_dim, num_ws * w_dim),
        )

        if gate_mode == "scalar":
            self.gate_head = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(feat_dim // 2, num_ws),
            )
        elif gate_mode == "channel":
            self.gate_head = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(feat_dim, num_ws * w_dim),
            )
        else:
            raise ValueError(f"Unsupported gate_mode: {gate_mode}")

        if w_avg is None:
            w_avg_plus = torch.zeros(num_ws, w_dim)
        else:
            # w_avg: [w_dim] -> [num_ws, w_dim]
            w_avg_plus = w_avg.detach().clone().unsqueeze(0).repeat(num_ws, 1)

        self.register_buffer("w_avg_plus", w_avg_plus)

    def forward(self, masked_image: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        中文：
            输入可见区域，输出 anchor 初始化结果。

        English:
            Take visible-region input and output anchored initialization.
        """
        x = torch.cat([masked_image, mask], dim=1)  # [B,4,H,W]
        feat = self.encoder(x)                      # [B,feat_dim]

        delta_w = self.delta_head(feat).view(-1, self.num_ws, self.w_dim)
        delta_w = torch.tanh(delta_w) * self.delta_scale

        if self.gate_mode == "scalar":
            gate = self.gate_head(feat).view(-1, self.num_ws, 1)
            gate = torch.sigmoid(gate)
        else:
            gate = self.gate_head(feat).view(-1, self.num_ws, self.w_dim)
            gate = torch.sigmoid(gate)

        init_w_plus = self.w_avg_plus.unsqueeze(0) + gate * delta_w

        return {
            "init_w_plus": init_w_plus,
            "delta_w": delta_w,
            "gate": gate,
            "visible_feat": feat,
        }