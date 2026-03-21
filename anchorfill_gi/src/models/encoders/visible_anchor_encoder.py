# -*- coding: utf-8 -*-
"""
中文：
    visible_anchor_encoder.py
    轻量的可见区域编码器。
    输入为 masked_image 和 mask 的拼接，输出一个全局 visible feature。

English:
    visible_anchor_encoder.py
    Lightweight encoder for visible-region features.
    It takes the concatenation of masked_image and mask as input,
    and outputs a global visible feature vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    中文：
        基础卷积块：Conv -> BN -> LeakyReLU

    English:
        Basic convolution block: Conv -> BN -> LeakyReLU
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VisibleAnchorEncoder(nn.Module):
    """
    中文：
        从可见区域输入中提取一个全局特征。
        输入:
            [B, 4, H, W] = masked_image(3) + mask(1)
        输出:
            [B, feat_dim]

    English:
        Extract a global feature from visible-region input.
        Input:
            [B, 4, H, W] = masked_image(3) + mask(1)
        Output:
            [B, feat_dim]
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 64,
        max_channels: int = 512,
        num_down: int = 5,
        feat_dim: int = 1024,
    ):
        super().__init__()

        channels = [base_channels]
        for _ in range(num_down - 1):
            channels.append(min(channels[-1] * 2, max_channels))

        self.stem = ConvBlock(in_channels, channels[0], stride=1)

        blocks = []
        in_ch = channels[0]
        for out_ch in channels:
            blocks.append(ConvBlock(in_ch, out_ch, stride=2))
            in_ch = out_ch
        self.down_blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * 4 * 4, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        feat = self.down_blocks(feat)
        feat = self.pool(feat)
        feat = self.head(feat)
        return feat