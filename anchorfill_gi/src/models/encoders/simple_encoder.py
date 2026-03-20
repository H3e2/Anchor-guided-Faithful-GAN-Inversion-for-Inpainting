# -*- coding: utf-8 -*-
"""
中文：
    simple_encoder.py
    这是第一版 baseline 使用的最简单 inversion encoder。
    它输入 masked image 与 mask 的拼接张量，输出 W+ latent code。

English:
    simple_encoder.py
    This is the simplest inversion encoder for the first baseline.
    It takes the concatenation of masked image and mask as input,
    and outputs a W+ latent code.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    中文：
        基础卷积块：Conv -> Norm -> LeakyReLU
        支持 stride=2 做下采样。

    English:
        Basic convolution block: Conv -> Norm -> LeakyReLU.
        It supports downsampling with stride=2.
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


class SimpleEncoder(nn.Module):
    """
    中文：
        第一版 baseline 的简单编码器。
        输入:
            [B, 4, H, W]，其中 4 通道 = masked_rgb(3) + mask(1)
        输出:
            [B, 18, 512]，表示 StyleGAN W+ latent

    English:
        A simple encoder for the first baseline.
        Input:
            [B, 4, H, W], where 4 channels = masked_rgb(3) + mask(1)
        Output:
            [B, 18, 512], representing StyleGAN W+ latent
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 64,
        max_channels: int = 512,
        num_down: int = 5,
        num_layers: int = 18,
        latent_dim: int = 512,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.out_dim = num_layers * latent_dim

        channels: List[int] = [base_channels]
        for _ in range(num_down - 1):
            channels.append(min(channels[-1] * 2, max_channels))

        # Stem / 初始特征提取层
        self.stem = ConvBlock(in_channels, channels[0], stride=1)

        # Down blocks / 下采样层
        down_blocks = []
        in_ch = channels[0]
        for out_ch in channels:
            down_blocks.append(ConvBlock(in_ch, out_ch, stride=2))
            in_ch = out_ch
        self.down_blocks = nn.Sequential(*down_blocks)

        # Head / 输出头
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * 4 * 4, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, self.out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        中文：
            前向传播，输出 W+ latent。

        English:
            Forward pass that outputs W+ latent.
        """
        feat = self.stem(x)
        feat = self.down_blocks(feat)
        feat = self.pool(feat)
        latent = self.head(feat)  # [B, 18*512]
        latent = latent.view(x.size(0), self.num_layers, self.latent_dim)
        return latent