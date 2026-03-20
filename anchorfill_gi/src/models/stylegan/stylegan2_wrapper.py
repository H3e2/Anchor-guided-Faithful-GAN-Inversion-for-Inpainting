# -*- coding: utf-8 -*-
"""
中文：
    stylegan2_wrapper.py
    这是 StyleGAN2 生成器包装器。
    当前阶段先提供一个 mock backend，用于打通整个训练链路。
    后续我们会把真实的 StyleGAN2/StyleGAN2-ADA 生成器接进来。

English:
    stylegan2_wrapper.py
    This is a wrapper for StyleGAN2 generator.
    At the current stage, we only provide a mock backend to make the whole
    training pipeline runnable.
    Later, we will replace it with a real StyleGAN2 / StyleGAN2-ADA backend.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class MockStyleBlock(nn.Module):
    """
    中文：
        mock generator 的上采样卷积块。
        用 nearest upsample + conv 的方式逐步恢复空间分辨率。

    English:
        Upsampling block for the mock generator.
        It progressively restores spatial resolution using nearest upsampling + conv.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MockStyleGAN2Generator(nn.Module):
    """
    中文：
        一个简化版 mock generator。
        作用不是复现真实 StyleGAN2，而是：
            1. 接受 [B, 18, 512] 的 W+ latent
            2. 输出 [B, 3, H, W] 图像
            3. 让训练链路先跑通

    English:
        A simplified mock generator.
        Its goal is NOT to reproduce real StyleGAN2, but to:
            1. accept [B, 18, 512] W+ latent
            2. output [B, 3, H, W] images
            3. make the training pipeline runnable first
    """

    def __init__(self, image_size: int = 256, num_layers: int = 18, latent_dim: int = 512):
        super().__init__()

        self.image_size = image_size
        self.num_layers = num_layers
        self.latent_dim = latent_dim

        # 先把 18x512 的 latent 聚合成一个全局向量
        # First aggregate the 18x512 latent into one global vector.
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_layers * latent_dim, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, 512 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 根据输出分辨率构建若干上采样层
        # Build enough upsampling stages according to output resolution.
        num_upsamples = int(math.log2(image_size) - 2)  # 4 -> 8 -> ... -> image_size
        channels = [512, 256, 128, 64, 32, 16]

        blocks = []
        in_ch = 512
        for i in range(num_upsamples):
            out_ch = channels[min(i, len(channels) - 1)]
            blocks.append(MockStyleBlock(in_ch, out_ch))
            in_ch = out_ch

        self.blocks = nn.Sequential(*blocks)
        self.to_rgb = nn.Conv2d(in_ch, 3, kernel_size=1)

    def forward(self, w_plus: torch.Tensor) -> torch.Tensor:
        """
        中文：
            输入 W+ latent，输出归一化到 [-1, 1] 的 RGB 图像。

        English:
            Take W+ latent as input and output an RGB image normalized to [-1, 1].
        """
        x = self.proj(w_plus).view(w_plus.size(0), 512, 4, 4)
        x = self.blocks(x)
        x = self.to_rgb(x)
        x = torch.tanh(x)
        return x


class StyleGAN2Wrapper(nn.Module):
    """
    中文：
        StyleGAN2 包装器。
        当前支持：
            - mock backend

        后续计划支持：
            - real stylegan2-ada-pytorch backend

    English:
        Wrapper for StyleGAN2.
        Currently supports:
            - mock backend

        Planned future support:
            - real stylegan2-ada-pytorch backend
    """

    def __init__(
        self,
        backend: str = "mock",
        image_size: int = 256,
        num_layers: int = 18,
        latent_dim: int = 512,
        checkpoint: Optional[str] = None,
        freeze_generator: bool = False,
    ):
        super().__init__()

        self.backend = backend.lower()

        if self.backend == "mock":
            self.generator = MockStyleGAN2Generator(
                image_size=image_size,
                num_layers=num_layers,
                latent_dim=latent_dim,
            )
        else:
            raise NotImplementedError(
                f"Backend '{backend}' is not implemented yet. "
                "For now, please use backend='mock'."
            )

        if freeze_generator:
            for p in self.generator.parameters():
                p.requires_grad = False

    def forward(self, w_plus: torch.Tensor) -> torch.Tensor:
        """
        中文：
            统一生成器前向接口。

        English:
            Unified generator forward interface.
        """
        return self.generator(w_plus)