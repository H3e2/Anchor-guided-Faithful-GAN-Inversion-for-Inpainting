# -*- coding: utf-8 -*-
"""
中文：
    visible_anchor_encoder.py
    提供两种可见区域编码器：
        1. SimpleVisibleAnchorEncoder
        2. MSResidualVisibleAnchorEncoder

English:
    visible_anchor_encoder.py
    Provide two visible-region encoders:
        1. SimpleVisibleAnchorEncoder
        2. MSResidualVisibleAnchorEncoder
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    """
    中文：
        基础卷积块：Conv -> BN -> LeakyReLU

    English:
        Basic conv block: Conv -> BN -> LeakyReLU
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SEBlock(nn.Module):
    """
    中文：
        轻量通道注意力模块。

    English:
        Lightweight channel attention module.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResidualAttentionBlock(nn.Module):
    """
    中文：
        轻量残差注意力块，用于保留局部细节。

    English:
        Lightweight residual attention block for preserving local details.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels, kernel_size=3, stride=1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.se = SEBlock(channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.dwconv(out)
        out = self.se(out)
        out = self.conv2(out)
        out = out + identity
        out = self.act(out)
        return out


class DownStage(nn.Module):
    """
    中文：
        下采样 stage：stride=2 卷积 + 轻量残差注意力块

    English:
        Downsampling stage: stride-2 conv + lightweight residual attention block.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = ConvNormAct(in_channels, out_channels, kernel_size=3, stride=2)
        self.refine = ResidualAttentionBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.refine(x)
        return x


class TopDownFuse(nn.Module):
    """
    中文：
        FPN 风格的 top-down 特征融合。

    English:
        FPN-style top-down feature fusion.
    """

    def __init__(self, low_channels: int, high_channels: int, out_channels: int):
        super().__init__()
        self.low_proj = nn.Conv2d(low_channels, out_channels, kernel_size=1, bias=False)
        self.high_proj = nn.Conv2d(high_channels, out_channels, kernel_size=1, bias=False)
        self.fuse = ResidualAttentionBlock(out_channels)

    def forward(self, low_feat: torch.Tensor, high_feat: torch.Tensor) -> torch.Tensor:
        high_up = F.interpolate(high_feat, size=low_feat.shape[-2:], mode="bilinear", align_corners=False)
        low = self.low_proj(low_feat)
        high = self.high_proj(high_up)
        out = low + high
        out = self.fuse(out)
        return out


class SimpleVisibleAnchorEncoder(nn.Module):
    """
    中文：
        简单版可见区域编码器。
        输入：
            [B, C, H, W]
        输出：
            dict:
                global_vec: [B, feat_dim]

    English:
        Simple visible-region encoder.
        Input:
            [B, C, H, W]
        Output:
            dict:
                global_vec: [B, feat_dim]
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 64,
        max_channels: int = 512,
        num_down: int = 5,
        feat_dim: int = 512,
    ):
        super().__init__()

        channels = [base_channels]
        for _ in range(num_down - 1):
            channels.append(min(channels[-1] * 2, max_channels))

        self.stem = ConvNormAct(in_channels, channels[0], stride=1)

        blocks = []
        in_ch = channels[0]
        for out_ch in channels:
            blocks.append(ConvNormAct(in_ch, out_ch, stride=2))
            in_ch = out_ch
        self.down_blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * 4 * 4, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, feat_dim),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.stem(x)
        feat = self.down_blocks(feat)
        feat = self.pool(feat)
        feat = self.head(feat)

        return {
            "global_vec": feat,
        }


class MSResidualVisibleAnchorEncoder(nn.Module):
    """
    中文：
        多尺度残差可见区域编码器。
        输入：
            [B, C, H, W]
        输出：
            dict:
                coarse_vec: [B, feat_dim]
                mid_vec   : [B, feat_dim]
                fine_vec  : [B, feat_dim]

    English:
        Multi-scale residual visible-region encoder.
        Input:
            [B, C, H, W]
        Output:
            dict:
                coarse_vec: [B, feat_dim]
                mid_vec   : [B, feat_dim]
                fine_vec  : [B, feat_dim]
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 64,
        max_channels: int = 512,
        num_down: int = 4,
        feat_dim: int = 512,
    ):
        super().__init__()

        c1 = base_channels
        c2 = min(c1 * 2, max_channels)
        c3 = min(c2 * 2, max_channels)
        c4 = min(c3 * 2, max_channels)
        c5 = min(c4 * 2, max_channels)

        self.stem = nn.Sequential(
            ConvNormAct(in_channels, c1, kernel_size=3, stride=1),
            ResidualAttentionBlock(c1),
        )

        self.stage1 = DownStage(c1, c2)  # 1/2
        self.stage2 = DownStage(c2, c3)  # 1/4
        self.stage3 = DownStage(c3, c4)  # 1/8
        self.stage4 = DownStage(c4, c5)  # 1/16

        self.fuse_mid = TopDownFuse(low_channels=c4, high_channels=c5, out_channels=c4)
        self.fuse_fine = TopDownFuse(low_channels=c3, high_channels=c4, out_channels=c3)

        self.coarse_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c5, feat_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.mid_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c4, feat_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fine_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c3, feat_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        s0 = self.stem(x)
        s1 = self.stage1(s0)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)

        mid_feat = self.fuse_mid(s3, s4)
        fine_feat = self.fuse_fine(s2, mid_feat)

        coarse_vec = self.coarse_head(s4)
        mid_vec = self.mid_head(mid_feat)
        fine_vec = self.fine_head(fine_feat)

        return {
            "coarse_vec": coarse_vec,
            "mid_vec": mid_vec,
            "fine_vec": fine_vec,
            "coarse_feat": s4,
            "mid_feat": mid_feat,
            "fine_feat": fine_feat,
        }