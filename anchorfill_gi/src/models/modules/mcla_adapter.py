# -*- coding: utf-8 -*-
"""
中文：
    mcla_adapter.py
    MCLA 的核心 adapter。
    当前版本：
        - 不使用 teacher
        - 直接从 masked input 学 clean latent proposal
        - 使用 RGB / edge / frequency 多分支信息

English:
    mcla_adapter.py
    Core adapter of MCLA.
    Current version:
        - no teacher
        - directly learns a clean latent proposal from masked input
        - uses RGB / edge / frequency multi-branch information
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.modules.visible_preprocessor import VisiblePreprocessor
from src.models.encoders.visible_anchor_encoder import (
    SimpleVisibleAnchorEncoder,
    MSResidualVisibleAnchorEncoder,
)


def build_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """
    中文：
        简单 MLP 头。

    English:
        Simple MLP head.
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


class BranchProjector(nn.Module):
    """
    中文：
        将不同分支特征投影到统一的 feat_dim。

    English:
        Project different branch features into a unified feat_dim.
    """

    def __init__(self, in_dim: int, feat_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, feat_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MCLAAdapter(nn.Module):
    """
    中文：
        Mask-conditioned Clean Latent Adapter

        输入：
            masked_image: [B, 3, H, W]
            mask:         [B, 1, H, W]

        输出：
            init_w_plus: [B, num_ws, w_dim]

    English:
        Mask-conditioned Clean Latent Adapter

        Input:
            masked_image: [B, 3, H, W]
            mask:         [B, 1, H, W]

        Output:
            init_w_plus: [B, num_ws, w_dim]
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 64,
        max_channels: int = 512,
        feat_dim: int = 512,
        num_down: int = 4,
        num_ws: int = 18,
        w_dim: int = 512,
        layer_splits: list[int] | tuple[int, int, int] = (4, 6, 8),
        delta_scale: float = 0.35,
        gate_mode: str = "channel",
        use_edge_branch: bool = True,
        use_freq_branch: bool = True,
        w_avg: torch.Tensor | None = None,
    ):
        super().__init__()

        self.num_ws = num_ws
        self.w_dim = w_dim
        self.layer_splits = list(layer_splits)
        self.delta_scale = delta_scale
        self.gate_mode = gate_mode
        self.use_edge_branch = use_edge_branch
        self.use_freq_branch = use_freq_branch

        assert sum(self.layer_splits) == self.num_ws, "Sum(layer_splits) must equal num_ws."

        # ---------------------------------------------------------
        # 1. 输入预处理：构造 rgb / edge / frequency 分支
        # 1. Input preprocessing: build rgb / edge / frequency branches
        # ---------------------------------------------------------
        self.preprocessor = VisiblePreprocessor()

        # ---------------------------------------------------------
        # 2. RGB 主分支：更强的多尺度残差编码器
        # 2. RGB main branch: stronger multi-scale residual encoder
        # ---------------------------------------------------------
        self.rgb_encoder = MSResidualVisibleAnchorEncoder(
            in_channels=4,  # visible_rgb + mask
            base_channels=base_channels,
            max_channels=max_channels,
            num_down=num_down,
            feat_dim=feat_dim,
        )

        # ---------------------------------------------------------
        # 3. edge 分支：轻量 simple encoder
        # 3. edge branch: lightweight simple encoder
        # ---------------------------------------------------------
        if self.use_edge_branch:
            self.edge_encoder = SimpleVisibleAnchorEncoder(
                in_channels=2,  # edge + mask
                base_channels=max(base_channels // 2, 32),
                max_channels=max(max_channels // 2, 128),
                num_down=4,
                feat_dim=feat_dim,
            )
            self.edge_proj = BranchProjector(feat_dim, feat_dim)

        # ---------------------------------------------------------
        # 4. frequency 分支：轻量 simple encoder
        # 4. frequency branch: lightweight simple encoder
        # ---------------------------------------------------------
        if self.use_freq_branch:
            self.freq_encoder = SimpleVisibleAnchorEncoder(
                in_channels=7,  # low(3) + high(3) + mask(1)
                base_channels=max(base_channels // 2, 32),
                max_channels=max(max_channels // 2, 128),
                num_down=4,
                feat_dim=feat_dim,
            )
            self.freq_proj = BranchProjector(feat_dim, feat_dim)

        # ---------------------------------------------------------
        # 5. 多分支融合
        # 5. Multi-branch fusion
        # ---------------------------------------------------------
        n_branches = 1 + int(self.use_edge_branch) + int(self.use_freq_branch)
        self.fuse = nn.Sequential(
            nn.Linear(n_branches * feat_dim, feat_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

        c_ws, m_ws, f_ws = self.layer_splits

        # ---------------------------------------------------------
        # 6. grouped clean latent correction 头
        # 6. grouped clean latent correction heads
        # ---------------------------------------------------------
        self.delta_head_coarse = build_mlp(feat_dim, feat_dim, c_ws * w_dim)
        self.delta_head_mid = build_mlp(feat_dim, feat_dim, m_ws * w_dim)
        self.delta_head_fine = build_mlp(feat_dim, feat_dim, f_ws * w_dim)

        # ---------------------------------------------------------
        # 7. grouped gates
        # 7. grouped gates
        # ---------------------------------------------------------
        if gate_mode == "scalar":
            self.gate_head_coarse = build_mlp(feat_dim, feat_dim // 2, c_ws)
            self.gate_head_mid = build_mlp(feat_dim, feat_dim // 2, m_ws)
            self.gate_head_fine = build_mlp(feat_dim, feat_dim // 2, f_ws)
        elif gate_mode == "channel":
            self.gate_head_coarse = build_mlp(feat_dim, feat_dim, c_ws * w_dim)
            self.gate_head_mid = build_mlp(feat_dim, feat_dim, m_ws * w_dim)
            self.gate_head_fine = build_mlp(feat_dim, feat_dim, f_ws * w_dim)
        else:
            raise ValueError(f"Unsupported gate_mode: {gate_mode}")

        # ---------------------------------------------------------
        # 8. generator 平均 latent
        # 8. generator average latent
        # ---------------------------------------------------------
        if w_avg is None:
            w_avg_plus = torch.zeros(num_ws, w_dim)
        else:
            w_avg_plus = w_avg.detach().clone().unsqueeze(0).repeat(num_ws, 1)

        self.register_buffer("w_avg_plus", w_avg_plus)

    def _reshape_delta(self, x: torch.Tensor, n_ws: int) -> torch.Tensor:
        """
        中文：
            将 MLP 输出 reshape 成 [B, n_ws, w_dim]，
            并通过 tanh 限制修正量范围。

        English:
            Reshape MLP output into [B, n_ws, w_dim],
            and limit the correction range via tanh.
        """
        return torch.tanh(x).view(-1, n_ws, self.w_dim) * self.delta_scale

    def _reshape_gate(self, x: torch.Tensor, n_ws: int) -> torch.Tensor:
        """
        中文：
            将 gate 输出 reshape 成 scalar gate 或 channel gate。

        English:
            Reshape gate output into scalar gate or channel gate.
        """
        if self.gate_mode == "scalar":
            return torch.sigmoid(x).view(-1, n_ws, 1)
        return torch.sigmoid(x).view(-1, n_ws, self.w_dim)

    def forward(self, masked_image: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        中文：
            前向流程：
                1. 预处理输入
                2. 三分支提特征
                3. 融合
                4. 预测 grouped delta_w / gate
                5. 构造 init_w_plus

        English:
            Forward pipeline:
                1. preprocess input
                2. extract features from three branches
                3. fuse them
                4. predict grouped delta_w / gate
                5. build init_w_plus
        """
        prep = self.preprocessor(masked_image, mask)

        # RGB 主分支
        rgb_out = self.rgb_encoder(prep["rgb_mask"])
        rgb_feat = (rgb_out["coarse_vec"] + rgb_out["mid_vec"] + rgb_out["fine_vec"]) / 3.0
        feats = [rgb_feat]

        # edge 分支
        if self.use_edge_branch:
            edge_out = self.edge_encoder(prep["edge_mask"])
            edge_feat = self.edge_proj(edge_out["global_vec"])
            feats.append(edge_feat)

        # frequency 分支
        if self.use_freq_branch:
            freq_out = self.freq_encoder(prep["freq_mask"])
            freq_feat = self.freq_proj(freq_out["global_vec"])
            feats.append(freq_feat)

        # 融合多分支特征
        fused_feat = self.fuse(torch.cat(feats, dim=1))

        c_ws, m_ws, f_ws = self.layer_splits

        # grouped delta_w
        delta_c = self._reshape_delta(self.delta_head_coarse(fused_feat), c_ws)
        delta_m = self._reshape_delta(self.delta_head_mid(fused_feat), m_ws)
        delta_f = self._reshape_delta(self.delta_head_fine(fused_feat), f_ws)
        delta_w = torch.cat([delta_c, delta_m, delta_f], dim=1)

        # grouped gates
        gate_c = self._reshape_gate(self.gate_head_coarse(fused_feat), c_ws)
        gate_m = self._reshape_gate(self.gate_head_mid(fused_feat), m_ws)
        gate_f = self._reshape_gate(self.gate_head_fine(fused_feat), f_ws)
        gate = torch.cat([gate_c, gate_m, gate_f], dim=1)

        # 最终 clean latent proposal
        init_w_plus = self.w_avg_plus.unsqueeze(0) + gate * delta_w

        return {
            "init_w_plus": init_w_plus,
            "delta_w": delta_w,
            "gate": gate,
            "fused_feat": fused_feat,
            "prep": prep,
        }