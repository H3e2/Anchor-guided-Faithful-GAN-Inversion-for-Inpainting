# -*- coding: utf-8 -*-
"""
中文：
    sanity_optimize_latent.py
    这是对官方 NVlabs stylegan2-ada-pytorch 仓库的轻量包装器。
    当前阶段我们只做推理/投影相关用途，不训练生成器。

English:
    stylegan2ada_wrapper.py
    A lightweight wrapper around the official NVlabs stylegan2-ada-pytorch repo.
    At the current stage, we only use it for inference / projection,
    and keep the generator frozen.
"""

from __future__ import annotations

import os
import sys
import pickle
from typing import Optional

import torch
import torch.nn as nn


class StyleGAN2ADAWrapper(nn.Module):
    """
    中文：
        真实 StyleGAN2-ADA 生成器包装器。
        主要提供：
            1. 加载官方 .pkl 预训练权重
            2. mapping(z) -> w
            3. w -> w+ 扩展
            4. synthesis(w+) -> image

    English:
        Wrapper for the real StyleGAN2-ADA generator.
        Main functionalities:
            1. load official pretrained .pkl
            2. mapping(z) -> w
            3. expand w into w+
            4. synthesis(w+) -> image
    """

    def __init__(
        self,
        repo_root: str,
        checkpoint: str,
        freeze_generator: bool = True,
        noise_mode: str = "const",
        truncation_psi: float = 1.0,
        device: str = "cuda",
    ):
        super().__init__()

        self.repo_root = repo_root
        self.checkpoint = checkpoint
        self.freeze_generator = freeze_generator
        self.noise_mode = noise_mode
        self.truncation_psi = truncation_psi
        self.device_name = device

        # 动态加入官方 repo 路径
        # Dynamically append the official repo path.
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        try:
            import legacy  # type: ignore
        except Exception as e:
            raise ImportError(
                f"Failed to import official stylegan2-ada-pytorch repo from: {repo_root}\n"
                f"Please make sure the repo is cloned correctly.\n"
                f"Original error: {repr(e)}"
            )

        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"StyleGAN checkpoint not found: {checkpoint}")

        # 读取官方 pkl
        # Load official .pkl checkpoint.
        with open(checkpoint, "rb") as f:
            network_dict = legacy.load_network_pkl(f)

        # 官方 projector.py 一般使用 G_ema
        # Official projector.py typically uses G_ema.
        self.G = network_dict["G_ema"].to(device)

        if freeze_generator:
            for p in self.G.parameters():
                p.requires_grad = False

        self.z_dim = self.G.z_dim
        self.w_dim = self.G.w_dim
        self.num_ws = self.G.num_ws
        self.img_resolution = self.G.img_resolution
        self.img_channels = self.G.img_channels

        # 保存 w_avg，后面初始化 W+ 时会用到
        # Save w_avg for later W+ initialization.
        if hasattr(self.G.mapping, "w_avg") and self.G.mapping.w_avg is not None:
            self.register_buffer("w_avg", self.G.mapping.w_avg.detach().clone())
        else:
            self.register_buffer("w_avg", torch.zeros(self.w_dim, device=device))

    @torch.no_grad()
    def mapping(self, z: torch.Tensor) -> torch.Tensor:
        """
        中文：
            将 z 映射到 w。
            输入:
                z: [B, z_dim]
            输出:
                w: [B, num_ws, w_dim]

        English:
            Map z into w.
            Input:
                z: [B, z_dim]
            Output:
                w: [B, num_ws, w_dim]
        """
        c = None  # FFHQ 是 unconditional，条件输入为空
        ws = self.G.mapping(z, c, truncation_psi=self.truncation_psi, truncation_cutoff=None)
        return ws

    def make_wplus_from_w(self, w: torch.Tensor) -> torch.Tensor:
        """
        中文：
            如果输入是 [B, w_dim]，则扩展成 [B, num_ws, w_dim]。
            如果输入已经是 W+ 形状，则原样返回。

        English:
            Expand [B, w_dim] into [B, num_ws, w_dim].
            If already in W+ shape, return as is.
        """
        if w.dim() == 2:
            return w.unsqueeze(1).repeat(1, self.num_ws, 1)
        elif w.dim() == 3:
            return w
        else:
            raise ValueError(f"Unsupported latent shape: {tuple(w.shape)}")

    def init_wplus(self, batch_size: int = 1) -> torch.Tensor:
        """
        中文：
            用 w_avg 初始化一个可优化的 W+ latent。

        English:
            Initialize an optimizable W+ latent from w_avg.
        """
        w = self.w_avg.unsqueeze(0).repeat(batch_size, 1)                 # [B, w_dim]
        w_plus = self.make_wplus_from_w(w)                                # [B, num_ws, w_dim]
        return w_plus.clone().detach()

    def synthesize(self, w_plus: torch.Tensor) -> torch.Tensor:
        """
        中文：
            使用 synthesis 网络从 W+ latent 生成图像。
            输出范围通常为 [-1, 1]。

        English:
            Use the synthesis network to generate images from W+ latent.
            Output is typically in [-1, 1].
        """
        img = self.G.synthesis(w_plus, noise_mode=self.noise_mode)
        return img

    def forward(self, w_plus: torch.Tensor) -> torch.Tensor:
        """
        中文：
            前向别名，等价于 synthesize。

        English:
            Forward alias for synthesize.
        """
        return self.synthesize(w_plus)