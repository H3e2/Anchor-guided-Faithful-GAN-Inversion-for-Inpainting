# -*- coding: utf-8 -*-
"""
中文：
    stylegan2ada_wrapper.py
    这是对 NVLabs StyleGAN2-ADA 生成器的轻量封装。
    这个版本解决了当前训练需要的关键接口：
        1. 支持加载预训练 pkl
        2. 支持 forward(w_plus)
        3. 支持读取 w_avg / z_dim
        4. 支持 mapping(z)
        5. 支持 noise buffer 的读取 / 初始化 / 写回

English:
    stylegan2ada_wrapper.py
    Lightweight wrapper for the NVLabs StyleGAN2-ADA generator.
    This version provides the key interfaces needed for current training:
        1. load pretrained pkl
        2. support forward(w_plus)
        3. expose w_avg / z_dim
        4. support mapping(z)
        5. support reading / initializing / writing noise buffers
"""

from __future__ import annotations

import os
import sys
import pickle
from pathlib import Path

import torch
import torch.nn as nn


class StyleGAN2ADAWrapper(nn.Module):
    """
    中文：
        StyleGAN2-ADA 生成器封装类。

    English:
        Wrapper class for the StyleGAN2-ADA generator.
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

        self.repo_root = str(repo_root)
        self.checkpoint = str(checkpoint)
        self.freeze_generator = freeze_generator
        self.noise_mode = noise_mode
        self.truncation_psi = truncation_psi
        self.device_str = device
        self.device = torch.device(device)

        self._setup_stylegan2ada_imports()
        self.G = self._load_generator().to(self.device)

        if self.freeze_generator:
            for p in self.G.parameters():
                p.requires_grad = False
            self.G.eval()

    def _setup_stylegan2ada_imports(self):
        """
        中文：
            将 StyleGAN2-ADA repo 加入 import 路径。

        English:
            Add the StyleGAN2-ADA repo to Python import paths.
        """
        repo_root = Path(self.repo_root)
        if not repo_root.exists():
            raise FileNotFoundError(f"StyleGAN2-ADA repo not found: {repo_root}")

        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

    def _load_generator(self):
        """
        中文：
            从 pkl 文件中加载预训练生成器 G_ema。

        English:
            Load the pretrained generator G_ema from a pkl file.
        """
        try:
            import dnnlib
            import legacy
        except Exception as e:
            raise ImportError(
                f"Failed to import StyleGAN2-ADA modules from repo_root={self.repo_root}. "
                f"Original error: {repr(e)}"
            )

        if not os.path.isfile(self.checkpoint):
            raise FileNotFoundError(f"StyleGAN2-ADA checkpoint not found: {self.checkpoint}")

        with open(self.checkpoint, "rb") as f:
            network_pkl = pickle.load(f)

        if "G_ema" in network_pkl:
            G = network_pkl["G_ema"]
        else:
            # fallback 兼容一些非标准 pkl / fallback for non-standard pkls
            G = network_pkl

        return G

    # =========================================================
    # Properties
    # =========================================================
    @property
    def z_dim(self) -> int:
        """
        中文：
            返回输入随机向量 z 的维度。

        English:
            Return the dimension of input latent z.
        """
        return int(self.G.z_dim)

    @property
    def w_avg(self) -> torch.Tensor:
        """
        中文：
            返回 mapping 网络中的平均 style 向量，形状 [w_dim]

        English:
            Return the average style vector from the mapping network, shape [w_dim]
        """
        return self.G.mapping.w_avg.detach()

    @property
    def num_ws(self) -> int:
        """
        中文：
            返回 synthesis 网络使用的 style 层数。

        English:
            Return the number of style inputs used by the synthesis network.
        """
        return int(self.G.num_ws)

    @property
    def w_dim(self) -> int:
        """
        中文：
            返回每个 style 向量的维度。

        English:
            Return the dimensionality of each style vector.
        """
        return int(self.G.w_dim)

    # =========================================================
    # Latent interfaces
    # =========================================================
    def mapping(self, z: torch.Tensor) -> torch.Tensor:
        """
        中文：
            将 z 映射到 W+，输出形状 [B, num_ws, w_dim]

        English:
            Map z into W+, output shape [B, num_ws, w_dim]
        """
        z = z.to(self.device)
        ws = self.G.mapping(z, None, truncation_psi=self.truncation_psi)
        return ws

    def init_wplus(self, batch_size: int = 1) -> torch.Tensor:
        """
        中文：
            用 w_avg 初始化 W+，输出 [B, num_ws, w_dim]

        English:
            Initialize W+ with w_avg, output [B, num_ws, w_dim]
        """
        w_avg = self.w_avg.to(self.device)  # [w_dim]
        w_plus = w_avg.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.num_ws, 1)
        return w_plus

    # =========================================================
    # Noise interfaces
    # =========================================================
    def get_noise_buffers(self):
        """
        中文：
            获取 synthesis 网络中所有 noise_const buffer。

        English:
            Collect all noise_const buffers in the synthesis network.
        """
        noise_bufs = {
            name: buf
            for name, buf in self.G.synthesis.named_buffers()
            if "noise_const" in name
        }
        return noise_bufs

    def init_noise_buffers(self):
        """
        中文：
            将所有 noise buffer 初始化为标准高斯噪声，并开启梯度。

        English:
            Initialize all noise buffers with standard Gaussian noise and enable gradients.
        """
        noise_bufs = self.get_noise_buffers()
        for _, buf in noise_bufs.items():
            buf.data = torch.randn_like(buf)
            buf.requires_grad = True
        return noise_bufs

    def set_noise_buffers(self, noise_bufs: dict[str, torch.Tensor]):
        """
        中文：
            将外部 noise buffer 写回当前 synthesis 网络。

        English:
            Write external noise buffers back into the current synthesis network.
        """
        own_noise_bufs = self.get_noise_buffers()

        with torch.no_grad():
            for name, buf in own_noise_bufs.items():
                if name in noise_bufs:
                    src = noise_bufs[name].to(buf.device, dtype=buf.dtype)
                    if src.shape != buf.shape:
                        raise ValueError(
                            f"Noise shape mismatch for {name}: "
                            f"expected {tuple(buf.shape)}, got {tuple(src.shape)}"
                        )
                    buf.copy_(src)

    def clone_noise_buffers(self):
        """
        中文：
            克隆当前 noise buffers。

        English:
            Clone current noise buffers.
        """
        out = {}
        for name, buf in self.get_noise_buffers().items():
            out[name] = buf.detach().clone()
        return out

    # =========================================================
    # Synthesis / forward
    # =========================================================
    def synthesize(self, w_plus: torch.Tensor) -> torch.Tensor:
        """
        中文：
            从 W+ 渲染图像。

        English:
            Render image from W+.
        """
        w_plus = w_plus.to(self.device)

        # 如果传进来的是 [B, w_dim]，自动扩展到 [B, num_ws, w_dim]
        # If input is [B, w_dim], automatically expand it to [B, num_ws, w_dim].
        if w_plus.dim() == 2:
            w_plus = w_plus.unsqueeze(1).repeat(1, self.num_ws, 1)

        if w_plus.dim() != 3:
            raise ValueError(f"Expected w_plus shape [B, num_ws, w_dim], got {tuple(w_plus.shape)}")

        img = self.G.synthesis(w_plus, noise_mode=self.noise_mode, force_fp32=True)
        return img

    def forward(self, w_plus: torch.Tensor) -> torch.Tensor:
        """
        中文：
            前向等价于 synthesize。

        English:
            Forward is equivalent to synthesize.
        """
        return self.synthesize(w_plus)