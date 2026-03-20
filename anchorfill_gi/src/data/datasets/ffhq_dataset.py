# -*- coding: utf-8 -*-
"""
中文：
    ffhq_dataset.py
    这是 FFHQ 风格目录的数据集读取类。
    当前支持：
        1. 从 image_dir 读取人脸图像
        2. 在线随机生成 irregular mask
        3. 构造 masked image
        4. 计算 boundary band
        5. 返回训练所需字典

English:
    ffhq_dataset.py
    Dataset class for FFHQ-style directory structure.
    Current features:
        1. Load face images from image_dir
        2. Generate irregular masks online
        3. Construct masked images
        4. Compute boundary band
        5. Return a training-ready sample dictionary
"""

from __future__ import annotations

import os
import glob
from typing import Dict, Any, List, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from src.data.mask_generators import build_mask_generator, compute_boundary_band


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_image_file(path: str) -> bool:
    """
    中文：
        判断文件是否为常见图像格式。

    English:
        Check whether a file is a common image file.
    """
    return os.path.splitext(path)[1].lower() in IMG_EXTENSIONS


def _scan_images(image_dir: str) -> List[str]:
    """
    中文：
        扫描目录中的图像文件并排序。

    English:
        Scan image files in the given directory and sort them.
    """
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    image_paths = []
    for ext in IMG_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
        image_paths.extend(glob.glob(os.path.join(image_dir, f"*{ext.upper()}")))

    image_paths = sorted(list(set(image_paths)))

    if len(image_paths) == 0:
        raise RuntimeError(f"No image files found in: {image_dir}")

    return image_paths


class FFHQDataset(Dataset):
    """
    中文：
        FFHQ 数据集读取类。

        输入目录结构示例：
            data/processed/ffhq256/train/images
            data/processed/ffhq256/val/images
            data/processed/ffhq256/test/images

        返回的样本字典包含：
            - image: 标准化后的原图，shape [3,H,W]
            - mask: 二值掩码，1=洞，0=已知，shape [1,H,W]
            - masked_image: 遮挡后的输入图，shape [3,H,W]
            - known_region: 已知区域掩码，shape [1,H,W]
            - boundary_band: 边界带，shape [1,H,W]
            - path: 原图路径

    English:
        Dataset class for FFHQ.

        Example input structure:
            data/processed/ffhq256/train/images
            data/processed/ffhq256/val/images
            data/processed/ffhq256/test/images

        Returned sample dictionary contains:
            - image: normalized full image, shape [3,H,W]
            - mask: binary mask, 1=hole, 0=known, shape [1,H,W]
            - masked_image: masked input, shape [3,H,W]
            - known_region: known-region mask, shape [1,H,W]
            - boundary_band: boundary band, shape [1,H,W]
            - path: original image path
    """

    def __init__(
        self,
        image_dir: str,
        image_size: int,
        mask_cfg: Dict[str, Any],
        normalize_cfg: Dict[str, Any],
        return_path: bool = True,
        max_samples: Optional[int] = None,
        fixed_mask: bool = False,
    ):
        """
        中文：
            初始化数据集。

        English:
            Initialize the dataset.
        """
        self.image_dir = image_dir
        self.image_size = image_size
        self.return_path = return_path

        self.image_paths = _scan_images(image_dir)
        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]

        self.mask_generator = build_mask_generator(mask_cfg)
        self.boundary_band_width = int(mask_cfg.get("boundary_band_width", 7))

        self.mean = normalize_cfg.get("mean", [0.5, 0.5, 0.5])
        self.std = normalize_cfg.get("std", [0.5, 0.5, 0.5])

        # 是否使用固定 mask（用于单图过拟合调试）
        # Whether to use a fixed mask (for single-image overfitting debug).
        self.fixed_mask = fixed_mask
        self.cached_mask = None
        self.cached_boundary_band = None

    def __len__(self) -> int:
        """
        中文：
            返回数据集样本数。

        English:
            Return the number of samples.
        """
        return len(self.image_paths)

    def _load_image(self, path: str) -> torch.Tensor:
        """
        中文：
            读取单张图像并 resize 到目标分辨率，再转换为 tensor 并归一化。

        English:
            Load one image, resize it to target resolution, convert to tensor,
            and normalize it.
        """
        img = Image.open(path).convert("RGB")
        img = TF.resize(img, [self.image_size, self.image_size], interpolation=Image.BICUBIC)
        img = TF.to_tensor(img)  # [3,H,W], range [0,1]
        img = TF.normalize(img, mean=self.mean, std=self.std)
        return img

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        中文：
            读取样本并返回训练所需字典。

        English:
            Load one sample and return a training-ready dictionary.
        """
        path = self.image_paths[index]
        image = self._load_image(path)  # [3,H,W], normalized

        _, h, w = image.shape

               # 生成 mask
        # Generate mask
        if self.fixed_mask:
            # 如果开启固定 mask，则第一次生成后缓存，后续重复使用
            # If fixed_mask is enabled, generate once and reuse afterwards.
            if self.cached_mask is None:
                self.cached_mask = self.mask_generator.generate(h, w)  # [1,H,W]
                self.cached_boundary_band = compute_boundary_band(
                    self.cached_mask,
                    self.boundary_band_width
                )
            mask = self.cached_mask.clone()
            boundary_band = self.cached_boundary_band.clone()
        else:
            # 默认模式：每次随机生成新的 mask
            # Default mode: generate a new random mask every time.
            mask = self.mask_generator.generate(h, w)  # [1,H,W]
            boundary_band = compute_boundary_band(mask, self.boundary_band_width)

        known_region = 1.0 - mask

        # 用 0 填充缺失区域（在归一化空间里）
        # Fill the hole region with zeros in normalized space.
        masked_image = image * known_region

        sample = {
            "image": image,                    # 原图 / full image
            "mask": mask,                      # 1=hole / masked region
            "known_region": known_region,      # 1=visible / known region
            "masked_image": masked_image,      # 输入图 / masked input
            "boundary_band": boundary_band,    # 边界带 / boundary band
        }

        if self.return_path:
            sample["path"] = path

        return sample