# -*- coding: utf-8 -*-
"""
中文：
    datamodule.py
    这个文件负责根据配置构建 train / val / test 的 Dataset 和 DataLoader。
    当前不依赖 PyTorch Lightning，只是借用 datamodule 这个命名习惯。

English:
    datamodule.py
    This file builds train / val / test datasets and dataloaders from config.
    We do not depend on PyTorch Lightning for now; the term "datamodule"
    is only used as a naming convention.
"""

from __future__ import annotations

from typing import Dict, Any, Optional

from torch.utils.data import DataLoader

from src.data.datasets.ffhq_dataset import FFHQDataset


class InpaintingDataModule:
    """
    中文：
        图像修复任务的数据模块。
        负责构建训练、验证、测试集及对应 DataLoader。

    English:
        Data module for image inpainting.
        It builds train/val/test datasets and corresponding dataloaders.
    """

    def __init__(self, cfg: Dict[str, Any]):
        """
        中文：
            使用合并后的总配置进行初始化。

        English:
            Initialize the data module with the merged configuration.
        """
        self.cfg = cfg

        self.train_dataset: Optional[FFHQDataset] = None
        self.val_dataset: Optional[FFHQDataset] = None
        self.test_dataset: Optional[FFHQDataset] = None

    def setup(self):
        """
        中文：
            根据配置创建 train / val / test 数据集。

        English:
            Build train / val / test datasets from config.
        """
        dataset_cfg = self.cfg["dataset"]
        mask_cfg = self.cfg["mask"]
        normalize_cfg = self.cfg["normalize"]

        debug_cfg = self.cfg.get("debug", {})
        fixed_mask = bool(debug_cfg.get("fixed_mask", False))
        max_samples = None
        if debug_cfg.get("enabled", False) and debug_cfg.get("overfit_batch", False):
            max_samples = int(debug_cfg.get("overfit_num_samples", 8))

        self.train_dataset = FFHQDataset(
            image_dir=dataset_cfg["train_image_dir"],
            image_size=dataset_cfg["image_size"],
            mask_cfg=mask_cfg,
            normalize_cfg=normalize_cfg,
            return_path=True,
            max_samples=max_samples,
            fixed_mask=fixed_mask,
        )

        self.val_dataset = FFHQDataset(
            image_dir=dataset_cfg["val_image_dir"],
            image_size=dataset_cfg["image_size"],
            mask_cfg=mask_cfg,
            normalize_cfg=normalize_cfg,
            return_path=True,
            max_samples=max_samples,
            fixed_mask=fixed_mask,
        )

        self.test_dataset = FFHQDataset(
            image_dir=dataset_cfg["test_image_dir"],
            image_size=dataset_cfg["image_size"],
            mask_cfg=mask_cfg,
            normalize_cfg=normalize_cfg,
            return_path=True,
            max_samples=max_samples,
            fixed_mask=fixed_mask,
        )

    def train_dataloader(self) -> DataLoader:
        """
        中文：
            构建训练集 DataLoader。

        English:
            Build the train dataloader.
        """
        loader_cfg = self.cfg["loader"]

        return DataLoader(
            self.train_dataset,
            batch_size=loader_cfg["batch_size"],
            shuffle=loader_cfg["shuffle"],
            num_workers=loader_cfg["num_workers"],
            pin_memory=loader_cfg["pin_memory"],
            persistent_workers=loader_cfg["persistent_workers"],
            drop_last=loader_cfg["drop_last"],
        )

    def val_dataloader(self) -> DataLoader:
        """
        中文：
            构建验证集 DataLoader。

        English:
            Build the validation dataloader.
        """
        loader_cfg = self.cfg["loader"]

        return DataLoader(
            self.val_dataset,
            batch_size=loader_cfg["val_batch_size"],
            shuffle=False,
            num_workers=loader_cfg["num_workers"],
            pin_memory=loader_cfg["pin_memory"],
            persistent_workers=loader_cfg["persistent_workers"],
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        """
        中文：
            构建测试集 DataLoader。

        English:
            Build the test dataloader.
        """
        loader_cfg = self.cfg["loader"]

        return DataLoader(
            self.test_dataset,
            batch_size=loader_cfg["test_batch_size"],
            shuffle=False,
            num_workers=loader_cfg["num_workers"],
            pin_memory=loader_cfg["pin_memory"],
            persistent_workers=loader_cfg["persistent_workers"],
            drop_last=False,
        )


def build_datamodule(cfg: Dict[str, Any]) -> InpaintingDataModule:
    """
    中文：
        根据配置构建数据模块。

    English:
        Build the data module from config.
    """
    dm = InpaintingDataModule(cfg)
    dm.setup()
    return dm