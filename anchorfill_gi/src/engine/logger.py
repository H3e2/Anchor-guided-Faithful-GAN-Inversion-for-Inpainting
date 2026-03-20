# -*- coding: utf-8 -*-
"""
中文：
    logger.py
    提供一个简单的日志记录器，同时支持：
        1. 控制台输出
        2. 文件日志
        3. TensorBoard

English:
    logger.py
    A simple logger utility that supports:
        1. console logging
        2. file logging
        3. TensorBoard logging
"""

from __future__ import annotations

import logging
import os
from typing import Dict

from torch.utils.tensorboard import SummaryWriter


def build_logger(log_dir: str, name: str = "anchorfill") -> logging.Logger:
    """
    中文：
        创建标准 logger，同时输出到终端和文件。

    English:
        Build a standard logger that writes to both console and file.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 防止重复添加 handler
    # Avoid duplicated handlers.
    if logger.handlers:
        return logger

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def build_writer(log_dir: str) -> SummaryWriter:
    """
    中文：
        创建 TensorBoard writer。

    English:
        Create a TensorBoard SummaryWriter.
    """
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)


def log_scalar_dict(writer: SummaryWriter, prefix: str, scalar_dict: Dict[str, float], step: int):
    """
    中文：
        将一个标量字典写入 TensorBoard。

    English:
        Write a dictionary of scalars into TensorBoard.
    """
    for k, v in scalar_dict.items():
        writer.add_scalar(f"{prefix}/{k}", v, step)