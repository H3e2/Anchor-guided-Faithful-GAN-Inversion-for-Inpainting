# -*- coding: utf-8 -*-
"""
中文：
    exp_logger.py
    统一实验输出目录、日志、时间戳管理。

English:
    exp_logger.py
    Unified experiment output directory, logging, and timestamp management.
"""

from __future__ import annotations

import os
import time
import logging
from datetime import datetime


def get_timestamp() -> str:
    """
    中文：
        返回统一格式的时间戳字符串。

    English:
        Return a timestamp string in a unified format.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def prepare_exp_dir(root_dir: str, exp_name: str) -> dict[str, str]:
    """
    中文：
        创建带时间戳的实验目录，并返回常用路径字典。

    English:
        Create a timestamped experiment directory and return common subpaths.
    """
    timestamp = get_timestamp()
    exp_dir = os.path.join(root_dir, exp_name, timestamp)

    paths = {
        "exp_dir": exp_dir,
        "ckpt_dir": os.path.join(exp_dir, "checkpoints"),
        "sample_dir": os.path.join(exp_dir, "samples"),
        "log_dir": os.path.join(exp_dir, "logs"),
        "metric_dir": os.path.join(exp_dir, "metrics"),
    }

    for _, p in paths.items():
        os.makedirs(p, exist_ok=True)

    return paths


def setup_logger(log_file: str, logger_name: str = "mcla") -> logging.Logger:
    """
    中文：
        创建同时输出到终端和文件的 logger。

    English:
        Create a logger that writes to both console and file.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


class AverageMeter:
    """
    中文：
        简单平均计量器。

    English:
        Simple average meter.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count