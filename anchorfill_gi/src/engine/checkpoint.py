# -*- coding: utf-8 -*-
"""
中文：
    checkpoint.py
    负责保存和加载训练检查点。

English:
    checkpoint.py
    Utilities for saving and loading training checkpoints.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    save_path: str,
    model,
    optimizer=None,
    scaler=None,
    epoch: int = 0,
    global_step: int = 0,
    extra: Optional[Dict[str, Any]] = None,
):
    """
    中文：
        保存 checkpoint。

    English:
        Save a checkpoint.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    state = {
        "model": model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }

    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()

    if scaler is not None:
        state["scaler"] = scaler.state_dict()

    if extra is not None:
        state["extra"] = extra

    torch.save(state, save_path)


def load_checkpoint(
    ckpt_path: str,
    model,
    optimizer=None,
    scaler=None,
    map_location: str = "cpu",
):
    """
    中文：
        加载 checkpoint，并恢复模型、优化器和 scaler。

    English:
        Load checkpoint and restore model / optimizer / scaler.
    """
    checkpoint = torch.load(ckpt_path, map_location=map_location)

    model.load_state_dict(checkpoint["model"], strict=True)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("global_step", 0)

    return epoch, global_step, checkpoint