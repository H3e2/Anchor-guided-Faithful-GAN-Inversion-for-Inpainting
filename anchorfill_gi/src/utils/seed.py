# -*- coding: utf-8 -*-
"""
中文：
    seed.py
    用于固定随机种子，保证实验可复现。

English:
    seed.py
    Utility for setting random seeds to ensure experiment reproducibility.
"""

from __future__ import annotations

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False):
    """
    中文：
        设置 Python / NumPy / PyTorch 的随机种子。

    English:
        Set random seeds for Python / NumPy / PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    os.environ["PYTHONHASHSEED"] = str(seed)