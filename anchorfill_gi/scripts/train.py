# -*- coding: utf-8 -*-
"""
中文：
    train.py
    这是项目训练入口脚本。
    当前功能：
        1. 读取实验配置
        2. 设置随机种子
        3. 构建 datamodule
        4. 构建 baseline system
        5. 启动 trainer

English:
    train.py
    Training entry script for the project.
    Current features:
        1. load experiment config
        2. set random seed
        3. build datamodule
        4. build baseline system
        5. launch trainer
"""

from __future__ import annotations

import os
import sys
import argparse

# 将项目根目录加入 PYTHONPATH
# Add project root to PYTHONPATH.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 将 torch hub 缓存放到数据盘
# Put torch hub cache on the data disk.
os.environ["TORCH_HOME"] = os.path.join(PROJECT_ROOT, ".cache", "torch")

from src.utils.config import load_exp_config, make_output_dir, save_final_config
from src.utils.seed import set_seed
from src.data.datamodule import build_datamodule
from src.models.systems.baseline_system import BaselineInversionSystem
from src.engine.trainer import Trainer


def parse_args():
    """
    中文：
        解析命令行参数。

    English:
        Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment yaml config.",
    )
    return parser.parse_args()


def main():
    """
    中文：
        训练主函数。

    English:
        Main function for training.
    """
    args = parse_args()

    cfg = load_exp_config(args.config)
    out_dir = make_output_dir(cfg)
    save_final_config(cfg, out_dir)

    set_seed(cfg["project"]["seed"])

    datamodule = build_datamodule(cfg)
    model = BaselineInversionSystem(cfg)
    trainer = Trainer(cfg, model, datamodule)

    trainer.fit()


if __name__ == "__main__":
    main()