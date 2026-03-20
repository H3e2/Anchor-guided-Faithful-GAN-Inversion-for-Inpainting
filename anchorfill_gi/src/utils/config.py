# -*- coding: utf-8 -*-
"""
中文：
    config.py
    负责加载实验配置，并将 base / data / model / train / loss 五类配置合并成一个总配置。

English:
    config.py
    Load experiment config and merge base / data / model / train / loss
    into one final configuration dictionary.
"""

from __future__ import annotations

import os
import copy
import yaml
from typing import Dict, Any


def load_yaml(path: str) -> Dict[str, Any]:
    """
    中文：
        读取单个 YAML 文件。

    English:
        Load one YAML file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge_dict(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    中文：
        递归深度合并两个字典。
        new 中的值会覆盖 base 中的同名项。

    English:
        Recursively deep-merge two dictionaries.
        Values in `new` override those in `base`.
    """
    merged = copy.deepcopy(base)
    for k, v in new.items():
        if (
            k in merged
            and isinstance(merged[k], dict)
            and isinstance(v, dict)
        ):
            merged[k] = deep_merge_dict(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged


def load_exp_config(exp_config_path: str) -> Dict[str, Any]:
    """
    中文：
        加载实验配置。
        exp yaml 里会给出 base/data/model/train/loss 五个子配置路径，
        然后我们将它们依次合并，再应用 exp yaml 中的“额外覆盖项”。

    English:
        Load experiment config.
        The exp yaml provides paths to base/data/model/train/loss configs.
        We merge them first, then apply extra override fields from exp yaml.
    """
    exp_cfg = load_yaml(exp_config_path)

    required_keys = ["base", "data", "model", "train", "loss"]
    for key in required_keys:
        if key not in exp_cfg:
            raise KeyError(f"Missing required config key: {key}")

    cfg: Dict[str, Any] = {}

    # 1) 先加载五个子配置文件
    # 1) First load the five sub-config files.
    for key in required_keys:
        sub_cfg_path = exp_cfg[key]
        sub_cfg = load_yaml(sub_cfg_path)
        cfg = deep_merge_dict(cfg, sub_cfg)

    # 2) 再应用 exp yaml 中除 base/data/model/train/loss 之外的额外覆盖项
    # 2) Apply extra overrides in exp yaml except the path fields.
    exp_override = {
        k: v for k, v in exp_cfg.items()
        if k not in required_keys
    }

    cfg = deep_merge_dict(cfg, exp_override)

    return cfg


def ensure_dir(path: str):
    """
    中文：
        如果目录不存在则创建。

    English:
        Create the directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def make_output_dir(cfg: Dict[str, Any]) -> str:
    """
    中文：
        根据配置创建实验输出目录。

    English:
        Create experiment output directory from config.
    """
    exp_name = cfg["project"]["exp_name"]
    output_root = cfg["paths"]["output_root"]

    out_dir = os.path.join(output_root, exp_name)
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "checkpoints"))
    ensure_dir(os.path.join(out_dir, "samples"))
    ensure_dir(os.path.join(out_dir, "logs"))

    return out_dir


def save_final_config(cfg: Dict[str, Any], out_dir: str):
    """
    中文：
        保存最终合并后的配置，便于复现。

    English:
        Save the merged final config for reproducibility.
    """
    save_path = os.path.join(out_dir, "merged_config.yaml")
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)