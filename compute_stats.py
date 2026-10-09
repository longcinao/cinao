# -*- coding: utf-8 -*-
"""
计算逐波段均值/标准差（供 z-score 归一化使用）
====================================================
仅在 config.NORMALIZATION = "zscore" 时需要。
默认的 "minmax" 归一化是逐样本自包含的，无需运行本脚本。
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.console import setup_console
setup_console()
import numpy as np

import config
from data.dataset import list_sample_paths
from data.band_folder_io import read_band_cube
from data.mat_io import read_mat_cube


def _read(path):
    if os.path.isdir(path):
        return read_band_cube(path, config.SELECTED_WAVELENGTHS)
    return read_mat_cube(path)


def main():
    files = list_sample_paths(
        config.SAMPLE_DIR,
        wavelengths=config.SELECTED_WAVELENGTHS,
        labels_csv=config.LABELS_CSV if os.path.isfile(config.LABELS_CSV) else None,
    )
    if not files:
        raise SystemExit(f"没有样本：{config.SAMPLE_DIR}")
    selected = np.asarray(config.SELECTED_BANDS)

    sample = _read(files[0])[selected]
    c, h, w = sample.shape

    sum_ = np.zeros(c, dtype=np.float64)
    sum_sq = np.zeros(c, dtype=np.float64)
    count = 0

    for i, p in enumerate(files):
        cube = _read(p)[selected].astype(np.float64)
        sum_ += cube.reshape(c, -1).sum(axis=1)
        sum_sq += (cube.reshape(c, -1) ** 2).sum(axis=1)
        count += cube.shape[1] * cube.shape[2]
        if (i + 1) % 50 == 0:
            print(f"  已处理 {i + 1}/{len(files)} 个文件")

    mean = sum_ / count
    var = sum_sq / count - mean ** 2
    std = np.sqrt(np.maximum(var, 0.0))

    out = {
        "selected_bands": config.SELECTED_BANDS,
        "selected_wavelengths": config.SELECTED_WAVELENGTHS,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    with open(config.STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n统计完成，保存到 {config.STATS_FILE}")
    print("mean:", np.round(mean, 4).tolist())
    print("std :", np.round(std, 4).tolist())


if __name__ == "__main__":
    main()
