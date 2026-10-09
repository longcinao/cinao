# -*- coding: utf-8 -*-
"""
.mat 文件读取
====================================================
本数据库的 .mat 是 MATLAB 7.3 (HDF5) 格式，scipy.io.loadmat 读不了，
必须用 h5py 读取；这里同时兼容旧版 v7（scipy 回退）。

每个文件内部只有一个 3D 变量 cube：shape (31, 256, 256)，dtype float32，
含义是「31 个光谱波段 × 256 高 × 256 宽」，即波段在第一维。
"""

import os
import numpy as np
import h5py
import scipy.io


def list_mat_files(mat_dir):
    """返回目录下按文件名排序的 .mat 文件绝对路径列表。"""
    files = sorted(
        f for f in os.listdir(mat_dir)
        if f.lower().endswith(".mat")
    )
    return [os.path.join(mat_dir, f) for f in files]


def _as_bands_first(arr):
    """
    把 3D 数组统一成 (波段, 高, 宽) 顺序。
    启发式：若最小维在最后一维（常见于 (H, W, C) 存储），则转置到第一维。
    本数据库的 cube 已经是 (波段, 高, 宽)，此函数主要做兜底保护。
    """
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"期望 3D 数据，实际得到 shape={arr.shape}")
    if arr.shape[-1] == min(arr.shape) and arr.shape[-1] < arr.shape[0]:
        arr = np.transpose(arr, (2, 0, 1))
    return arr.astype(np.float32)


def read_mat_cube(path):
    """
    读取单个 .mat 文件，返回 (波段, 高, 宽) 的 float32 数组。

    - v7.3 (HDF5)  -> h5py 读取
    - v7 及更早     -> scipy.io.loadmat 读取
    """
    # 先按 HDF5(v7.3) 尝试
    try:
        with h5py.File(path, "r") as f:
            for key in f.keys():
                obj = f[key]
                if isinstance(obj, h5py.Dataset):
                    arr = obj[()]
                    arr = np.asarray(arr)
                    # 跳过 cell/struct 等引用型数据，只取真正的 3D 图像
                    if arr.ndim == 3 and arr.dtype != object:
                        return _as_bands_first(arr)
    except OSError:
        pass  # 不是 HDF5，走下面的 scipy 回退

    # 旧版 v7 回退
    raw = scipy.io.loadmat(path)
    for k, v in raw.items():
        if k.startswith("__"):
            continue
        v = np.asarray(v)
        if v.ndim == 3 and v.dtype != object:
            return _as_bands_first(v)

    raise ValueError(f"在 {path} 中未找到 3D 图像数据（cube）")
