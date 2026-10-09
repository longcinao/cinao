# -*- coding: utf-8 -*-
"""
配准后的窄带文件夹读取
====================================================
E:\\label\\04_encoded_task\\dataset\\raw_registered\\<样本名>\\ 下，
每个波长一张灰度图，文件名是中心波长，例如 590.jpeg。

本期只读四个特征波段：590（薄苔）、610（裂纹）、650（齿痕）、695（厚苔）。
返回 (C, H, W) float32，通道顺序与传入的 wavelengths 一致。
"""

import os
import numpy as np
from PIL import Image


def _stem_of(path):
    name = os.path.basename(path.rstrip("\\/"))
    if os.path.isdir(path):
        return name
    return os.path.splitext(name)[0]


def band_file(folder, wavelength):
    """返回 folder 中该波长图像路径；找不到则 None。"""
    for ext in (".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"):
        path = os.path.join(folder, f"{int(wavelength)}{ext}")
        if os.path.isfile(path):
            return path
    return None


def list_bands(folder):
    bands = []
    if not os.path.isdir(folder):
        return bands
    for name in os.listdir(folder):
        stem, ext = os.path.splitext(name)
        if ext.lower() in {".jpeg", ".jpg", ".png"} and stem.isdigit():
            bands.append(int(stem))
    bands.sort()
    return bands


def list_band_folders(raw_dir, wavelengths):
    """列出含齐指定波长文件的样本文件夹，按名字排序。"""
    if not os.path.isdir(raw_dir):
        return []
    wanted = [int(w) for w in wavelengths]
    out = []
    for name in sorted(os.listdir(raw_dir)):
        folder = os.path.join(raw_dir, name)
        if not os.path.isdir(folder):
            continue
        if all(band_file(folder, w) for w in wanted):
            out.append(folder)
    return out


def read_band_cube(folder, wavelengths):
    """
    读取指定波长，堆成 (C, H, W) float32。
    各波段空间尺寸不一致时，缩放到第一张的大小。
    """
    paths = []
    for w in wavelengths:
        path = band_file(folder, w)
        if path is None:
            raise FileNotFoundError(f"{folder} 缺少波长 {w}")
        paths.append(path)

    planes = []
    target_hw = None
    for path in paths:
        img = Image.open(path).convert("L")
        arr = np.asarray(img, dtype=np.float32)
        if target_hw is None:
            target_hw = arr.shape
        elif arr.shape != target_hw:
            img = img.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32)
        planes.append(arr)
    return np.stack(planes, axis=0)
