# -*- coding: utf-8 -*-
"""
光谱数据预处理与数据增强
====================================================
输入输出都是 numpy 数组，shape (C, H, W)，dtype float32，取值范围 [0,1]。

设计原则（8 波段 LED 光谱数据的特殊性）：
1. 不用 torchvision 的 RGB 增强（ColorJitter 等按 RGB 三通道假设设计，
   会破坏光谱之间的物理关系）。
2. 空间增强（翻转/旋转/裁剪）对所有波段一视同仁，物理上安全。
3. 逐波段「LED 亮度抖动」是合理的：本数据是 8 路 LED 分别照明采集，
   每路 LED 的实际亮度存在波动，逐波段乘以一个小的随机系数正是模拟这种波动。
4. 归一化放在增强之前，保证 minmax 归一化（自包含）与增强不互相抵消。

重要：部署到树莓派时，推理端必须复用本文件的 preprocess_cube()
（读取 -> 选波段 -> resize -> 归一化），不能另写一套，否则精度会掉。
"""

import json
import numpy as np
from scipy.ndimage import zoom, rotate as nd_rotate, shift as nd_shift


# ============================================================
# 一、基础预处理（训练与推理共用）
# ============================================================
def resize_cube(cube, size):
    """把 (C, H, W) 双线性缩放到 (C, size, size)。"""
    h, w = cube.shape[1], cube.shape[2]
    if h == size and w == size:
        return cube
    return zoom(cube, (1.0, size / h, size / w), order=1)


def percentile_minmax(cube, low=1.0, high=99.0):
    """
    逐波段做分位数裁剪后缩放到 [0,1]（每样本自包含，无需全局统计）。
    先用 1%~99% 分位数压掉离群像素，再 min-max，鲁棒性更好。
    """
    out = np.empty_like(cube, dtype=np.float32)
    for c in range(cube.shape[0]):
        b = cube[c].astype(np.float32)
        lo = np.percentile(b, low)
        hi = np.percentile(b, high)
        out[c] = np.clip((b - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return out


def zscore_norm(cube, mean, std):
    """逐波段 z-score 标准化。mean/std 形状为 (C,)，来自 compute_stats.py。"""
    mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(std, dtype=np.float32).reshape(-1, 1, 1)
    return (cube.astype(np.float32) - mean) / (std + 1e-8)


def preprocess_cube(cube, selected_bands, img_size=224,
                    normalization="minmax", stats=None):
    """
    完整预处理：选波段 -> 缩放 -> 归一化。
    - cube: (31, H, W) 原始光谱立方体
    - selected_bands: 要保留的波段索引列表（长度 = IN_CHANNELS）
    - stats: zscore 模式下的 (mean, std)，两者 shape 均为 (len(selected_bands),)
    返回 (C, img_size, img_size) 的 float32 数组。
    """
    cube = np.asarray(cube, dtype=np.float32)[selected_bands]
    cube = resize_cube(cube, img_size)
    if normalization == "minmax":
        cube = percentile_minmax(cube)
    elif normalization == "zscore":
        if stats is None:
            raise ValueError("zscore 模式需要提供 stats=(mean, std)，请先运行 compute_stats.py")
        mean, std = stats
        cube = zscore_norm(cube, mean, std)
    else:
        raise ValueError(f"未知的归一化方式: {normalization}")
    return cube


def load_stats(stats_file, selected_bands):
    """
    读取 compute_stats.py 生成的统计文件，返回 (mean, std)。
    统计文件里存的就是「选中波段」的逐波段统计，直接返回。
    """
    with open(stats_file, "r", encoding="utf-8") as f:
        d = json.load(f)
    mean = np.asarray(d["mean"], dtype=np.float32)
    std = np.asarray(d["std"], dtype=np.float32)
    if len(mean) != len(selected_bands):
        raise ValueError(
            f"统计文件波段数({len(mean)}) 与 selected_bands({len(selected_bands)}) 不一致"
        )
    return mean, std


# ============================================================
# 二、数据增强（仅训练集使用）
# ============================================================
class SpectralAugmentation:
    """
    针对 8 波段光谱舌象数据的安全增强。

    参数 AUGMENT 是一个 dict，含义见 config.py 中的 AUGMENT 配置。
    所有增强都作用在 [0,1] 范围的数据上，结束时裁剪回 [0,1]。
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, cube):
        cube = cube.astype(np.float32, copy=False)

        # 1) 水平翻转
        if np.random.rand() < self.p.get("flip_h", 0.0):
            cube = cube[:, :, ::-1]

        # 2) 垂直翻转（默认关闭，舌象一般不做）
        if np.random.rand() < self.p.get("flip_v", 0.0):
            cube = cube[:, ::-1, :]

        # 3) 小角度旋转（逐波段，mode='reflect' 避免黑边）
        if np.random.rand() < self.p.get("rotate", 0.0):
            ang = np.random.uniform(-self.p.get("max_angle", 10.0),
                                    self.p.get("max_angle", 10.0))
            cube = np.stack([
                nd_rotate(c, ang, reshape=False, order=1, mode="reflect")
                for c in cube
            ], axis=0)

        # 4) 随机缩放裁剪（随机取一块子区域再放大回原尺寸）
        if np.random.rand() < self.p.get("crop", 0.0):
            lo, hi = self.p.get("crop_scale", (0.85, 1.0))
            scale = np.random.uniform(lo, hi)
            h, w = cube.shape[1], cube.shape[2]
            ch, cw = int(h * scale), int(w * scale)
            top = np.random.randint(0, h - ch + 1)
            left = np.random.randint(0, w - cw + 1)
            crop = cube[:, top:top + ch, left:left + cw]
            cube = zoom(crop, (1.0, h / ch, w / cw), order=1)

        # 5) 全局亮度抖动（整幅图所有波段同乘一个系数，模拟总光照变化）
        if np.random.rand() < self.p.get("brightness", 0.0):
            lo, hi = self.p.get("brightness_range", (0.85, 1.15))
            cube = cube * np.random.uniform(lo, hi)

        # 6) 逐波段 LED 亮度抖动（每路 LED 独立照明，各自独立波动）
        if np.random.rand() < self.p.get("band_jitter", 0.0):
            lo, hi = self.p.get("band_jitter_range", (0.9, 1.1))
            gain = np.random.uniform(lo, hi, size=(cube.shape[0], 1, 1)).astype(np.float32)
            cube = cube * gain

        # 7) 高斯噪声（模拟传感器噪声）
        if np.random.rand() < self.p.get("noise", 0.0):
            cube = cube + np.random.normal(0.0, self.p.get("noise_sigma", 0.01),
                                           size=cube.shape).astype(np.float32)

        # 8) 随机擦除（cutout，同一位置跨波段一起置 0，模拟遮挡/反光）
        if np.random.rand() < self.p.get("cutout", 0.0):
            lo, hi = self.p.get("cutout_ratio", (0.02, 0.15))
            ratio = np.random.uniform(lo, hi)
            h, w = cube.shape[1], cube.shape[2]
            bh, bw = int(h * ratio), int(w * ratio)
            top = np.random.randint(0, max(1, h - bh))
            left = np.random.randint(0, max(1, w - bw))
            cube[:, top:top + bh, left:left + bw] = 0.0

        # 结束裁剪回 [0,1]
        return np.clip(cube, 0.0, 1.0).astype(np.float32)


def get_train_transform(augment_cfg):
    """返回训练集的增强函数。"""
    return SpectralAugmentation(augment_cfg)


def get_val_transform():
    """验证集/推理端不做增强，直接返回原数据。"""
    return None
