# -*- coding: utf-8 -*-
"""
全局配置文件
====================================================
所有路径、波段选择、任务定义、训练超参都在这里统一修改。
其他脚本（train.py / export_*.py / deploy/*）都从本文件读取配置，
保证训练、导出、部署三者使用的波段选择与归一化方式完全一致。

当前数据源（本机 Linux，2026-10-09）：
  配准窄带图：/home/server/wxl/hyperspctral_image
  标注 JSON：本目录 hsi/（原 dataset/labels/hsi）
  可用监督样本见 LABELS_CSV（由 build_labels_from_encoded_task.py 生成）。
"""

import os
import numpy as np

# ============================================================
# 1. 路径
# ============================================================
# 代码、权重、CSV、日志都写在本目录。
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = PROJECT_DIR
OUTPUT_DIR = PROJECT_DIR
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---- 高光谱数据（本机）----
# 标注 JSON 在项目内 hsi/；配准波段图在上级目录的 hyperspctral_image/。
# 补标后重新导出 CSV，再在本目录训练。
LABEL_JSON_DIR = os.path.join(PROJECT_DIR, "hsi")
RAW_REGISTERED_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "hyperspctral_image")
COLOR_REGISTERED_DIR = os.path.join(PROJECT_DIR, "color_images_registered")  # 本机可选，没有则导出 CSV 时 color 列为空
LABEL_CLASSES_FILE = os.path.join(PROJECT_DIR, "labels", "classes.txt")
LABEL_PROTOCOL_FILE = os.path.join(PROJECT_DIR, "labels", "annotation_protocol.md")
WAVELENGTH_TABLE = os.path.join(PROJECT_DIR, "波长对照.xlsx")
# 兼容仍引用旧变量名的脚本
ENCODED_TASK_ROOT = PROJECT_DIR
DATASET_DIR = PROJECT_DIR

# band_folder：每个样本一个文件夹，文件名是波长（590.jpeg 等）
# mat：旧的单个 .mat 立方体，当前训练不用它
DATA_FORMAT = "band_folder"
SAMPLE_DIR = RAW_REGISTERED_DIR
MAT_DIR = SAMPLE_DIR  # 兼容旧脚本参数名，实际指向配准波段文件夹

# 标签 CSV：build_labels_from_encoded_task.py 从 LabelMe JSON 导出
# 列：文件名,薄苔,裂纹,齿痕,厚苔,（其后为计数与路径附加列）
# 未标注填 -1。已有薄厚苔、无齿痕框的样本，齿痕记 0（确定无齿痕）。
LABELS_CSV = os.path.join(OUTPUT_DIR, "labels_4band.csv")

# ============================================================
# 2. 波段选择（重要：训练/导出/部署必须完全一致）
# ============================================================
# 文件夹内图像以 2024 基准波长命名。2025 及以后的灯已在配准时对齐到最近基准灯，
# 因此 590.jpeg / 610.jpeg / 650.jpeg / 695.jpeg 在绝大多数样本中都存在。
# 物理中心波长若与文件名差若干 nm，见 波长对照.xlsx，不要跨年强行合并解释吸收峰。
SELECTED_WAVELENGTHS = [590, 610, 650, 695]  # 薄苔、裂纹、齿痕、厚苔
# 读完四张图后立方体已是 4 通道，预处理里按通道下标取层
SELECTED_BANDS = [0, 1, 2, 3]
IN_CHANNELS = len(SELECTED_WAVELENGTHS)

# ============================================================
# 3. 图像 / 归一化
# ============================================================
IMG_SIZE = 224

# 归一化方式：
#   "minmax"  —— 逐样本、逐波段做 1%~99% 分位数裁剪后缩放到 [0,1]。
#                优点：完全自包含，部署到树莓派时无需额外统计量，推荐使用。
#   "zscore"  —— 用数据集统计的逐波段 mean/std 做标准化（需先运行 compute_stats.py）。
NORMALIZATION = "minmax"

# zscore 模式使用的统计文件（由 compute_stats.py 生成）
STATS_FILE = os.path.join(OUTPUT_DIR, "band_stats.json")

# ============================================================
# 4. 任务定义（4 个多任务头，与四波长一一对应）
# ============================================================
TASK_NAMES = ["薄苔", "裂纹", "齿痕", "厚苔"]
NUM_CLASSES = [2, 2, 2, 2]

TASK_NAMES_EN = ["thin_coating", "crack", "tooth_marks", "thick_coating"]

CLASS_NAMES = {
    "薄苔": ["无薄苔", "有薄苔"],
    "裂纹": ["无裂纹", "有裂纹"],
    "齿痕": ["无齿痕", "有齿痕"],
    "厚苔": ["无厚苔", "有厚苔"],
}

TASK_LOSS_WEIGHTS = [1.0, 1.0, 1.0, 1.0]

# ============================================================
# 5. 模型
# ============================================================
MODEL_NAME = "mobilenet_v3_large"   # "mobilenet_v3_large" | "mobilenet_v3_small"
PRETRAINED = False                  # 输入是 4 通道，无法直接复用 ImageNet 的 3 通道权重
HEAD_DROPOUT = 0.2

# ============================================================
# 6. 训练超参
# ============================================================
BATCH_SIZE = 16
NUM_WORKERS = 4                  # Linux 可用多进程加速读图；Windows 若卡死改回 0
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 60
VAL_SPLIT = 0.2
SEED = 42

USE_CLASS_WEIGHTS = True
LR_SCHEDULER = "cosine"
WARMUP_EPOCHS = 2

# ============================================================
# 7. 数据增强开关（针对光谱窄带，见 data/transforms.py）
# ============================================================
AUGMENT = {
    "flip_h": 0.5,
    "flip_v": 0.0,
    "rotate": 0.5,
    "max_angle": 10.0,
    "crop": 0.5,
    "crop_scale": (0.85, 1.0),
    "brightness": 0.5,
    "brightness_range": (0.85, 1.15),
    "band_jitter": 0.5,
    "band_jitter_range": (0.9, 1.1),
    "noise": 0.3,
    "noise_sigma": 0.01,
    "cutout": 0.3,
    "cutout_ratio": (0.02, 0.15),
}
