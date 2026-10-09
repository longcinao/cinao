# -*- coding: utf-8 -*-
"""数据读取与预处理子包。

刻意不在此处 import dataset（dataset 依赖 pandas + torch）。
这样树莓派上（deploy/infer_tflite.py）只读 .mat 做预处理时，
不会被动引入 torch/pandas 这两个在 ARM 上不必要/难装的重依赖。
"""
from .mat_io import read_mat_cube, list_mat_files
