# -*- coding: utf-8 -*-
"""
树莓派 4B 上的 TFLite 推理脚本
====================================================
读取单个 .mat 文件 -> 预处理（与训练完全一致）-> TFLite 推理 -> 打印 6 个任务预测结果。

依赖（树莓派上）：
    pip install numpy scipy h5py
    pip install tflite-runtime        # 推荐，轻量
    # 若 tflite-runtime 装不上，可改用 ai-edge-litert 或完整 tensorflow

用法：
    python infer_tflite.py --model model.tflite --mat "某个.mat 文件"

重要：
    预处理（选波段/resize/归一化）必须与训练一致，这里直接复用项目的
    data.mat_io 与 data.transforms，避免另写一套造成精度下降。
"""

import os
import sys
import argparse

# 把项目根目录加入 path，以便复用 data/config 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.console import setup_console
setup_console()

import numpy as np

import config
from data.mat_io import read_mat_cube
from data.transforms import preprocess_cube, load_stats


def load_tflite_interpreter(model_path):
    """按可用性顺序加载 TFLite 解释器（tflite-runtime / ai-edge-litert / tensorflow）。"""
    try:
        import tflite_runtime.interpreter as tflite
        return tflite.Interpreter(model_path=model_path,
                                  num_threads=4)
    except ImportError:
        pass
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter(model_path=model_path, num_threads=4)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=model_path, num_threads=4)
    except ImportError:
        raise ImportError("未找到 TFLite 运行时，请安装 tflite-runtime 或 ai-edge-litert")


def preprocess_mat(mat_path):
    """读取并预处理单个 .mat，返回 (1, C, H, W) float32。"""
    cube = read_mat_cube(mat_path)                       # (31, 256, 256)

    stats = None
    if config.NORMALIZATION == "zscore":
        stats = load_stats(config.STATS_FILE, config.SELECTED_BANDS)

    x = preprocess_cube(
        cube, config.SELECTED_BANDS, config.IMG_SIZE,
        config.NORMALIZATION, stats,
    )                                                    # (C, 224, 224) [0,1]
    return x[None, ...].astype(np.float32)               # (1, C, 224, 224)


def main():
    parser = argparse.ArgumentParser(description="树莓派 TFLite 推理")
    parser.add_argument("--model", type=str, default="model.tflite")
    parser.add_argument("--mat", type=str, required=True, help="待预测的 .mat 文件")
    args = parser.parse_args()

    # 1) 预处理
    x = preprocess_mat(args.mat)

    # 2) 加载模型
    interpreter = load_tflite_interpreter(args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"输入: {input_details[0]['shape']}  dtype={input_details[0]['dtype']}")
    print(f"输出数量: {len(output_details)}")

    # 3) 推理
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()

    # 4) 取 6 个输出概率，打印每个任务的预测类别
    print("\n==== 预测结果 ====")
    for i, out in enumerate(output_details):
        prob = interpreter.get_tensor(out["index"])[0]   # (num_classes,)
        pred = int(np.argmax(prob))
        task_cn = config.TASK_NAMES[i]
        cls_names = config.CLASS_NAMES.get(task_cn, [str(k) for k in range(len(prob))])
        cls_name = cls_names[pred] if pred < len(cls_names) else str(pred)
        top_probs = ", ".join(
            f"{cls_names[k]}:{prob[k]:.2f}" for k in np.argsort(prob)[::-1][:3]
        )
        print(f"[{task_cn}] 预测={cls_name}  (置信度 {prob[pred]:.3f})")
        print(f"           Top3 -> {top_probs}")


if __name__ == "__main__":
    main()
