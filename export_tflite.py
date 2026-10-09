# -*- coding: utf-8 -*-
"""
ONNX -> TFLite 转换（树莓派 4B 部署用）
====================================================
依赖：
    pip install onnx2tf tensorflow-cpu   # 转换环节在 PC 上做
转换流程：
    model.onnx  --onnx2tf-->  saved_model  --TFLiteConverter-->  model.tflite

支持：
    - float32 直接转换
    - int8 量化（需少量校准数据，进一步提速/瘦身，树莓派 CPU 上更友好）

用法：
    python export_tflite.py                     # 默认 float32
    python export_tflite.py --quantize int8     # int8 量化
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.console import setup_console
setup_console()
import numpy as np

import config


def convert_onnx_to_savedmodel(onnx_path, saved_model_dir):
    """用 onnx2tf 把 onnx 转成 TensorFlow SavedModel。"""
    import onnx2tf
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=saved_model_dir,
        non_verbose=True,
    )
    print(f"[完成] SavedModel 已生成: {saved_model_dir}")


def representative_dataset(num_samples=20):
    """int8 量化校准数据：随机生成与训练输入同分布的 [0,1] 数据（示例）。"""
    for _ in range(num_samples):
        # 训练时输入是 [0,1]（minmax 归一化），这里用均匀随机近似
        data = np.random.rand(1, config.IN_CHANNELS,
                              config.IMG_SIZE, config.IMG_SIZE).astype(np.float32)
        yield [data]


def convert_savedmodel_to_tflite(saved_model_dir, out_path, quantize="float32"):
    """把 SavedModel 转成 TFLite，可选 int8 量化。"""
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    if quantize == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
    else:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"[完成] TFLite 已生成: {out_path}  ({os.path.getsize(out_path)/1e6:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="ONNX -> TFLite 转换")
    parser.add_argument("--onnx", type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "model.onnx"))
    parser.add_argument("--quantize", type=str, default="float32",
                        choices=["float32", "int8"])
    args = parser.parse_args()

    saved_model_dir = os.path.join(config.CHECKPOINT_DIR, "saved_model")
    out_path = os.path.join(config.CHECKPOINT_DIR, "model.tflite")

    print("== 步骤 1/2: ONNX -> SavedModel (onnx2tf) ==")
    convert_onnx_to_savedmodel(args.onnx, saved_model_dir)

    print("== 步骤 2/2: SavedModel -> TFLite ==")
    convert_savedmodel_to_tflite(saved_model_dir, out_path, args.quantize)

    print("\n部署：把 model.tflite 与 deploy/infer_tflite.py 一起复制到树莓派。")
    print("详见 deploy/README_树莓派部署.md")


if __name__ == "__main__":
    main()
