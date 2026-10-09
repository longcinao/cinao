# -*- coding: utf-8 -*-
"""
导出 ONNX
====================================================
把训练好的 best_model.pth 导出为 ONNX，输出 6 个 Softmax 概率。
用法：
    python export_onnx.py --checkpoint checkpoints/best_model.pth
导出结果：
    checkpoints/model.onnx
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.console import setup_console
setup_console()
import torch

import config
from models import build_model, ExportWrapper


def main():
    parser = argparse.ArgumentParser(description="导出多任务模型为 ONNX")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()

    device = "cpu"   # 导出统一用 CPU，便于后续 TFLite 转换

    # 1) 重建模型（结构与训练完全一致）
    model = build_model(config.NUM_CLASSES, config.IN_CHANNELS,
                        config.MODEL_NAME, config.HEAD_DROPOUT)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    # 兼容两种保存格式：纯 state_dict 或带 'model' 键的 checkpoint
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    # 2) 套一层 Softmax，让导出模型直接输出概率
    export_model = ExportWrapper(model).to(device).eval()

    # 3) 示例输入 (batch=1, 8通道, 224, 224)
    dummy = torch.randn(1, config.IN_CHANNELS, config.IMG_SIZE, config.IMG_SIZE, device=device)

    out_path = os.path.join(config.CHECKPOINT_DIR, "model.onnx")
    output_names = config.TASK_NAMES_EN

    torch.onnx.export(
        export_model,
        dummy,
        out_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes={"input": {0: "batch"}},   # 仅输入支持动态 batch（输出数量固定为 6 个）
    )
    print(f"[完成] ONNX 已导出: {out_path}")
    print(f"输出顺序: {list(zip(output_names, config.TASK_NAMES))}")

    # 4) 用 onnxruntime 做一次校验（若已安装）
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        outs = sess.run(None, {"input": dummy.numpy()})
        print("[校验] onnxruntime 推理成功，输出形状:")
        for name, o in zip(output_names, outs):
            print(f"        {name}: {o.shape}  概率和={o.sum():.4f}")
    except ImportError:
        print("[提示] 未安装 onnxruntime，跳过校验。可执行: pip install onnxruntime")


if __name__ == "__main__":
    main()
