# 树莓派 4B 部署指南（PyTorch → ONNX → TFLite）

本文档说明如何把训练好的多任务 MobileNetV3 模型部署到树莓派 4B 上做推理。

## 一、整体流程

```
训练 (PC, GPU)                      转换 (PC)                       推理 (树莓派 4B)
best_model.pth ──export_onnx.py──▶ model.onnx ──export_tflite.py──▶ model.tflite ──infer_tflite.py──▶ 预测结果
                                          └─(onnx2tf → tflite)─┘
```

> 为什么不直接用 PyTorch 在树莓派上推理？因为树莓派 4B 是 ARM CPU，
> 跑 PyTorch 又慢又吃内存；TFLite（尤其 int8 量化后）在 ARM CPU 上快得多、省内存。

## 二、PC 端：导出 ONNX 与 TFLite

在电脑上（本项目根目录，已安装 requirements.txt）执行：

```bash
# 1) 导出 ONNX（会自动做 onnxruntime 校验）
python export_onnx.py --checkpoint checkpoints/best_model.pth

# 2) 安装转换依赖（仅 PC 需要，树莓派不用装 tensorflow）
pip install onnx2tf tensorflow-cpu onnxruntime

# 3) ONNX -> TFLite（默认 float32）
python export_tflite.py
# 或 int8 量化（体积更小、ARM CPU 更快，树莓派上推荐）
python export_tflite.py --quantize int8
```

产物：
- `checkpoints/model.onnx`
- `checkpoints/model.tflite`

> 补充：还有一种更直接的方式是 Google 官方的 `ai-edge-torch`（PyTorch 直接转 TFLite，
> 无需经过 ONNX）。若 onnx2tf 在你的环境遇到算子不支持，可改用：
> `pip install ai-edge-torch`，然后用 `ai_edge_torch.convert(model, (x,))` 直接导出 .tflite。
> 本项目的 `ExportWrapper` 已把 Softmax 内置，两种方式都直接输出 4 个任务的概率
> （薄苔、裂纹、齿痕、厚苔）。

## 三、树莓派 4B 端：安装依赖

在树莓派上（Raspberry Pi OS，建议 64-bit）：

```bash
# 基础科学计算（用于读 .mat 和预处理）
sudo apt update
sudo apt install -y python3-pip python3-numpy python3-scipy
pip3 install h5py

# TFLite 运行时（推荐轻量的 tflite-runtime；装不上就用 ai-edge-litert）
pip3 install tflite-runtime
# 备用：
# pip3 install ai-edge-litert
```

## 四、复制文件到树莓派

把整个项目目录复制过去（包含 `config.py`、`data/`、`deploy/`，
因为推理脚本复用了 `data/` 里的预处理函数，保证与训练一致）：

```bash
scp -r 高光谱舌象库 pi@树莓派IP:~/tongue/
```

## 五、运行推理

```bash
cd ~/tongue/高光谱舌象库
python deploy/infer_tflite.py --model checkpoints/model.tflite --mat 某个.mat
```

当前训练读的是 `raw_registered` 里的四张波长图。`infer_tflite.py` 仍走旧的 `.mat` 立方体接口，部署前要确认输入和 `config.py` 里的 4 个波长一致。

输出按 `config.TASK_NAMES` 打印，顺序是薄苔、裂纹、齿痕、厚苔。

## 六、关键注意事项

1. **预处理必须与训练完全一致**：波段选择（`config.SELECTED_BANDS`）、
   缩放（224）、归一化方式（`config.NORMALIZATION`）三者必须和训练时相同。
   `deploy/infer_tflite.py` 直接复用 `data/transforms.py` 里的 `preprocess_cube`，
   避免了另写一套导致的精度下降。**不要**单独重写预处理。

2. **Softmax 已内置**：导出时用了 `ExportWrapper`，模型输出的就是 4 个任务的 Softmax 概率，
   推理端无需再做 softmax。

3. **int8 量化**：树莓派 4B 的 CPU 对 int8 有加速，`--quantize int8` 一般能再快 2~3 倍、
   体积缩小到 1/4。如果量化后精度明显下降，可回到 float32 版本。

4. **性能参考**：MobileNetV3-Large（4 通道输入）单张 224×224 在树莓派 4B 上，
   float32 约数十毫秒量级，int8 更快，实时性足够。

5. **输出顺序**：4 个输出按 `config.TASK_NAMES` 顺序
   （薄苔/裂纹/齿痕/厚苔），对应 ONNX 输出名 `config.TASK_NAMES_EN`。
