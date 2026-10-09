# 高光谱舌象 · 4 波段多任务分类

以后的训练代码都在这个目录里改、在这个目录里跑。  
高光谱图像和标注软件不在这里，仍在：

`E:\label\04_encoded_task`

标注只完成了一部分。新标完的样本继续放在那个目录，然后回到本目录重新导出标签、再训练。

## 现在在训什么

PyTorch + MobileNetV3-Large，4 个二分类任务，各对应一个波长：

| 任务 | 波长 | 类别 |
|---|---|---|
| 薄苔 | 590 nm | 无 / 有 |
| 裂纹 | 610 nm | 无 / 有 |
| 齿痕 | 650 nm | 无 / 有 |
| 厚苔 | 695 nm | 无 / 有 |

每个样本是 `dataset\raw_registered\<样本名>\` 下的一组灰度图，文件名就是波长，例如 `590.jpeg`。模型只读上面四个波长。

## 目录

```
高光谱舌象库/
├── config.py                          # 数据地址、波长、任务、超参
├── train.py                           # 训练
├── build_labels_from_encoded_task.py  # 标注 JSON → labels_4band.csv
├── labels_4band.csv                   # 当前已导出的监督标签
├── train_log.csv                      # 最近一次 60 epoch 训练记录
├── data/                              # 读波段图、预处理、DataLoader
├── models/                            # 4 通道 MobileNetV3，4 个分类头
├── utils/                             # 损失、指标、控制台编码
├── compute_stats.py                   # 仅 zscore 归一化时需要
├── export_onnx.py / export_tflite.py  # 导出
├── select_bands.py                    # 旧 .mat 立方体的波段筛选，当前训练不用
├── deploy/                            # 树莓派推理说明
└── checkpoints/                       # 本地权重，不上传 GitHub
```

## 日常命令

在本目录打开终端：

```bash
pip install -r requirements.txt
python build_labels_from_encoded_task.py
python train.py --dry-run
python train.py --labels labels_4band.csv
```

补标之后只要重跑导出脚本，再训练。不必把图像拷进本目录。

断点续训：

```bash
python train.py --labels labels_4band.csv --resume checkpoints/last_checkpoint.pth
```

## 标签规则

`labels_4band.csv` 由标注 JSON 生成。

- 勾了「舍弃」或「抖动不适合标注」的样本不写入。
- 薄苔、裂纹、厚苔：有对应形状为 1，没有为 0。
- 齿痕：有框为 1；已经标了薄苔或厚苔但没有齿痕框，记为确定无齿痕 0；还没标苔的样本记 -1，训练时忽略。

2026-10-09 这份表里有 341 条。薄苔和厚苔阳性都是 332 / 阴性 9，准确率会被多数类抬高，看结果时要单独看裂纹和齿痕。

## 数据从哪里来

| 内容 | 位置 |
|---|---|
| 配准窄带图 | `E:\label\04_encoded_task\dataset\raw_registered` |
| 配准彩色图 | `E:\label\04_encoded_task\dataset\color_images_registered` |
| LabelMe JSON | `E:\label\04_encoded_task\dataset\labels\hsi` |
| 标注协议 | `E:\label\04_encoded_task\dataset\labels\annotation_protocol.md` |
| 波长对照 | `E:\label\04_encoded_task\波长对照.xlsx` |

换电脑时只改 `config.py` 里的 `ENCODED_TASK_ROOT`。权重和日志仍写在代码旁边。

旧的 `.mat` 立方体还在 `D:\舌象数据库\高光谱舌象数据库\其他hyperspectral`，当前 4 波段训练不读它。
