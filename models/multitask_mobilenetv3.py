# -*- coding: utf-8 -*-
"""
多任务 MobileNetV3 模型
====================================================
在 torchvision 的 MobileNetV3 基础上做了两处改动：
1. 把第一层卷积的输入通道数从 3 改成 4（对应 590/610/650/695 nm）。
2. 删掉原 1000 类分类头，换成 4 个互相独立的多任务分类头。

每个任务头输出「logits」（未经 Softmax），训练时用 nn.CrossEntropyLoss
（其内部自带 Softmax）。导出/部署时用 ExportWrapper 额外套一层 Softmax，
使导出的模型直接输出概率。
"""

import torch
import torch.nn as nn
from torchvision import models


class TaskHead(nn.Module):
    """
    单个任务的分类头：
    Linear -> ReLU -> Dropout -> Linear(logits)

    说明：这里输出 logits 而非概率，Softmax 由损失函数内部完成；
    推理/导出时通过 ExportWrapper 或 predict() 显式做 Softmax。
    """

    def __init__(self, in_features, num_classes, dropout=0.2, hidden=None):
        super().__init__()
        hidden = hidden if hidden is not None else max(128, in_features // 2)
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.head(x)


class MultiTaskMobileNetV3(nn.Module):
    """MobileNetV3 骨干 + 4 个独立分类头的多任务模型。"""

    def __init__(self, num_classes_list, in_channels=4,
                 model_name="mobilenet_v3_large", dropout=0.2):
        super().__init__()
        self.num_classes_list = list(num_classes_list)
        self.in_channels = in_channels

        # ---- 1) 构建 MobileNetV3 骨干（不加载预训练权重）----
        if model_name == "mobilenet_v3_large":
            backbone = models.mobilenet_v3_large(weights=None)
        elif model_name == "mobilenet_v3_small":
            backbone = models.mobilenet_v3_small(weights=None)
        else:
            raise ValueError(f"不支持的模型: {model_name}")

        # ---- 2) 修改第一层卷积：输入通道 3 -> in_channels ----
        # torchvision 中 features[0] 是 Conv2dNormActivation(Conv+BN+Hardswish)，
        # 其中 features[0][0] 就是第一层 Conv2d(3, 16, 3, stride=2, padding=1, bias=False)。
        first_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=(first_conv.bias is not None),
        )
        backbone.features[0][0] = new_conv

        # ---- 3) 取骨干最终特征维度，删除原分类头 ----
        # MobileNetV3 结构：features -> avgpool -> classifier[Linear(960/576, ...)]
        # 用 classifier[0] 的 in_features 作为各任务头的输入维度。
        in_features = backbone.classifier[0].in_features
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        # 原分类头不再需要（删除引用，避免误用）
        del backbone.classifier

        # ---- 4 个独立的多任务分类头 ----
        self.heads = nn.ModuleList([
            TaskHead(in_features, n, dropout=dropout) for n in num_classes_list
        ])

    def _extract_feature(self, x):
        """提取骨干特征向量 (B, in_features)。"""
        feat = self.features(x)          # (B, C, 7, 7)
        feat = self.avgpool(feat)        # (B, C, 1, 1)
        return torch.flatten(feat, 1)    # (B, C)

    def forward(self, x):
        """返回各任务的 logits 列表，每个形状 (B, num_classes_i)。"""
        feat = self._extract_feature(x)
        return [head(feat) for head in self.heads]

    @torch.no_grad()
    def predict(self, x):
        """推理：返回各任务的 Softmax 概率列表。"""
        logits = self.forward(x)
        return [torch.softmax(lg, dim=-1) for lg in logits]

    # ---- 可选：冻结/解冻骨干 ----
    def freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = True


class ExportWrapper(nn.Module):
    """
    导出专用包装器：在模型输出上加 Softmax，使导出的 ONNX/TFLite 直接输出概率。
    注意：训练仍用原模型的 logits，二者不冲突。
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits = self.model(x)
        return [torch.softmax(lg, dim=-1) for lg in logits]


def build_model(num_classes_list, in_channels=4, model_name="mobilenet_v3_large",
                dropout=0.2):
    """工厂函数：构建多任务 MobileNetV3 模型。"""
    return MultiTaskMobileNetV3(
        num_classes_list=num_classes_list,
        in_channels=in_channels,
        model_name=model_name,
        dropout=dropout,
    )
