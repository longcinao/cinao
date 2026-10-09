# -*- coding: utf-8 -*-
"""
多任务损失
====================================================
每个任务使用一个 CrossEntropyLoss（内部自带 Softmax），
按 task_weights 加权求和得到总损失。
支持：
- 任务级权重 task_weights（不同任务重要性/难易不同）
- 类别级权重 class_weights（应对类别不均衡）
- ignore_index=-1（未标注的样本自动忽略，支持部分标注）
"""

import numpy as np
import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    def __init__(self, num_classes_list, task_weights=None, class_weights=None,
                 label_smoothing=0.0):
        """
        - num_classes_list: 每个任务的类别数列表
        - task_weights:     每个任务的损失权重（默认全 1）
        - class_weights:    每个任务一个类别权重列表（或 None），形状需与类别数一致
        """
        super().__init__()
        n = len(num_classes_list)
        self.task_weights = [float(w) for w in (task_weights or [1.0] * n)]
        class_weights = class_weights or [None] * n

        self.criterions = nn.ModuleList()
        for n_cls, cw in zip(num_classes_list, class_weights):
            w = None
            if cw is not None:
                w = torch.tensor(cw, dtype=torch.float32)
            self.criterions.append(
                nn.CrossEntropyLoss(weight=w,
                                    label_smoothing=label_smoothing,
                                    ignore_index=-1)
            )

    def forward(self, logits_list, labels):
        """
        - logits_list: 模型输出的 logits 列表，每个 (B, num_classes_i)
        - labels:      (B, num_tasks) 的 LongTensor，-1 表示未标注（被忽略）
        返回 (total_loss, per_task_losses) ，per_task_losses 是 Python 列表（已 detach）。
        """
        total = torch.tensor(0.0, device=logits_list[0].device)
        per_task = []
        for i, (logits, criterion) in enumerate(zip(logits_list, self.criterions)):
            loss = criterion(logits, labels[:, i].long())
            per_task.append(float(loss.detach().cpu()))
            total = total + self.task_weights[i] * loss
        return total, per_task


def compute_class_weights(labels_array, num_classes_list):
    """
    根据训练集标签统计，计算每个任务的类别权重（应对类别不均衡）。
    - labels_array: (N, T) 的 numpy 数组，-1 表示未标注（不计入统计）
    - num_classes_list: 每个任务的类别数
    返回：list，每个元素是长度等于该任务类别数的权重列表（或 None）。
    公式（加权损失）：w_c = 样本总数 / (类别数 * 该类样本数)
    """
    labels_array = np.asarray(labels_array)
    class_weights = []
    for t, n_cls in enumerate(num_classes_list):
        col = labels_array[:, t]
        col = col[col >= 0]                      # 去掉未标注
        if len(col) == 0:
            class_weights.append(None)
            continue
        counts = np.bincount(col, minlength=n_cls).astype(np.float32)
        # 对出现次数为 0 的类别给一个极小权重，避免除零
        counts = np.where(counts == 0, 1e-6, counts)
        total = len(col)
        w = total / (n_cls * counts)
        class_weights.append(w.tolist())
    return class_weights
