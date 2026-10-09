# -*- coding: utf-8 -*-
"""
多任务评估指标
====================================================
主要输出：每个任务的准确率（accuracy），以及多任务平均准确率。
"""

import torch


def accuracy_per_task(logits_list, labels):
    """
    计算每个任务的准确率。
    - logits_list: 每个任务的 logits，形状 (B, num_classes_i)
    - labels:      (B, num_tasks) LongTensor，-1 表示未标注（跳过不计）
    返回：Python 浮点数列表（长度 = 任务数）。
    """
    accs = []
    for i, logits in enumerate(logits_list):
        pred = logits.argmax(dim=1)
        target = labels[:, i].long()
        valid = target >= 0
        if valid.sum() == 0:
            accs.append(0.0)
        else:
            correct = (pred[valid] == target[valid]).float().sum()
            accs.append((correct / valid.sum()).item())
    return accs


class AverageMeter:
    """简单的运行平均值累加器。"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0.0
