# -*- coding: utf-8 -*-
"""
训练主脚本
====================================================
用法：
    # 真实训练（需要先准备好 labels_4band.csv）
    python train.py --labels labels_4band.csv

    # 无标签干跑（验证数据加载、模型前向、反向传播是否正常）
    python train.py --dry-run

常用可选参数：--epochs 60 --batch-size 16 --lr 1e-3
"""

import os
import sys
import json
import time
import argparse

# 保证能 import 本项目模块（config / data / models / utils）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.console import setup_console
setup_console()

import numpy as np
import torch
import torch.nn as nn

import config
from data.dataset import build_dataloaders, MatDataset, list_sample_paths
from data.transforms import get_train_transform, get_val_transform, load_stats
from models import build_model, MultiTaskMobileNetV3
from utils.losses import MultiTaskLoss, compute_class_weights
from utils.metrics import accuracy_per_task, AverageMeter


# ============================================================
# 工具函数
# ============================================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def collect_train_labels(full_dataset, train_indices=None):
    """收集训练集全部标签为 (N, T) numpy 数组，用于计算类别权重。"""
    if full_dataset.labels is None:
        return None
    all_labels = []
    for stem in full_dataset.filenames():
        stem = os.path.splitext(stem)[0]
        lab = full_dataset.labels.get(stem)
        all_labels.append(lab if lab is not None
                          else np.full(len(full_dataset.task_names), -1, dtype=np.int64))
    return np.stack(all_labels, axis=0)


# ============================================================
# 训练 / 验证各一个 epoch
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    acc_meters = [AverageMeter() for _ in config.TASK_NAMES]

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss, per_task = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), x.size(0))
        accs = accuracy_per_task([lg.detach() for lg in logits], y)
        for m, a in zip(acc_meters, accs):
            m.update(a, x.size(0))

    return loss_meter.avg, [m.avg for m in acc_meters]


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meters = [AverageMeter() for _ in config.TASK_NAMES]

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss, per_task = criterion(logits, y)

        loss_meter.update(loss.item(), x.size(0))
        accs = accuracy_per_task(logits, y)
        for m, a in zip(acc_meters, accs):
            m.update(a, x.size(0))

    return loss_meter.avg, [m.avg for m in acc_meters]


def build_scheduler(optimizer, epochs, warmup_epochs):
    """构建学习率调度器（含可选预热）。"""
    if config.LR_SCHEDULER == "cosine":
        main = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
    elif config.LR_SCHEDULER == "step":
        main = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    else:
        return None

    if warmup_epochs > 0:
        def warmup_lr(ep):
            return min(1.0, (ep + 1) / warmup_epochs)
        warm = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr)
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warm, main], milestones=[warmup_epochs]
        )
    return main


# ============================================================
# 干跑模式：验证整个管线（不需要真实标签）
# ============================================================
def dry_run(device):
    print("\n[干跑模式] 无标签，用占位标签验证数据加载 + 前向 + 反向。\n")

    # 用 -1 占位标签构造一个临时标签表，驱动 Dataset 走完整流程
    files = list_sample_paths(
        config.SAMPLE_DIR, wavelengths=config.SELECTED_WAVELENGTHS,
    )
    stems = [os.path.basename(p.rstrip("\\/")) for p in files[:64]]

    stats = load_stats(config.STATS_FILE, config.SELECTED_BANDS) \
        if config.NORMALIZATION == "zscore" else None

    train_ds = MatDataset(
        config.SAMPLE_DIR, config.TASK_NAMES, config.SELECTED_BANDS,
        config.IMG_SIZE, config.NORMALIZATION, stats,
        labels_csv=None, transform=get_train_transform(config.AUGMENT),
        wavelengths=config.SELECTED_WAVELENGTHS,
    )
    # 干跑只走前 64 个文件夹，避免把 2000+ 未标注样本全读一遍
    train_ds.paths = train_ds.paths[:64]

    # 手工塞一个假的 labels dict 以演示带标签路径
    train_ds.labels = {s: np.zeros(len(config.TASK_NAMES), dtype=np.int64) for s in stems}
    train_ds.has_labels = True

    loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=False, num_workers=0)

    model = build_model(config.NUM_CLASSES, config.IN_CHANNELS, config.MODEL_NAME, config.HEAD_DROPOUT).to(device)
    criterion = MultiTaskLoss(config.NUM_CLASSES, config.TASK_LOSS_WEIGHTS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    print(f"数据集样本数: {len(train_ds)}")
    print(f"模型参数总量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    print(f"设备: {device}")

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        print(f"  batch {i}: x.shape={tuple(x.shape)}, y.shape={tuple(y.shape)}, "
              f"x 值域=[{x.min().item():.3f}, {x.max().item():.3f}]")
        logits = model(x)
        print(f"           logits 各任务形状: {[tuple(lg.shape) for lg in logits]}")
        loss, per_task = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accs = accuracy_per_task(logits, y)
        print(f"           loss={loss.item():.4f}, 各任务acc={[f'{a:.3f}' for a in accs]}")
        if i >= 2:   # 跑 3 个 batch 足够验证
            break

    print("\n[干跑通过] 数据加载、模型前向、反向传播均正常。")
    print("下一步：1) python build_labels_from_encoded_task.py 从 LabelMe JSON 生成 labels_4band.csv；")
    print("       2) python train.py --labels <labels_4band.csv> 正式训练。\n")


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="多任务 MobileNetV3 舌象分类训练")
    parser.add_argument("--labels", type=str, default=config.LABELS_CSV,
                        help="标签 CSV 路径（None 则进入干跑）")
    parser.add_argument("--dry-run", action="store_true", help="无标签干跑验证管线")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--resume", type=str, default=None, help="断点续训 checkpoint 路径")
    args = parser.parse_args()

    device = get_device()
    set_seed(config.SEED)

    labels_csv = args.labels
    if args.dry_run or labels_csv is None or not os.path.exists(labels_csv):
        if labels_csv is not None and not os.path.exists(labels_csv) and not args.dry_run:
            print(f"[警告] 未找到标签文件: {labels_csv}，自动进入干跑模式。")
        dry_run(device)
        return

    print(f"\n==== 开始正式训练 ====")
    print(f"标签文件: {labels_csv}")
    print(f"设备: {device}")
    print(f"数据目录: {config.SAMPLE_DIR}")
    print(f"特征波长: {config.SELECTED_WAVELENGTHS}  (共 {config.IN_CHANNELS} 通道)")

    # ---- 归一化统计 ----
    stats = load_stats(config.STATS_FILE, config.SELECTED_BANDS) \
        if config.NORMALIZATION == "zscore" else None

    # ---- 数据 ----
    train_loader, val_loader, full_ds = build_dataloaders(
        config.SAMPLE_DIR, config.TASK_NAMES, config.SELECTED_BANDS,
        img_size=config.IMG_SIZE, normalization=config.NORMALIZATION, stats=stats,
        labels_csv=labels_csv, batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS, val_split=config.VAL_SPLIT, seed=config.SEED,
        train_transform=get_train_transform(config.AUGMENT),
        val_transform=get_val_transform(),
        wavelengths=config.SELECTED_WAVELENGTHS,
    )
    print(f"训练样本数: {len(train_loader.dataset)}, 验证样本数: {len(val_loader.dataset)}")

    # ---- 类别权重（不均衡处理）----
    class_weights = None
    if config.USE_CLASS_WEIGHTS:
        labels_arr = collect_train_labels(full_ds)
        class_weights = compute_class_weights(labels_arr, config.NUM_CLASSES)
        print("已按训练集类别频率计算加权损失权重。")

    # ---- 模型 / 损失 / 优化器 ----
    model = build_model(config.NUM_CLASSES, config.IN_CHANNELS,
                        config.MODEL_NAME, config.HEAD_DROPOUT).to(device)
    criterion = MultiTaskLoss(config.NUM_CLASSES, config.TASK_LOSS_WEIGHTS, class_weights)
    criterion.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=config.WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, args.epochs, config.WARMUP_EPOCHS)

    start_epoch = 0
    best_acc = -1.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", -1.0)
        print(f"已从 {args.resume} 续训，起始 epoch={start_epoch}")

    # ---- 日志 ----
    log_path = os.path.join(config.OUTPUT_DIR, "train_log.csv")
    log_file = open(log_path, "w", encoding="utf-8")
    header = ["epoch", "train_loss", "val_loss"] + \
             [f"train_{n}" for n in config.TASK_NAMES] + \
             [f"val_{n}" for n in config.TASK_NAMES] + ["mean_val_acc", "lr"]
    log_file.write(",".join(header) + "\n")

    # ---- 训练循环 ----
    print(f"\n开始训练 {args.epochs} epochs ...\n")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_accs = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accs = validate(model, val_loader, criterion, device)
        mean_val = float(np.mean(val_accs))

        if scheduler is not None:
            scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        # 保存最佳（按多任务平均验证准确率）
        is_best = mean_val > best_acc
        if is_best:
            best_acc = mean_val
            torch.save(model.state_dict(),
                       os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))

        # 每轮保存 checkpoint（可断点续训）
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "best_acc": best_acc},
                   os.path.join(config.CHECKPOINT_DIR, "last_checkpoint.pth"))

        # 写日志
        row = [epoch, f"{train_loss:.4f}", f"{val_loss:.4f}"] + \
              [f"{a:.4f}" for a in train_accs] + [f"{a:.4f}" for a in val_accs] + \
              [f"{mean_val:.4f}", f"{lr_now:.6f}"]
        log_file.write(",".join(str(x) for x in row) + "\n")
        log_file.flush()

        # 控制台打印
        acc_str = "  ".join(f"{n}={a:.3f}" for n, a in zip(config.TASK_NAMES, val_accs))
        print(f"[Epoch {epoch:3d}/{args.epochs}] "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"mean_val_acc={mean_val:.4f}  ({'best' if is_best else '     '})  "
              f"耗时={time.time()-t0:.1f}s")
        print(f"            验证各任务准确率 -> {acc_str}")

    log_file.close()
    print(f"\n==== 训练完成 ====")
    print(f"最佳平均验证准确率: {best_acc:.4f}")
    print(f"最佳权重已保存到: {os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')}")
    print(f"训练日志: {log_path}")


if __name__ == "__main__":
    main()
