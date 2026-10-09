# -*- coding: utf-8 -*-
"""
Dataset 与 DataLoader 构建
====================================================
默认读取 E:\\label\\04_encoded_task\\dataset\\raw_registered 下的样本文件夹：
四张窄带图 → 选 590/610/650/695 → 缩放 224 → 归一化 → 增强 → 张量。

标签从 labels_4band.csv 读取；-1 表示未标注（训练时 loss/指标会忽略）。
已有薄厚苔、但没有齿痕框的样本，齿痕为 0（确定无齿痕）。

训练/验证切分使用 Subset + 各自独立的 transform，避免 random_split
共享底层 dataset 导致 train/val 的增强相互污染。
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .band_folder_io import list_band_folders, read_band_cube, _stem_of
from .mat_io import list_mat_files, read_mat_cube


def _is_folder_sample(path):
    return os.path.isdir(path)


def list_sample_paths(sample_dir, wavelengths=None, labels_csv=None):
    """列出可训练样本路径。配准库为文件夹；旧数据为 .mat。"""
    if wavelengths:
        paths = list_band_folders(sample_dir, wavelengths)
    else:
        paths = list_mat_files(sample_dir)
        paths = [os.path.join(sample_dir, os.path.basename(p)) if not os.path.isabs(p) else p
                 for p in paths]
    if labels_csv:
        df = pd.read_csv(labels_csv)
        allowed = {
            str(x).strip().rsplit(".", 1)[0]
            for x in df.iloc[:, 0].tolist()
        }
        paths = [p for p in paths if _stem_of(p) in allowed]
    return paths


class MatDataset(Dataset):
    """高光谱舌象数据集（含标签）。兼容波段文件夹与 .mat。"""

    def __init__(self, sample_dir, task_names, selected_bands, img_size=224,
                 normalization="minmax", stats=None, labels_csv=None,
                 transform=None, return_filename=False, wavelengths=None):
        self.wavelengths = list(wavelengths) if wavelengths else None
        self.paths = list_sample_paths(sample_dir, self.wavelengths, labels_csv)
        self.task_names = list(task_names)
        self.selected_bands = list(selected_bands)
        self.img_size = img_size
        self.normalization = normalization
        self.stats = stats
        self.transform = transform
        self.return_filename = return_filename

        self.labels = self._load_labels(labels_csv) if labels_csv else None
        self.has_labels = self.labels is not None

    def _load_labels(self, csv_path):
        df = pd.read_csv(csv_path)
        n_tasks = len(self.task_names)
        mapping = {}
        for _, row in df.iterrows():
            stem = str(row.iloc[0]).strip().rsplit(".", 1)[0]
            mapping[stem] = row.iloc[1:1 + n_tasks].values.astype(np.int64)
        return mapping

    def __len__(self):
        return len(self.paths)

    def filenames(self):
        return [_stem_of(p) for p in self.paths]

    def _read_cube(self, path):
        if _is_folder_sample(path):
            if not self.wavelengths:
                raise ValueError(f"读取波段文件夹需要 wavelengths，收到 {path}")
            return read_band_cube(path, self.wavelengths)
        return read_mat_cube(path)

    def _load_sample(self, idx, transform):
        from .transforms import preprocess_cube

        path = self.paths[idx]
        stem = _stem_of(path)

        cube = self._read_cube(path)
        cube = preprocess_cube(
            cube, self.selected_bands, self.img_size,
            self.normalization, self.stats,
        )
        if transform is not None:
            cube = transform(cube)

        x = torch.from_numpy(cube.astype(np.float32))

        if self.has_labels:
            label = self.labels.get(stem, np.full(len(self.task_names), -1, dtype=np.int64))
            label = torch.from_numpy(label)
            if self.return_filename:
                return x, label, stem
            return x, label
        if self.return_filename:
            return x, stem
        return x

    def __getitem__(self, idx):
        return self._load_sample(idx, self.transform)


class Subset(Dataset):
    """带独立 transform 的数据子集（代替 torch 的 random_split 共享问题）。"""

    def __init__(self, base: MatDataset, indices, transform=None):
        self.base = base
        self.indices = list(indices)
        self.transform = transform
        self.has_labels = base.has_labels
        self.task_names = base.task_names

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base._load_sample(self.indices[idx], self.transform)


def _list_mat_files(mat_dir):
    """兼容旧名：现在可能返回波段文件夹。"""
    return list_sample_paths(mat_dir)


def build_dataloaders(sample_dir, task_names, selected_bands, img_size=224,
                      normalization="minmax", stats=None, labels_csv=None,
                      batch_size=16, num_workers=2, val_split=0.2, seed=42,
                      train_transform=None, val_transform=None, wavelengths=None):
    """
    构建训练集 / 验证集 DataLoader，返回 (train_loader, val_loader, full_dataset)。
    """
    full = MatDataset(
        sample_dir, task_names, selected_bands, img_size,
        normalization, stats, labels_csv, transform=None,
        wavelengths=wavelengths,
    )

    n_total = len(full)
    if n_total == 0:
        raise RuntimeError(
            f"没有可用样本。请检查 SAMPLE_DIR={sample_dir} 与 LABELS_CSV={labels_csv}"
        )
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    if n_train <= 0 or n_val < 0:
        raise RuntimeError(f"划分后训练集为空：N={n_total}, val_split={val_split}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()

    train_ds = Subset(full, train_idx, transform=train_transform)
    val_ds = Subset(full, val_idx, transform=val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=(len(train_ds) >= batch_size),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader, full
