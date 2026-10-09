# -*- coding: utf-8 -*-
"""
从 E:\\label\\04_encoded_task 的 LabelMe JSON 生成 labels_4band.csv。

规则：
- flags.舍弃 = True 的样本整条丢弃，不进训练表。
- 薄苔 / 厚苔 / 裂纹：有对应形状为 1，没有为 0。
- 齿痕：已有薄苔或厚苔标注的样本，有框为 1，无框为 0（确定无齿痕）；
  尚未标薄厚苔的样本，齿痕仍为 -1，不把「没开始标」当成阴性。
"""

from __future__ import annotations

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.console import setup_console
setup_console()

import config


def band_file(folder, wavelength):
    for ext in (".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"):
        path = os.path.join(folder, f"{int(wavelength)}{ext}")
        if os.path.isfile(path):
            return path
    return None

ALIAS = {
    "薄苔": "thin",
    "厚苔": "thick",
    "舌苔": "thick",
    "thin": "thin",
    "thick": "thick",
    "coating": "thick",
    "裂纹": "crack",
    "crack": "crack",
    "fissure": "crack",
    "齿痕": "teeth",
    "teeth_mark": "teeth",
    "teeth": "teeth",
    "teeth_line": "teeth",
}


def _color_path(stem):
    for ext in (".jpeg", ".jpg", ".png"):
        path = os.path.join(config.COLOR_REGISTERED_DIR, stem + ext)
        if os.path.isfile(path):
            return path
    return ""


def main():
    json_dir = config.LABEL_JSON_DIR
    raw_dir = config.RAW_REGISTERED_DIR
    waves = list(config.SELECTED_WAVELENGTHS)
    rows = []
    skipped = 0
    missing_raw = 0
    missing_band = 0

    for name in sorted(os.listdir(json_dir)):
        if not name.lower().endswith(".json"):
            continue
        path = os.path.join(json_dir, name)
        with open(path, encoding="utf-8") as f:
            ann = json.load(f)
        flags = ann.get("flags") or {}
        if flags.get("舍弃") or flags.get("抖动不适合标注"):
            skipped += 1
            continue
        stem = os.path.splitext(name)[0]
        folder = os.path.join(raw_dir, stem)
        if not os.path.isdir(folder):
            missing_raw += 1
            continue
        if any(band_file(folder, w) is None for w in waves):
            missing_band += 1
            continue

        counts = {"thin": 0, "thick": 0, "crack": 0, "teeth": 0}
        for shape in ann.get("shapes") or []:
            key = ALIAS.get(str(shape.get("label", "")).strip())
            if key:
                counts[key] += 1

        thin = 1 if counts["thin"] else 0
        thick = 1 if counts["thick"] else 0
        crack = 1 if counts["crack"] else 0
        has_coating = counts["thin"] > 0 or counts["thick"] > 0
        if counts["teeth"]:
            teeth = 1
        elif has_coating:
            teeth = 0
        else:
            teeth = -1
        unsure = 1 if flags.get("不确定") else 0
        rows.append([
            stem, thin, crack, teeth, thick, unsure,
            counts["thin"], counts["thick"], counts["crack"], counts["teeth"],
            path, folder, _color_path(stem),
        ])

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out_path = config.LABELS_CSV
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "文件名", "薄苔", "裂纹", "齿痕", "厚苔", "不确定",
            "薄苔形状数", "厚苔形状数", "裂纹框数", "齿痕框数",
            "json路径", "raw路径", "color路径",
        ])
        writer.writerows(rows)

    n = len(rows)
    n_thin = sum(r[1] == 1 for r in rows)
    n_crack = sum(r[2] == 1 for r in rows)
    n_teeth = sum(r[3] == 1 for r in rows)
    n_teeth_ignore = sum(r[3] == -1 for r in rows)
    n_thick = sum(r[4] == 1 for r in rows)
    print(f"写入 {n} 条 -> {out_path}")
    print(f"  舍弃跳过 {skipped}，缺文件夹 {missing_raw}，缺四波长 {missing_band}")
    print(f"  薄苔阳性 {n_thin} / 裂纹阳性 {n_crack} / 厚苔阳性 {n_thick}")
    n_teeth_neg = sum(r[3] == 0 for r in rows)
    print(f"  齿痕阳性 {n_teeth} / 确定无齿痕 {n_teeth_neg} / 未标忽略 {n_teeth_ignore}")
    print("训练：python train.py --labels", out_path)


if __name__ == "__main__":
    main()
