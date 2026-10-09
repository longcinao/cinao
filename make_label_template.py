# -*- coding: utf-8 -*-
"""
生成标签 CSV
====================================================
当前监督来自 E:\\label\\04_encoded_task\\dataset\\labels\\hsi 的 LabelMe JSON，
请运行 build_labels_from_encoded_task.py，不要手工从空模板填 2000 行。

本脚本保留为兼容入口：若已配置波段文件夹与 JSON，则转调上述导出；
否则仍扫描 SAMPLE_DIR 生成全 -1 模板。
"""

import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.console import setup_console
setup_console()
import config
from data.dataset import list_sample_paths


def main():
    json_dir = getattr(config, "LABEL_JSON_DIR", "")
    if json_dir and os.path.isdir(json_dir) and any(
        name.lower().endswith(".json") for name in os.listdir(json_dir)
    ):
        from build_labels_from_encoded_task import main as export_main
        export_main()
        return

    files = list_sample_paths(
        config.SAMPLE_DIR, wavelengths=getattr(config, "SELECTED_WAVELENGTHS", None),
    )
    out_path = os.path.join(config.OUTPUT_DIR, "labels_4band.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["文件名"] + config.TASK_NAMES)
        for p in files:
            stem = os.path.basename(p.rstrip("\\/"))
            if os.path.isfile(p):
                stem = os.path.splitext(stem)[0]
            writer.writerow([stem] + [-1] * len(config.TASK_NAMES))
    print(f"共生成 {len(files)} 行空标签模板 -> {out_path}")


if __name__ == "__main__":
    main()
