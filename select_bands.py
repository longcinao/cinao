# -*- coding: utf-8 -*-
"""
无监督波段筛选：从 31 个光谱波段中选出 K 个「信息量大、相互独立」的波段
====================================================================
背景：
  每个 .mat 内含 cube: (31, 256, 256)，31 个波段对应采集时拍的 31 张不同波段图。
  当前没有标签，因此「特征最明显」按无监督口径理解：
      = 信息量大（标准差/熵高） + 波段间不冗余（相关性低） + 信噪比尚可。

方法（三步）：
  1) 质量筛查：逐波段统计全图均值(亮度)与标准差，暗/噪波段一眼可见。
  2) 信息量排序：用「跨全部样本汇合的逐波段标准差」衡量信息量。
  3) 去冗余贪心选择：先取信息量最大的波段，之后每一步取
     「信息量高 且 与已选波段平均相关性低」的波段，直到选满 K 个。

用法：
  python select_bands.py --n-bands 8
  python select_bands.py --n-bands 8 --n-files 50        # 先拿前 50 个文件快速跑
  python select_bands.py --n-bands 8 --exclude 0 1 2     # 排除明显暗/噪波段
  python select_bands.py --n-bands 8 --alpha 1.0         # 更强调去冗余

输出：
  - 控制台：逐波段指标表 + 推荐 K 个波段 + 可直接粘贴进 config.py 的 SELECTED_BANDS
  - band_metrics.csv：逐波段指标（mean/std/entropy/平均相关性/是否选中）
  - band_selection.png：信息量曲线 + 相关性热图 + 选中波段预览拼接图
"""

import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.mat_io import read_mat_cube, list_mat_files


def shannon_entropy(x, bins=256):
    """单波段图像的 Shannon 信息熵（1%~99% 分位数范围内做直方图）。"""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return 0.0
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if hi <= lo:
        return 0.0
    hist, _ = np.histogram(x, bins=bins, range=(lo, hi))
    p = hist.astype(np.float64) / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def greedy_select(score, avg_corr, k, alpha=0.5, exclude=()):
    """贪心去冗余选择：信息量高且与已选波段平均相关性低。"""
    exclude = set(exclude)
    remain = [b for b in range(len(score)) if b not in exclude]
    selected = []
    while len(selected) < k and remain:
        if not selected:
            i = max(remain, key=lambda b: score[b])
        else:
            best_i, best_v = None, -np.inf
            for b in remain:
                redun = float(np.mean(np.abs(avg_corr[b, selected])))
                v = score[b] - alpha * redun
                if v > best_v:
                    best_v, best_i = v, b
            i = best_i
        selected.append(i)
        remain.remove(i)
    return selected


def main():
    ap = argparse.ArgumentParser(description="无监督光谱波段筛选")
    ap.add_argument("--mat-dir", type=str,
                    default=r"D:\舌象数据库\高光谱舌象数据库\其他hyperspectral")
    ap.add_argument("--n-bands", type=int, default=8, help="要选出的波段数")
    ap.add_argument("--n-files", type=int, default=None,
                    help="只取前 N 个文件加速（None=全部）")
    ap.add_argument("--max-pixels", type=int, default=20000,
                    help="每个文件采样用于相关性计算的像素数")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="冗余惩罚强度（0=纯信息量，越大越倾向选低相关波段）")
    ap.add_argument("--exclude", type=int, nargs="*", default=[],
                    help="直接排除的波段索引，如 --exclude 0 1 2")
    ap.add_argument("--no-plot", action="store_true", help="不画图")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = list_mat_files(args.mat_dir)
    if args.n_files is not None:
        files = files[: args.n_files]
    print(f"读取 {len(files)} 个 .mat 文件 ...")

    rng = np.random.default_rng(args.seed)
    B = None
    sum_b = sumsq_b = ent_sum = None
    corr_sum = corr_cnt = None
    n_pix = 0

    t0 = time.time()
    for fi, path in enumerate(files):
        cube = read_mat_cube(path).astype(np.float64)          # (B, H, W)
        if B is None:
            B = cube.shape[0]
            sum_b = np.zeros(B, np.float64)
            sumsq_b = np.zeros(B, np.float64)
            ent_sum = np.zeros(B, np.float64)
            corr_sum = np.zeros((B, B), np.float64)
            corr_cnt = 0

        flat = cube.reshape(B, -1)
        sum_b += flat.sum(axis=1)
        sumsq_b += (flat * flat).sum(axis=1)
        n_pix += flat.shape[1]

        for b in range(B):
            ent_sum[b] += shannon_entropy(cube[b])

        # 相关性：每文件采样 max_pixels 个像素，标准化后求 (B,B) 相关矩阵
        k = min(args.max_pixels, flat.shape[1])
        idx = rng.choice(flat.shape[1], size=k, replace=False)
        sub = flat[:, idx]
        sub = sub - sub.mean(axis=1, keepdims=True)
        sub = sub / (sub.std(axis=1, keepdims=True) + 1e-8)
        corr_sum += (sub @ sub.T) / k
        corr_cnt += 1

        if (fi + 1) % 25 == 0 or fi + 1 == len(files):
            print(f"  {fi + 1}/{len(files)}  用时 {time.time() - t0:.1f}s")

    mean = sum_b / n_pix
    std = np.sqrt(np.maximum(sumsq_b / n_pix - mean ** 2, 0.0))
    entropy = ent_sum / len(files)
    avg_corr = corr_sum / corr_cnt
    np.fill_diagonal(avg_corr, 0.0)

    def norm01(x):
        x = np.asarray(x, np.float64)
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    score = norm01(std)   # 信息量得分（std 归一化到 [0,1]）
    selected = greedy_select(score, avg_corr, args.n_bands, args.alpha, args.exclude)

    # ---- 控制台输出 ----
    print("\n===== 逐波段指标（按波段号，* 为选中）=====")
    print(f"{'band':>4} {'mean':>10} {'std':>10} {'entropy':>9} {'avg|corr|':>10} {'信息量':>7}")
    for b in range(B):
        mark = " *" if b in selected else ""
        print(f"{b:>4} {mean[b]:>10.4f} {std[b]:>10.4f} {entropy[b]:>9.3f} "
              f"{np.mean(np.abs(avg_corr[b])):>10.3f} {score[b]:>7.3f}{mark}")

    # 暗/噪波段提醒
    low_mean = np.argsort(mean)[:5]
    low_std = np.argsort(std)[:5]
    print("\n[提示] 亮度最低的 5 个波段：", low_mean.tolist(),
          "（可能偏暗/噪）")
    print("[提示] 标准差最低的 5 个波段：", low_std.tolist(),
          "（信息量最低，可考虑 --exclude 排除）")

    sel_sorted = sorted(selected)
    print(f"\n===== 推荐的 {len(selected)} 个波段 =====")
    print("按波段号升序：", sel_sorted)
    print("按信息量优先（贪心顺序）：", selected)
    print("\n可直接粘贴进 config.py：")
    print(f"SELECTED_BANDS = {sel_sorted}")

    # ---- 保存 CSV ----
    here = os.path.dirname(os.path.abspath(__file__))
    out_csv = os.path.join(here, "band_metrics.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("band,mean,std,entropy,avg_abs_corr,info_score,selected\n")
        for b in range(B):
            f.write(f"{b},{mean[b]:.4f},{std[b]:.4f},{entropy[b]:.3f},"
                    f"{np.mean(np.abs(avg_corr[b])):.3f},{score[b]:.3f},"
                    f"{1 if b in selected else 0}\n")
    print(f"[已保存] 逐波段指标 -> {out_csv}")

    # ---- 画图 ----
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            matplotlib.rcParams["font.sans-serif"] = [
                "Microsoft YaHei", "SimHei", "Arial Unicode MS"]
            matplotlib.rcParams["axes.unicode_minus"] = False

            fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
            axes[0].plot(range(B), std, "o-", color="tab:blue")
            axes[0].set_xticks(range(B))
            axes[0].set_xlabel("波段")
            axes[0].set_ylabel("标准差")
            axes[0].set_title("逐波段信息量 (std)")
            for b in selected:
                axes[0].axvline(b, color="tab:red", ls="--", lw=1)

            im = axes[1].imshow(avg_corr, cmap="RdBu_r", vmin=-1, vmax=1)
            axes[1].set_title("波段间相关性")
            axes[1].set_xticks(range(B))
            axes[1].set_yticks(range(B))
            fig.colorbar(im, ax=axes[1], fraction=0.046)

            cube0 = read_mat_cube(files[0])

            def to_img(b):
                x = cube0[b]
                lo, hi = np.percentile(x, 1), np.percentile(x, 99)
                return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)

            cols = 4
            rows = int(np.ceil(len(sel_sorted) / cols))
            h, w = cube0.shape[1], cube0.shape[2]
            montage = np.zeros((rows * h, cols * w))
            for j, b in enumerate(sel_sorted):
                r, c = divmod(j, cols)
                montage[r * h:(r + 1) * h, c * w:(c + 1) * w] = to_img(b)
            axes[2].imshow(montage, cmap="gray")
            axes[2].set_title(f"选中 {len(sel_sorted)} 个波段预览（首个样本）")
            axes[2].axis("off")

            fig.tight_layout()
            out_png = os.path.join(here, "band_selection.png")
            fig.savefig(out_png, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"[已保存] 可视化 -> {out_png}")
        except Exception as e:
            print(f"[警告] 画图失败（不影响筛选结果）：{e}")


if __name__ == "__main__":
    main()
