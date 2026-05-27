#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
combine_figure_panels.py
=========================
把单独的子图 PNG 合并为发表级多面板大图。

输入目录默认: /data/usersdir/liuzhengxin/Stepbystep/7.figure/otherfigures/Figure
输出目录默认: 同目录下 combined/

合并方案：
  Figure 3 (4 panels, 2×2):  A=Interaction_heatmap, B=AdditiveInteraction,
                              C=ROC, D=Calibration
  Figure 4 (2 panels, 1×2):  A=DCA, B=CV_10fold
  Figure 5 (4 panels, 2×2):  A=NIVA_brain_map, B=SplitHalf,
                              C=Bootstrap, D=HansenCorrelation

特性：
  - 自动按各子图原始宽高比布局（不变形拉伸）
  - 在每个面板左上角加白底黑字 A/B/C/D 标签
  - 输出 300 dpi PNG + 矢量 PDF
  - 用 matplotlib，不引入新依赖

用法：
  python combine_figure_panels.py
  python combine_figure_panels.py --src /path/to/Figure --out /path/to/combined
  python combine_figure_panels.py --only 3       # 只合 Figure 3
  python combine_figure_panels.py --dpi 600      # 提高 dpi
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

# ---------------- 各图面板规格 ----------------
# 每个面板: (file_glob_pattern, panel_label)
FIGURE_SPECS = {
    3: {
        "panels": [
            ("Fig_3A_Interaction_heatmap.png", "A"),
            ("Fig_3B_AdditiveInteraction.png", "B"),
            ("Fig_3C_ROC.png",                  "C"),
            ("Fig_3D_Calibration.png",          "D"),
        ],
        "grid": (2, 2),         # rows, cols
        "figsize": (16, 14),    # inches
        "title": "Figure 3 | Neuro-immune dual-burden interaction analysis",
    },
    4: {
        "panels": [
            ("Fig_4A_DCA.png",      "A"),
            ("Fig_4B_CV_10fold.png", "B"),
        ],
        "grid": (1, 2),
        "figsize": (16, 7),
        "title": "Figure 4 | Predictive performance versus mechanistic insight",
    },
    5: {
        "panels": [
            ("Fig_5A_NIVA_brain_map.png",     "A"),
            ("Fig_5B_SplitHalf.png",          "B"),
            ("Fig_5C_Bootstrap.png",          "C"),
            ("Fig_5D_HansenCorrelation.png",  "D"),
        ],
        "grid": (2, 2),
        "figsize": (16, 14),
        "title": "Figure 5 | Neuro-immune vulnerability atlas (NIVA) and validation",
    },
}


def combine_one_figure(fig_num: int, spec: dict, src_dir: Path, out_dir: Path, dpi: int = 300):
    panels = spec["panels"]
    n_rows, n_cols = spec["grid"]

    # 检查所有 panel 文件是否存在
    missing = []
    panel_paths = []
    for fname, label in panels:
        p = src_dir / fname
        if not p.exists():
            missing.append(fname)
        panel_paths.append((p, label))

    if missing:
        print(f"  ❌ Figure {fig_num} 缺少子图: {missing}")
        return False

    # 创建画布
    fig = plt.figure(figsize=spec["figsize"], facecolor="white")
    gs = GridSpec(n_rows, n_cols, figure=fig,
                  wspace=0.04, hspace=0.10,
                  left=0.02, right=0.99, top=0.95, bottom=0.02)

    # 主标题
    fig.suptitle(spec["title"], fontsize=18, fontweight="bold",
                 x=0.02, y=0.985, ha="left", color="#111827")

    for idx, (img_path, label) in enumerate(panel_paths):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        img = mpimg.imread(str(img_path))
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # 面板标签 A/B/C/D：白底黑字，左上角
        ax.text(0.01, 0.99, label,
                transform=ax.transAxes,
                fontsize=22, fontweight="bold",
                color="#111827",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white",
                          edgecolor="#374151",
                          linewidth=1.0,
                          alpha=0.92))

    # 输出
    out_dir.mkdir(parents=True, exist_ok=True)
    png_out = out_dir / f"Figure_{fig_num}_combined.png"
    pdf_out = out_dir / f"Figure_{fig_num}_combined.pdf"

    fig.savefig(png_out, dpi=dpi, bbox_inches="tight",
                facecolor="white", pad_inches=0.15)
    fig.savefig(pdf_out, bbox_inches="tight",
                facecolor="white", pad_inches=0.15)
    plt.close(fig)

    sz_png = png_out.stat().st_size / 1024
    sz_pdf = pdf_out.stat().st_size / 1024
    print(f"  ✅ Figure {fig_num}: "
          f"{png_out.name} ({sz_png:.0f} KB)  +  "
          f"{pdf_out.name} ({sz_pdf:.0f} KB)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--src",
        default="/data/usersdir/liuzhengxin/Stepbystep/7.figure/otherfigures/Figure",
        help="放各 Fig_*.png 的目录",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="输出目录（默认 <src>/combined）",
    )
    parser.add_argument(
        "--only", type=int, choices=[3, 4, 5],
        help="只合并指定图（3 / 4 / 5），默认全部",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG 输出 dpi（默认 300）")
    args = parser.parse_args()

    src_dir = Path(args.src).resolve()
    if not src_dir.exists():
        sys.exit(f"❌ 源目录不存在: {src_dir}")

    out_dir = Path(args.out).resolve() if args.out else src_dir / "combined"

    targets = [args.only] if args.only else sorted(FIGURE_SPECS.keys())
    print(f"源目录: {src_dir}")
    print(f"输出:   {out_dir}")
    print(f"将合并: {targets}")
    print("=" * 70)

    n_ok = 0
    for fig_num in targets:
        if combine_one_figure(fig_num, FIGURE_SPECS[fig_num], src_dir, out_dir, dpi=args.dpi):
            n_ok += 1

    print("=" * 70)
    print(f"✅ 完成: {n_ok} / {len(targets)} 张多面板图已生成")
    print(f"位置:   {out_dir}/")


if __name__ == "__main__":
    main()
