#!/usr/bin/env python3
"""
Figure 3: Neuro-Immune Interaction Analysis
=============================================
A. 交互热图 (NT × Inflammation → 12-month mRS)
B. Simple Slope Plot (剂量-效应梯度)
C. NRI/IDI 增量分析 (临床预测价值)
D. 校准曲线 (Calibration Plot)

输出路径:
  /data/usersdir/liuzhengxin/Stepbystep/7.figure/figure3/

数据来源:
  merged_neuro_data.csv (裸名NT列 + 炎症列 + m12_mRS)
"""
# ── 直接从 plot_figure2.py 导入所有共享定义 ──
from plot_figure2 import (
    _INFLAM_KEYWORDS, _INFLAM_KNOWN, _INFLAM_RENAME,
    _NT_SYSTEMS_ORDERED, _KNOWN_NT, _NT_RENAME,
    _fdr_correct, _clean_name,
    _plot_interaction_heatmap,
    _plot_simple_slopes,
    _nri_idi_analysis,
    _find_mrs, _get_inflam_cols,
)
import os
from pathlib import Path

OUTPUT_ROOT = Path("/data/usersdir/liuzhengxin/Stepbystep/7.figure/figure3")


def main():
    data_path = (
        "/data/usersdir/liuzhengxin/Stepbystep/"
        "6.NeurotransmitterMapping/3.variable_outcom_merge_data/"
        "merged_neuro_data.csv"
    )
    if not os.path.exists(data_path):
        print(f"  ⚠️ 未找到数据: {data_path}")
        return

    import pandas as pd
    df = pd.read_csv(data_path, low_memory=False)

    # 三级回退
    resid_cols = [c for c in df.columns if c.startswith("Resid_")]
    if not resid_cols:
        resid_cols = [c for c in df.columns if c.startswith("Load_")]
    if not resid_cols:
        resid_cols = [c for c in _KNOWN_NT if c in df.columns]
    if not resid_cols:
        print("  ⚠️ 无 NT 列")
        return

    print(f"  NT 列: {len(resid_cols)} 个")
    out = OUTPUT_ROOT
    out.mkdir(parents=True, exist_ok=True)

    print("\n[Fig 3A] 交互热图...")
    _plot_interaction_heatmap(df, resid_cols, out)
    # 重命名输出
    _rename(out, "fig_interaction_heatmap.png", "figure3a_interaction_heatmap.png")
    _rename(out, "fig_interaction_heatmap.pdf", "figure3a_interaction_heatmap.pdf")
    _rename(out, "interaction_q_values.csv", "figure3a_q_values.csv")

    print("\n[Fig 3B] Simple Slope Plot...")
    _plot_simple_slopes(df, resid_cols, out)
    # 重命名 top1/2/3
    for i in range(1, 4):
        for f in out.glob(f"simple_slope_top{i}_*.png"):
            new_name = f.name.replace("simple_slope_top", "figure3b_simple_slope_")
            _rename(out, f.name, new_name)

    print("\n[Fig 3C-D] NRI/IDI + 校准曲线...")
    _nri_idi_analysis(df, resid_cols, out)
    _rename(out, "nri_idi_incremental.png", "figure3c_nri_idi_incremental.png")
    _rename(out, "nri_idi_results.csv", "figure3c_nri_idi_results.csv")
    _rename(out, "calibration_plot.png", "figure3d_calibration_plot.png")

    print(f"\n{'='*60}")
    print(f"  ✅ Figure 3 全部完成！→ {out}")
    print(f"{'='*60}")


def _rename(directory, old, new):
    old_path = directory / old
    new_path = directory / new
    if old_path.exists():
        old_path.rename(new_path)
        print(f"  → {new}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Figure 3: Neuro-Immune Interaction Analysis")
    print("=" * 60)
    main()
