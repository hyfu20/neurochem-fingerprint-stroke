#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tables_to_images.py
====================
把 publication_ready/ 目录里所有 Table_*.csv 渲染成专业的 PNG + PDF 表格图。

特性：
  - 自动检测数值列、p 值列、效应量列并选择合适的精度格式化
  - 列宽自适应；表头加粗灰底；交替行底色
  - 全局 FDR q < 0.05 的行高亮（红棕）
  - 同时输出 PNG（300 dpi，给 Word/PPT）+ PDF（矢量，给打印 / 导师阅读）
  - 单文件出图，也支持批处理整个目录

用法：
  python tables_to_images.py /path/to/publication_ready
  python tables_to_images.py /path/to/publication_ready --out figures_tables
  python tables_to_images.py file.csv                 # 单文件
"""
import argparse
import sys
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

# 中文字体兜底（在 macOS / Linux 服务器上都尽量找到一个能用的）
import matplotlib.font_manager as fm
for candidate in ["PingFang SC", "Heiti SC", "Songti SC",
                  "Noto Sans CJK SC", "WenQuanYi Zen Hei",
                  "Microsoft YaHei", "SimHei", "DejaVu Sans"]:
    try:
        fm.findfont(candidate, fallback_to_default=False)
        plt.rcParams["font.family"] = candidate
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

# ---------------- 表格友好标题映射（CSV 文件名 → 图上标题） ----------------
TABLE_TITLES = {
    "table1_baseline.csv":
        "Table 1. Baseline characteristics of the analytical cohort (N = 3,582)",
    "Table_1_Baseline.csv":
        "Table 1. Baseline characteristics of the analytical cohort (N = 3,582)",
    "Table_2_NT_Acute_Outcome.csv":
        "Table 2. Top neurotransmitter predictors of discharge mRS (Model C)",
    "Table_3_CST_Adjusted_Model_D.csv":
        "Table 3. NT systems surviving CST adjustment (Model D)",
    "Table_4_NT_Inflammation_Interactions.csv":
        "Table 4. FDR-significant NT × inflammation interactions",
    "Table_S1_Deep_Phenotyping_FDR.csv":
        "Supplementary Table S1. Deep-phenotyping FDR results (all domains)",
    "Table_S2_NRI_IDI_AUC.csv":
        "Supplementary Table S2. Incremental prediction metrics (NRI / IDI / AUC)",
    "Table_S3_AdditiveInteraction_summary.csv":
        "Supplementary Table S3a. Additive interaction summary (most-significant pair: Medial cholinergic pathway × hsCRP)",
    "Table_S3_AdditiveInteraction_4cell.csv":
        "Supplementary Table S3b. Four-cell counts and absolute risks for the same pair (LL/LH/HL/HH)",
    "Table_S4_AdditiveInteraction_AllSigPairs.csv":
        "Supplementary Table S4. All seven FDR-significant NT × inflammation pairs",
    "Table_S5_SimpleSlope_CHA_IL6.csv":
        "Supplementary Table S5. Tertile-stratified CHA × IL-6 simple-slope models",
}

# 高亮列匹配（出现 q 值列就触发显著性高亮）
Q_VAL_PATTERNS = ["q_global", "q_value", "q_BH", "FDR_q", "global_q"]


def _fmt_cell(val, col_name):
    """格式化单个单元格内容"""
    if pd.isna(val):
        return "—"
    col_low = col_name.lower()

    # 数值
    if isinstance(val, (int, np.integer)):
        return f"{int(val):,}"
    if isinstance(val, (float, np.floating)):
        # 极小的 P 值 / q 值用科学计数
        if any(k in col_low for k in ["p_value", "p value", "pval", "q_", "_q", "p_perm", "p_spin"]):
            if val < 1e-3:
                return f"{val:.2e}"
            return f"{val:.3f}"
        # OR / CI / beta
        if any(k in col_low for k in ["or", "ci", "beta", "β", "lower", "upper",
                                       "delta", "Δ", "slope", "ratio", "rho", "auc"]):
            return f"{val:.3f}"
        # 百分比
        if "pct" in col_low or "percent" in col_low or "rate" in col_low:
            if abs(val) < 1.0:
                return f"{val*100:.1f}%"
            return f"{val:.1f}"
        # 默认 3 位有效数字
        if abs(val) < 0.001 or abs(val) >= 1e4:
            return f"{val:.2e}"
        return f"{val:.3f}"

    # 字符串清理
    s = str(val).strip()
    return s if s else "—"


def _detect_q_col(df):
    """找出 q 值列名（FDR），用于高亮"""
    for c in df.columns:
        cl = c.lower().replace(" ", "_")
        for pat in Q_VAL_PATTERNS:
            if pat.lower() in cl:
                return c
    return None


def render_table(csv_path: Path, out_dir: Path, max_rows: int = 60):
    """把单个 CSV 渲染成 PNG + PDF"""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  ❌ 读取失败 {csv_path.name}: {e}")
        return False

    if df.empty:
        print(f"  ⚠️ {csv_path.name} 为空，跳过")
        return False

    title = TABLE_TITLES.get(csv_path.name, csv_path.stem.replace("_", " "))

    # 大表截断（同时保存完整 CSV 副本到 out_dir）
    if len(df) > max_rows:
        print(f"  ℹ️ {csv_path.name}: {len(df)} 行 > {max_rows}，仅在图上显示前 {max_rows} 行（完整数据见 CSV）")
        df_view = df.head(max_rows).copy()
        truncated = True
    else:
        df_view = df.copy()
        truncated = False

    # 格式化所有单元格
    formatted = df_view.copy().astype(object)
    for col in df_view.columns:
        formatted[col] = [_fmt_cell(v, col) for v in df_view[col]]

    # q 列高亮
    q_col = _detect_q_col(df_view)
    sig_rows = set()
    if q_col is not None:
        for idx, v in enumerate(df_view[q_col]):
            try:
                if pd.notna(v) and float(v) < 0.05:
                    sig_rows.add(idx)
            except Exception:
                pass

    # 画布尺寸自适应
    n_rows, n_cols = formatted.shape
    col_widths = []
    for col in formatted.columns:
        max_len = max(len(str(col)),
                      *[len(str(x)) for x in formatted[col]])
        col_widths.append(min(max(max_len, 8), 40))
    total_width = sum(col_widths) * 0.12 + 1.5
    total_height = max(2.5, n_rows * 0.35 + 2.0)

    fig, ax = plt.subplots(figsize=(min(total_width, 22), min(total_height, 30)))
    ax.axis("off")

    # 表格
    table = ax.table(
        cellText=formatted.values.tolist(),
        colLabels=[str(c) for c in formatted.columns],
        cellLoc="center",
        loc="upper center",
        colWidths=[w / sum(col_widths) for w in col_widths],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    # 表头样式
    n_disp_rows = n_rows  # 不算标题行
    for col_i in range(n_cols):
        cell = table[(0, col_i)]
        cell.set_facecolor("#374151")  # 深灰
        cell.set_text_props(color="white", weight="bold")
        cell.set_edgecolor("#1f2937")

    # 行交替底色 + 显著性高亮
    for row_i in range(n_disp_rows):
        # 行交替
        base_color = "#f9fafb" if row_i % 2 == 0 else "#ffffff"
        is_sig = row_i in sig_rows
        for col_i in range(n_cols):
            cell = table[(row_i + 1, col_i)]
            if is_sig:
                cell.set_facecolor("#fef2f2")  # 红棕 hint
                cell.set_text_props(color="#7f1d1d", weight="bold")
            else:
                cell.set_facecolor(base_color)
            cell.set_edgecolor("#d1d5db")

    # 标题
    ax.set_title(title, fontsize=12, weight="bold", loc="left", pad=15, color="#111827")

    # 底注
    notes = []
    if q_col is not None and sig_rows:
        notes.append(f"红色加粗行：{q_col} < 0.05（FDR-significant，共 {len(sig_rows)} 行）")
    if truncated:
        notes.append(f"图上仅显示前 {max_rows} 行；完整数据见同名 CSV（共 {len(df)} 行）")
    if notes:
        fig.text(0.02, 0.01, "  •  ".join(notes), fontsize=8,
                 color="#6b7280", ha="left", va="bottom")

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])

    # 输出
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  ✅ {csv_path.name}  →  {png_path.name}  +  {pdf_path.name}")
    return True


def combine_pdfs(folder: Path, out_pdf: Path):
    """把所有 Table*.pdf 合成一份 All_Tables.pdf 方便发给导师"""
    pdfs = sorted(set(folder.glob("Table_*.pdf")) | set(folder.glob("table1_*.pdf")))
    if not pdfs:
        return
    # 用 matplotlib 不直接合并 PDF；改用 pypdf
    try:
        from pypdf import PdfWriter
    except ImportError:
        try:
            from PyPDF2 import PdfWriter
        except ImportError:
            print("  ℹ️ 未安装 pypdf；跳过 PDF 合并。pip install pypdf 可启用此功能。")
            return
    writer = PdfWriter()
    for p in pdfs:
        writer.append(str(p))
    with open(out_pdf, "wb") as f:
        writer.write(f)
    print(f"\n📚 合并 PDF: {out_pdf}  （共 {len(pdfs)} 张表）")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", help="包含 Table_*.csv 的目录，或单个 .csv 文件")
    parser.add_argument("--out", default=None,
                        help="输出目录（默认在 target 下创建 figures_tables/）")
    parser.add_argument("--max-rows", type=int, default=60,
                        help="表格图最多显示的行数（默认 60）")
    args = parser.parse_args()

    target = Path(args.target).resolve()
    if not target.exists():
        sys.exit(f"❌ 找不到 {target}")

    # 黑名单：已被新表取代或为诊断中间文件，跳过
    BLACKLIST_SUBSTR = (
        "_DIAG_", "diag_",
        "table_S1_nt_loads_by_outcome",  # 被 Table_S1_Deep_Phenotyping_FDR 取代
        "Table_S_",                       # 旧的无编号占位文件名
    )

    if target.is_file():
        csv_files = [target]
        out_dir = Path(args.out).resolve() if args.out else target.parent / "figures_tables"
    else:
        # 主表 Table_*.csv + 兼容 table1_baseline.csv（小写、无前缀）
        candidates = set(target.glob("Table_*.csv"))
        candidates |= set(target.glob("table1_*.csv"))
        candidates |= set(target.glob("Table[0-9]*.csv"))
        csv_files = sorted(p for p in candidates
                           if not any(b in p.name for b in BLACKLIST_SUBSTR))
        out_dir = Path(args.out).resolve() if args.out else target / "figures_tables"

    if not csv_files:
        sys.exit(f"❌ 在 {target} 下没找到 Table_*.csv 或 table1_*.csv")

    print(f"待渲染 CSV: {len(csv_files)} 个")
    print(f"输出目录:   {out_dir}")
    print("=" * 70)

    n_ok = 0
    for csv_p in csv_files:
        if render_table(csv_p, out_dir, max_rows=args.max_rows):
            n_ok += 1

    print("=" * 70)
    print(f"成功渲染: {n_ok} / {len(csv_files)}")

    # 合并 PDF
    combine_pdfs(out_dir, out_dir / "All_Tables.pdf")


if __name__ == "__main__":
    main()
