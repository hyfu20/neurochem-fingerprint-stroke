#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多维结局分析 — NT 递质对认知/ADL/情绪/运动的独立效应
====================================================
结局变量:
  MoCA 总分 (6m: VF6V01_145, 12m: VF12A1_116) — 认知
  MoCA 子项: 语言(VF6V01_126), 记忆(VF6V01_132), 定向(VF6V01_138),
             注意(VF6V01_122), 命名(VF6V01_118), 视空间(VF6V01_114)
  Barthel/ADL 总分 (VA6_98) — 日常生活能力
  PHQ-9 总分 (VA4_72) — 抑郁
  GAD-7 总分 (VA5_80) — 焦虑
  SIS 量表相关项

方法: Koch 残差 + OLS 回归 (连续结局), 控制 TLV+NIHSS+Age+Sex+CST

耗时: < 2 分钟

用法:
  python3 outcome_analysis.py
"""
import argparse, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
warnings.filterwarnings("ignore")


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    return None

def zscore(s):
    sd = s.std()
    return (s - s.mean()) / sd if sd > 1e-10 else s - s.mean()

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="merged_neuro_data.csv")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    csv_path = args.input
    if not Path(csv_path).exists():
        cand = "/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/3.variable_outcom_merge_data/merged_neuro_data.csv"
        if Path(cand).exists(): csv_path = cand
        else: print("❌ 找不到数据"); sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"\n{'=' * 70}")
    print(f"  多维结局分析 — NT 对认知/ADL/情绪的独立效应")
    print(f"{'=' * 70}")
    print(f"  数据: {csv_path}\n  样本: {len(df)} × {len(df.columns)}")

    # ── 基础变量 ──
    tlv = find_col(df, ["TLV", "TLV_mm3"])
    nihss = find_col(df, ["A_NIHSS", "NIHSS"])
    age = find_col(df, ["AGE", "Age"])
    sex = find_col(df, ["SEX", "Sex"])
    cst = find_col(df, ["CST_Load", "CST_load"])

    KNOWN_NT = ["5HT1a","5HT1b","5HT2a","5HT4","5HT6","5HTT",
                "A4B2","D1","D2","DAT","M1","NAT","VAChT",
                "human_CHA","JHU_EC","Lateral_Path","Medial_Path"]
    nt_cols = [c for c in df.columns if c in KNOWN_NT]
    if not nt_cols:
        nt_cols = [c for c in df.columns if c.startswith("Load_")]

    for c in [tlv, nihss, age, sex, cst] + nt_cols:
        if c: df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"  NT: {len(nt_cols)} 个, TLV: {tlv}, CST: {cst}")

    # ── 结局变量定义 ──
    OUTCOMES = {
        # 认知 MoCA
        "MoCA_6m_Total":     ("VF6V01_145", "认知总分(6m)", "continuous"),
        "MoCA_12m_Total":    ("VF12A1_116", "认知总分(12m)", "continuous"),
        "MoCA_6m_Language":  ("VF6V01_126", "语言(6m)", "continuous"),
        "MoCA_6m_Memory":    ("VF6V01_132", "延迟回忆(6m)", "continuous"),
        "MoCA_6m_Orientation":("VF6V01_138","定向(6m)", "continuous"),
        "MoCA_6m_Attention": ("VF6V01_122", "注意(6m)", "continuous"),
        "MoCA_6m_Naming":    ("VF6V01_118", "命名(6m)", "continuous"),
        "MoCA_6m_Visuospatial":("VF6V01_114","视空间(6m)", "continuous"),
        # ADL / Barthel
        "Barthel_Total":     ("VA6_98",  "Barthel/ADL总分", "continuous"),
        # 情绪
        "PHQ9_Total":        ("VA4_72",  "抑郁PHQ-9总分", "continuous"),
        "GAD7_Total":        ("VA5_80",  "焦虑GAD-7总分", "continuous"),
        # SIS 卒中影响量表 — 基线/早期 (VA5_*)
        "SIS_BSL_Strength":      ("VA5_38",  "SIS-力量(BSL)", "continuous"),
        "SIS_BSL_Memory":        ("VA5_45",  "SIS-记忆(BSL)", "continuous"),
        "SIS_BSL_Emotion":       ("VA5_54",  "SIS-情绪(BSL)", "continuous"),
        "SIS_BSL_Communication": ("VA5_61",  "SIS-交流(BSL)", "continuous"),
        "SIS_BSL_ADL":           ("VA5_71",  "SIS-日常活动(BSL)", "continuous"),
        "SIS_BSL_Mobility":      ("VA5_80",  "SIS-活动能力(BSL)", "continuous"),
        "SIS_BSL_HandFunction":  ("VA5_85",  "SIS-手功能(BSL)", "continuous"),
        "SIS_BSL_Participation": ("VA5_93",  "SIS-社会参与(BSL)", "continuous"),
        # SIS 6 月 (VF6V13_*)
        "SIS_6m_Strength":       ("VF6V13_71","SIS-力量(6m)", "continuous"),
        "SIS_6m_Memory":         ("VF6V13_73","SIS-记忆(6m)", "continuous"),
        "SIS_6m_Emotion":        ("VF6V13_75","SIS-情绪(6m)", "continuous"),
        "SIS_6m_Communication":  ("VF6V13_77","SIS-交流(6m)", "continuous"),
        "SIS_6m_ADL":            ("VF6V13_80","SIS-日常活动(6m)", "continuous"),
        "SIS_6m_Mobility":       ("VF6V13_83","SIS-活动能力(6m)", "continuous"),
        "SIS_6m_HandFunction":   ("VF6V13_85","SIS-手功能(6m)", "continuous"),
        "SIS_6m_Participation":  ("VF6V13_87","SIS-社会参与(6m)", "continuous"),
        "SIS_6m_Total":          ("VF6V13_88","SIS-总分(6m)", "continuous"),
        # SIS 12 月 (VF12A5_*)
        "SIS_12m_Strength":      ("VF12A5_73","SIS-力量(12m)", "continuous"),
        "SIS_12m_Emotion":       ("VF12A5_75","SIS-情绪(12m)", "continuous"),
        "SIS_12m_ADL":           ("VF12A5_76","SIS-日常活动(12m)", "continuous"),
        "SIS_12m_Participation": ("VF12A5_79","SIS-社会参与(12m)", "continuous"),
        "SIS_12m_Total":         ("VF12A5_80","SIS-总分(12m)", "continuous"),
        # 12月 MoCA 子项
        "MoCA_12m_Language": ("VF12A1_98", "语言(12m)", "continuous"),
        "MoCA_12m_Memory":   ("VF12A1_104","延迟回忆(12m)", "continuous"),
        "MoCA_12m_Attention":("VF12A1_94", "注意(12m)", "continuous"),
        # 12月 Barthel (VF12A6_*)
        "Barthel_12m_Total": ("VF12A6_98", "Barthel/ADL总分(12m)", "continuous"),
        "SIS_12m_Scale":     ("VF12A6_122","SIS影响量表(12m)", "continuous"),
        # 睡眠
        "ESS_Total":         ("VA2_118",  "Epworth嗜睡量表", "continuous"),
        "PSQI_Total":        ("VA2_106",  "PSQI睡眠质量", "continuous"),
    }

    # 检查哪些结局存在
    available = {}
    for label, (col, desc, vtype) in OUTCOMES.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            n_valid = df[col].notna().sum()
            pct = n_valid / len(df) * 100
            available[label] = (col, desc, vtype)
            print(f"  ✓ {label:<25s} ({col}): N={n_valid} ({pct:.1f}%) — {desc}")
        else:
            print(f"  ✗ {label:<25s} ({col}): 不存在")

    if not available:
        print("❌ 无可用结局变量"); sys.exit(1)

    # ── Koch 残差 ──
    print(f"\n  计算 Koch 残差...")
    resid_cols = []
    resid_map = {}
    for nt in nt_cols:
        valid = df[[nt, tlv]].dropna()
        if len(valid) > 30:
            slope, intercept, _, _, _ = stats.linregress(valid[tlv], valid[nt])
            rname = f"Resid_{nt.replace('Load_','')}"
            df[rname] = np.nan
            df.loc[valid.index, rname] = df.loc[valid.index, nt] - (intercept + slope * df.loc[valid.index, tlv])
            resid_cols.append(rname)
            resid_map[nt] = rname
    print(f"  ✅ {len(resid_cols)} 个残差列")

    # ── 协变量 ──
    covars = [c for c in [tlv, nihss, age, sex, cst] if c]

    # ══════════════════════════════════════════════════
    # 回归分析: 每个结局 × 每个 NT
    # ══════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print(f"  回归: Resid_NT → 各结局 (控制 TLV+NIHSS+Age+Sex+CST)")
    print(f"{'─' * 70}")

    all_results = []
    for outcome_label, (outcome_col, desc, vtype) in available.items():
        n_sig = 0
        for nt_col in resid_cols:
            nt_name = nt_col.replace("Resid_", "")
            preds = [nt_col] + covars
            sub = df[[outcome_col] + preds].dropna()
            if len(sub) < 30:
                continue

            sub_z = sub.copy()
            for p in preds:
                sub_z[p] = zscore(sub_z[p])
            sub_z[outcome_col] = zscore(sub_z[outcome_col])

            try:
                X = sm.add_constant(sub_z[preds])
                res = sm.OLS(sub_z[outcome_col], X).fit()
                beta = res.params[nt_col]
                pval = res.pvalues[nt_col]
                ci = res.conf_int().loc[nt_col]

                all_results.append({
                    "Outcome": outcome_label,
                    "Outcome_Col": outcome_col,
                    "Description": desc,
                    "NT": nt_name,
                    "Beta": beta,
                    "Beta_CI_lower": ci[0],
                    "Beta_CI_upper": ci[1],
                    "P_value": pval,
                    "R2_model": res.rsquared,
                    "N": len(sub),
                })
                if pval < 0.05:
                    n_sig += 1
            except Exception:
                pass

        print(f"  {outcome_label:<25s}: {n_sig}/{len(resid_cols)} NT 显著 (p<0.05)")

    rdf = pd.DataFrame(all_results)
    if rdf.empty:
        print("❌ 无结果"); return

    # FDR 校正 (按结局分组)
    from statsmodels.stats.multitest import multipletests
    for outcome in rdf["Outcome"].unique():
        mask = rdf["Outcome"] == outcome
        pvals = rdf.loc[mask, "P_value"].values
        valid = np.isfinite(pvals)
        q = np.full_like(pvals, np.nan)
        if valid.sum() > 0:
            _, q[valid], _, _ = multipletests(pvals[valid], method="fdr_bh")
        rdf.loc[mask, "FDR_q"] = q

    # ── 打印 Top hits ──
    print(f"\n{'─' * 70}")
    print(f"  Top Hits (按 P 值排序, 前20)")
    print(f"{'─' * 70}")
    top = rdf.nsmallest(20, "P_value")
    print(f"  {'Outcome':<22s} {'NT':<15s} {'β':>8s} {'P':>12s} {'FDR q':>10s} {'N':>6s}")
    print(f"  {'─'*22} {'─'*15} {'─'*8} {'─'*12} {'─'*10} {'─'*6}")
    for _, r in top.iterrows():
        print(f"  {r['Outcome']:<22s} {r['NT']:<15s} {r['Beta']:>+8.3f} "
              f"{r['P_value']:>12.2e}{sig_stars(r['P_value']):3s} "
              f"{r['FDR_q']:>10.3f} {r['N']:>6.0f}")

    # ── 按结局汇总 ──
    print(f"\n{'─' * 70}")
    print(f"  按结局汇总 Top NT")
    print(f"{'─' * 70}")
    for outcome in available:
        sub = rdf[rdf["Outcome"] == outcome].nsmallest(3, "P_value")
        if sub.empty: continue
        best = sub.iloc[0]
        n_sig = (rdf[rdf["Outcome"] == outcome]["P_value"] < 0.05).sum()
        n_fdr = (rdf[rdf["Outcome"] == outcome]["FDR_q"] < 0.05).sum()
        print(f"  {outcome:<25s}: Top={best['NT']} β={best['Beta']:+.3f} p={best['P_value']:.2e}"
              f"  | sig={n_sig}, FDR={n_fdr}")

    # ══════════════════════════════════════════════════
    # 输出
    # ══════════════════════════════════════════════════
    if args.output:
        out_dir = Path(args.output)
    else:
        p = Path(csv_path).resolve().parent
        while p != p.parent:
            if "6.NeurotransmitterMapping" in p.name:
                out_dir = p / "6.furtherv4"; break
            p = p.parent
        else:
            out_dir = Path(csv_path).parent / "6.furtherv4"
    out_dir.mkdir(parents=True, exist_ok=True)

    rdf.to_csv(out_dir / "outcome_analysis_results.csv", index=False)
    print(f"\n  💾 {out_dir / 'outcome_analysis_results.csv'}")

    # ══════════════════════════════════════════════════
    # 绘图: 热力图 + 总表
    # ══════════════════════════════════════════════════
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from datetime import datetime

    plt.rcParams.update({"font.sans-serif": ["Arial","DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})

    # ── Figure 1: β 热力图 (NT × Outcome) ──
    pivot_beta = rdf.pivot_table(index="NT", columns="Outcome", values="Beta", aggfunc="first")
    pivot_p = rdf.pivot_table(index="NT", columns="Outcome", values="P_value", aggfunc="first")

    # 按 MoCA 总分排序
    if "MoCA_6m_Total" in pivot_beta.columns:
        sort_col = "MoCA_6m_Total"
    elif pivot_beta.columns.size > 0:
        sort_col = pivot_beta.columns[0]
    else:
        sort_col = None
    if sort_col:
        pivot_beta = pivot_beta.sort_values(sort_col, ascending=True)
        pivot_p = pivot_p.reindex(pivot_beta.index)

    fig, ax = plt.subplots(figsize=(max(10, len(available)*1.8), max(6, len(resid_cols)*0.45)))
    try:
        import seaborn as sns
        sns.heatmap(pivot_beta, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                    ax=ax, linewidths=0.5, cbar_kws={"label": "Standardized β"})
        # 标星号
        for i in range(len(pivot_beta.index)):
            for j in range(len(pivot_beta.columns)):
                p = pivot_p.iloc[i, j]
                if pd.notna(p) and p < 0.05:
                    ax.text(j+0.5, i+0.82, "★", ha="center", va="center",
                            fontsize=9, color="gold", fontweight="bold")
    except ImportError:
        ax.imshow(pivot_beta.values, cmap="RdBu_r", aspect="auto")

    ax.set_title("NT → Multi-dimensional Outcomes (Standardized β)\n"
                 "★ = p < 0.05 | Controlled: TLV + NIHSS + Age + Sex + CST",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "OUTCOME_HEATMAP.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  📊 {out_dir / 'OUTCOME_HEATMAP.png'}")

    # ── Figure 2: Top Hits 总表图 ──
    top20 = rdf.nsmallest(25, "P_value")
    fig2, ax_t = plt.subplots(figsize=(18, max(4, 0.45*len(top20)+3)))
    ax_t.axis("off")

    rows = []
    for _, r in top20.iterrows():
        rows.append([
            r["Outcome"], r["Description"], r["NT"],
            f"{r['Beta']:+.3f}", f"[{r['Beta_CI_lower']:+.3f}, {r['Beta_CI_upper']:+.3f}]",
            f"{r['P_value']:.2e} {sig_stars(r['P_value'])}",
            f"{r['FDR_q']:.3f}" if pd.notna(r['FDR_q']) else "—",
            f"{r['N']:.0f}",
        ])

    cols = ["Outcome", "Description", "NT", "β", "95% CI", "P", "FDR q", "N"]
    t = ax_t.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1.0, 1.7)
    for j in range(len(cols)):
        t[0, j].set_facecolor("#2C3E50")
        t[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows)+1):
        for j in range(len(cols)):
            t[i, j].set_facecolor("#F7F9FC" if i%2==0 else "#FFFFFF")
            t[i, j].set_edgecolor("#DEE2E6")

    ax_t.set_title("Multi-Outcome Analysis: Top NT → Cognitive / ADL / Mood\n"
                   f"(Generated {datetime.now().strftime('%Y-%m-%d %H:%M')})",
                   fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    fig2.savefig(out_dir / "OUTCOME_TABLE.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"  📊 {out_dir / 'OUTCOME_TABLE.png'}")

    # ── Figure 3: 按结局的 Forest Plot (每个结局 Top 5 NT) ──
    n_outcomes = len(available)
    if n_outcomes > 0:
        fig3, axes = plt.subplots(1, min(n_outcomes, 4),
                                   figsize=(5*min(n_outcomes, 4), max(6, len(resid_cols)*0.35)))
        if n_outcomes == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for idx, (outcome_label, (_, desc, _)) in enumerate(list(available.items())[:4]):
            if idx >= len(axes): break
            ax = axes[idx]
            sub = rdf[rdf["Outcome"] == outcome_label].sort_values("P_value")
            if sub.empty: continue

            y = np.arange(len(sub))
            colors = ["#E64B35" if p < 0.05 and b > 0
                      else "#4DBBD5" if p < 0.05 and b < 0
                      else "#CCCCCC"
                      for p, b in zip(sub["P_value"], sub["Beta"])]

            ax.barh(y, sub["Beta"].values, color=colors, edgecolor="black",
                    linewidth=0.3, height=0.6, alpha=0.85)
            ax.errorbar(sub["Beta"].values, y,
                        xerr=[sub["Beta"].values - sub["Beta_CI_lower"].values,
                              sub["Beta_CI_upper"].values - sub["Beta"].values],
                        fmt="none", ecolor="black", capsize=2, linewidth=0.8)
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels(sub["NT"].values, fontsize=7)
            ax.set_xlabel("Standardized β")
            ax.set_title(f"{desc}", fontsize=9, fontweight="bold")

        plt.suptitle("NT Effect on Multi-dimensional Outcomes\n"
                     "(Red=harmful p<0.05, Blue=protective p<0.05)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig3.savefig(out_dir / "OUTCOME_FOREST.png", dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig3)
        print(f"  📊 {out_dir / 'OUTCOME_FOREST.png'}")

    print(f"\n{'=' * 70}")
    print(f"  ✅ 多维结局分析完成!")
    print(f"  📸 截图: OUTCOME_HEATMAP.png + OUTCOME_TABLE.png + OUTCOME_FOREST.png")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
