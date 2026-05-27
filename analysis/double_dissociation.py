#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双重解离分析: NAT(体力引擎) vs A4B2(精准操作) × SIS 子维度
============================================================
Nature Neuroscience 级证据:
  1. NAT 预测重体力项 (VA6_89重家务, VA6_93快步走, VA6_97拿重物)
  2. A4B2 预测精细动作项 (VA6_82穿衣, VA6_83洗浴, VA6_92移动)
  3. 雷达图: 高/低 NAT 损伤组 vs 高/低 A4B2 损伤组 在 SIS 16 项上的差异
  4. MoCA 认知子维度: A4B2/human_CHA 预测空间执行功能
  5. 客观验证: NAT 预测 10m 步行时间

用法:
  python3 double_dissociation.py --input merged_neuro_data.csv

后台:
  nohup python3 double_dissociation.py --input merged_neuro_data.csv > dissociation.log 2>&1 &
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.multitest import multipletests


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def zscore(s):
    sd = s.std()
    return (s - s.mean()) / sd if sd > 1e-10 else s - s.mean()


def fdr_correct(pvals):
    p = np.asarray(pvals, dtype=float)
    valid = np.isfinite(p) & (p >= 0)
    q = np.full_like(p, np.nan, dtype=float)
    if valid.sum() > 0:
        _, q_vals, _, _ = multipletests(p[valid], method="fdr_bh")
        q[valid] = q_vals
    return q


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""


# SIS 条目定义 (基于你的变量说明)
SIS_ITEMS = {
    # 精细动作 / ADL (A4B2 主导)
    "VA6_82": "穿衣 (上半身)",
    "VA6_83": "洗浴",
    "VA6_84": "扣纽扣",
    "VA6_85": "拉拉链",
    "VA6_86": "梳头/刷牙",
    "VA6_87": "开罐/瓶",
    # 重体力 / 移动 (NAT 主导)
    "VA6_88": "购物",
    "VA6_89": "做重家务",
    "VA6_90": "做轻家务",
    "VA6_91": "端碗/杯",
    "VA6_92": "床→椅转移",
    "VA6_93": "快步走",
    "VA6_94": "上下楼梯",
    "VA6_95": "进出车/公交",
    "VA6_96": "翻身/侧卧",
    "VA6_97": "拿重物",
}

# 功能分组
FINE_MOTOR_ITEMS = ["VA6_82", "VA6_83", "VA6_84", "VA6_85", "VA6_86", "VA6_87"]
GROSS_MOTOR_ITEMS = ["VA6_88", "VA6_89", "VA6_93", "VA6_94", "VA6_95", "VA6_97"]
MOBILITY_ITEMS = ["VA6_90", "VA6_91", "VA6_92", "VA6_96"]

SIS_DOMAIN = {}
for item in FINE_MOTOR_ITEMS:
    SIS_DOMAIN[item] = "Fine Motor (精细动作)"
for item in GROSS_MOTOR_ITEMS:
    SIS_DOMAIN[item] = "Gross Motor (重体力)"
for item in MOBILITY_ITEMS:
    SIS_DOMAIN[item] = "Basic Mobility (基础移动)"

# MoCA 子维度
MOCA_ITEMS = {
    "VF6V01_114": "空间与执行功能",
    "VF6V02_114": "命名",
    "VF6V03_114": "注意力",
    "VF6V04_114": "语言",
    "VF6V05_114": "抽象思维",
    "VF6V06_114": "延迟回忆",
    "VF6V07_114": "定向力",
}


def main():
    parser = argparse.ArgumentParser(description="双重解离分析")
    parser.add_argument("-i", "--input", default="merged_neuro_data.csv")
    parser.add_argument("-o", "--output", default="/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/5.further_data_analysis_code/double_dissociation_results")
    args = parser.parse_args()

    csv_path = args.input
    if not Path(csv_path).exists():
        for alt in [
            "/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/3.variable_outcom_merge_data/merged_neuro_data.csv",
            "/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/variable_outcom_merge_data/merged_neuro_data.csv",
        ]:
            if Path(alt).exists():
                csv_path = alt
                break
        else:
            print(f"❌ 找不到数据文件")
            sys.exit(1)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"\n{'=' * 72}")
    print(f"  双重解离分析: NAT(体力引擎) vs A4B2(精准操作)")
    print(f"  Nature Neuroscience 级证据")
    print(f"  数据: {df.shape[0]} × {df.shape[1]}")
    print(f"{'=' * 72}")

    # ── 识别列 ──
    tlv = find_col(df, ["TLV", "TLV_mm3"])
    nihss = find_col(df, ["A_NIHSS", "NIHSS"])
    age = find_col(df, ["AGE", "Age"])
    sex = find_col(df, ["SEX", "Sex"])
    covars = [c for c in [tlv, nihss, age, sex] if c]

    nat_col = find_col(df, ["Load_NAT", "NAT"])
    a4b2_col = find_col(df, ["Load_A4B2", "A4B2"])
    cha_col = find_col(df, ["Load_human_CHA", "human_CHA"])
    sht6_col = find_col(df, ["Load_5HT6", "5HT6"])
    dat_col = find_col(df, ["Load_DAT", "DAT"])
    vacht_col = find_col(df, ["Load_VAChT", "VAChT"])

    top_nts = {
        "NAT": nat_col, "A4B2": a4b2_col, "human_CHA": cha_col,
        "5HT6": sht6_col, "DAT": dat_col, "VAChT": vacht_col,
    }
    top_nts = {k: v for k, v in top_nts.items() if v is not None}

    for c in list(top_nts.values()) + covars:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"  NT 变量: {list(top_nts.keys())}")
    print(f"  协变量: {covars}")

    # 识别可用的 SIS 条目
    sis_available = {k: v for k, v in SIS_ITEMS.items() if k in df.columns}
    for c in sis_available:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"  SIS 条目: {len(sis_available)}/{len(SIS_ITEMS)} 可用")

    # 识别可用的 MoCA 子维度
    moca_available = {k: v for k, v in MOCA_ITEMS.items() if k in df.columns}
    for c in moca_available:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"  MoCA 子维度: {len(moca_available)}/{len(MOCA_ITEMS)} 可用")

    # 10m 步行
    walk_col = find_col(df, ["VA7_121", "walk_10m", "TenMeterWalk"])
    if walk_col:
        df[walk_col] = pd.to_numeric(df[walk_col], errors="coerce")
        print(f"  10m 步行: {walk_col}")

    # ==================================================================
    # Part 1: 全 SIS 条目 × Top NT 回归矩阵
    # ==================================================================
    print(f"\n{'─' * 72}")
    print(f"  [Part 1] SIS 条目 × NT 回归矩阵 (双重解离检验)")
    print(f"{'─' * 72}")

    sis_results = []
    if sis_available:
        for sis_item, sis_label in sis_available.items():
            domain = SIS_DOMAIN.get(sis_item, "Other")
            for nt_name, nt_col in top_nts.items():
                predictors = [nt_col] + covars
                sub = df[[sis_item] + predictors].dropna()
                if len(sub) < 30:
                    continue

                sub_z = sub.copy()
                for p in predictors:
                    sub_z[p] = zscore(sub_z[p])

                try:
                    X = sm.add_constant(sub_z[predictors])
                    y = sub_z[sis_item].astype(float)
                    res = sm.OLS(y, X).fit()

                    sis_results.append({
                        "SIS_Item": sis_item,
                        "SIS_Label": sis_label,
                        "Domain": domain,
                        "NT": nt_name,
                        "Beta": res.params[nt_col],
                        "P_value": res.pvalues[nt_col],
                        "Partial_R2": res.rsquared - sm.OLS(y,
                            sm.add_constant(sub_z[covars])).fit().rsquared
                            if covars else res.rsquared,
                        "N": len(sub),
                    })
                except Exception:
                    continue

    sis_df = pd.DataFrame(sis_results)
    if not sis_df.empty:
        sis_df["FDR_q"] = fdr_correct(sis_df["P_value"].values)
        sis_df["Sig"] = sis_df["P_value"] < 0.05
        sis_df = sis_df.sort_values(["NT", "P_value"])
        sis_df.to_csv(out / "sis_x_nt_matrix.csv", index=False)

        # ── 打印解离表 ──
        print(f"\n  {'SIS Item':<12s} {'Label':<18s} {'Domain':<22s}", end="")
        for nt_name in top_nts:
            print(f" {nt_name:>10s}", end="")
        print()
        print("  " + "─" * (52 + 11 * len(top_nts)))

        for sis_item in sis_available:
            label = sis_available[sis_item][:16]
            domain = SIS_DOMAIN.get(sis_item, "Other")[:20]
            print(f"  {sis_item:<12s} {label:<18s} {domain:<22s}", end="")
            for nt_name in top_nts:
                match = sis_df[(sis_df["SIS_Item"] == sis_item) & (sis_df["NT"] == nt_name)]
                if not match.empty:
                    p = match.iloc[0]["P_value"]
                    star = sig_stars(p)
                    beta_sign = "+" if match.iloc[0]["Beta"] > 0 else "-"
                    print(f" {beta_sign}{star:>8s}", end="")
                else:
                    print(f" {'—':>10s}", end="")
            print()

        # ── 双重解离统计检验 ──
        if "NAT" in top_nts and "A4B2" in top_nts:
            print(f"\n  ── 双重解离统计检验 ──")

            # NAT 在重体力 vs 精细动作的效应差异
            nat_gross = sis_df[(sis_df["NT"] == "NAT") &
                               (sis_df["Domain"] == "Gross Motor (重体力)")]["Beta"].abs()
            nat_fine = sis_df[(sis_df["NT"] == "NAT") &
                              (sis_df["Domain"] == "Fine Motor (精细动作)")]["Beta"].abs()

            # A4B2 在精细动作 vs 重体力的效应差异
            a4b2_fine = sis_df[(sis_df["NT"] == "A4B2") &
                               (sis_df["Domain"] == "Fine Motor (精细动作)")]["Beta"].abs()
            a4b2_gross = sis_df[(sis_df["NT"] == "A4B2") &
                                (sis_df["Domain"] == "Gross Motor (重体力)")]["Beta"].abs()

            if len(nat_gross) >= 2 and len(nat_fine) >= 2:
                u, p = stats.mannwhitneyu(nat_gross, nat_fine, alternative="greater")
                print(f"  NAT: 重体力|β| > 精细|β|?  U-test p={p:.4f} "
                      f"{'✓ 是' if p < 0.05 else '— 否'}")

            if len(a4b2_fine) >= 2 and len(a4b2_gross) >= 2:
                u, p = stats.mannwhitneyu(a4b2_fine, a4b2_gross, alternative="greater")
                print(f"  A4B2: 精细|β| > 重体力|β|?  U-test p={p:.4f} "
                      f"{'✓ 是' if p < 0.05 else '— 否'}")

    # ==================================================================
    # Part 2: 雷达图 — 高/低 NT 损伤组的 SIS 功能指纹
    # ==================================================================
    print(f"\n{'─' * 72}")
    print(f"  [Part 2] 雷达图: 高/低 NT 损伤组的 SIS 功能剖面")
    print(f"{'─' * 72}")

    if sis_available and len(sis_available) >= 6:
        for nt_name, nt_col in [("NAT", nat_col), ("A4B2", a4b2_col)]:
            if not nt_col:
                continue

            items = list(sis_available.keys())
            labels = [sis_available[k][:10] for k in items]

            sub = df[[nt_col] + items].dropna()
            if len(sub) < 50:
                continue

            # 按中位数分组
            median_val = sub[nt_col].median()
            high = sub[sub[nt_col] >= median_val]
            low = sub[sub[nt_col] < median_val]

            high_means = [high[item].mean() for item in items]
            low_means = [low[item].mean() for item in items]

            # 雷达图
            angles = np.linspace(0, 2 * np.pi, len(items), endpoint=False).tolist()
            high_plot = high_means + [high_means[0]]
            low_plot = low_means + [low_means[0]]
            angles += [angles[0]]

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            ax.fill(angles, high_plot, alpha=0.15, color="#E64B35")
            ax.fill(angles, low_plot, alpha=0.15, color="#4DBBD5")
            ax.plot(angles, high_plot, "o-", color="#E64B35", linewidth=2,
                    markersize=5, label=f"High {nt_name} Load (worse)")
            ax.plot(angles, low_plot, "o-", color="#4DBBD5", linewidth=2,
                    markersize=5, label=f"Low {nt_name} Load (better)")
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_title(f"SIS Functional Profile by {nt_name} Damage Level\n"
                         f"(High N={len(high)}, Low N={len(low)})",
                         fontsize=12, fontweight="bold", pad=20)
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
            plt.tight_layout()
            fig.savefig(fig_dir / f"radar_{nt_name}_SIS.png")
            plt.close(fig)
            print(f"  📊 radar_{nt_name}_SIS.png")

            # 每项的 t 检验
            print(f"\n  {nt_name} 高损伤 vs 低损伤:")
            for item, label in zip(items, [sis_available[k] for k in items]):
                t, p = stats.ttest_ind(high[item].dropna(), low[item].dropna())
                domain = SIS_DOMAIN.get(item, "")
                star = sig_stars(p)
                diff = high[item].mean() - low[item].mean()
                print(f"    {item} {label:<16s} Δ={diff:+.2f}  p={p:.3e} {star}  [{domain}]")

    # ==================================================================
    # Part 3: MoCA 认知子维度 × NT
    # ==================================================================
    print(f"\n{'─' * 72}")
    print(f"  [Part 3] MoCA 认知子维度 × NT (躯体-认知双维模型)")
    print(f"{'─' * 72}")

    moca_results = []
    if moca_available:
        for moca_item, moca_label in moca_available.items():
            for nt_name, nt_col in top_nts.items():
                predictors = [nt_col] + covars
                sub = df[[moca_item] + predictors].dropna()
                if len(sub) < 30:
                    continue

                sub_z = sub.copy()
                for p in predictors:
                    sub_z[p] = zscore(sub_z[p])

                try:
                    X = sm.add_constant(sub_z[predictors])
                    y = sub_z[moca_item].astype(float)
                    res = sm.OLS(y, X).fit()

                    moca_results.append({
                        "MoCA_Item": moca_item,
                        "MoCA_Label": moca_label,
                        "NT": nt_name,
                        "Beta": res.params[nt_col],
                        "P_value": res.pvalues[nt_col],
                        "N": len(sub),
                    })
                except Exception:
                    continue

    moca_df = pd.DataFrame(moca_results)
    if not moca_df.empty:
        moca_df["FDR_q"] = fdr_correct(moca_df["P_value"].values)
        moca_df.to_csv(out / "moca_x_nt.csv", index=False)

        print(f"\n  {'MoCA Item':<16s} {'Label':<18s}", end="")
        for nt_name in top_nts:
            print(f" {nt_name:>12s}", end="")
        print()
        print("  " + "─" * (34 + 13 * len(top_nts)))
        for moca_item in moca_available:
            label = moca_available[moca_item][:16]
            print(f"  {moca_item:<16s} {label:<18s}", end="")
            for nt_name in top_nts:
                match = moca_df[(moca_df["MoCA_Item"] == moca_item) & (moca_df["NT"] == nt_name)]
                if not match.empty:
                    p = match.iloc[0]["P_value"]
                    b = match.iloc[0]["Beta"]
                    star = sig_stars(p)
                    sign = "+" if b > 0 else "-"
                    print(f"  {sign}p={p:.1e}{star}", end="")
                else:
                    print(f" {'—':>12s}", end="")
            print()
    else:
        print("  无可用 MoCA 子维度数据")

    # ==================================================================
    # Part 4: 客观验证 — 10m 步行时间
    # ==================================================================
    print(f"\n{'─' * 72}")
    print(f"  [Part 4] 客观验证: NT → 10m 步行时间")
    print(f"{'─' * 72}")

    if walk_col:
        walk_results = []
        for nt_name, nt_col in top_nts.items():
            predictors = [nt_col] + covars
            sub = df[[walk_col] + predictors].dropna()
            if len(sub) < 30:
                continue

            sub_z = sub.copy()
            for p in predictors:
                sub_z[p] = zscore(sub_z[p])

            try:
                X = sm.add_constant(sub_z[predictors])
                y = sub_z[walk_col].astype(float)
                res = sm.OLS(y, X).fit()

                walk_results.append({
                    "NT": nt_name,
                    "Beta": res.params[nt_col],
                    "P_value": res.pvalues[nt_col],
                    "Direction": "步行变慢 ↑" if res.params[nt_col] > 0 else "步行变快 ↓",
                    "N": len(sub),
                })
                star = sig_stars(res.pvalues[nt_col])
                print(f"  {nt_name:<12s} β={res.params[nt_col]:+.4f}  "
                      f"p={res.pvalues[nt_col]:.2e} {star}")
            except Exception:
                continue

        walk_df = pd.DataFrame(walk_results)
        if not walk_df.empty:
            walk_df.to_csv(out / "walk_10m_x_nt.csv", index=False)
    else:
        print("  未找到 10m 步行变量")

    # ==================================================================
    # Part 5: 综合解离热图 (SIS + MoCA + 步行)
    # ==================================================================
    print(f"\n{'─' * 72}")
    print(f"  [Part 5] 综合解离热图")
    print(f"{'─' * 72}")

    all_outcomes = []

    # SIS
    if not sis_df.empty:
        for _, r in sis_df.iterrows():
            all_outcomes.append({
                "Outcome": r["SIS_Item"],
                "Label": r["SIS_Label"],
                "Category": "SIS-" + r["Domain"].split("(")[0].strip(),
                "NT": r["NT"],
                "neg_log_p": -np.log10(max(r["P_value"], 1e-30)),
                "Beta": r["Beta"],
            })

    # MoCA
    if not moca_df.empty:
        for _, r in moca_df.iterrows():
            all_outcomes.append({
                "Outcome": r["MoCA_Item"],
                "Label": r["MoCA_Label"],
                "Category": "MoCA",
                "NT": r["NT"],
                "neg_log_p": -np.log10(max(r["P_value"], 1e-30)),
                "Beta": r["Beta"],
            })

    all_df = pd.DataFrame(all_outcomes)
    if not all_df.empty and len(all_df["NT"].unique()) >= 2:
        # 透视表
        heat = all_df.pivot_table(index="Label", columns="NT",
                                   values="neg_log_p", aggfunc="first")
        heat_beta = all_df.pivot_table(index="Label", columns="NT",
                                        values="Beta", aggfunc="first")

        if heat.shape[0] >= 3 and heat.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(max(8, len(heat.columns) * 1.8),
                                             max(6, len(heat) * 0.35)))
            try:
                import seaborn as sns
                sns.heatmap(heat, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                            linewidths=0.5, cbar_kws={"label": "$-\\log_{10}(p)$"})
                # 标注显著 + 方向
                for i in range(len(heat.index)):
                    for j in range(len(heat.columns)):
                        val = heat.iloc[i, j]
                        b = heat_beta.iloc[i, j]
                        if pd.notna(val) and val > -np.log10(0.05):
                            sign = "+" if b > 0 else "−"
                            ax.text(j + 0.5, i + 0.82, f"{sign}★",
                                    ha="center", va="center", fontsize=8,
                                    color="white", fontweight="bold")
            except ImportError:
                im = ax.imshow(heat.values, cmap="YlOrRd", aspect="auto")
                ax.set_xticks(range(len(heat.columns)))
                ax.set_xticklabels(heat.columns, rotation=45, ha="right")
                ax.set_yticks(range(len(heat.index)))
                ax.set_yticklabels(heat.index, fontsize=7)
                plt.colorbar(im, ax=ax)

            ax.set_title("Double Dissociation: NT × Functional Sub-domain\n"
                         "(SIS Physical Items + MoCA Cognitive Items)",
                         fontsize=12, fontweight="bold")
            plt.tight_layout()
            fig.savefig(fig_dir / "double_dissociation_heatmap.png")
            plt.close(fig)
            print(f"  📊 double_dissociation_heatmap.png")

        all_df.to_csv(out / "full_dissociation_matrix.csv", index=False)

    # ==================================================================
    # 总结
    # ==================================================================
    print(f"\n{'=' * 72}")
    print(f"  ✅ 双重解离分析完成！")
    print(f"{'=' * 72}")
    print(f"  📁 {out.resolve()}")
    print(f"\n  📋 输出文件:")
    for f in sorted(out.glob("*.csv")):
        print(f"     • {f.name}")
    for f in sorted(fig_dir.glob("*.png")):
        print(f"     • figures/{f.name}")

    print(f"\n  🎯 论文关键图表:")
    print(f"     Figure 2A: figures/radar_NAT_SIS.png (NAT 功能指纹)")
    print(f"     Figure 2B: figures/radar_A4B2_SIS.png (A4B2 功能指纹)")
    print(f"     Figure 2C: figures/double_dissociation_heatmap.png (解离热图)")
    print(f"\n  📝 写作要点:")
    print(f"     1. NAT 主管: 重家务/快步走/拿重物 → '体力引擎'")
    print(f"     2. A4B2 主管: 穿衣/洗浴/扣纽扣 → '精准操作器'")
    print(f"     3. 如果 MoCA 空间执行 被 A4B2 预测 → '躯体-认知双维模型'")
    print(f"     4. 如果 10m 步行被 NAT 预测 → '客观行为验证'")


if __name__ == "__main__":
    main()
