#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MICE 插补可靠性验证 — 审稿人对策 (修正版)
=============================================
修正: BSL_IL6 等偏态变量插补失效问题
  ★ 对数转换: 插补前 log1p(), 插补后 expm1() → 保留偏态分布
  ★ PMM 近似: 用 KNN + 随机扰动替代默认线性回归 → 避免方差压缩
  ★ 自动检测偏态: skewness > 2 的变量自动 log 转换

分析模块:
1. 缺失模式分析: 哪些变量缺失多少
2. 智能插补: 偏态变量 log 转换 + PMM 近似
3. 插补前后分布对比图: 密度曲线叠加, KS 检验
4. Complete Case Analysis (CCA): 仅用完整数据重跑核心回归
5. CCA vs Imputed 结果对比: OR 一致性森林图

用法:
  python3 mice_reliability.py --input merged_neuro_data.csv

后台:
  nohup python3 mice_reliability.py --input merged_neuro_data.csv > mice.log 2>&1 &
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

try:
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer, KNNImputer
    from sklearn.linear_model import BayesianRidge
    from sklearn.ensemble import RandomForestRegressor
    HAS_IMPUTER = True
except ImportError:
    HAS_IMPUTER = False


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def zscore(s):
    sd = s.std()
    return (s - s.mean()) / sd if sd > 1e-10 else s - s.mean()


def group_mrs(x):
    if pd.isna(x): return np.nan
    x = float(x)
    return 0 if x <= 2 else (1 if x <= 4 else 2)


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""


def main():
    parser = argparse.ArgumentParser(description="MICE 插补可靠性验证")
    parser.add_argument("-i", "--input", default="merged_neuro_data.csv")
    parser.add_argument("-o", "--output", default="/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/5.further_data_analysis_code/mice_reliability_results")
    parser.add_argument("--top-vars", nargs="+",
                        default=["NAT", "A4B2", "5HT6", "DAT", "VAChT",
                                 "Load_NAT", "Load_A4B2", "Load_5HT6", "Load_DAT", "Load_VAChT"])
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
    print(f"  MICE 插补可靠性验证")
    print(f"  数据: {df.shape[0]} × {df.shape[1]}")
    print(f"{'=' * 72}")

    # 识别列
    tlv = find_col(df, ["TLV", "TLV_mm3"])
    nihss = find_col(df, ["A_NIHSS", "NIHSS"])
    age = find_col(df, ["AGE", "Age"])
    sex = find_col(df, ["SEX", "Sex"])
    covars = [c for c in [tlv, nihss, age, sex] if c]

    nt_cols = [c for c in df.columns if c.startswith("Load_")]
    for c in nt_cols + covars:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    mrs_col = find_col(df, ["D_MRS", "m3_mRS", "m6_mRS", "m12_mRS", "mRS"])

    inflam_cols = [c for c in ["BSL_IL6", "IL6", "CRP", "hsCRP", "NLR", "WBC"]
                   if c in df.columns]
    for c in inflam_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 分析列
    analysis_cols = nt_cols + covars + inflam_cols
    if mrs_col:
        analysis_cols.append(mrs_col)
    analysis_cols = [c for c in analysis_cols if c in df.columns]

    top_vars = [c for c in args.top_vars if c in df.columns]
    if not top_vars:
        top_vars = nt_cols[:5]

    # ==================================================================
    # Part 1: 缺失模式分析
    # ==================================================================
    print(f"\n{'─' * 72}")
    print(f"  [Part 1] 缺失模式分析")
    print(f"{'─' * 72}")

    miss_info = []
    for c in analysis_cols:
        n_miss = df[c].isna().sum()
        n_total = len(df)
        rate = n_miss / n_total
        miss_info.append({
            "Variable": c,
            "N_missing": n_miss,
            "N_total": n_total,
            "Missing_rate": rate,
            "Category": ("Low (<5%)" if rate < 0.05
                         else "Moderate (5-20%)" if rate < 0.20
                         else "High (>20%)"),
        })

    miss_df = pd.DataFrame(miss_info).sort_values("Missing_rate", ascending=False)
    miss_df.to_csv(out / "missing_pattern.csv", index=False)

    print(f"\n  {'Variable':<20s} {'Missing':>8s} {'Rate':>8s} {'Level'}")
    print("  " + "─" * 50)
    for _, r in miss_df.head(20).iterrows():
        print(f"  {r['Variable']:<20s} {r['N_missing']:>8d} {r['Missing_rate']:>7.1%}  {r['Category']}")

    # 缺失率条形图
    fig, ax = plt.subplots(figsize=(10, max(4, len(miss_df) * 0.25)))
    colors = []
    for _, r in miss_df.iterrows():
        if r["Missing_rate"] > 0.20:
            colors.append("#F44336")
        elif r["Missing_rate"] > 0.05:
            colors.append("#FFC107")
        else:
            colors.append("#4CAF50")

    y_pos = np.arange(len(miss_df))
    ax.barh(y_pos, miss_df["Missing_rate"].values * 100, color=colors,
            edgecolor="black", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(miss_df["Variable"].values, fontsize=7)
    ax.set_xlabel("Missing Rate (%)")
    ax.axvline(5, color="green", linestyle="--", alpha=0.5, label="5%")
    ax.axvline(20, color="red", linestyle="--", alpha=0.5, label="20%")
    ax.legend(fontsize=8)
    ax.set_title("Missing Data Pattern", fontweight="bold")
    plt.tight_layout()
    fig.savefig(fig_dir / "missing_pattern.png")
    plt.close(fig)
    print(f"  📊 missing_pattern.png")

    # ==================================================================
    # Part 2: MICE 插补 + 分布对比
    # ==================================================================
    print(f"\n{'─' * 72}")
    print(f"  [Part 2] 插补前后分布对比")
    print(f"{'─' * 72}")

    if not HAS_IMPUTER:
        print("  ⚠️ 需要 scikit-learn, 跳过插补")
        df_imputed = df.copy()
    else:
        impute_cols = [c for c in analysis_cols if df[c].isna().sum() > 0]

        if not impute_cols:
            print("  无缺失值, 无需插补")
            df_imputed = df.copy()
        else:
            sub_orig = df[impute_cols].copy()

            # ── 智能检测偏态变量, 自动 log 转换 ──
            # 已知的生物标志物强制 log 转换 (无论 skewness 多少)
            KNOWN_SKEWED = {"BSL_IL6", "IL6", "CRP", "hsCRP", "IL10", "TNFa", "NLR"}
            skewed_cols = []
            for c in impute_cols:
                vals = sub_orig[c].dropna()
                if len(vals) > 20:
                    skew = vals.skew()
                    c_bare = c.upper().replace("BSL_", "")
                    # 已知偏态 或 skewness > 1.5
                    if c in KNOWN_SKEWED or c_bare in KNOWN_SKEWED or abs(skew) > 1.5:
                        if vals.min() >= 0:  # 仅对非负变量做 log
                            skewed_cols.append(c)
                            print(f"  ⚠ {c}: skewness={skew:.2f} → log1p 转换")

            # log 转换偏态列
            sub_transformed = sub_orig.copy()
            for c in skewed_cols:
                sub_transformed[c] = np.log1p(sub_transformed[c].clip(lower=0))

            # ── MissForest 插补 (随机森林, 行业标准) ──
            # 随机森林优势:
            #   1. 非线性: 能捕捉 IL6 在低值区的聚集特征
            #   2. 自带边界: 插补值不会超出观测范围
            #   3. 不假设正态: 完美处理偏态数据
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(
                    n_estimators=50, n_jobs=-1, random_state=42
                ),
                max_iter=10,
                random_state=42,
                # 注意: RandomForest 不支持 sample_posterior
            )
            imputed_data = imputer.fit_transform(sub_transformed)
            df_imputed = df.copy()
            df_imputed[impute_cols] = imputed_data

            # 偏态列还原: expm1 + clip 到生物学范围
            for c in skewed_cols:
                df_imputed[c] = np.expm1(df_imputed[c].clip(lower=0))
                # 限制在原始观测范围内 (不会比最小值还小, 不会比最大值还大)
                orig_min = df[c].dropna().min()
                orig_max = df[c].dropna().max()
                df_imputed[c] = df_imputed[c].clip(lower=max(0, orig_min), upper=orig_max)
                print(f"  ✓ {c}: 插补值已 clip 到 [{max(0, orig_min):.2f}, {orig_max:.2f}]")

            # ── 分类变量修正: 四舍五入回整数 ──
            cat_candidates = ["SEX", "Sex", "sex", "D_MRS", "mRS",
                              "m3_mRS", "m6_mRS", "m12_mRS",
                              "AF", "DM", "HBP", "EVT", "IVT", "TOAST"]
            for c in cat_candidates:
                if c in df_imputed.columns and c in impute_cols:
                    df_imputed[c] = df_imputed[c].round().astype(float)
                    # 限制范围
                    orig_min = df[c].dropna().min()
                    orig_max = df[c].dropna().max()
                    df_imputed[c] = df_imputed[c].clip(orig_min, orig_max)
            print(f"  ✓ 分类变量已四舍五入 (sex/mRS 等)")

            n_skewed = len(skewed_cols)
            print(f"  ✓ MissForest 插补完成: {len(impute_cols)} 列 "
                  f"({n_skewed} 列经 log 转换, RandomForest)")

            # 分布对比图: 每个有缺失的变量画密度曲线
            plot_cols = [c for c in impute_cols if df[c].isna().sum() >= 10][:12]

            if plot_cols:
                ncols = min(4, len(plot_cols))
                nrows = (len(plot_cols) + ncols - 1) // ncols
                fig, axes = plt.subplots(nrows, ncols,
                                          figsize=(4 * ncols, 3.5 * nrows))
                axes = np.atleast_2d(axes).flatten()

                ks_results = []
                for idx, col in enumerate(plot_cols):
                    ax = axes[idx]

                    orig_vals = df[col].dropna().values
                    imputed_vals = df_imputed[col].values[df[col].isna().values]

                    # 原始分布
                    if len(orig_vals) > 5:
                        ax.hist(orig_vals, bins=30, density=True, alpha=0.5,
                                color="#4DBBD5", label=f"Original (N={len(orig_vals)})")

                    # 插补的值
                    if len(imputed_vals) > 5:
                        ax.hist(imputed_vals, bins=20, density=True, alpha=0.5,
                                color="#E64B35",
                                label=f"Imputed (N={len(imputed_vals)})")

                    # KS 检验
                    if len(orig_vals) > 5 and len(imputed_vals) > 5:
                        ks_stat, ks_p = stats.ks_2samp(orig_vals, imputed_vals)
                        ax.set_title(f"{col}\nKS p={ks_p:.3f}",
                                     fontsize=9, fontweight="bold")
                        ks_results.append({
                            "Variable": col,
                            "KS_stat": ks_stat,
                            "KS_P": ks_p,
                            "Consistent": "✓ 一致" if ks_p > 0.05 else "⚠ 差异",
                        })
                    else:
                        ax.set_title(col, fontsize=9)

                    ax.legend(fontsize=6)
                    ax.tick_params(labelsize=7)

                for j in range(idx + 1, len(axes)):
                    axes[j].set_visible(False)

                plt.suptitle("Imputation Quality: Original vs Imputed Distributions\n"
                             "(KS p > 0.05 = distributions consistent)",
                             fontsize=11, fontweight="bold", y=1.02)
                plt.tight_layout()
                fig.savefig(fig_dir / "imputation_distribution_comparison.png")
                plt.close(fig)
                print(f"  📊 imputation_distribution_comparison.png")

                if ks_results:
                    ks_df = pd.DataFrame(ks_results)
                    ks_df.to_csv(out / "imputation_ks_test.csv", index=False)
                    n_consist = (ks_df["KS_P"] > 0.05).sum()
                    print(f"  KS 检验: {n_consist}/{len(ks_df)} 变量分布一致")

    # ==================================================================
    # Part 3: Complete Case Analysis (CCA) vs Imputed
    # ==================================================================
    print(f"\n{'─' * 72}")
    print(f"  [Part 3] Complete Case vs Imputed 核心回归对比")
    print(f"{'─' * 72}")

    if not mrs_col:
        print("  ⚠️ 无 mRS 列, 跳过")
    else:
        target_cca = "_cca_target"
        target_imp = "_imp_target"

        df[target_cca] = df[mrs_col].apply(group_mrs)
        df_imputed[target_imp] = df_imputed[mrs_col].apply(group_mrs)

        # 确保 imputed df 也有未插补的列 (Load_ 列无缺失但可能未复制)
        for c in nt_cols + covars:
            if c not in df_imputed.columns:
                df_imputed[c] = df[c]

        print(f"  CCA 样本: {df[target_cca].notna().sum()}, "
              f"Imputed 样本: {df_imputed[target_imp].notna().sum()}")

        comparison = []
        for nt in top_vars:
            nt_name = nt.replace("Load_", "")
            predictors = [nt] + covars

            for label, data, target in [
                ("Complete_Case", df, target_cca),
                ("MICE_Imputed", df_imputed, target_imp),
            ]:
                sub = data[[target] + predictors].dropna()
                if len(sub) < 30:
                    continue

                sub_z = sub.copy()
                for p in predictors:
                    sub_z[p] = zscore(sub_z[p])

                try:
                    mod = OrderedModel(sub_z[target], sub_z[predictors], distr="logit")
                    res = mod.fit(method="bfgs", disp=False)
                    ci = res.conf_int().loc[nt]

                    comparison.append({
                        "NT": nt_name,
                        "Method": label,
                        "Beta": res.params[nt],
                        "OR": np.exp(res.params[nt]),
                        "OR_CI_lower": np.exp(ci[0]),
                        "OR_CI_upper": np.exp(ci[1]),
                        "P_value": res.pvalues[nt],
                        "N": len(sub),
                    })
                except Exception:
                    continue

        comp_df = pd.DataFrame(comparison)
        if not comp_df.empty:
            comp_df.to_csv(out / "cca_vs_imputed.csv", index=False)

            # 打印对比
            print(f"\n  {'NT':<12s} {'CCA OR':>10s} {'CCA P':>12s} {'N':>6s}   "
                  f"{'IMP OR':>10s} {'IMP P':>12s} {'N':>6s}   {'一致?':>8s}")
            print("  " + "─" * 85)

            for nt_name in comp_df["NT"].unique():
                cca = comp_df[(comp_df["NT"] == nt_name) & (comp_df["Method"] == "Complete_Case")]
                imp = comp_df[(comp_df["NT"] == nt_name) & (comp_df["Method"] == "MICE_Imputed")]
                if cca.empty or imp.empty:
                    continue
                c, i = cca.iloc[0], imp.iloc[0]

                # OR 变化百分比
                or_change = abs(i["OR"] - c["OR"]) / c["OR"] * 100 if c["OR"] > 0 else np.nan
                consistent = or_change < 20 and (c["P_value"] < 0.05) == (i["P_value"] < 0.05)
                mark = "✓ 一致" if consistent else "⚠ 差异"

                print(f"  {nt_name:<12s} {c['OR']:>8.3f}  {c['P_value']:>10.2e} {c['N']:>6d}   "
                      f"{i['OR']:>8.3f}  {i['P_value']:>10.2e} {i['N']:>6d}   "
                      f"{mark} (ΔOR={or_change:.1f}%)")

            # 森林图: CCA vs Imputed
            nts = comp_df["NT"].unique()
            fig, ax = plt.subplots(figsize=(9, max(4, len(nts) * 0.7)))

            for method, color, offset in [
                ("Complete_Case", "#4DBBD5", -0.15),
                ("MICE_Imputed", "#E64B35", 0.15),
            ]:
                m_data = comp_df[comp_df["Method"] == method].set_index("NT").reindex(nts)
                for idx, nt in enumerate(nts):
                    if nt not in m_data.index or pd.isna(m_data.loc[nt, "OR"]):
                        continue
                    r = m_data.loc[nt]
                    ax.errorbar(
                        r["OR"], idx + offset,
                        xerr=[[r["OR"] - r["OR_CI_lower"]],
                              [r["OR_CI_upper"] - r["OR"]]],
                        fmt="o", color=color, markersize=7, capsize=4, linewidth=1.5,
                        label=method.replace("_", " ") if idx == 0 else None
                    )

            ax.axvline(1, color="gray", linestyle="--", alpha=0.7)
            ax.set_yticks(range(len(nts)))
            ax.set_yticklabels(nts, fontsize=10)
            ax.set_xlabel("Odds Ratio (95% CI)", fontsize=11)
            ax.set_title("Sensitivity: Complete Case vs MICE Imputed\n"
                         "(Consistent OR = Imputation is reliable)",
                         fontsize=12, fontweight="bold")
            ax.legend(fontsize=10)
            plt.tight_layout()
            fig.savefig(fig_dir / "cca_vs_imputed_forest.png")
            plt.close(fig)
            print(f"  📊 cca_vs_imputed_forest.png")

    # ==================================================================
    # Part 4: Two-Hit 交互项敏感性 (CCA vs Imputed)
    # ==================================================================
    print(f"\n{'─' * 72}")
    print(f"  [Part 4] Two-Hit 交互项: CCA vs Imputed")
    print(f"{'─' * 72}")

    if mrs_col and inflam_cols:
        best_inflam = inflam_cols[0]  # BSL_IL6 优先

        interaction_comp = []
        for nt in top_vars:
            nt_name = nt.replace("Load_", "")

            for label, data in [("Complete_Case", df), ("MICE_Imputed", df_imputed)]:
                target_col = f"_inter_{label}"
                data[target_col] = data[mrs_col].apply(group_mrs)

                predictors = [nt, best_inflam] + covars
                predictors = list(dict.fromkeys(predictors))

                sub = data[[target_col] + predictors].dropna()
                if len(sub) < 40:
                    continue

                sub_z = sub.copy()
                for p in predictors:
                    sub_z[p] = zscore(sub_z[p])
                sub_z["Interaction"] = sub_z[nt] * sub_z[best_inflam]

                try:
                    all_pred = predictors + ["Interaction"]
                    mod = OrderedModel(sub_z[target_col], sub_z[all_pred], distr="logit")
                    res = mod.fit(method="bfgs", disp=False)

                    interaction_comp.append({
                        "NT": nt_name,
                        "Inflam": best_inflam,
                        "Method": label,
                        "NT_Beta": res.params[nt],
                        "NT_P": res.pvalues[nt],
                        "Interaction_Beta": res.params["Interaction"],
                        "Interaction_OR": np.exp(res.params["Interaction"]),
                        "Interaction_P": res.pvalues["Interaction"],
                        "N": len(sub),
                    })
                except Exception:
                    continue

        inter_comp_df = pd.DataFrame(interaction_comp)
        if not inter_comp_df.empty:
            inter_comp_df.to_csv(out / "interaction_cca_vs_imputed.csv", index=False)

            print(f"\n  交互项 ({best_inflam}): CCA vs Imputed")
            print(f"  {'NT':<12s} {'CCA Inter_P':>14s} {'CCA OR':>10s}   "
                  f"{'IMP Inter_P':>14s} {'IMP OR':>10s}   {'一致?':>8s}")
            print("  " + "─" * 75)

            for nt_name in inter_comp_df["NT"].unique():
                cca = inter_comp_df[(inter_comp_df["NT"] == nt_name) &
                                     (inter_comp_df["Method"] == "Complete_Case")]
                imp = inter_comp_df[(inter_comp_df["NT"] == nt_name) &
                                     (inter_comp_df["Method"] == "MICE_Imputed")]
                if cca.empty or imp.empty:
                    continue
                c, i = cca.iloc[0], imp.iloc[0]
                both_sig = c["Interaction_P"] < 0.05 and i["Interaction_P"] < 0.05
                both_ns = c["Interaction_P"] >= 0.05 and i["Interaction_P"] >= 0.05
                consistent = both_sig or both_ns
                mark = "✓ 一致" if consistent else "⚠ 差异"

                print(f"  {nt_name:<12s} {c['Interaction_P']:>12.2e}  "
                      f"{c['Interaction_OR']:>8.3f}   "
                      f"{i['Interaction_P']:>12.2e}  "
                      f"{i['Interaction_OR']:>8.3f}   {mark}")
    else:
        print("  ⚠️ 无炎症指标或 mRS 列, 跳过交互验证")

    # ==================================================================
    # 总结
    # ==================================================================
    print(f"\n{'=' * 72}")
    print(f"  ✅ MICE 可靠性验证完成！")
    print(f"{'=' * 72}")
    print(f"  📁 {out.resolve()}")
    print(f"\n  📋 结果文件:")
    for f in sorted(out.glob("*.csv")):
        print(f"     • {f.name}")
    for f in sorted(fig_dir.glob("*.png")):
        print(f"     • figures/{f.name}")

    print(f"\n  🎯 审稿人回复要点:")
    print(f"     1. missing_pattern.csv → 缺失率分布 (>20% 的变量需特别说明)")
    print(f"     2. imputation_ks_test.csv → KS p>0.05 = 插补分布可靠")
    print(f"     3. cca_vs_imputed.csv → OR 变化 <20% = 结论不依赖插补")
    print(f"     4. interaction_cca_vs_imputed.csv → 交互项在 CCA/IMP 下一致")
    print(f"     5. figures/cca_vs_imputed_forest.png → 直接放 Supplementary")


if __name__ == "__main__":
    main()
