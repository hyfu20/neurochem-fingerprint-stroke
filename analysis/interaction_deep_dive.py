#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互分析汇总 + NAT×A4B2 双雄交互检验
=====================================
1. 从已有结果中提取 NT×炎症 交互显著结果
2. 跑 NAT×A4B2 协同打击分析 (递质×递质)
3. 决定合写还是拆写

用法:
  python3 interaction_deep_dive.py --input merged_neuro_data.csv
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

warnings.filterwarnings("ignore")

import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="merged_neuro_data.csv")
    parser.add_argument("-o", "--output", default="/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/5.further_data_analysis_code/interaction_deep_dive_results")
    parser.add_argument("--existing-results", default=None,
                        help="已有的 interaction.csv 路径 (可选)")
    args = parser.parse_args()

    csv_path = args.input
    if not Path(csv_path).exists():
        server = ("/data/usersdir/liuzhengxin/Stepbystep/"
                  "6.NeurotransmitterMapping/3.variable_outcom_merge_data/"
                  "merged_neuro_data.csv")
        if Path(server).exists():
            csv_path = server
        else:
            print(f"❌ 找不到: {args.input}")
            sys.exit(1)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"\n{'=' * 72}")
    print(f"  交互分析深度挖掘")
    print(f"  数据: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"{'=' * 72}")

    # ── 识别列 ──
    tlv = find_col(df, ["TLV", "TLV_mm3"])
    nihss = find_col(df, ["A_NIHSS", "NIHSS"])
    age = find_col(df, ["AGE", "Age"])
    sex = find_col(df, ["SEX", "Sex"])
    covars = [c for c in [tlv, nihss, age, sex] if c]

    nt_cols = [c for c in df.columns if c.startswith("Load_")]
    for c in nt_cols + covars:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 找 NAT 和 A4B2 的列名
    nat_col = find_col(df, ["Load_NAT", "NAT"])
    a4b2_col = find_col(df, ["Load_A4B2", "A4B2"])

    mrs_candidates = ["D_MRS", "m3_mRS", "m6_mRS", "m12_mRS", "mRS"]
    mrs_found = [c for c in mrs_candidates if c in df.columns]

    inflam_cols = [c for c in ["BSL_IL6", "IL6", "CRP", "hsCRP", "IL10", "NLR", "WBC"]
                   if c in df.columns]
    for c in inflam_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"  NAT: {nat_col}, A4B2: {a4b2_col}")
    print(f"  mRS: {mrs_found}")
    print(f"  炎症: {inflam_cols}")
    print(f"  协变量: {covars}")

    # ==================================================================
    # Part 1: 已有的 NT × 炎症 交互结果汇总
    # ==================================================================
    print(f"\n{'─' * 72}")
    print(f"  [Part 1] NT × 炎症 交互分析 (重新跑 Top 5 NT)")
    print(f"{'─' * 72}")

    top5_nts = ["Load_NAT", "Load_A4B2", "Load_5HT6", "Load_DAT", "Load_VAChT"]
    top5_nts = [c for c in top5_nts if c in df.columns]

    inflam_results = []
    for mrs_col in mrs_found:
        target = f"_t_{mrs_col}"
        df[target] = df[mrs_col].apply(group_mrs)

        for nt in top5_nts:
            nt_name = nt.replace("Load_", "")
            for inflam in inflam_cols:
                predictors = [nt, inflam] + covars
                predictors = list(dict.fromkeys(predictors))

                sub = df[[target] + predictors].dropna()
                if len(sub) < 40:
                    continue

                sub_z = sub.copy()
                for p in predictors:
                    sub_z[p] = zscore(sub_z[p])
                sub_z["Interaction"] = sub_z[nt] * sub_z[inflam]

                try:
                    all_pred = predictors + ["Interaction"]
                    mod = OrderedModel(sub_z[target], sub_z[all_pred], distr="logit")
                    res = mod.fit(method="bfgs", disp=False)

                    inflam_results.append({
                        "Outcome": mrs_col,
                        "NT": nt_name,
                        "Inflam": inflam,
                        "Interaction_Beta": res.params["Interaction"],
                        "Interaction_OR": np.exp(res.params["Interaction"]),
                        "Interaction_P": res.pvalues["Interaction"],
                        "NT_Beta": res.params[nt],
                        "NT_P": res.pvalues[nt],
                        "Direction": "炎症放大损伤 ↑" if res.params["Interaction"] > 0
                                     else "炎症缓冲损伤 ↓",
                        "N": len(sub),
                    })
                except Exception:
                    continue

    inflam_df = pd.DataFrame(inflam_results)
    if not inflam_df.empty:
        inflam_df = inflam_df.sort_values("Interaction_P")
        inflam_df.to_csv(out / "nt_x_inflammation.csv", index=False)

        sig = inflam_df[inflam_df["Interaction_P"] < 0.05]
        print(f"\n  显著交互 (p<0.05): {len(sig)}/{len(inflam_df)}")
        print(f"\n  {'Outcome':<12s} {'NT':<10s} {'Inflam':<12s} {'OR':>8s} {'P':>12s} {'方向'}")
        print("  " + "─" * 65)
        for _, r in sig.head(15).iterrows():
            stars = "***" if r["Interaction_P"] < 0.001 else "**" if r["Interaction_P"] < 0.01 else "*"
            print(f"  {r['Outcome']:<12s} {r['NT']:<10s} {r['Inflam']:<12s} "
                  f"{r['Interaction_OR']:>8.3f} {r['Interaction_P']:>10.2e} {stars} {r['Direction']}")
    else:
        print("  无炎症交互结果")

    # ==================================================================
    # Part 2: NAT × A4B2 双雄协同打击
    # ==================================================================
    print(f"\n{'─' * 72}")
    print(f"  [Part 2] NAT × A4B2 双雄协同打击分析")
    print(f"{'─' * 72}")

    if not nat_col or not a4b2_col:
        print("  ❌ 未找到 NAT 或 A4B2 列")
    else:
        dual_results = []

        for mrs_col in mrs_found:
            target = f"_dual_{mrs_col}"
            df[target] = df[mrs_col].apply(group_mrs)

            predictors = [nat_col, a4b2_col] + covars
            sub = df[[target] + predictors].dropna()
            if len(sub) < 40:
                continue

            sub_z = sub.copy()
            for p in predictors:
                sub_z[p] = zscore(sub_z[p])

            # ── 模型 1: 各自独立效应 ──
            try:
                mod1 = OrderedModel(sub_z[target], sub_z[predictors], distr="logit")
                res1 = mod1.fit(method="bfgs", disp=False)
                nat_indep_p = res1.pvalues[nat_col]
                a4b2_indep_p = res1.pvalues[a4b2_col]
                nat_indep_or = np.exp(res1.params[nat_col])
                a4b2_indep_or = np.exp(res1.params[a4b2_col])
            except Exception:
                continue

            # ── 模型 2: 加入交互项 ──
            sub_z["NAT_x_A4B2"] = sub_z[nat_col] * sub_z[a4b2_col]
            try:
                mod2 = OrderedModel(sub_z[target],
                                     sub_z[predictors + ["NAT_x_A4B2"]],
                                     distr="logit")
                res2 = mod2.fit(method="bfgs", disp=False)
                inter_beta = res2.params["NAT_x_A4B2"]
                inter_p = res2.pvalues["NAT_x_A4B2"]
                inter_or = np.exp(inter_beta)
            except Exception:
                inter_beta = np.nan
                inter_p = np.nan
                inter_or = np.nan

            dual_results.append({
                "Outcome": mrs_col,
                "NAT_OR_indep": nat_indep_or,
                "NAT_P_indep": nat_indep_p,
                "A4B2_OR_indep": a4b2_indep_or,
                "A4B2_P_indep": a4b2_indep_p,
                "Interaction_Beta": inter_beta,
                "Interaction_OR": inter_or,
                "Interaction_P": inter_p,
                "N": len(sub),
            })

            # 打印
            stars_nat = "***" if nat_indep_p < 0.001 else "**" if nat_indep_p < 0.01 else "*" if nat_indep_p < 0.05 else ""
            stars_a4b2 = "***" if a4b2_indep_p < 0.001 else "**" if a4b2_indep_p < 0.01 else "*" if a4b2_indep_p < 0.05 else ""
            stars_inter = "***" if inter_p < 0.001 else "**" if inter_p < 0.01 else "*" if inter_p < 0.05 else ""

            print(f"\n  ── {mrs_col} (N={len(sub)}) ──")
            print(f"  NAT  独立效应:  OR={nat_indep_or:.3f}, p={nat_indep_p:.2e} {stars_nat}")
            print(f"  A4B2 独立效应:  OR={a4b2_indep_or:.3f}, p={a4b2_indep_p:.2e} {stars_a4b2}")
            print(f"  NAT×A4B2 交互:  OR={inter_or:.3f}, p={inter_p:.2e} {stars_inter}")

            if inter_p < 0.05:
                if inter_beta > 0:
                    print(f"  🔥 显著协同打击! 两个系统同时受损 → 预后断崖式恶化")
                else:
                    print(f"  🛡 显著拮抗效应! 一个系统的损伤可被另一个代偿")
            else:
                print(f"  → 交互不显著: 两个系统独立发挥作用, 可平行讨论")

        dual_df = pd.DataFrame(dual_results)
        if not dual_df.empty:
            dual_df.to_csv(out / "nat_x_a4b2_interaction.csv", index=False)

    # ==================================================================
    # Part 3: 所有 Top5 NT 两两交互矩阵
    # ==================================================================
    print(f"\n{'─' * 72}")
    print(f"  [Part 3] Top 5 NT 两两交互矩阵")
    print(f"{'─' * 72}")

    mrs_col = find_col(df, ["D_MRS", "m3_mRS", "m6_mRS", "m12_mRS"])
    if mrs_col and len(top5_nts) >= 2:
        target = f"_pair_{mrs_col}"
        df[target] = df[mrs_col].apply(group_mrs)

        pair_results = []
        for i in range(len(top5_nts)):
            for j in range(i + 1, len(top5_nts)):
                nt_a, nt_b = top5_nts[i], top5_nts[j]
                name_a = nt_a.replace("Load_", "")
                name_b = nt_b.replace("Load_", "")

                predictors = [nt_a, nt_b] + covars
                sub = df[[target] + predictors].dropna()
                if len(sub) < 40:
                    continue

                sub_z = sub.copy()
                for p in predictors:
                    sub_z[p] = zscore(sub_z[p])
                sub_z["Inter"] = sub_z[nt_a] * sub_z[nt_b]

                try:
                    mod = OrderedModel(sub_z[target],
                                        sub_z[predictors + ["Inter"]],
                                        distr="logit")
                    res = mod.fit(method="bfgs", disp=False)
                    pair_results.append({
                        "NT_A": name_a,
                        "NT_B": name_b,
                        "Interaction_OR": np.exp(res.params["Inter"]),
                        "Interaction_P": res.pvalues["Inter"],
                        "Direction": "协同 ↑↑" if res.params["Inter"] > 0 else "拮抗 ↑↓",
                    })
                except Exception:
                    continue

        pair_df = pd.DataFrame(pair_results)
        if not pair_df.empty:
            pair_df = pair_df.sort_values("Interaction_P")
            pair_df.to_csv(out / "pairwise_nt_interaction.csv", index=False)

            print(f"\n  {'NT_A':<10s} {'NT_B':<10s} {'OR':>8s} {'P':>12s} {'方向'}")
            print("  " + "─" * 50)
            for _, r in pair_df.iterrows():
                stars = "***" if r["Interaction_P"] < 0.001 else "**" if r["Interaction_P"] < 0.01 else "*" if r["Interaction_P"] < 0.05 else ""
                print(f"  {r['NT_A']:<10s} {r['NT_B']:<10s} "
                      f"{r['Interaction_OR']:>8.3f} {r['Interaction_P']:>10.2e} {stars} {r['Direction']}")

    # ==================================================================
    # 最终决策建议
    # ==================================================================
    print(f"\n{'=' * 72}")
    print(f"  ✅ 交互分析完成！")
    print(f"{'=' * 72}")
    print(f"  📁 {out.resolve()}")
    print(f"\n  📋 写作决策指南:")
    print(f"     • nt_x_inflammation.csv → NT×炎症, 有显著则写'协同打击'")
    print(f"     • nat_x_a4b2_interaction.csv → NAT×A4B2:")
    print(f"       - P<0.05 → 合写: '多系统协同打击'")
    print(f"       - P>0.05 → 可拆: '平行独立通路'")
    print(f"     • pairwise_nt_interaction.csv → 所有两两交互总览")


if __name__ == "__main__":
    main()
