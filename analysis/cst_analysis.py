#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CST_Load 专项分析 — 修复残差崩溃版
"""
import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import warnings
warnings.filterwarnings("ignore")

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns: return c
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
    print(f"\n{'=' * 65}\n  CST_Load 专项分析 (含实时 Koch 残差计算)\n{'=' * 65}")
    print(f"  数据: {csv_path}\n  样本: {len(df)} × {len(df.columns)}")

    cst = find_col(df, ["CST_Load", "CST_load", "cst_load"])
    tlv = find_col(df, ["TLV", "TLV_mm3", "tlv"])
    nihss = find_col(df, ["A_NIHSS", "NIHSS"])
    age = find_col(df, ["AGE", "Age"])
    sex = find_col(df, ["SEX", "Sex"])
    mrs_col = find_col(df, ["D_MRS", "m3_mRS", "m6_mRS", "m12_mRS"])

    if not cst:
        print("❌ 未找到 CST_Load 列!"); sys.exit(1)

    KNOWN_NT = ["5HT1a","5HT1b","5HT2a","5HT4","5HT6","5HTT",
                "A4B2","D1","D2","DAT","M1","NAT","VAChT",
                "human_CHA","JHU_EC","Lateral_Path","Medial_Path"]
    nt_cols = [c for c in df.columns if c in KNOWN_NT]
    if not nt_cols: nt_cols = [c for c in df.columns if c.startswith("Load_")]
    print(f"  NT 列: {len(nt_cols)} 个, 前5: {nt_cols[:5]}")

    for c in [cst, tlv, nihss, age, sex, mrs_col]:
        if c: df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in nt_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ════════════════════════════════════════════════
    # 1. 描述统计
    # ════════════════════════════════════════════════
    print(f"\n{'─' * 65}\n  [1] CST_Load 描述统计\n{'─' * 65}")
    cst_vals = df[cst].dropna()
    n_total = len(cst_vals)
    n_zero = (cst_vals == 0).sum()
    n_pos = (cst_vals > 0).sum()
    print(f"  N={n_total}, 非零={n_pos} ({n_pos/n_total*100:.1f}%)")
    print(f"  Mean={cst_vals.mean():.2f}, Median={cst_vals.median():.2f}, SD={cst_vals.std():.2f}")
    print(f"  Range=[{cst_vals.min():.2f}, {cst_vals.max():.2f}]")

    # ════════════════════════════════════════════════
    # 2. 相关性
    # ════════════════════════════════════════════════
    print(f"\n{'─' * 65}\n  [2] CST_Load 相关性\n{'─' * 65}")
    for label, col in [("TLV", tlv), ("NIHSS", nihss), ("mRS", mrs_col)] + [(c, c) for c in nt_cols[:5]]:
        if not col: continue
        valid = df[[cst, col]].dropna()
        if len(valid) < 20: continue
        r, p = stats.spearmanr(valid[cst], valid[col])
        print(f"  CST vs {label:<18s}: r={r:+.3f}, p={p:.2e} {sig_stars(p)}")

    # ════════════════════════════════════════════════
    # 3. CST → mRS 三层敏感性
    # ════════════════════════════════════════════════
    print(f"\n{'─' * 65}\n  [3] CST_Load → mRS 有序回归\n{'─' * 65}")
    target = "_cst_target"
    df[target] = df[mrs_col].apply(group_mrs)

    cst_results = []
    for mname, preds in [("A_Unadjusted", [cst]),
                          ("B_+Demo", [cst]+[c for c in [age,sex] if c]),
                          ("C_+Full", [cst]+[c for c in [tlv,nihss,age,sex] if c])]:
        sub = df[[target]+preds].dropna()
        if len(sub) < 30: continue
        sub_z = sub.copy()
        for p in preds: sub_z[p] = zscore(sub_z[p])
        try:
            res = OrderedModel(sub_z[target], sub_z[preds], distr="logit").fit(method="bfgs", disp=False)
            ci = res.conf_int().loc[cst]
            cst_results.append({"Model": mname, "OR": np.exp(res.params[cst]),
                                "OR_CI_lower": np.exp(ci[0]), "OR_CI_upper": np.exp(ci[1]),
                                "P_value": res.pvalues[cst], "N": len(sub)})
            print(f"  {mname:<15s}: OR={cst_results[-1]['OR']:.3f} [{np.exp(ci[0]):.2f}-{np.exp(ci[1]):.2f}], p={res.pvalues[cst]:.2e} {sig_stars(res.pvalues[cst])}")
        except Exception as e:
            print(f"  {mname}: 失败 — {e}")

    # ════════════════════════════════════════════════
    # 4. 现场计算 Koch 残差 + 有无 CST 控制对比
    # ════════════════════════════════════════════════
    print(f"\n{'─' * 65}\n  [4] Koch 残差 + CST 控制对比\n{'─' * 65}")
    print("  计算 Resid_NT = Load_NT - β×TLV ...")
    resid_cols = []
    for nt in nt_cols:
        valid = df[[nt, tlv]].dropna()
        if len(valid) > 30:
            slope, intercept, _, _, _ = stats.linregress(valid[tlv], valid[nt])
            rname = f"Resid_{nt.replace('Load_','')}"
            df[rname] = np.nan
            df.loc[valid.index, rname] = df.loc[valid.index, nt] - (intercept + slope * df.loc[valid.index, tlv])
            resid_cols.append(rname)
    print(f"  ✅ {len(resid_cols)} 个残差列")

    base_no_cst = [c for c in [tlv, nihss, age, sex] if c]
    base_with_cst = [c for c in [tlv, nihss, cst, age, sex] if c]

    compare_results = []
    for nt_col in resid_cols:
        nt_name = nt_col.replace("Resid_", "")
        for label, covars in [("无CST", base_no_cst), ("+CST", base_with_cst)]:
            preds = [nt_col] + covars
            sub = df[[target]+preds].dropna()
            if len(sub) < 30: continue
            sub_z = sub.copy()
            for p in preds: sub_z[p] = zscore(sub_z[p])
            try:
                res = OrderedModel(sub_z[target], sub_z[preds], distr="logit").fit(method="bfgs", disp=False)
                ci = res.conf_int().loc[nt_col]
                compare_results.append({"NT": nt_name, "Control": label,
                    "OR": np.exp(res.params[nt_col]), "P": res.pvalues[nt_col],
                    "OR_CI_lower": np.exp(ci[0]), "OR_CI_upper": np.exp(ci[1]), "N": len(sub)})
            except Exception:
                pass

    if compare_results:
        cdf = pd.DataFrame(compare_results)
        print(f"\n  {'NT':<18s} {'无CST OR':>10s} {'p':>10s}  │  {'+CST OR':>10s} {'p':>10s}  │ 判定")
        print(f"  {'─'*18} {'─'*10} {'─'*10}  │  {'─'*10} {'─'*10}  │ {'─'*6}")
        for nt in cdf["NT"].unique():
            r0 = cdf[(cdf["NT"]==nt)&(cdf["Control"]=="无CST")]
            r1 = cdf[(cdf["NT"]==nt)&(cdf["Control"]=="+CST")]
            if r0.empty or r1.empty: continue
            tag = "✅保持" if r1.iloc[0]["P"] < 0.05 else "❌衰减"
            print(f"  {nt:<18s} {r0.iloc[0]['OR']:>10.3f} {r0.iloc[0]['P']:>10.2e}{sig_stars(r0.iloc[0]['P']):3s} │  "
                  f"{r1.iloc[0]['OR']:>10.3f} {r1.iloc[0]['P']:>10.2e}{sig_stars(r1.iloc[0]['P']):3s} │ {tag}")

    # ════════════════════════════════════════════════
    # 输出目录
    # ════════════════════════════════════════════════
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

    if cst_results:
        pd.DataFrame(cst_results).to_csv(out_dir / "cst_analysis_results.csv", index=False)
    if compare_results:
        cdf.to_csv(out_dir / "cst_nt_comparison.csv", index=False)

    # ════════════════════════════════════════════════
    # 绘图: 4-Panel 总图 + 总表
    # ════════════════════════════════════════════════
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from datetime import datetime
    plt.rcParams.update({"font.sans-serif": ["Arial","DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("CST_Load Analysis — Summary\n"
                 f"(Generated {datetime.now().strftime('%Y-%m-%d %H:%M')})",
                 fontsize=16, fontweight="bold", y=0.98)

    # Panel A: 分布
    ax1 = fig.add_subplot(2, 3, 1)
    pos_vals = cst_vals[cst_vals > 0]
    ax1.hist(pos_vals, bins=50, color="#E64B35", edgecolor="black", linewidth=0.3, alpha=0.85)
    ax1.axvline(pos_vals.median(), color="navy", linestyle="--", linewidth=1.5,
                label=f"Median={pos_vals.median():.0f}")
    ax1.set_xlabel("CST_Load (mm³)"); ax1.set_ylabel("Count")
    ax1.set_title(f"A. CST_Load Distribution\n(N={n_total}, {n_pos} overlap [{n_pos/n_total*100:.0f}%])", fontweight="bold")
    ax1.legend(fontsize=8)

    # Panel B: CST vs TLV
    ax2 = fig.add_subplot(2, 3, 2)
    if tlv:
        v = df[[cst,tlv]].dropna()
        ax2.scatter(v[tlv], v[cst], s=5, alpha=0.3, c="#4DBBD5")
        r_val, p_val = stats.spearmanr(v[tlv], v[cst])
        ax2.set_xlabel("TLV (mm³)"); ax2.set_ylabel("CST_Load (mm³)")
        ax2.set_title(f"B. CST vs TLV (r={r_val:.3f}, p={p_val:.1e})", fontweight="bold")

    # Panel C: CST → mRS 三层
    ax3 = fig.add_subplot(2, 3, 3)
    if cst_results:
        rdf = pd.DataFrame(cst_results)
        y = np.arange(len(rdf))
        colors = ["#FFA500","#4DBBD5","#E64B35"]
        ax3.barh(y, rdf["OR"].values-1, left=1, color=colors[:len(rdf)],
                 edgecolor="black", linewidth=0.5, height=0.5, alpha=0.85)
        ax3.errorbar(rdf["OR"].values, y,
                     xerr=[rdf["OR"].values-rdf["OR_CI_lower"].values,
                           rdf["OR_CI_upper"].values-rdf["OR"].values],
                     fmt="none", ecolor="black", capsize=4)
        ax3.axvline(1, color="gray", linestyle="--")
        ax3.set_yticks(y)
        ax3.set_yticklabels([f"{r['Model']}\nOR={r['OR']:.3f} p={r['P_value']:.1e}"
                             for _,r in rdf.iterrows()], fontsize=8)
        ax3.set_xlabel("Odds Ratio")
        ax3.set_title("C. CST_Load → mRS", fontweight="bold")

    # Panel D: NT ± CST 对比
    ax4 = fig.add_subplot(2, 1, 2)
    if compare_results:
        cdf_plot = pd.DataFrame(compare_results)
        nt_order = cdf_plot[cdf_plot["Control"]=="无CST"].sort_values("P")["NT"].values
        y = np.arange(len(nt_order)); w = 0.35
        for ctrl, off, col, lab in [("无CST",-w/2,"#4DBBD5","Without CST"),
                                     ("+CST",+w/2,"#E64B35","With CST")]:
            sc = cdf_plot[cdf_plot["Control"]==ctrl].set_index("NT")
            ors = [sc.loc[nt,"OR"] if nt in sc.index else 1.0 for nt in nt_order]
            ax4.barh(y+off, np.array(ors)-1, w, left=1, color=col,
                     edgecolor="black", linewidth=0.3, alpha=0.85, label=lab)
            for i,nt in enumerate(nt_order):
                if nt in sc.index:
                    star = sig_stars(sc.loc[nt,"P"])
                    if star: ax4.text(sc.loc[nt,"OR"]+0.003, i+off, star, va="center", fontsize=7, fontweight="bold")
        ax4.axvline(1, color="gray", linestyle="--")
        ax4.set_yticks(y); ax4.set_yticklabels(nt_order, fontsize=8)
        ax4.set_xlabel("Odds Ratio (per SD)")
        ax4.set_title("D. NT Effect ± CST Control\n(mRS ~ Resid_NT + TLV + NIHSS + Age + Sex ± CST)", fontweight="bold")
        ax4.legend(fontsize=9, loc="lower right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "CST_SUMMARY.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  📊 {out_dir / 'CST_SUMMARY.png'}")

    # 总表图
    if compare_results:
        fig2, ax_t = plt.subplots(figsize=(18, max(4, 0.5*len(nt_order)+3)))
        ax_t.axis("off")
        rows = []
        for nt in nt_order:
            r0 = cdf[(cdf["NT"]==nt)&(cdf["Control"]=="无CST")]
            r1 = cdf[(cdf["NT"]==nt)&(cdf["Control"]=="+CST")]
            if r0.empty or r1.empty: continue
            o0,p0 = r0.iloc[0]["OR"], r0.iloc[0]["P"]
            o1,p1 = r1.iloc[0]["OR"], r1.iloc[0]["P"]
            dpct = (o1-o0)/(o0-1)*100 if abs(o0-1)>0.001 else 0
            tag = "依然显著 ✅" if p1<0.05 else "不显著 ❌"
            rows.append([nt, f"{o0:.3f}", f"{p0:.2e} {sig_stars(p0)}",
                         f"{o1:.3f}", f"{p1:.2e} {sig_stars(p1)}",
                         f"{dpct:+.1f}%", tag])
        cols = ["NT", "OR (无CST)", "P", "OR (+CST)", "P", "ΔOR%", "独立性"]
        t = ax_t.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
        t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1.0, 2.0)
        for j in range(len(cols)):
            t[0,j].set_facecolor("#2C3E50")
            t[0,j].set_text_props(color="white", fontweight="bold")
        for i in range(1, len(rows)+1):
            for j in range(len(cols)):
                t[i,j].set_facecolor("#F7F9FC" if i%2==0 else "#FFFFFF")
                t[i,j].set_edgecolor("#DEE2E6")
        ax_t.set_title("CST Control: Does Neurochemistry Survive Tract Disconnection?\n"
                       f"(Generated {datetime.now().strftime('%Y-%m-%d %H:%M')})",
                       fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()
        fig2.savefig(out_dir / "CST_TABLE.png", dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig2)
        print(f"  📊 {out_dir / 'CST_TABLE.png'}")

    print(f"\n{'=' * 65}\n  ✅ 完成! 截图: CST_SUMMARY.png + CST_TABLE.png\n{'=' * 65}")

if __name__ == "__main__":
    main()
