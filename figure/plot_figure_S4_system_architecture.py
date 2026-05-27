#!/usr/bin/env python3
"""
Figure 6: System-Level Architecture & Dose-Response
=====================================================
A. 系统水平 PCA 聚合 — 5 大系统的 OR 条形图
B. 突触前(Transporter) vs 突触后(Receptor) 效应量对比
C. 剂量-效应：Top NT 四分位 mRS 分布

输出: /data/usersdir/liuzhengxin/Stepbystep/7.figure/figure6/
"""
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

OUTPUT_ROOT = Path("/data/usersdir/liuzhengxin/Stepbystep/7.figure/figure6")

NT_SYSTEMS = {
    "Serotonergic":      ["5HT1a","5HT1b","5HT2a","5HT4","5HT6","5HTT"],
    "Cholinergic":       ["A4B2","M1","VAChT","human_CHA"],
    "Catecholaminergic": ["D1","D2","DAT","NAT"],
    "Chol. Tract":       ["JHU_EC","Lateral_Path","Medial_Path"],
}
NT_RENAME = {
    "5HT1a":"5-HT₁ₐ","5HT1b":"5-HT₁ᵦ","5HT2a":"5-HT₂ₐ",
    "5HT4":"5-HT₄","5HT6":"5-HT₆","5HTT":"SERT",
    "A4B2":"α4β2","M1":"M₁","VAChT":"VAChT","human_CHA":"AChE",
    "D1":"D₁","D2":"D₂","DAT":"DAT","NAT":"NAT",
    "JHU_EC":"EC Tract","Lateral_Path":"Lat. Path","Medial_Path":"Med. Path",
}
SYS_COLORS = {
    "Serotonergic":"#F39B7F","Cholinergic":"#4DBBD5",
    "Catecholaminergic":"#E64B35","Chol. Tract":"#00A087",
}

# 突触分类
PRE_SYNAPTIC = ["DAT","NAT","5HTT","VAChT"]  # transporters
POST_SYNAPTIC = ["A4B2","M1","5HT1a","5HT1b","5HT2a","5HT4","5HT6","D1","D2"]  # receptors

def find_col(df, cands):
    for c in cands:
        if c in df.columns: return c
    return None

def main():
    data_path = ("/data/usersdir/liuzhengxin/Stepbystep/"
                 "6.NeurotransmitterMapping/3.variable_outcom_merge_data/merged_neuro_data.csv")
    if not os.path.exists(data_path):
        print(f"⚠️ 未找到: {data_path}"); return

    import pandas as pd
    import statsmodels.api as sm
    from sklearn.decomposition import PCA

    df = pd.read_csv(data_path, low_memory=False)
    print(f"数据: {df.shape}")

    mrs_col = find_col(df, ["D_MRS","d_mrs","mRS"])
    if not mrs_col: print("⚠️ 无 mRS"); return

    # NT 列
    resid_cols = [c for c in df.columns if c.startswith("Resid_")]
    pfx = "Resid_"
    if not resid_cols:
        resid_cols = [c for c in df.columns if c.startswith("Load_")]; pfx = "Load_"
    if not resid_cols:
        all_nt = [nt for nts in NT_SYSTEMS.values() for nt in nts]
        resid_cols = [c for c in all_nt if c in df.columns]; pfx = ""
    if not resid_cols: print("⚠️ 无NT列"); return

    # 协变量
    covars = []
    for cands in [["TLV"],["A_NIHSS","BSL_NIHSS","NIHSS"],["AGE","Age"],["SEX","Sex"]]:
        c = find_col(df, cands)
        if c: covars.append(c)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Figure 6A: 系统水平 PCA → OR 条形图
    # ================================================================
    print("\n[6A] System-Level PCA...")

    # 只用受体/转运体系统（不含 Chol. Tract）
    pca_systems = {
        "Serotonergic": ["5HT1a","5HT1b","5HT2a","5HT4","5HT6","5HTT"],
        "Cholinergic":  ["A4B2","M1","VAChT","human_CHA"],
        "Dopaminergic": ["D1","D2","DAT"],
        "Noradrenergic":["NAT"],
    }
    pca_colors = {
        "Serotonergic":"#F39B7F","Cholinergic":"#4DBBD5",
        "Dopaminergic":"#8491B4","Noradrenergic":"#E64B35",
    }

    sys_results = []
    for sys_name, nts in pca_systems.items():
        nt_cols = [f"{pfx}{nt}" for nt in nts if f"{pfx}{nt}" in resid_cols]
        if len(nt_cols) < 2:
            # 单变量系统直接用原始列
            if len(nt_cols) == 1:
                pc1 = pd.to_numeric(df[nt_cols[0]], errors="coerce")
            else:
                continue
        else:
            sub_nt = df[nt_cols].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub_nt) < 100: continue
            # Z-score
            for c in nt_cols:
                s = sub_nt[c].std()
                if s > 1e-10: sub_nt[c] = (sub_nt[c] - sub_nt[c].mean()) / s
            pca = PCA(n_components=1)
            pc1_arr = pca.fit_transform(sub_nt)[:, 0]
            # 对齐方向：确保 PC1 与总 load 正相关
            total = sub_nt.sum(axis=1)
            if np.corrcoef(pc1_arr, total.values)[0, 1] < 0:
                pc1_arr = -pc1_arr
            pc1 = pd.Series(pc1_arr, index=sub_nt.index)
            var_explained = pca.explained_variance_ratio_[0]
            print(f"  {sys_name}: PC1 var={var_explained:.1%}, n={len(sub_nt)}")

        # 回归 PC1 → mRS
        needed = [mrs_col] + covars
        reg_df = df[needed].copy()
        reg_df["PC1"] = pc1
        reg_df = reg_df.apply(pd.to_numeric, errors="coerce").dropna()
        if len(reg_df) < 100: continue

        # Z-score
        for c in ["PC1"] + covars:
            if c.upper() not in ("SEX","GENDER"):
                s = reg_df[c].std()
                if s > 1e-10: reg_df[c] = (reg_df[c] - reg_df[c].mean()) / s

        X = sm.add_constant(reg_df[["PC1"] + covars])
        try:
            res = sm.OLS(reg_df[mrs_col].astype(float), X).fit()
            coef = res.params["PC1"]
            p_val = res.pvalues["PC1"]
            se = res.bse["PC1"]
            or_val = np.exp(coef)
            ci_lo = np.exp(coef - 1.96 * se)
            ci_hi = np.exp(coef + 1.96 * se)
            sys_results.append({
                "System": sys_name, "OR": or_val,
                "CI_lo": ci_lo, "CI_hi": ci_hi,
                "P": p_val, "N": len(reg_df),
                "color": pca_colors.get(sys_name, '#888'),
            })
            print(f"    OR={or_val:.3f} [{ci_lo:.3f}-{ci_hi:.3f}], P={p_val:.2e}")
        except Exception as e:
            print(f"    ⚠️ {sys_name}: {e}")

    if sys_results:
        sys_results.sort(key=lambda x: x["OR"], reverse=True)

        fig, ax = plt.subplots(figsize=(7, 4), facecolor='white')
        names = [r["System"] for r in sys_results]
        ors = [r["OR"] for r in sys_results]
        ci_los = [r["CI_lo"] for r in sys_results]
        ci_his = [r["CI_hi"] for r in sys_results]
        colors = [r["color"] for r in sys_results]
        x_pos = np.arange(len(names))

        bars = ax.bar(x_pos, ors, color=colors, edgecolor='black', linewidth=0.8, alpha=0.9)
        ax.errorbar(x_pos, ors,
                    yerr=[[o - lo for o, lo in zip(ors, ci_los)],
                          [hi - o for o, hi in zip(ors, ci_his)]],
                    fmt='none', ecolor='black', capsize=6, linewidth=1.5)

        for i, r in enumerate(sys_results):
            stars = '***' if r["P"]<0.001 else ('**' if r["P"]<0.01 else ('*' if r["P"]<0.05 else 'ns'))
            ax.text(i, r["CI_hi"] + 0.003, f'{r["OR"]:.3f}{stars}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.axhline(y=1.0, color='gray', linewidth=1, linestyle='--', alpha=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, fontsize=11, fontweight='bold')
        for i, c in enumerate(colors):
            ax.get_xticklabels()[i].set_color(c)
        ax.set_ylabel('OR per 1-SD PC1 increase', fontsize=12, fontweight='bold')
        ax.set_title('Figure 6A. System-Level Vulnerability (PCA PC1 → mRS)\n'
                     'Adjusted for TLV, NIHSS, Age, Sex',
                     fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.15)

        plt.tight_layout()
        fig.savefig(OUTPUT_ROOT / 'figure6a_system_PCA.png', dpi=300,
                    bbox_inches='tight', facecolor='white')
        fig.savefig(OUTPUT_ROOT / 'figure6a_system_PCA.pdf',
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  ✅ figure6a_system_PCA.png/.pdf")

    # ================================================================
    # Figure 6B: Pre-synaptic vs Post-synaptic 效应量
    # ================================================================
    print("\n[6B] Pre vs Post-synaptic...")

    pre_betas, post_betas = [], []

    for nc in resid_cols:
        bare = nc.replace(pfx, "")
        needed = [mrs_col, nc] + covars
        needed = [c for c in needed if c in df.columns]
        sub = df[needed].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) < 100: continue
        for c in needed:
            if c.upper() not in ("SEX","GENDER"):
                s = sub[c].std()
                if s > 1e-10: sub[c] = (sub[c] - sub[c].mean()) / s
        X = sm.add_constant(sub[[nc] + [c for c in covars if c in sub.columns]])
        try:
            res = sm.OLS(sub[mrs_col].astype(float), X).fit()
            beta_abs = abs(res.params[nc])
            if bare in PRE_SYNAPTIC:
                pre_betas.append(beta_abs)
            elif bare in POST_SYNAPTIC:
                post_betas.append(beta_abs)
        except: pass

    if pre_betas and post_betas:
        u_stat, p_mw = stats.mannwhitneyu(post_betas, pre_betas, alternative='greater')
        print(f"  Pre-synaptic |β|: {np.mean(pre_betas):.4f} ± {np.std(pre_betas):.4f} (n={len(pre_betas)})")
        print(f"  Post-synaptic |β|: {np.mean(post_betas):.4f} ± {np.std(post_betas):.4f} (n={len(post_betas)})")
        print(f"  Mann-Whitney P={p_mw:.4f}")

        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        bp = ax.boxplot([pre_betas, post_betas],
                       labels=['Pre-synaptic\n(Transporters)', 'Post-synaptic\n(Receptors)'],
                       patch_artist=True, widths=0.5,
                       medianprops=dict(color='black', linewidth=2))
        bp['boxes'][0].set_facecolor('#8491B4'); bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor('#E64B35'); bp['boxes'][1].set_alpha(0.7)

        # 散点
        np.random.seed(42)
        ax.scatter(np.random.normal(1, 0.06, len(pre_betas)), pre_betas,
                  alpha=0.6, s=40, color='#8491B4', edgecolors='black', linewidth=0.5, zorder=5)
        ax.scatter(np.random.normal(2, 0.06, len(post_betas)), post_betas,
                  alpha=0.6, s=40, color='#E64B35', edgecolors='black', linewidth=0.5, zorder=5)

        # 显著性标注
        y_max = max(max(pre_betas), max(post_betas))
        stars = '***' if p_mw < 0.001 else ('**' if p_mw < 0.01 else ('*' if p_mw < 0.05 else 'ns'))
        ax.plot([1, 1, 2, 2], [y_max+0.002, y_max+0.004, y_max+0.004, y_max+0.002],
               color='black', linewidth=1.5)
        ax.text(1.5, y_max+0.005, f'{stars}\nP={p_mw:.3f}', ha='center', fontsize=10, fontweight='bold')

        ax.set_ylabel('|Standardized β|', fontsize=12, fontweight='bold')
        ax.set_title('Figure 6B. Pre- vs Post-synaptic Effect Sizes\n'
                     'Post-synaptic receptor damage > Transporter damage',
                     fontsize=10, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        fig.savefig(OUTPUT_ROOT / 'figure6b_pre_vs_post_synaptic.png', dpi=300,
                    bbox_inches='tight', facecolor='white')
        fig.savefig(OUTPUT_ROOT / 'figure6b_pre_vs_post_synaptic.pdf',
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  ✅ figure6b_pre_vs_post_synaptic.png/.pdf")

    # ================================================================
    # Figure 6C: 剂量-效应 (Top-3 NT 四分位 mRS 分布)
    # ================================================================
    print("\n[6C] Dose-Response (Quartile mRS)...")
    top3_nt = ["NAT","A4B2","5HT6"]
    top3_cols = [f"{pfx}{nt}" for nt in top3_nt if f"{pfx}{nt}" in resid_cols]
    # 回退到 Load_ 或裸名
    if not top3_cols:
        top3_cols = [f"Load_{nt}" for nt in top3_nt if f"Load_{nt}" in df.columns]
    if not top3_cols:
        top3_cols = [nt for nt in top3_nt if nt in df.columns]

    if top3_cols:
        n_plots = len(top3_cols)
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5), facecolor='white')
        if n_plots == 1: axes = [axes]

        for ax, nc in zip(axes, top3_cols):
            bare = nc.replace(pfx, "").replace("Load_", "")
            label = NT_RENAME.get(bare, bare)

            sub = df[[mrs_col, nc]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < 100: continue

            # 四分位
            sub["Quartile"] = pd.qcut(sub[nc], 4, labels=["Q1","Q2","Q3","Q4"])

            # 每个四分位的 mRS 均值
            q_means = sub.groupby("Quartile", observed=True)[mrs_col].mean()
            q_sems = sub.groupby("Quartile", observed=True)[mrs_col].sem()
            q_ns = sub.groupby("Quartile", observed=True)[mrs_col].count()

            colors_q = ['#4DBBD5','#F1C40F','#E67E22','#E74C3C']
            x = np.arange(4)
            bars = ax.bar(x, q_means.values, yerr=q_sems.values,
                         color=colors_q, edgecolor='black', linewidth=0.8,
                         alpha=0.9, capsize=6)

            for i, (m, n) in enumerate(zip(q_means.values, q_ns.values)):
                ax.text(i, m + q_sems.values[i] + 0.05, f'{m:.2f}\n(n={n})',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')

            # Kruskal-Wallis
            groups = [g[mrs_col].values for _, g in sub.groupby("Quartile", observed=True)]
            h_stat, kw_p = stats.kruskal(*groups)
            # Spearman trend
            rho, sp_p = stats.spearmanr(sub[nc], sub[mrs_col])

            ax.set_xticks(x)
            ax.set_xticklabels(["Q1\n(lowest)","Q2","Q3","Q4\n(highest)"], fontsize=9)
            ax.set_ylabel('Mean mRS at discharge', fontsize=11, fontweight='bold')
            ax.set_title(f'{label}\nKW P={kw_p:.2e}, ρ={rho:.3f}',
                        fontsize=10, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.15)

        plt.suptitle('Figure 6C. Dose-Response: NT Load Quartiles → mRS',
                     fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        fig.savefig(OUTPUT_ROOT / 'figure6c_dose_response.png', dpi=300,
                    bbox_inches='tight', facecolor='white')
        fig.savefig(OUTPUT_ROOT / 'figure6c_dose_response.pdf',
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  ✅ figure6c_dose_response.png/.pdf")

    print(f"\n{'='*50}")
    print(f"  ✅ Figure 6 完成 → {OUTPUT_ROOT}")
    print(f"{'='*50}")

if __name__ == "__main__":
    print("="*50)
    print("  Figure 6: System Architecture & Dose-Response")
    print("="*50)
    main()
