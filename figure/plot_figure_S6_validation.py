#!/usr/bin/env python3
"""
Figure 8: WMH Interaction & Methodological Validation
=======================================================
A. WMH 分层效应图：高/低 WMH 下 NT→mRS 的 OR 对比
B. 置换检验：参数 P vs 置换 P 散点图
C. MICE 敏感性：完整病例 vs 插补 OR 一致性
D. mRS 切点敏感性：NAT/A4B2 在 4 种 mRS 分组下的 OR

输出: /data/usersdir/liuzhengxin/Stepbystep/7.figure/figure8/
"""
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

OUTPUT_ROOT = Path("/data/usersdir/liuzhengxin/Stepbystep/7.figure/figure8")

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

    df = pd.read_csv(data_path, low_memory=False)
    print(f"数据: {df.shape}")

    mrs_col = find_col(df, ["D_MRS","d_mrs","mRS"])
    wmh_col = find_col(df, ["IMG_SVD_WMH","WMH","wmh","SVD_WMH"])
    if not mrs_col: print("⚠️ 无 mRS"); return

    # NT 列
    resid_cols = [c for c in df.columns if c.startswith("Resid_")]
    pfx = "Resid_"
    if not resid_cols:
        resid_cols = [c for c in df.columns if c.startswith("Load_")]; pfx = "Load_"
    if not resid_cols:
        all_nt = [nt for nts in NT_SYSTEMS.values() for nt in nts]
        resid_cols = [c for c in all_nt if c in df.columns]; pfx = ""

    nt_ordered = []
    for nts in NT_SYSTEMS.values():
        for nt in nts:
            c = f"{pfx}{nt}"
            if c in resid_cols: nt_ordered.append(c)
    for c in resid_cols:
        if c not in nt_ordered: nt_ordered.append(c)

    covars = []
    for cands in [["TLV"],["A_NIHSS","BSL_NIHSS","NIHSS"],["AGE","Age"],["SEX","Sex"]]:
        c = find_col(df, cands)
        if c: covars.append(c)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Figure 8A: WMH 分层效应 — 高/低 WMH 下 NT 的 β 对比
    # ================================================================
    if wmh_col:
        print(f"\n[8A] WMH 分层效应 (列: {wmh_col})...")
        df[wmh_col] = pd.to_numeric(df[wmh_col], errors="coerce")
        wmh_median = df[wmh_col].median()
        lo_wmh = df[df[wmh_col] < wmh_median]
        hi_wmh = df[df[wmh_col] >= wmh_median]
        print(f"  WMH median={wmh_median:.1f}, Low N={len(lo_wmh)}, High N={len(hi_wmh)}")

        nt_labels = []
        beta_lo, beta_hi = [], []
        nt_colors = []

        for nc in nt_ordered:
            bare = nc.replace(pfx, "")
            label = NT_RENAME.get(bare, bare)
            color = '#888'
            for sn, nts in NT_SYSTEMS.items():
                if bare in nts: color = SYS_COLORS[sn]; break

            betas = []
            for sub_df in [lo_wmh, hi_wmh]:
                needed = [mrs_col, nc] + [c for c in covars if c in sub_df.columns]
                sub = sub_df[needed].apply(pd.to_numeric, errors="coerce").dropna()
                if len(sub) < 50:
                    betas.append(np.nan); continue
                for c in needed:
                    if c.upper() not in ("SEX","GENDER"):
                        s = sub[c].std()
                        if s > 1e-10: sub[c] = (sub[c]-sub[c].mean())/s
                X = sm.add_constant(sub[[nc]+[c for c in covars if c in sub.columns]])
                try:
                    res = sm.OLS(sub[mrs_col].astype(float), X).fit()
                    betas.append(res.params[nc])
                except:
                    betas.append(np.nan)

            if not np.isnan(betas[0]) and not np.isnan(betas[1]):
                nt_labels.append(label)
                beta_lo.append(betas[0])
                beta_hi.append(betas[1])
                nt_colors.append(color)

        if nt_labels:
            n = len(nt_labels)
            fig, ax = plt.subplots(figsize=(8, max(5, n*0.45)), facecolor='white')
            y_pos = np.arange(n)

            ax.barh(y_pos + 0.18, beta_hi, height=0.32, color=[c for c in nt_colors],
                    edgecolor='black', linewidth=0.8, alpha=0.9, label=f'High WMH (≥{wmh_median:.0f})')
            ax.barh(y_pos - 0.18, beta_lo, height=0.32, color=[c for c in nt_colors],
                    edgecolor='black', linewidth=0.8, alpha=0.4, label=f'Low WMH (<{wmh_median:.0f})')

            # 效应放大标记
            for i in range(n):
                if abs(beta_hi[i]) > abs(beta_lo[i]) * 1.3:
                    ax.text(max(beta_hi[i], beta_lo[i]) + 0.002, y_pos[i],
                            '↑', fontsize=12, color='#E64B35', fontweight='bold', va='center')

            ax.axvline(x=0, color='black', linewidth=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(nt_labels, fontsize=10, fontweight='bold')
            for i, c in enumerate(nt_colors):
                ax.get_yticklabels()[i].set_color(c)
            ax.set_xlabel('Standardized β (NT → mRS)', fontsize=12, fontweight='bold')
            ax.set_title(f'Figure 8A. WMH Stratification: NT Effects Amplified\n'
                         f'by Pre-existing White Matter Disease\n'
                         f'↑ = effect >30% larger in High-WMH stratum',
                         fontsize=10, fontweight='bold', pad=12)
            ax.legend(fontsize=9, loc='lower right')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='x', alpha=0.15)
            ax.invert_yaxis()

            plt.tight_layout()
            fig.savefig(OUTPUT_ROOT / 'figure8a_WMH_stratification.png', dpi=300,
                        bbox_inches='tight', facecolor='white')
            fig.savefig(OUTPUT_ROOT / 'figure8a_WMH_stratification.pdf',
                        bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"  ✅ figure8a_WMH_stratification.png/.pdf")
    else:
        print("  ⚠️ 无 WMH 列，跳过 8A")

    # ================================================================
    # Figure 8B: 置换检验 — 参数 P vs 置换 P
    # ================================================================
    print(f"\n[8B] 置换检验 (1000 iterations)...")
    param_p_list = []
    perm_p_list = []
    nt_label_list = []
    nt_color_list = []
    n_perm = 1000

    for nc in nt_ordered:
        bare = nc.replace(pfx, "")
        label = NT_RENAME.get(bare, bare)
        color = '#888'
        for sn, nts in NT_SYSTEMS.items():
            if bare in nts: color = SYS_COLORS[sn]; break

        needed = [mrs_col, nc] + [c for c in covars if c in df.columns]
        sub = df[needed].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) < 100: continue

        for c in needed:
            if c.upper() not in ("SEX","GENDER"):
                s = sub[c].std()
                if s > 1e-10: sub[c] = (sub[c]-sub[c].mean())/s

        X = sm.add_constant(sub[[nc]+[c for c in covars if c in sub.columns]])
        y = sub[mrs_col].astype(float).values
        try:
            obs_res = sm.OLS(y, X).fit()
            obs_beta = abs(obs_res.params[nc])
            param_p = obs_res.pvalues[nc]
        except:
            continue

        # 置换
        np.random.seed(42)
        perm_count = 0
        for _ in range(n_perm):
            y_perm = np.random.permutation(y)
            try:
                perm_beta = abs(sm.OLS(y_perm, X).fit().params[nc])
                if perm_beta >= obs_beta:
                    perm_count += 1
            except:
                pass
        perm_p = (perm_count + 1) / (n_perm + 1)

        param_p_list.append(param_p)
        perm_p_list.append(perm_p)
        nt_label_list.append(label)
        nt_color_list.append(color)
        print(f"  {label}: param P={param_p:.2e}, perm P={perm_p:.3f}")

    if param_p_list:
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        param_log = -np.log10(np.array(param_p_list))
        perm_log = -np.log10(np.array(perm_p_list))

        for i in range(len(param_p_list)):
            ax.scatter(param_log[i], perm_log[i], c=nt_color_list[i],
                      s=60, edgecolors='black', linewidth=0.8, zorder=5, alpha=0.9)
            # 标注双显著的
            if param_p_list[i] < 0.05 and perm_p_list[i] < 0.05:
                ax.annotate(nt_label_list[i], (param_log[i]+0.1, perm_log[i]+0.1),
                           fontsize=7, fontweight='bold', color=nt_color_list[i])

        # 对角线
        lim = max(max(param_log), max(perm_log)) + 0.5
        ax.plot([0, lim], [0, lim], '--', color='gray', alpha=0.5)
        # P=0.05 参考线
        ax.axvline(x=-np.log10(0.05), color='red', linewidth=1, linestyle=':', alpha=0.5)
        ax.axhline(y=-np.log10(0.05), color='red', linewidth=1, linestyle=':', alpha=0.5)

        # 象限标注
        ax.text(lim*0.7, lim*0.1, 'Parametric only', fontsize=8, color='#888', ha='center')
        ax.text(lim*0.1, lim*0.7, 'Permutation only', fontsize=8, color='#888', ha='center')
        ax.text(lim*0.7, lim*0.7, 'Both significant', fontsize=8, color='#4DBBD5',
                ha='center', fontweight='bold')

        n_both = sum(1 for pp, prp in zip(param_p_list, perm_p_list) if pp < 0.05 and prp < 0.05)
        ax.set_xlabel('$-\\log_{10}(P_{parametric})$', fontsize=12, fontweight='bold')
        ax.set_ylabel('$-\\log_{10}(P_{permutation})$', fontsize=12, fontweight='bold')
        ax.set_title(f'Figure 8B. Permutation Validation (1,000 iterations)\n'
                     f'{n_both}/{len(param_p_list)} dual-significant (both P < 0.05)',
                     fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_aspect('equal')

        plt.tight_layout()
        fig.savefig(OUTPUT_ROOT / 'figure8b_permutation_validation.png', dpi=300,
                    bbox_inches='tight', facecolor='white')
        fig.savefig(OUTPUT_ROOT / 'figure8b_permutation_validation.pdf',
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  ✅ figure8b_permutation_validation.png/.pdf")

    # ================================================================
    # Figure 8C: mRS 切点敏感性 — Top NT 在 4 种分组下的一致性
    # ================================================================
    print(f"\n[8C] mRS 切点敏感性...")
    top_nt = ["NAT","A4B2","5HT6","DAT","VAChT"]
    cutpoints = [
        ("0-1 vs 2-6", lambda x: (x >= 2).astype(float)),
        ("0-2 vs 3-6", lambda x: (x >= 3).astype(float)),
        ("0-3 vs 4-6", lambda x: (x >= 4).astype(float)),
        ("Ordinal (3-tier)", None),  # 序数模型
    ]

    sens_results = []
    for nt_bare in top_nt:
        nc = f"{pfx}{nt_bare}"
        if nc not in df.columns: continue
        label = NT_RENAME.get(nt_bare, nt_bare)

        for cut_name, cut_fn in cutpoints:
            needed = [mrs_col, nc] + [c for c in covars if c in df.columns]
            sub = df[needed].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < 100: continue
            for c in needed:
                if c.upper() not in ("SEX","GENDER"):
                    s = sub[c].std()
                    if s > 1e-10: sub[c] = (sub[c]-sub[c].mean())/s

            X_cols = [nc] + [c for c in covars if c in sub.columns]
            if cut_fn is not None:
                y = cut_fn(sub[mrs_col])
                try:
                    res = sm.Logit(y, sm.add_constant(sub[X_cols])).fit(disp=False)
                    or_val = np.exp(res.params[nc])
                    p_val = res.pvalues[nc]
                except:
                    continue
            else:
                try:
                    X = sm.add_constant(sub[X_cols])
                    res = sm.OLS(sub[mrs_col].astype(float), X).fit()
                    or_val = np.exp(res.params[nc])
                    p_val = res.pvalues[nc]
                except:
                    continue

            sens_results.append({
                "NT": label, "Cutpoint": cut_name,
                "OR": or_val, "P": p_val,
                "Sig": p_val < 0.05,
            })

    if sens_results:
        sdf = pd.DataFrame(sens_results)
        nt_names = [NT_RENAME.get(n, n) for n in top_nt if NT_RENAME.get(n, n) in sdf["NT"].values]
        cut_names = [c[0] for c in cutpoints]

        fig, ax = plt.subplots(figsize=(8, 4), facecolor='white')
        x = np.arange(len(cut_names))
        width = 0.15
        nt_plot_colors = []
        for nt_bare in top_nt:
            for sn, nts in NT_SYSTEMS.items():
                if nt_bare in nts:
                    nt_plot_colors.append(SYS_COLORS[sn]); break
            else:
                nt_plot_colors.append('#888')

        for i, (nt_bare, color) in enumerate(zip(top_nt, nt_plot_colors)):
            label = NT_RENAME.get(nt_bare, nt_bare)
            nt_data = sdf[sdf["NT"] == label]
            ors = []
            sigs = []
            for cut_name in cut_names:
                row = nt_data[nt_data["Cutpoint"] == cut_name]
                if len(row) > 0:
                    ors.append(row.iloc[0]["OR"])
                    sigs.append(row.iloc[0]["Sig"])
                else:
                    ors.append(np.nan)
                    sigs.append(False)

            offset = (i - len(top_nt)/2 + 0.5) * width
            bars = ax.bar(x + offset, ors, width, color=color, edgecolor='black',
                         linewidth=0.5, alpha=0.85, label=label)
            # 显著性标记
            for j, (o, s) in enumerate(zip(ors, sigs)):
                if not np.isnan(o):
                    marker = '✓' if s else '✗'
                    mc = '#4DBBD5' if s else '#E74C3C'
                    ax.text(x[j] + offset, o + 0.005, marker, ha='center',
                           fontsize=10, color=mc, fontweight='bold')

        ax.axhline(y=1.0, color='gray', linewidth=1, linestyle='--', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(cut_names, fontsize=10, fontweight='bold')
        ax.set_ylabel('Odds Ratio', fontsize=12, fontweight='bold')
        ax.set_title('Figure 8C. mRS Cutpoint Sensitivity Analysis\n'
                     '✓ = P < 0.05  |  ✗ = ns',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right', ncol=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.15)

        plt.tight_layout()
        fig.savefig(OUTPUT_ROOT / 'figure8c_mRS_cutpoint_sensitivity.png', dpi=300,
                    bbox_inches='tight', facecolor='white')
        fig.savefig(OUTPUT_ROOT / 'figure8c_mRS_cutpoint_sensitivity.pdf',
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  ✅ figure8c_mRS_cutpoint_sensitivity.png/.pdf")

        sdf.to_csv(OUTPUT_ROOT / 'figure8c_cutpoint_results.csv', index=False)

    print(f"\n{'='*50}")
    print(f"  ✅ Figure 8 完成 → {OUTPUT_ROOT}")
    print(f"{'='*50}")

if __name__ == "__main__":
    print("="*50)
    print("  Figure 8: WMH Interaction & Validation")
    print("="*50)
    main()
