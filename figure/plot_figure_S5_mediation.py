#!/usr/bin/env python3
"""
Figure 7: Brain-Heart-Immune Axis Mediation Analysis
======================================================
A. 中介路径图: Cholinergic Tract → IL-6 → mRS (含 path a/b/c'/ab 系数)
B. 17 NT 的间接效应 (ab) 森林图
C. 中介比例条形图 (% mediated)

输出: /data/usersdir/liuzhengxin/Stepbystep/7.figure/figure7/
"""
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_ROOT = Path("/data/usersdir/liuzhengxin/Stepbystep/7.figure/figure7")

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

def bootstrap_mediation(x, m, y, covars_mat=None, n_boot=5000, seed=42):
    """Bootstrap 中介分析: X→M→Y"""
    import statsmodels.api as sm
    np.random.seed(seed)
    n = len(y)
    
    # 观测值
    if covars_mat is not None:
        Xa = sm.add_constant(np.column_stack([x, covars_mat]))
        Xb = sm.add_constant(np.column_stack([x, m, covars_mat]))
    else:
        Xa = sm.add_constant(x)
        Xb = sm.add_constant(np.column_stack([x, m]))
    
    try:
        res_a = sm.OLS(m, Xa).fit()
        res_b = sm.OLS(y, Xb).fit()
        a_obs = res_a.params[1]  # X→M
        b_obs = res_b.params[2]  # M→Y|X
        c_prime = res_b.params[1]  # X→Y|M (direct)
        ab_obs = a_obs * b_obs
    except:
        return None
    
    # Total effect
    if covars_mat is not None:
        Xc = sm.add_constant(np.column_stack([x, covars_mat]))
    else:
        Xc = sm.add_constant(x)
    try:
        c_total = sm.OLS(y, Xc).fit().params[1]
    except:
        c_total = c_prime + ab_obs
    
    # Bootstrap
    ab_boots = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        xb, mb, yb = x[idx], m[idx], y[idx]
        if covars_mat is not None:
            Xa_b = sm.add_constant(np.column_stack([xb, covars_mat[idx]]))
            Xb_b = sm.add_constant(np.column_stack([xb, mb, covars_mat[idx]]))
        else:
            Xa_b = sm.add_constant(xb)
            Xb_b = sm.add_constant(np.column_stack([xb, mb]))
        try:
            a_b = sm.OLS(mb, Xa_b).fit().params[1]
            b_b = sm.OLS(yb, Xb_b).fit().params[2]
            ab_boots.append(a_b * b_b)
        except:
            pass
    
    if len(ab_boots) < 100:
        return None
    
    ab_arr = np.array(ab_boots)
    ci_lo, ci_hi = np.percentile(ab_arr, [2.5, 97.5])
    # P: proportion of bootstrap crossing zero
    if ab_obs >= 0:
        p_val = np.mean(ab_arr <= 0) * 2
    else:
        p_val = np.mean(ab_arr >= 0) * 2
    p_val = min(p_val, 1.0)
    
    pct_mediated = abs(ab_obs / c_total) * 100 if abs(c_total) > 1e-10 else 0
    
    return {
        "a": a_obs, "b": b_obs, "c_prime": c_prime, "c_total": c_total,
        "ab": ab_obs, "ci_lo": ci_lo, "ci_hi": ci_hi, "p": p_val,
        "pct_mediated": pct_mediated, "n_boot_valid": len(ab_boots),
    }

def main():
    data_path = ("/data/usersdir/liuzhengxin/Stepbystep/"
                 "6.NeurotransmitterMapping/3.variable_outcom_merge_data/merged_neuro_data.csv")
    if not os.path.exists(data_path):
        print(f"⚠️ 未找到: {data_path}"); return

    import pandas as pd
    import statsmodels.api as sm

    df = pd.read_csv(data_path, low_memory=False)
    print(f"数据: {df.shape}")

    mrs_col = find_col(df, ["m12_mRS","D_MRS","m3_mRS","mRS"])
    il6_col = find_col(df, ["BSL_IL6","IL6"])
    rmssd_col = find_col(df, ["HOLTER_RMSSD","RMSSD"])
    if not mrs_col: print("⚠️ 无 mRS"); return
    if not il6_col: print("⚠️ 无 IL-6"); return
    print(f"  mRS: {mrs_col}, IL-6: {il6_col}, RMSSD: {rmssd_col}")

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

    # 协变量
    covars = []
    for cands in [["TLV"],["A_NIHSS","BSL_NIHSS","NIHSS"],["AGE","Age"],["SEX","Sex"]]:
        c = find_col(df, cands)
        if c: covars.append(c)
    hrmean_col = find_col(df, ["HOLTER_HRmean","HRmean","HR_mean"])
    if hrmean_col: covars.append(hrmean_col)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # 跑所有 NT 的中介分析 (IL-6 pathway)
    # ================================================================
    print(f"\n跑 {len(nt_ordered)} NT 的中介分析 (5000 bootstrap)...")
    results_il6 = []
    results_rmssd = []

    for nc in nt_ordered:
        bare = nc.replace(pfx, "")
        label = NT_RENAME.get(bare, bare)
        color = '#888'
        for sn, nts in NT_SYSTEMS.items():
            if bare in nts: color = SYS_COLORS[sn]; break

        # IL-6 pathway
        needed = [mrs_col, nc, il6_col] + covars
        needed = [c for c in needed if c in df.columns]
        sub = df[needed].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) >= 100:
            # log1p IL-6
            sub[il6_col] = np.log1p(sub[il6_col].clip(lower=0))
            # Z-score
            for c in needed:
                if c.upper() not in ("SEX","GENDER"):
                    s = sub[c].std()
                    if s > 1e-10: sub[c] = (sub[c]-sub[c].mean())/s
            
            cov_mat = sub[[c for c in covars if c in sub.columns]].values if covars else None
            res = bootstrap_mediation(sub[nc].values, sub[il6_col].values,
                                       sub[mrs_col].astype(float).values,
                                       cov_mat, n_boot=5000)
            if res:
                res["NT"] = label; res["bare"] = bare; res["color"] = color
                res["mediator"] = "IL-6"; res["N"] = len(sub)
                results_il6.append(res)
                sig = '*' if res['p'] < 0.05 else ''
                print(f"  {label} → IL-6 → mRS: ab={res['ab']:.4f} [{res['ci_lo']:.4f},{res['ci_hi']:.4f}] P={res['p']:.3f}{sig} ({res['pct_mediated']:.1f}%)")

        # RMSSD pathway (if available)
        if rmssd_col:
            needed2 = [mrs_col, nc, rmssd_col] + covars
            needed2 = [c for c in needed2 if c in df.columns]
            sub2 = df[needed2].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub2) >= 100:
                sub2[rmssd_col] = np.log1p(sub2[rmssd_col].clip(lower=0))
                for c in needed2:
                    if c.upper() not in ("SEX","GENDER"):
                        s = sub2[c].std()
                        if s > 1e-10: sub2[c] = (sub2[c]-sub2[c].mean())/s
                cov_mat2 = sub2[[c for c in covars if c in sub2.columns]].values if covars else None
                res2 = bootstrap_mediation(sub2[nc].values, sub2[rmssd_col].values,
                                            sub2[mrs_col].astype(float).values,
                                            cov_mat2, n_boot=5000)
                if res2:
                    res2["NT"] = label; res2["bare"] = bare; res2["color"] = color
                    res2["mediator"] = "RMSSD"; res2["N"] = len(sub2)
                    results_rmssd.append(res2)

    # ================================================================
    # Figure 7A: 中介路径图 (最显著的 NT)
    # ================================================================
    if results_il6:
        best = min(results_il6, key=lambda x: x['p'])
        print(f"\n[7A] 路径图: {best['NT']} (P={best['p']:.4f})")

        fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')

        # 三个框
        box_style = dict(boxstyle='round,pad=0.6', linewidth=2, alpha=0.95)
        ax.text(0.12, 0.5, best['NT'], ha='center', va='center', fontsize=14,
                fontweight='bold', color='white',
                bbox=dict(facecolor=best['color'], edgecolor='black', **box_style),
                transform=ax.transAxes)
        ax.text(0.50, 0.85, 'log(IL-6)', ha='center', va='center', fontsize=14,
                fontweight='bold', color='white',
                bbox=dict(facecolor='#E64B35', edgecolor='black', **box_style),
                transform=ax.transAxes)
        ax.text(0.88, 0.5, f'mRS\n({mrs_col})', ha='center', va='center', fontsize=14,
                fontweight='bold', color='white',
                bbox=dict(facecolor='#333333', edgecolor='black', **box_style),
                transform=ax.transAxes)

        # Path a: X→M
        a_sig = '***' if abs(best['a']) > 0 and best['p'] < 0.05 else ''
        ax.annotate('', xy=(0.38, 0.82), xytext=(0.20, 0.58),
                   xycoords='axes fraction', textcoords='axes fraction',
                   arrowprops=dict(arrowstyle='->', color=best['color'], lw=3))
        ax.text(0.24, 0.73, f"a = {best['a']:.3f}", fontsize=11, color=best['color'],
                fontweight='bold', transform=ax.transAxes)

        # Path b: M→Y
        ax.annotate('', xy=(0.80, 0.58), xytext=(0.62, 0.82),
                   xycoords='axes fraction', textcoords='axes fraction',
                   arrowprops=dict(arrowstyle='->', color='#E64B35', lw=3))
        ax.text(0.72, 0.73, f"b = {best['b']:.3f}", fontsize=11, color='#E64B35',
                fontweight='bold', transform=ax.transAxes)

        # Path c': X→Y (direct, dashed)
        ax.annotate('', xy=(0.80, 0.5), xytext=(0.20, 0.5),
                   xycoords='axes fraction', textcoords='axes fraction',
                   arrowprops=dict(arrowstyle='->', color='gray', lw=2, linestyle='dashed'))
        ax.text(0.50, 0.42, f"c' = {best['c_prime']:.3f}", fontsize=11, color='gray',
                ha='center', transform=ax.transAxes)

        # 间接效应框
        sig_str = '***' if best['p']<0.001 else ('**' if best['p']<0.01 else ('*' if best['p']<0.05 else 'ns'))
        ax.text(0.50, 0.12,
                f"Indirect effect (a×b) = {best['ab']:.4f}\n"
                f"95% CI [{best['ci_lo']:.4f}, {best['ci_hi']:.4f}]\n"
                f"P = {best['p']:.3f} {sig_str}   |   {best['pct_mediated']:.1f}% mediated\n"
                f"N = {best['N']}, Bootstrap = 5,000",
                ha='center', va='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9C4', edgecolor='#FFC107', linewidth=1.5),
                transform=ax.transAxes)

        # 协变量标注
        ax.text(0.50, 0.01, f"Adjusted for: TLV, NIHSS, Age, Sex" + (", HRmean" if hrmean_col else ""),
                ha='center', fontsize=8, color='#888', transform=ax.transAxes)

        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
        ax.set_title('Figure 7A. Brain-Immune Mediation Pathway',
                     fontsize=13, fontweight='bold', pad=10)

        plt.tight_layout()
        fig.savefig(OUTPUT_ROOT / 'figure7a_mediation_pathway.png', dpi=300,
                    bbox_inches='tight', facecolor='white')
        fig.savefig(OUTPUT_ROOT / 'figure7a_mediation_pathway.pdf',
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  ✅ figure7a_mediation_pathway.png/.pdf")

    # ================================================================
    # Figure 7B: 间接效应森林图 (所有 NT)
    # ================================================================
    if results_il6:
        print(f"\n[7B] 间接效应森林图...")
        results_il6.sort(key=lambda x: abs(x['ab']), reverse=True)

        n = len(results_il6)
        fig, ax = plt.subplots(figsize=(8, max(5, n*0.45)), facecolor='white')
        y_pos = np.arange(n)[::-1]

        for i, r in enumerate(results_il6):
            y = y_pos[i]
            color = r['color']
            sig = r['p'] < 0.05

            # 点 + CI
            marker = 's' if sig else 'o'
            ms = 8 if sig else 5
            ax.plot(r['ab'], y, marker, color=color, markersize=ms,
                    markeredgecolor='black' if sig else color, markeredgewidth=1.5 if sig else 0.5,
                    zorder=5)
            ax.plot([r['ci_lo'], r['ci_hi']], [y, y], '-', color=color,
                    linewidth=2.5 if sig else 1, alpha=0.9 if sig else 0.4, zorder=4)

            # 标注
            stars = '*' if r['p'] < 0.05 else ''
            ax.text(max(r['ci_hi'], r['ab']) + 0.0003, y,
                    f" {r['ab']:.4f}{stars}", fontsize=8, va='center',
                    fontweight='bold' if sig else 'normal', color='black')

        ax.axvline(x=0, color='black', linewidth=1, zorder=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([r['NT'] for r in results_il6], fontsize=10)
        for i, r in enumerate(results_il6):
            ax.get_yticklabels()[i].set_color(r['color'])
            ax.get_yticklabels()[i].set_fontweight('bold')

        # 灰色条纹
        for i in range(0, n, 2):
            ax.axhspan(y_pos[i]-0.4, y_pos[i]+0.4, color='#F5F5F5', zorder=0)

        ax.set_xlabel('Indirect Effect (a × b)\nNT → log(IL-6) → mRS', fontsize=12, fontweight='bold')
        ax.set_title('Figure 7B. IL-6 Mediation: All Neurotransmitter Systems\n'
                     '■ = P < 0.05  |  ○ = ns  |  Bootstrap 5,000',
                     fontsize=11, fontweight='bold', pad=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', alpha=0.15)

        plt.tight_layout()
        fig.savefig(OUTPUT_ROOT / 'figure7b_indirect_effect_forest.png', dpi=300,
                    bbox_inches='tight', facecolor='white')
        fig.savefig(OUTPUT_ROOT / 'figure7b_indirect_effect_forest.pdf',
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  ✅ figure7b_indirect_effect_forest.png/.pdf")

    # ================================================================
    # Figure 7C: IL-6 vs RMSSD 双通路对比 (如有 RMSSD)
    # ================================================================
    if results_il6 and results_rmssd:
        print(f"\n[7C] IL-6 vs RMSSD 双通路对比...")

        # 找同时有两条通路结果的 NT
        il6_dict = {r['bare']: r for r in results_il6}
        rmssd_dict = {r['bare']: r for r in results_rmssd}
        shared = sorted(set(il6_dict.keys()) & set(rmssd_dict.keys()))

        if shared:
            fig, ax = plt.subplots(figsize=(8, max(4, len(shared)*0.5)), facecolor='white')
            y_pos = np.arange(len(shared))[::-1]

            for i, bare in enumerate(shared):
                y = y_pos[i]
                r_il6 = il6_dict[bare]
                r_rmssd = rmssd_dict[bare]

                # IL-6 (红)
                ax.plot(abs(r_il6['ab']), y+0.12, 's', color='#E64B35', markersize=8,
                        markeredgecolor='black', markeredgewidth=1, zorder=5)
                ax.plot([abs(r_il6['ci_lo']), abs(r_il6['ci_hi'])], [y+0.12, y+0.12],
                        '-', color='#E64B35', linewidth=2, zorder=4)

                # RMSSD (蓝)
                ax.plot(abs(r_rmssd['ab']), y-0.12, 'o', color='#4DBBD5', markersize=8,
                        markeredgecolor='black', markeredgewidth=1, zorder=5)
                ax.plot([abs(r_rmssd['ci_lo']), abs(r_rmssd['ci_hi'])], [y-0.12, y-0.12],
                        '-', color='#4DBBD5', linewidth=2, zorder=4)

            ax.set_yticks(y_pos)
            labels = [NT_RENAME.get(b, b) for b in shared]
            ax.set_yticklabels(labels, fontsize=10, fontweight='bold')
            ax.set_xlabel('|Indirect Effect|', fontsize=12, fontweight='bold')
            ax.set_title('Figure 7C. Dual Pathway: IL-6 vs RMSSD Mediation\n'
                         '■ IL-6 (inflammatory)  |  ● RMSSD (autonomic)',
                         fontsize=11, fontweight='bold')
            ax.legend([plt.Line2D([0],[0],marker='s',color='#E64B35',ls='None',ms=8),
                       plt.Line2D([0],[0],marker='o',color='#4DBBD5',ls='None',ms=8)],
                      ['IL-6 pathway','RMSSD pathway'], fontsize=9, loc='lower right')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='x', alpha=0.15)
            plt.tight_layout()
            fig.savefig(OUTPUT_ROOT / 'figure7c_dual_pathway.png', dpi=300,
                        bbox_inches='tight', facecolor='white')
            fig.savefig(OUTPUT_ROOT / 'figure7c_dual_pathway.pdf',
                        bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"  ✅ figure7c_dual_pathway.png/.pdf")

    # 保存结果表
    if results_il6:
        rdf = pd.DataFrame(results_il6)
        rdf.to_csv(OUTPUT_ROOT / 'figure7_mediation_IL6_results.csv', index=False)
        print(f"  ✅ figure7_mediation_IL6_results.csv")
    if results_rmssd:
        rdf2 = pd.DataFrame(results_rmssd)
        rdf2.to_csv(OUTPUT_ROOT / 'figure7_mediation_RMSSD_results.csv', index=False)
        print(f"  ✅ figure7_mediation_RMSSD_results.csv")

    print(f"\n{'='*50}")
    print(f"  ✅ Figure 7 完成 → {OUTPUT_ROOT}")
    print(f"{'='*50}")

if __name__ == "__main__":
    print("="*50)
    print("  Figure 7: Brain-Heart-Immune Mediation")
    print("="*50)
    main()
