#!/usr/bin/env python3
"""
Figure 5: Temporal Dynamics of Neurochemical Effects
=====================================================
展示 NT 预后效应从出院→3月→6月→12月的时间轨迹
"急性主导 → 代偿平台 → 晚期再现"

输出: /data/usersdir/liuzhengxin/Stepbystep/7.figure/figureS3/
"""
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_ROOT = Path("/data/usersdir/liuzhengxin/Stepbystep/7.figure/figureS3")

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
# 高亮 top 5 NT（其余灰色）
TOP_NT = ["NAT","A4B2","5HT6","DAT","VAChT"]

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

    # mRS 各时间点
    TIME_POINTS = [
        ("Discharge", ["D_MRS","d_mrs","mRS_discharge"]),
        ("3 months",  ["m3_mRS","M03_mRS","mRS_3m"]),
        ("6 months",  ["m6_mRS","M06_mRS","mRS_6m"]),
        ("12 months", ["m12_mRS","M12_mRS","mRS_12m"]),
    ]
    mrs_cols = {}
    for label, cands in TIME_POINTS:
        c = find_col(df, cands)
        if c:
            mrs_cols[label] = c
            print(f"  {label}: {c}")
    if len(mrs_cols) < 2:
        print("⚠️ mRS 时间点不足"); return

    # NT 列
    resid_cols = [c for c in df.columns if c.startswith("Resid_")]
    pfx = "Resid_"
    if not resid_cols:
        resid_cols = [c for c in df.columns if c.startswith("Load_")]; pfx = "Load_"
    if not resid_cols:
        all_nt = [nt for nts in NT_SYSTEMS.values() for nt in nts]
        resid_cols = [c for c in all_nt if c in df.columns]; pfx = ""
    if not resid_cols: print("⚠️ 无NT列"); return

    nt_ordered = []
    for nts in NT_SYSTEMS.values():
        for nt in nts:
            c = f"{pfx}{nt}"
            if c in resid_cols: nt_ordered.append(c)
    for c in resid_cols:
        if c not in nt_ordered: nt_ordered.append(c)

    # 协变量
    covars = []
    for cands in [["TLV"],["A_NIHSS","BSL_NIHSS","NIHSS"],["AGE","Age"],["SEX","Sex"],["CST_Load"]]:
        c = find_col(df, cands)
        if c: covars.append(c)

    # 计算每个 NT × 时间点 的 -log10(P) 和 β
    time_labels = list(mrs_cols.keys())
    n_nt = len(nt_ordered)
    n_time = len(time_labels)

    neglogp_mat = np.full((n_nt, n_time), np.nan)
    beta_mat = np.full((n_nt, n_time), np.nan)
    sig_mat = np.full((n_nt, n_time), False)

    print(f"\n计算 {n_nt} NT × {n_time} 时间点...")
    for j, t_label in enumerate(time_labels):
        mrs_c = mrs_cols[t_label]
        for i, nc in enumerate(nt_ordered):
            needed = [mrs_c, nc] + [c for c in covars if c in df.columns]
            sub = df[needed].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < 50: continue
            for col in needed:
                if col.upper() in ("SEX","GENDER"): continue
                s = sub[col].std()
                if s > 1e-10: sub[col] = (sub[col]-sub[col].mean())/s
            X = sm.add_constant(sub[[nc]+[c for c in covars if c in sub.columns]])
            try:
                res = sm.OLS(sub[mrs_c].astype(float), X).fit()
                p = res.pvalues[nc]
                neglogp_mat[i,j] = -np.log10(max(p, 1e-20))
                beta_mat[i,j] = res.params[nc]
                sig_mat[i,j] = p < 0.05
            except: pass

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Figure 5A: 折线图 — Top NT 的 -log10(P) 时间轨迹
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    x_pos = np.arange(n_time)

    for i, nc in enumerate(nt_ordered):
        bare = nc.replace(pfx, "")
        label = NT_RENAME.get(bare, bare)

        # 确定颜色和线宽
        if bare in TOP_NT:
            for sn, nts in NT_SYSTEMS.items():
                if bare in nts:
                    color = SYS_COLORS[sn]; break
            else:
                color = '#333333'
            lw = 2.5
            alpha = 1.0
            zorder = 10
            marker = 'o'
            ms = 7
        else:
            color = '#CCCCCC'
            lw = 1.0
            alpha = 0.4
            zorder = 1
            label = None  # 不显示在图例
            marker = ''
            ms = 0

        vals = neglogp_mat[i, :]
        valid = ~np.isnan(vals)
        if valid.sum() < 2: continue

        ax.plot(x_pos[valid], vals[valid], '-', color=color, linewidth=lw,
                alpha=alpha, zorder=zorder, marker=marker, markersize=ms,
                markeredgecolor='white', markeredgewidth=0.8, label=label)

    # P=0.05 参考线
    ax.axhline(y=-np.log10(0.05), color='red', linewidth=1.5, linestyle='--',
               alpha=0.7, zorder=5)
    ax.text(n_time - 0.9, -np.log10(0.05) + 0.15, 'P = 0.05',
            fontsize=9, color='red', alpha=0.7)

    # 时间阶段标注
    phase_colors = ['#FFEBEE', '#E8F5E9', '#FFF3E0', '#FFEBEE']
    phase_labels = ['Acute\nSignal', 'Compensatory\nPlateau', 'Late\nRe-emergence']
    if n_time >= 4:
        for idx, (start, end, pc) in enumerate([(0, 1, phase_colors[0]),
                                                  (1, 3, phase_colors[1]),
                                                  (3, n_time, phase_colors[3])]):
            if end <= n_time:
                ax.axvspan(start - 0.4, min(end - 0.6, n_time - 0.6),
                           alpha=0.3, color=pc, zorder=0)
                if idx < len(phase_labels):
                    ax.text((start + min(end, n_time) - 1) / 2, ax.get_ylim()[1] * 0.95,
                            phase_labels[idx], ha='center', fontsize=8,
                            color='#666', style='italic', va='top')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(time_labels, fontsize=11, fontweight='bold')
    ax.set_ylabel('$-\\log_{10}(P)$', fontsize=13, fontweight='bold')
    ax.set_xlabel('Follow-up Time Point', fontsize=13, fontweight='bold')
    ax.set_title('Figure 5. Temporal Dynamics of Neurochemical Effects\n'
                 'Top-5 NT highlighted; gray = other 12 NTs',
                 fontsize=12, fontweight='bold', pad=15)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9, ncol=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.15)

    plt.tight_layout()
    fig.savefig(OUTPUT_ROOT / 'figureS3_temporal_dynamics.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_ROOT / 'figureS3_temporal_dynamics.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ figureS3_temporal_dynamics.png/.pdf")

    # ================================================================
    # Figure 5B: 热图 — 全 NT × 时间点的显著性
    # ================================================================
    nt_labels = [NT_RENAME.get(c.replace(pfx,""), c.replace(pfx,"")) for c in nt_ordered]

    fig, ax = plt.subplots(figsize=(6, max(6, n_nt * 0.45 + 2)), facecolor='white')
    display = np.clip(neglogp_mat, 0, 10)
    im = ax.imshow(display, cmap='YlOrRd', vmin=0, vmax=10, aspect='auto')

    for i in range(n_nt):
        for j in range(n_time):
            val = neglogp_mat[i,j]
            if np.isnan(val):
                ax.text(j, i, '–', ha='center', va='center', fontsize=8, color='#CCC')
            elif val >= -np.log10(0.001):
                ax.text(j, i, '***', ha='center', va='center', fontsize=9,
                        color='white', fontweight='bold')
            elif val >= -np.log10(0.01):
                ax.text(j, i, '**', ha='center', va='center', fontsize=9,
                        color='white', fontweight='bold')
            elif val >= -np.log10(0.05):
                ax.text(j, i, '*', ha='center', va='center', fontsize=9,
                        color='black', fontweight='bold')

    ax.set_xticks(range(n_time))
    ax.set_xticklabels(time_labels, fontsize=10, fontweight='bold')
    ax.set_yticks(range(n_nt))
    ax.set_yticklabels(nt_labels, fontsize=9)
    for i, c in enumerate(nt_ordered):
        bare = c.replace(pfx,"")
        for sn, nts in NT_SYSTEMS.items():
            if bare in nts:
                ax.get_yticklabels()[i].set_color(SYS_COLORS[sn])
                ax.get_yticklabels()[i].set_fontweight('bold')
                break

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('$-\\log_{10}(P)$', fontsize=11, fontweight='bold')
    cbar.ax.axhline(y=-np.log10(0.05), color='black', linewidth=1.5, linestyle='--')

    # 统计每列显著数
    for j in range(n_time):
        n_sig = np.sum(sig_mat[:, j])
        ax.text(j, n_nt + 0.3, f'{n_sig}/17 sig.', ha='center', va='top',
                fontsize=8, color='#666', fontweight='bold')

    ax.set_title('Figure 5B. Significance Landscape Across Time\n'
                 '*** P<0.001  ** P<0.01  * P<0.05',
                 fontsize=11, fontweight='bold', pad=15)

    plt.tight_layout()
    fig.savefig(OUTPUT_ROOT / 'figureS3B_temporal_heatmap.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_ROOT / 'figureS3B_temporal_heatmap.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ figureS3B_temporal_heatmap.png/.pdf")

    # 保存数值
    pd.DataFrame(neglogp_mat, index=nt_labels, columns=time_labels).to_csv(
        OUTPUT_ROOT / 'figureS3_neglogp_matrix.csv')
    pd.DataFrame(beta_mat, index=nt_labels, columns=time_labels).to_csv(
        OUTPUT_ROOT / 'figureS3_beta_matrix.csv')
    print(f"✅ CSV saved")

    print(f"\n{'='*50}")
    print(f"  ✅ Figure 5 完成 → {OUTPUT_ROOT}")
    print(f"{'='*50}")

if __name__ == "__main__":
    print("="*50)
    print("  Figure 5: Temporal Dynamics")
    print("="*50)
    main()
