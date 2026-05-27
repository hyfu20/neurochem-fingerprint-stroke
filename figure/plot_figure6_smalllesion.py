#!/usr/bin/env python3
"""
Figure 6: Small-Lesion Severe-Outcome Phenotype
=================================================
A. 雷达图/条形图: 小病灶好预后 vs 小病灶差预后 的 NT 残差 profile
B. 箱线图: 两组的 IL-6 和 RMSSD 对比
C. 效应量热图: 17 NT 的 Cohen's d

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

def find_col(df, cands):
    for c in cands:
        if c in df.columns: return c
    return None

def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    s1, s2 = g1.std(), g2.std()
    pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    return (g1.mean() - g2.mean()) / pooled if pooled > 1e-10 else 0

def main():
    data_path = ("/data/usersdir/liuzhengxin/Stepbystep/"
                 "6.NeurotransmitterMapping/3.variable_outcom_merge_data/merged_neuro_data.csv")
    if not os.path.exists(data_path):
        print(f"⚠️ 未找到: {data_path}"); return

    import pandas as pd
    df = pd.read_csv(data_path, low_memory=False)
    print(f"数据: {df.shape}")

    # mRS 和 TLV
    mrs_col = find_col(df, ["m12_mRS","m3_mRS","D_MRS","d_mrs","mRS"])
    tlv_col = find_col(df, ["TLV","tlv"])
    if not mrs_col or not tlv_col: print("⚠️ 缺 mRS/TLV"); return

    df[mrs_col] = pd.to_numeric(df[mrs_col], errors="coerce")
    df[tlv_col] = pd.to_numeric(df[tlv_col], errors="coerce")

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

    # 炎症/自主神经
    il6_col = find_col(df, ["BSL_IL6","IL6","il6"])
    rmssd_col = find_col(df, ["HOLTER_RMSSD","RMSSD","rmssd"])

    # 定义小病灶 (TLV < Q1)
    q1 = df[tlv_col].quantile(0.25)
    small = df[df[tlv_col] < q1].copy()
    print(f"小病灶 (TLV < Q1={q1:.1f}): N={len(small)}")

    # 分组: 好预后 vs 差预后
    good = small[small[mrs_col] <= 2]
    poor = small[small[mrs_col] >= 3]
    print(f"  好预后 (mRS≤2): N={len(good)} ({len(good)/len(small)*100:.1f}%)")
    print(f"  差预后 (mRS≥3): N={len(poor)} ({len(poor)/len(small)*100:.1f}%)")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Figure 6A: NT 残差 Cohen's d 条形图 (排序)
    # ================================================================
    print("\n[6A] NT Cohen's d 条形图...")
    nt_labels = []
    d_values = []
    p_values = []
    nt_colors = []

    for nc in nt_ordered:
        bare = nc.replace(pfx, "")
        g = pd.to_numeric(good[nc], errors="coerce").dropna()
        p_grp = pd.to_numeric(poor[nc], errors="coerce").dropna()
        if len(g) < 10 or len(p_grp) < 10: continue

        d = cohens_d(p_grp, g)  # positive = poor group higher
        _, pv = stats.mannwhitneyu(g, p_grp, alternative='two-sided')

        nt_labels.append(NT_RENAME.get(bare, bare))
        d_values.append(d)
        p_values.append(pv)

        color = '#888888'
        for sn, nts in NT_SYSTEMS.items():
            if bare in nts: color = SYS_COLORS[sn]; break
        nt_colors.append(color)

    # 按 |d| 排序
    order = np.argsort(np.abs(d_values))[::-1]
    nt_labels = [nt_labels[i] for i in order]
    d_values = [d_values[i] for i in order]
    p_values = [p_values[i] for i in order]
    nt_colors = [nt_colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(5, len(nt_labels)*0.4)), facecolor='white')
    y_pos = np.arange(len(nt_labels))

    bars = ax.barh(y_pos, d_values, color=nt_colors, edgecolor='black',
                   linewidth=0.5, alpha=0.85, height=0.7)

    # 显著性标注
    for i, (d, p) in enumerate(zip(d_values, p_values)):
        stars = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        offset = 0.02 if d >= 0 else -0.02
        ha = 'left' if d >= 0 else 'right'
        ax.text(d + offset, i, f'{d:.3f}{stars}', va='center', ha=ha,
                fontsize=8, fontweight='bold' if p < 0.05 else 'normal',
                color='black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(nt_labels, fontsize=10, fontweight='bold')
    # Y 标签上色
    for i, c in enumerate(nt_colors):
        ax.get_yticklabels()[i].set_color(c)

    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.axvline(x=0.2, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.axvline(x=-0.2, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.set_xlabel("Cohen's d\n(positive = higher in severe group)", fontsize=12, fontweight='bold')
    ax.set_title(f"Figure 6A. Neurochemical Profile: Small-Lesion Phenotype\n"
                 f"TLV < Q1 ({q1:.0f} mm³): Good (mRS≤2, N={len(good)}) vs Severe (mRS≥3, N={len(poor)})\n"
                 f"*** P<0.001  ** P<0.01  * P<0.05",
                 fontsize=10, fontweight='bold', pad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.15)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(OUTPUT_ROOT / 'figure6A_smalllesion_cohens_d.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_ROOT / 'figure6A_smalllesion_cohens_d.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅ figure6A_smalllesion_cohens_d.png/.pdf")

    # ================================================================
    # Figure 6B: IL-6 和 RMSSD 箱线图
    # ================================================================
    print("\n[6B] IL-6 / RMSSD 箱线图...")
    bio_cols = []
    bio_labels = []
    if il6_col: bio_cols.append(il6_col); bio_labels.append("IL-6 (pg/mL)")
    if rmssd_col: bio_cols.append(rmssd_col); bio_labels.append("RMSSD (ms)")

    if bio_cols:
        n_bio = len(bio_cols)
        fig, axes = plt.subplots(1, n_bio, figsize=(4*n_bio, 5), facecolor='white')
        if n_bio == 1: axes = [axes]

        for ax, bc, bl in zip(axes, bio_cols, bio_labels):
            g_vals = pd.to_numeric(good[bc], errors="coerce").dropna()
            p_vals = pd.to_numeric(poor[bc], errors="coerce").dropna()

            # log1p 变换展示
            g_log = np.log1p(g_vals.clip(lower=0))
            p_log = np.log1p(p_vals.clip(lower=0))

            bp = ax.boxplot([g_log, p_log], labels=['Good\n(mRS≤2)', 'Severe\n(mRS≥3)'],
                           patch_artist=True, widths=0.5,
                           boxprops=dict(linewidth=1.5),
                           medianprops=dict(color='black', linewidth=2),
                           whiskerprops=dict(linewidth=1.5),
                           flierprops=dict(marker='o', markersize=3, alpha=0.3))
            bp['boxes'][0].set_facecolor('#4DBBD5')
            bp['boxes'][0].set_alpha(0.6)
            bp['boxes'][1].set_facecolor('#E64B35')
            bp['boxes'][1].set_alpha(0.6)

            # 散点
            ax.scatter(np.random.normal(1, 0.05, len(g_log)), g_log,
                      alpha=0.2, s=10, color='#4DBBD5', zorder=5)
            ax.scatter(np.random.normal(2, 0.05, len(p_log)), p_log,
                      alpha=0.2, s=10, color='#E64B35', zorder=5)

            # 统计检验
            _, p_mw = stats.mannwhitneyu(g_vals, p_vals, alternative='two-sided')
            d = cohens_d(p_vals, g_vals)
            stars = '***' if p_mw < 0.001 else ('**' if p_mw < 0.01 else ('*' if p_mw < 0.05 else 'ns'))

            y_max = max(g_log.max(), p_log.max())
            ax.plot([1, 1, 2, 2], [y_max+0.1, y_max+0.2, y_max+0.2, y_max+0.1],
                   color='black', linewidth=1.5)
            ax.text(1.5, y_max+0.25, f'{stars}\nd={d:.2f}', ha='center', fontsize=10,
                   fontweight='bold')

            ax.set_ylabel(f'log₁ₚ({bl})', fontsize=11, fontweight='bold')
            ax.set_title(bl, fontsize=12, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.suptitle(f'Figure 6B. Systemic Biomarkers in Small-Lesion Phenotype\n'
                     f'(TLV < Q1, Good N={len(good)}, Severe N={len(poor)})',
                     fontsize=11, fontweight='bold', y=1.02)
        plt.tight_layout()
        fig.savefig(OUTPUT_ROOT / 'figure6B_biomarker_boxplot.png', dpi=300,
                    bbox_inches='tight', facecolor='white')
        fig.savefig(OUTPUT_ROOT / 'figure6B_biomarker_boxplot.pdf',
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  ✅ figure6B_biomarker_boxplot.png/.pdf")
    else:
        print("  ⚠️ 无 IL-6/RMSSD 列，跳过 6B")

    # ================================================================
    # Figure 6C: 机制路径图（文字框 + 箭头）
    # ================================================================
    print("\n[6C] 机制路径图...")
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='white')

    boxes = [
        (0.08, 0.5, 'Small Lesion\n(TLV < Q1)', '#F5F5F5', 'black'),
        (0.30, 0.5, 'Strategic\nCholinergic\nDisconnection', '#4DBBD5', 'white'),
        (0.52, 0.5, 'Loss of\nAnti-inflammatory\nBrake', '#FFC107', 'black'),
        (0.74, 0.5, 'Systemic\nIL-6 ↑↑', '#E64B35', 'white'),
        (0.92, 0.5, 'Poor\nOutcome\n(mRS ≥ 3)', '#B71C1C', 'white'),
    ]

    for x, y, txt, bg, fc in boxes:
        ax.text(x, y, txt, ha='center', va='center', fontsize=10,
                fontweight='bold', color=fc,
                bbox=dict(boxstyle='round,pad=0.5', facecolor=bg,
                         edgecolor='black', linewidth=1.5, alpha=0.9),
                transform=ax.transAxes)

    # 箭头
    arrow_style = dict(arrowstyle='->', color='#333333', lw=2.5,
                       connectionstyle='arc3,rad=0')
    for x1, x2 in [(0.15, 0.23), (0.37, 0.45), (0.59, 0.67), (0.81, 0.87)]:
        ax.annotate('', xy=(x2, 0.5), xytext=(x1, 0.5),
                   xycoords='axes fraction', textcoords='axes fraction',
                   arrowprops=arrow_style)

    # 标注关键证据
    ax.text(0.30, 0.15, 'Lat. Path d=0.22*\nMed. Path d=0.27**',
            ha='center', fontsize=8, color='#4DBBD5', fontweight='bold',
            transform=ax.transAxes)
    ax.text(0.74, 0.15, 'IL-6 d=0.55***',
            ha='center', fontsize=8, color='#E64B35', fontweight='bold',
            transform=ax.transAxes)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Figure 6C. Mechanistic Pathway: "Small Lesion, Poor Outcome"',
                 fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()
    fig.savefig(OUTPUT_ROOT / 'figure6C_mechanism_pathway.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_ROOT / 'figure6C_mechanism_pathway.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅ figure6C_mechanism_pathway.png/.pdf")

    # 保存统计表
    results_df = pd.DataFrame({
        'NT': nt_labels, "Cohen's d": d_values, 'P': p_values,
    })
    results_df.to_csv(OUTPUT_ROOT / 'figure6_cohens_d_table.csv', index=False)
    print(f"  ✅ figure6_cohens_d_table.csv")

    print(f"\n{'='*50}")
    print(f"  ✅ Figure 6 完成 → {OUTPUT_ROOT}")
    print(f"{'='*50}")

if __name__ == "__main__":
    print("="*50)
    print("  Figure 6: Small-Lesion Severe-Outcome Phenotype")
    print("="*50)
    main()
