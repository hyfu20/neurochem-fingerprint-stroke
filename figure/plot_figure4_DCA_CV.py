#!/usr/bin/env python3
"""
Figure 4: Decision Curve Analysis + Cross-Validation AUC
=========================================================
A. DCA: 基础模型 vs 加NT模型 的净收益曲线
B. 10-fold CV AUC 条形图

输出: /data/usersdir/liuzhengxin/Stepbystep/7.figure/figure5/
"""
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_ROOT = Path("/data/usersdir/liuzhengxin/Stepbystep/7.figure/figure4")

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
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression

    df = pd.read_csv(data_path, low_memory=False)
    print(f"数据: {df.shape}")

    # mRS
    mrs_col = find_col(df, ["m12_mRS","m3_mRS","D_MRS","d_mrs","mRS"])
    if not mrs_col: print("⚠️ 无 mRS"); return
    print(f"结局: {mrs_col}")

    # 二分结局: mRS 0-2 = good, 3-6 = poor
    outcome = pd.to_numeric(df[mrs_col], errors="coerce")
    y_bin = (outcome >= 3).astype(float)

    # 协变量
    tlv = find_col(df, ["TLV","tlv"])
    nihss = find_col(df, ["A_NIHSS","BSL_NIHSS","NIHSS"])
    age = find_col(df, ["AGE","Age","age"])
    sex = find_col(df, ["SEX","Sex","sex"])
    base_cols = [c for c in [tlv, nihss, age, sex] if c]
    if len(base_cols) < 3: print("⚠️ 基线变量不足"); return

    # NT 列 (top 5)
    nt_all = [c for c in df.columns if c.startswith("Resid_")]
    if not nt_all:
        nt_all = [c for c in df.columns if c.startswith("Load_")]
    if not nt_all:
        known = ["NAT","A4B2","5HT6","DAT","VAChT","D1","5HTT","D2","M1","5HT1a"]
        nt_all = [c for c in known if c in df.columns]
    nt_top5 = nt_all[:5]
    print(f"基线: {base_cols}")
    print(f"NT top5: {nt_top5}")

    # 准备数据
    all_cols = base_cols + nt_top5
    sub = df[all_cols].apply(pd.to_numeric, errors="coerce")
    sub["y"] = y_bin
    sub = sub.dropna()
    print(f"完整样本: N={len(sub)}, events={int(sub['y'].sum())} ({sub['y'].mean()*100:.1f}%)")

    # log1p TLV
    if tlv and tlv in sub.columns:
        sub[tlv] = np.log1p(sub[tlv].clip(lower=0))

    # Z-score (性别除外)
    for c in all_cols:
        if c and c.upper() not in ("SEX","GENDER"):
            s = sub[c].std()
            if s > 1e-10:
                sub[c] = (sub[c] - sub[c].mean()) / s

    # ── Dual-Hit 交互项（与 Fig 3 一致）──
    inflam_cands = ["BSL_IL6","BSL_hsCRP","BSL_CRP","BSL_BR1_WBC","NLR"]
    inf_col = find_col(df, inflam_cands)
    dual_hit_cols = []
    if inf_col and nt_top5:
        sub[inf_col] = pd.to_numeric(sub.get(inf_col, df[inf_col]), errors="coerce")
        if inf_col not in sub.columns:
            sub[inf_col] = pd.to_numeric(df.loc[sub.index, inf_col], errors="coerce")
        inf_log = np.log1p(sub[inf_col].clip(lower=0))
        nt_med = sub[nt_top5[0]].median()
        inf_med = inf_log.median()
        sub["grp_NT_only"] = ((sub[nt_top5[0]] > nt_med) & (inf_log <= inf_med)).astype(int)
        sub["grp_Inf_only"] = ((sub[nt_top5[0]] <= nt_med) & (inf_log > inf_med)).astype(int)
        sub["grp_DualHit"] = ((sub[nt_top5[0]] > nt_med) & (inf_log > inf_med)).astype(int)
        dual_hit_cols = ["grp_NT_only", "grp_Inf_only", "grp_DualHit"]
        print(f"Dual-Hit: NT={nt_top5[0]}, Inf={inf_col}")
        print(f"  DualHit N={sub['grp_DualHit'].sum()}")
    
    full_cols = all_cols + dual_hit_cols
    sub = sub.dropna(subset=full_cols + ["y"])

    y = sub["y"].values
    X_base = sub[base_cols].values
    X_full = sub[full_cols].values

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Figure 5A: Decision Curve Analysis
    # ================================================================
    print("\n[5A] Decision Curve Analysis...")

    def calc_net_benefit(y_true, y_pred, threshold):
        """计算净收益"""
        n = len(y_true)
        y_pos = (y_pred >= threshold).astype(int)
        tp = np.sum((y_pos == 1) & (y_true == 1))
        fp = np.sum((y_pos == 1) & (y_true == 0))
        nb = tp / n - fp / n * (threshold / (1 - threshold))
        return nb

    thresholds = np.arange(0.01, 0.99, 0.01)

    # 拟合模型
    lr_base = LogisticRegression(max_iter=1000, solver='lbfgs')
    lr_base.fit(X_base, y)
    p_base = lr_base.predict_proba(X_base)[:, 1]

    lr_full = LogisticRegression(max_iter=1000, solver='lbfgs')
    lr_full.fit(X_full, y)
    p_full = lr_full.predict_proba(X_full)[:, 1]

    # 净收益
    nb_base = [calc_net_benefit(y, p_base, t) for t in thresholds]
    nb_full = [calc_net_benefit(y, p_full, t) for t in thresholds]
    nb_all = [y.mean() - (1 - y.mean()) * (t / (1 - t)) for t in thresholds]  # treat all
    nb_none = [0] * len(thresholds)  # treat none

    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    ax.plot(thresholds, nb_base, '-', color='#4DBBD5', linewidth=2.5,
            label='Base (TLV+NIHSS+Age+Sex)')
    ax.plot(thresholds, nb_full, '-', color='#E64B35', linewidth=2.5,
            label='Full (Base + NT + Dual-Hit)')
    ax.plot(thresholds, nb_all, '--', color='gray', linewidth=1, alpha=0.7,
            label='Treat All')
    ax.plot(thresholds, nb_none, '-', color='black', linewidth=1, alpha=0.5,
            label='Treat None')

    # 高亮净收益区间
    nb_base_arr = np.array(nb_base)
    nb_full_arr = np.array(nb_full)
    gain_mask = nb_full_arr > nb_base_arr
    if gain_mask.any():
        ax.fill_between(thresholds, nb_base_arr, nb_full_arr,
                        where=gain_mask, alpha=0.15, color='#E64B35',
                        label='Net gain from NT')

    ax.set_xlim(0.05, 0.95)
    y_min = min(min(nb_base), min(nb_full), 0) - 0.02
    y_max = max(max(nb_base), max(nb_full), y.mean()) + 0.05
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Threshold Probability', fontsize=13, fontweight='bold')
    ax.set_ylabel('Net Benefit', fontsize=13, fontweight='bold')
    ax.set_title('Figure 4A. Decision Curve Analysis\n'
                 f'({mrs_col}: mRS \u2265 3)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.15)
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_ROOT / 'figure4a_DCA.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_ROOT / 'figure4a_DCA.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅ figure4a_DCA.png/.pdf")

    # ================================================================
    # Figure 5B: 10-fold Cross-Validation AUC
    # ================================================================
    print("\n[5B] 10-fold Cross-Validation AUC...")

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    auc_base_folds = []
    auc_full_folds = []

    for train_idx, test_idx in skf.split(X_full, y):
        # Base model
        lr_b = LogisticRegression(max_iter=1000, solver='lbfgs')
        lr_b.fit(X_base[train_idx], y[train_idx])
        p_b = lr_b.predict_proba(X_base[test_idx])[:, 1]
        auc_base_folds.append(roc_auc_score(y[test_idx], p_b))

        # Full model
        lr_f = LogisticRegression(max_iter=1000, solver='lbfgs')
        lr_f.fit(X_full[train_idx], y[train_idx])
        p_f = lr_f.predict_proba(X_full[test_idx])[:, 1]
        auc_full_folds.append(roc_auc_score(y[test_idx], p_f))

    auc_base_mean = np.mean(auc_base_folds)
    auc_full_mean = np.mean(auc_full_folds)
    auc_base_std = np.std(auc_base_folds)
    auc_full_std = np.std(auc_full_folds)
    delta_auc = auc_full_mean - auc_base_mean

    print(f"  Base AUC: {auc_base_mean:.4f} ± {auc_base_std:.4f}")
    print(f"  Full AUC: {auc_full_mean:.4f} ± {auc_full_std:.4f}")
    print(f"  ΔAUC: {delta_auc:+.4f}")

    # Paired t-test
    from scipy.stats import ttest_rel
    t_stat, p_val = ttest_rel(auc_full_folds, auc_base_folds)
    print(f"  Paired t-test: t={t_stat:.3f}, P={p_val:.4f}")

    # 条形图
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
    models = ['Clinical Base\n(TLV+NIHSS+Age+Sex)', '+ Neurochemical\nFingerprint']
    means = [auc_base_mean, auc_full_mean]
    stds = [auc_base_std, auc_full_std]
    colors = ['#4DBBD5', '#E64B35']

    bars = ax.bar(models, means, yerr=stds, color=colors, edgecolor='black',
                  linewidth=0.8, alpha=0.9, capsize=8, error_kw={'linewidth': 1.5})

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.005,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    # ΔAUC 标注
    sig_str = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
    ax.annotate(f'ΔAUC = {delta_auc:+.3f}\nP = {p_val:.3f} {sig_str}',
                xy=(1, auc_full_mean + auc_full_std + 0.01),
                xytext=(0.5, max(means) + max(stds) + 0.03),
                fontsize=9, ha='center', fontweight='bold',
                color='#E64B35',
                arrowprops=dict(arrowstyle='->', color='#E64B35', lw=1.5))

    ax.set_ylabel('AUC (10-fold CV)', fontsize=13, fontweight='bold')
    ax.set_title('Figure 4B. Cross-Validation: Incremental Value\n'
                 f'(N={len(sub)}, 10-fold stratified CV)',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(min(means) - 0.08, max(means) + max(stds) + 0.08)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2)

    # 单个 fold 散点
    ax.scatter([0]*10, auc_base_folds, color='#4DBBD5', s=25, alpha=0.5, zorder=5)
    ax.scatter([1]*10, auc_full_folds, color='#E64B35', s=25, alpha=0.5, zorder=5)

    plt.tight_layout()
    fig.savefig(OUTPUT_ROOT / 'figure4b_CV_AUC.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_ROOT / 'figure4b_CV_AUC.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅ figure4b_CV_AUC.png/.pdf")

    # 保存数值
    cv_df = pd.DataFrame({
        'Fold': range(1, 11),
        'AUC_Base': auc_base_folds,
        'AUC_Full': auc_full_folds,
        'Delta': [f - b for f, b in zip(auc_full_folds, auc_base_folds)],
    })
    cv_df.to_csv(OUTPUT_ROOT / 'figure4b_CV_results.csv', index=False)
    print(f"  ✅ figure4b_CV_results.csv")

    print(f"\n{'='*50}")
    print(f"  ✅ Figure 4 完成 → {OUTPUT_ROOT}")
    print(f"{'='*50}")

if __name__ == "__main__":
    print("="*50)
    print("  Figure 4: DCA + Cross-Validation")
    print("="*50)
    main()
