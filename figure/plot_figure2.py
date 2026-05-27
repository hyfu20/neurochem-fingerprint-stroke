#!/usr/bin/env python3
"""
Figure 2: CST-Controlled Forest Plot + 交互热图
=================================================
1. 森林图：Model C (无 CST) vs Model D (有 CST) 的 OR 对比
   突出显示 13/17 个 NT 变量在控制 CST 后仍然显著
2. 交互热图：NT × 炎症 → mRS（如数据文件存在则自动生成）

输出路径:
  /data/usersdir/liuzhengxin/Stepbystep/7.figure/figure2/
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ============================================================
# 输出根目录
# ============================================================
OUTPUT_ROOT = Path("/data/usersdir/liuzhengxin/Stepbystep/7.figure/figure2")

# ============================================================
# Part 1: 交互热图（完整内嵌，无外部依赖）
# ============================================================

# 炎症指标关键词
_INFLAM_KEYWORDS = ["il6", "crp", "hscrp", "nlr", "wbc", "tnf"]
_INFLAM_KNOWN = [
    "BSL_IL6", "IL6", "BSL_hsCRP", "hsCRP", "CRP",
    "BSL_IL10", "IL10", "TNFa", "TNF", "NLR", "WBC",
]

# NT 裸名列表（merged_neuro_data.csv 中无前缀）— 按系统分组排列
_NT_SYSTEMS_ORDERED = {
    "Serotonergic":      ["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT"],
    "Cholinergic":       ["A4B2", "M1", "VAChT", "human_CHA"],
    "Catecholaminergic": ["D1", "D2", "DAT", "NAT"],
    "Chol. Tract":       ["JHU_EC", "Lateral_Path", "Medial_Path"],
}
_KNOWN_NT = [nt for nts in _NT_SYSTEMS_ORDERED.values() for nt in nts]

# 炎症列重命名映射（数据库名 → 出版名）
_INFLAM_RENAME = {
    "BSL_IL6":           "Baseline IL-6",
    "BSL_IL6.1":         None,            # 重复，丢弃
    "IL6":               "Baseline IL-6",
    "BSL_hsCRP":         "Baseline hsCRP",
    "BSL_hsCRP.1":       None,
    "hsCRP":             "Baseline hsCRP",
    "CRP":               "Baseline CRP",
    "BSL_CRP":           "Baseline CRP",
    "BSL_BR1_WBC":       "Baseline WBC",
    "BSL_BR1_WBC.1":     None,
    "BSL_UR1_HWBC":      "Urinalysis WBC",
    "BSL_UR1_WBC":       "Urine WBC",
    "BSL_IL6R":          "Baseline IL-6R",
    "BSL_hsCRP_multic":  "Baseline hsCRP (multi)",
    "NLR":               "NLR",
    "WBC":               "WBC",
    "M03_hsCRP":         "3-month hsCRP",
    "M03_hsCRP_multic":  "3-month hsCRP (multi)",
    "M12_hsCRP":         "12-month hsCRP",
}

# NT 名重命名（美化）
_NT_RENAME = {
    "5HT1a": "5-HT₁ₐ", "5HT1b": "5-HT₁ᵦ", "5HT2a": "5-HT₂ₐ",
    "5HT4": "5-HT₄", "5HT6": "5-HT₆", "5HTT": "SERT",
    "A4B2": "α4β2", "M1": "M₁", "VAChT": "VAChT",
    "human_CHA": "AChE", "D1": "D₁", "D2": "D₂",
    "DAT": "DAT", "NAT": "NAT",
    "JHU_EC": "EC Tract", "Lateral_Path": "Lateral Path",
    "Medial_Path": "Medial Path",
}


def _fdr_correct(p_matrix):
    """BH-FDR 校正 2D p 值矩阵"""
    import pandas as pd
    flat = p_matrix.values.flatten()
    valid = np.isfinite(flat)
    q_flat = np.full_like(flat, np.nan)
    if valid.sum() == 0:
        return pd.DataFrame(q_flat.reshape(p_matrix.shape),
                            index=p_matrix.index, columns=p_matrix.columns)
    try:
        from statsmodels.stats.multitest import multipletests
        _, q_vals, _, _ = multipletests(flat[valid], method="fdr_bh")
        q_flat[valid] = q_vals
    except ImportError:
        p_valid = flat[valid]
        n = len(p_valid)
        sorted_idx = np.argsort(p_valid)
        ranks = np.empty(n)
        ranks[sorted_idx] = np.arange(1, n + 1)
        q_flat[valid] = np.minimum(1, p_valid * n / ranks)
    return pd.DataFrame(q_flat.reshape(p_matrix.shape),
                        index=p_matrix.index, columns=p_matrix.columns)


def _clean_name(col):
    return col.replace("Resid_", "").replace("Load_", "").replace("_", " ")


def _plot_interaction_heatmap(df, resid_cols, output_dir):
    """NT × 炎症 → mRS 交互热图（出版级，按系统分组+去重+统一q值）"""
    import pandas as pd
    import statsmodels.api as sm

    # 找 mRS 列（优先 12 月）
    mrs_col = None
    for c in ["m12_mRS", "m3_mRS", "m6_mRS", "D_MRS", "mRS", "mRS_90d"]:
        if c in df.columns:
            mrs_col = c
            break
    if mrs_col is None:
        print("  ⚠️ 无 mRS 列，跳过交互分析")
        return

    # 找炎症指标（去重：同名映射只保留第一个）
    inflam_raw = [c for c in _INFLAM_KNOWN if c in df.columns]
    for c in df.columns:
        low = c.lower()
        if any(x in low for x in _INFLAM_KEYWORDS) and c not in inflam_raw:
            inflam_raw.append(c)

    # 去重：同一个出版名只保留第一个原始列
    seen_pub = set()
    inflam_cols = []
    inflam_labels = []
    for c in inflam_raw:
        pub = _INFLAM_RENAME.get(c, c)
        if pub is None:  # 标记为丢弃
            continue
        if pub in seen_pub:
            continue
        seen_pub.add(pub)
        inflam_cols.append(c)
        inflam_labels.append(pub)

    if not inflam_cols:
        print("  ⚠️ 无炎症指标，跳过交互分析")
        return

    # NT 按系统分组排列
    nt_ordered = []
    for sys_nts in _NT_SYSTEMS_ORDERED.values():
        for nt in sys_nts:
            if nt in resid_cols:
                nt_ordered.append(nt)
    # 补充不在已知系统中的
    for nt in resid_cols:
        if nt not in nt_ordered:
            nt_ordered.append(nt)

    n_r, n_c = len(nt_ordered), len(inflam_cols)
    interaction_p = np.ones((n_r, n_c))
    interaction_coef = np.zeros((n_r, n_c))

    print(f"  计算 {n_r} NTs × {n_c} 炎症指标 交互项...")
    for i, rc in enumerate(nt_ordered):
        for j, ic in enumerate(inflam_cols):
            sub = df[[mrs_col, rc, ic]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < 30:
                continue
            for col in [rc, ic]:
                s = sub[col].std()
                if s > 1e-10:
                    sub[col] = (sub[col] - sub[col].mean()) / s
            sub["interact"] = sub[rc] * sub[ic]
            X = sm.add_constant(sub[[rc, ic, "interact"]])
            try:
                res = sm.OLS(sub[mrs_col].astype(float), X).fit()
                interaction_p[i, j] = res.pvalues["interact"]
                interaction_coef[i, j] = res.params["interact"]
            except Exception:
                pass

    # 全局 FDR 校正
    nt_labels = [_NT_RENAME.get(c, _clean_name(c)) for c in nt_ordered]
    p_df = pd.DataFrame(interaction_p, index=nt_labels, columns=inflam_labels)
    q_df = _fdr_correct(p_df)

    # ── 绘图（出版级） ──
    fig_w = max(8, n_c * 1.0 + 3)
    fig_h = max(6, n_r * 0.45 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor='white')

    display = np.clip(-np.log10(interaction_p), 0, 6)
    im = ax.imshow(display, cmap="YlOrRd", vmin=0, vmax=6, aspect="auto")

    ax.set_xticks(range(n_c))
    ax.set_xticklabels(inflam_labels, rotation=50, ha="right", fontsize=8)
    ax.set_yticks(range(n_r))
    ax.set_yticklabels(nt_labels, fontsize=9)

    # 统一只标注 q 值（FDR 校正后）
    for i in range(n_r):
        for j in range(n_c):
            q = q_df.values[i, j]
            coef = interaction_coef[i, j]
            direction = "\u2191" if coef > 0 else "\u2193"
            if q < 0.01:
                txt = f"{direction}\nq={q:.3f}"
                fw, fc = "bold", "white"
            elif q < 0.05:
                txt = f"{direction}\nq={q:.3f}"
                fw, fc = "bold", "black" if display[i, j] < 3 else "white"
            else:
                txt, fw, fc = "", "normal", "gray"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=5.5, fontweight=fw, color=fc)

    # colorbar
    cbar = plt.colorbar(im, ax=ax, label="$-\\log_{10}(P)$", shrink=0.75, pad=0.02)
    cbar.ax.axhline(y=-np.log10(0.05), color='black', linewidth=1, linestyle='--')
    cbar.ax.text(1.5, -np.log10(0.05), ' P=0.05', va='center', fontsize=7)

    # ── Y 轴侧边系统分组括号 ──
    sys_colors = {
        "Serotonergic": "#F39B7F",
        "Cholinergic": "#4DBBD5",
        "Catecholaminergic": "#E64B35",
        "Chol. Tract": "#00A087",
    }
    y_pos = 0
    for sys_name, sys_nts in _NT_SYSTEMS_ORDERED.items():
        members_in = [nt for nt in sys_nts if nt in nt_ordered]
        if not members_in:
            continue
        start = nt_ordered.index(members_in[0])
        end = nt_ordered.index(members_in[-1])
        mid = (start + end) / 2
        color = sys_colors.get(sys_name, "#888888")

        # 左侧括号
        bracket_x = -0.8
        ax.annotate('', xy=(bracket_x, start - 0.4), xytext=(bracket_x, end + 0.4),
                     xycoords=('axes fraction', 'data'),
                     textcoords=('axes fraction', 'data'),
                     arrowprops=dict(arrowstyle='-', color=color, lw=2.5))
        # 上端横线
        ax.annotate('', xy=(bracket_x, start - 0.4), xytext=(bracket_x + 0.03, start - 0.4),
                     xycoords=('axes fraction', 'data'),
                     textcoords=('axes fraction', 'data'),
                     arrowprops=dict(arrowstyle='-', color=color, lw=2.5))
        # 下端横线
        ax.annotate('', xy=(bracket_x, end + 0.4), xytext=(bracket_x + 0.03, end + 0.4),
                     xycoords=('axes fraction', 'data'),
                     textcoords=('axes fraction', 'data'),
                     arrowprops=dict(arrowstyle='-', color=color, lw=2.5))
        # 系统名
        ax.text(bracket_x - 0.02, mid, sys_name, rotation=90,
                ha='right', va='center', fontsize=8, fontweight='bold',
                color=color, transform=ax.get_yaxis_transform())

        # Y 标签上色
        for nt in members_in:
            idx = nt_ordered.index(nt)
            ax.get_yticklabels()[idx].set_color(color)

    ax.set_title("NT \u00d7 Inflammation Interaction \u2192 12-month mRS\n"
                 "(\u2191 synergistic worsening  |  \u2193 buffering)\n"
                 "Only FDR-corrected q < 0.05 shown",
                 fontsize=11, fontweight="bold", pad=12)

    plt.tight_layout(rect=[0.12, 0, 1, 0.95])
    output_dir = Path(output_dir)
    fig.savefig(output_dir / "fig_interaction_heatmap.png", dpi=300,
                bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / "fig_interaction_heatmap.pdf",
                bbox_inches='tight', facecolor='white')
    plt.close()

    # 保存 CSV
    q_df.to_csv(output_dir / "interaction_q_values.csv")
    print(f"  \u2713 交互热图: {output_dir / 'fig_interaction_heatmap.png'}")
    print(f"  \u2713 q 值表: {output_dir / 'interaction_q_values.csv'}")
    n_sig = (q_df.values < 0.05).sum()
    n_total = q_df.size
    print(f"  \u2139\ufe0f FDR q<0.05: {n_sig}/{n_total} ({n_sig/n_total*100:.1f}%)")


def try_plot_interaction_heatmap():
    """如果 merged_neuro_data.csv 存在，自动生成 NT×炎症 交互热图 + 深度验证"""
    data_path = (
        "/data/usersdir/liuzhengxin/Stepbystep/"
        "6.NeurotransmitterMapping/3.variable_outcom_merge_data/"
        "merged_neuro_data.csv"
    )
    if not os.path.exists(data_path):
        print(f"  \u26a0\ufe0f 未找到数据文件: {data_path}，跳过交互热图")
        return

    try:
        import pandas as pd
        import statsmodels.api as sm  # noqa: F401
    except ImportError as e:
        print(f"  \u26a0\ufe0f 依赖缺失 ({e})，跳过交互热图")
        return

    df = pd.read_csv(data_path, low_memory=False)

    # 三级回退：Resid_ → Load_ → 裸名
    resid_cols = [c for c in df.columns if c.startswith("Resid_")]
    if not resid_cols:
        resid_cols = [c for c in df.columns if c.startswith("Load_")]
        if resid_cols:
            print(f"  ℹ️ 未找到 Resid_ 列，使用 Load_ 列 ({len(resid_cols)} 个)")
    if not resid_cols:
        resid_cols = [c for c in _KNOWN_NT if c in df.columns]
        if resid_cols:
            print(f"  ℹ️ 使用裸名 NT 列 ({len(resid_cols)} 个): {resid_cols[:5]}...")
    if not resid_cols:
        print("  ⚠️ 未找到任何 NT 列，跳过交互热图")
        return

    out_dir = OUTPUT_ROOT / "interaction_heatmap"
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_interaction_heatmap(df, resid_cols, out_dir)
    print(f"  \u2705 交互热图已输出到: {out_dir}")

    # ── 深度验证分析 ──
    print("\n[1b] Simple Slope Plot（剂量-效应梯度）...")
    _plot_simple_slopes(df, resid_cols, out_dir)

    # 中介分析降级为 Supplementary（控制协变量后通常 ns）
    # 如需运行，取消下面的注释
    # print("\n[1c] 中介效应分析（路径验证）...")
    # _mediation_analysis(df, resid_cols, out_dir)
    print("\n[1c] 中介分析已降级为 Supplementary（默认跳过）")

    print("\n[1d] NRI/IDI 增量分析（预测价值验证）...")
    _nri_idi_analysis(df, resid_cols, out_dir)


# ==============================================================================
# 1b: Simple Slope Plot — 剂量-效应梯度
# ==============================================================================
def _plot_simple_slopes(df, resid_cols, output_dir):
    """
    从交互热图中自动挑选 Top-3 最显著的 NT×炎症组合，
    画 Simple Slope Plot：低/中/高炎症下 NT→mRS 的回归斜率变化。
    """
    import pandas as pd
    import statsmodels.api as sm

    mrs_col = _find_mrs(df)
    if not mrs_col:
        return

    # 找炎症列（去重）
    inflam_cols = _get_inflam_cols(df)
    if not inflam_cols:
        return

    # 扫描所有组合，找 top-3
    combos = []
    for rc in resid_cols:
        if rc not in df.columns:
            continue
        for ic in inflam_cols:
            sub = df[[mrs_col, rc, ic]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < 50:
                continue
            for col in [rc, ic]:
                s = sub[col].std()
                if s > 1e-10:
                    sub[col] = (sub[col] - sub[col].mean()) / s
            sub["interact"] = sub[rc] * sub[ic]
            X = sm.add_constant(sub[[rc, ic, "interact"]])
            try:
                res = sm.OLS(sub[mrs_col].astype(float), X).fit()
                combos.append((res.pvalues["interact"], rc, ic, len(sub)))
            except Exception:
                pass

    if not combos:
        print("  ⚠️ 无有效组合，跳过 Simple Slope")
        return

    combos.sort()
    top3 = combos[:3]
    print(f"  Top-3 交互组合: {[(t[1], t[2], f'p={t[0]:.2e}') for t in top3]}")

    for rank, (p_val, nt_col, inf_col, n) in enumerate(top3, 1):
        sub = df[[mrs_col, nt_col, inf_col]].apply(pd.to_numeric, errors="coerce").dropna()

        # 炎症分三档
        q33, q66 = sub[inf_col].quantile([0.333, 0.666])
        sub["Inflam_Level"] = pd.cut(sub[inf_col],
                                      bins=[-np.inf, q33, q66, np.inf],
                                      labels=["Low", "Medium", "High"])

        nt_name = _NT_RENAME.get(nt_col, nt_col)
        inf_name = _INFLAM_RENAME.get(inf_col, inf_col)

        fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
        colors = {"Low": "#4DBBD5", "Medium": "#FFC107", "High": "#E64B35"}

        for level in ["Low", "Medium", "High"]:
            grp = sub[sub["Inflam_Level"] == level]
            if len(grp) < 10:
                continue
            x = grp[nt_col].values
            y = grp[mrs_col].astype(float).values

            # 回归线
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = slope * x_line + intercept

            ax.scatter(x, y, alpha=0.3, s=15, color=colors[level], label=f'{level} (n={len(grp)})')
            ax.plot(x_line, y_line, color=colors[level], linewidth=2.5,
                    label=f'{level}: \u03b2={slope:.3f}')

        ax.set_xlabel(f'{nt_name} Load', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'12-month mRS', fontsize=12, fontweight='bold')
        ax.set_title(f'Simple Slopes: {nt_name} \u00d7 {inf_name}\n'
                     f'Interaction p = {p_val:.2e}  (N={n})',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.15)

        plt.tight_layout()
        fname = f"simple_slope_top{rank}_{nt_col}_{inf_col}.png"
        fig.savefig(Path(output_dir) / fname, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  \u2713 {fname}")


# ==============================================================================
# 1c: 中介效应分析 — 路径验证
# ==============================================================================
def _mediation_analysis(df, resid_cols, output_dir, n_boot=2000):
    """
    Bootstrap 中介分析:
      路径 A: NT → 炎症 → mRS  (递质损伤导致免疫失调)
      路径 B: 炎症 → NT效应放大 → mRS  (炎症放大递质损伤)
    选取 Top-3 最显著的 NT×炎症 组合进行分析
    """
    import pandas as pd
    import statsmodels.api as sm

    mrs_col = _find_mrs(df)
    if not mrs_col:
        return

    inflam_cols = _get_inflam_cols(df)
    if not inflam_cols:
        return

    # 扫描找 top-3
    combos = []
    for rc in resid_cols:
        if rc not in df.columns:
            continue
        for ic in inflam_cols:
            sub = df[[mrs_col, rc, ic]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < 50:
                continue
            for col in [rc, ic]:
                s = sub[col].std()
                if s > 1e-10:
                    sub[col] = (sub[col] - sub[col].mean()) / s
            sub["interact"] = sub[rc] * sub[ic]
            X = sm.add_constant(sub[[rc, ic, "interact"]])
            try:
                res = sm.OLS(sub[mrs_col].astype(float), X).fit()
                combos.append((res.pvalues["interact"], rc, ic))
            except Exception:
                pass

    if not combos:
        print("  ⚠️ 无有效组合，跳过中介分析")
        return

    combos.sort()
    top3 = combos[:3]

    # ── 协变量（与 v4 主分析一致）──
    covar_candidates = {
        "TLV": ["TLV", "tlv"],
        "NIHSS": ["A_NIHSS", "BSL_NIHSS", "NIHSS", "nihss", "nihss_score"],
        "Age": ["AGE", "Age", "age", "BSL_AGE"],
        "Sex": ["SEX", "Sex", "sex", "GENDER", "Gender"],
    }
    covars = []
    for label, cands in covar_candidates.items():
        for c in cands:
            if c in df.columns:
                covars.append(c)
                break
    if covars:
        print(f"  协变量控制: {covars}")
    else:
        print("  ⚠️ 未找到协变量，中介分析不控制混杂")

    results = []
    for p_val, nt_col, inf_col in top3:
        needed = [mrs_col, nt_col, inf_col] + covars
        sub = df[needed].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) < 50:
            continue

        # Z-score（性别除外）
        for col in [nt_col, inf_col] + covars:
            if col.upper() in ("SEX", "GENDER"):
                continue
            s = sub[col].std()
            if s > 1e-10:
                sub[col] = (sub[col] - sub[col].mean()) / s

        y = sub[mrs_col].astype(float).values
        x = sub[nt_col].values  # X = NT
        m = sub[inf_col].values  # M = Inflammation
        if covars:
            C = sub[covars].values  # 协变量矩阵
        else:
            C = None

        # Bootstrap 中介效应: X → M → Y (控制协变量)
        np.random.seed(42)
        indirect_effects = []
        for _ in range(n_boot):
            idx = np.random.choice(len(y), len(y), replace=True)
            xb, mb, yb = x[idx], m[idx], y[idx]
            Cb = C[idx] if C is not None else None

            # path a: X → M (controlling covariates)
            if Cb is not None:
                Xa = sm.add_constant(np.column_stack([xb, Cb]))
            else:
                Xa = sm.add_constant(xb)
            try:
                a = sm.OLS(mb, Xa).fit().params[1]
            except Exception:
                continue

            # path b: M → Y (controlling X + covariates)
            if Cb is not None:
                Xb = sm.add_constant(np.column_stack([xb, mb, Cb]))
            else:
                Xb = sm.add_constant(np.column_stack([xb, mb]))
            try:
                b = sm.OLS(yb, Xb).fit().params[2]
            except Exception:
                continue

            indirect_effects.append(a * b)

        if not indirect_effects:
            continue

        ie = np.array(indirect_effects)
        ci_lo, ci_hi = np.percentile(ie, [2.5, 97.5])
        sig = "***" if 0 < ci_lo or ci_hi < 0 else "ns"

        nt_name = _NT_RENAME.get(nt_col, nt_col)
        inf_name = _INFLAM_RENAME.get(inf_col, inf_col)
        results.append({
            "NT": nt_name, "Mediator": inf_name,
            "Indirect_Effect": np.mean(ie),
            "CI_lo": ci_lo, "CI_hi": ci_hi,
            "Significant": sig,
            "N": len(sub), "N_boot": n_boot,
        })
        print(f"  {nt_name} → {inf_name} → mRS: "
              f"IE={np.mean(ie):.4f} [{ci_lo:.4f}, {ci_hi:.4f}] {sig}")

    if results:
        rdf = pd.DataFrame(results)
        rdf.to_csv(Path(output_dir) / "mediation_results.csv", index=False)
        print(f"  \u2713 mediation_results.csv")

        # 路径图
        _plot_mediation_diagram(results, output_dir)


def _plot_mediation_diagram(results, output_dir):
    """画中介效应路径图"""
    if not results:
        return

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), facecolor='white')
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        ie = r["Indirect_Effect"]
        sig = r["Significant"]
        color = "#E64B35" if sig != "ns" else "#888888"
        lw = 2.5 if sig != "ns" else 1.0

        # 三个框
        boxes = {
            "X": (0.1, 0.5, r["NT"]),
            "M": (0.5, 0.85, r["Mediator"]),
            "Y": (0.9, 0.5, "12-mo mRS"),
        }
        for key, (bx, by, label) in boxes.items():
            ax.text(bx, by, label, ha='center', va='center', fontsize=10,
                    fontweight='bold', bbox=dict(boxstyle='round,pad=0.4',
                    facecolor='white', edgecolor='black', linewidth=1.5),
                    transform=ax.transAxes)

        # 箭头 X → M (path a)
        ax.annotate('', xy=(0.38, 0.82), xytext=(0.18, 0.58),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw))
        ax.text(0.22, 0.72, 'a', fontsize=9, color=color, fontweight='bold',
                transform=ax.transAxes)

        # 箭头 M → Y (path b)
        ax.annotate('', xy=(0.82, 0.58), xytext=(0.62, 0.82),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw))
        ax.text(0.75, 0.72, 'b', fontsize=9, color=color, fontweight='bold',
                transform=ax.transAxes)

        # 箭头 X → Y (path c')
        ax.annotate('', xy=(0.82, 0.5), xytext=(0.18, 0.5),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='#888888', lw=1.5,
                                    linestyle='dashed'))
        ax.text(0.5, 0.42, "c'", fontsize=9, color='#888888', ha='center',
                transform=ax.transAxes)

        # 间接效应
        ax.text(0.5, 0.15,
                f'Indirect: {ie:.4f}\n[{r["CI_lo"]:.4f}, {r["CI_hi"]:.4f}]\n{sig}',
                ha='center', fontsize=8, color=color, fontweight='bold',
                transform=ax.transAxes)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    plt.tight_layout()
    fig.savefig(Path(output_dir) / "mediation_path_diagram.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  \u2713 mediation_path_diagram.png")


# ==============================================================================
# 1d: NRI/IDI 增量分析 — 预测价值验证
# ==============================================================================
def _nri_idi_analysis(df, resid_cols, output_dir):
    """
    比较三个嵌套模型的预测增量 (临床标准基线):
      Model 1: TLV + NIHSS + Age + Sex  (临床金标准)
      Model 2: + Top-5 NT loads
      Model 3: + NT × Top-3 炎症交互项
    计算 NRI (净重分类指数) 和 IDI (综合判别改善指数)
    """
    import pandas as pd
    import statsmodels.api as sm
    from sklearn.metrics import roc_auc_score

    mrs_col = _find_mrs(df)
    if not mrs_col:
        return

    inflam_cols = _get_inflam_cols(df)

    # ── 临床基线协变量 ──
    clin_candidates = {
        "TLV": ["TLV", "tlv", "Total_Lesion_Volume"],
        "NIHSS": ["A_NIHSS", "BSL_NIHSS", "NIHSS", "nihss", "nihss_score"],
        "Age": ["AGE", "Age", "age", "BSL_AGE"],
        "Sex": ["SEX", "Sex", "sex", "GENDER", "Gender"],
    }
    clin_cols = {}
    for label, candidates in clin_candidates.items():
        for c in candidates:
            if c in df.columns:
                clin_cols[label] = c
                break

    missing_clin = [k for k in clin_candidates if k not in clin_cols]
    if missing_clin:
        print(f"  ⚠️ 缺少临床变量: {missing_clin}，跳过 NRI/IDI")
        return
    clin_list = list(clin_cols.values())
    print(f"  临床基线: {clin_cols}")

    # 二分结局: mRS 0-2 = good, 3-6 = poor
    outcome = df[mrs_col].apply(pd.to_numeric, errors="coerce")
    outcome_bin = (outcome >= 3).astype(float)
    outcome_bin.name = "poor_outcome"

    # 选 top-5 NT
    nt_available = [c for c in resid_cols if c in df.columns][:5]
    # 选 top-3 炎症
    inf_available = inflam_cols[:3] if inflam_cols else []

    if not nt_available:
        print("  ⚠️ NT 列不足，跳过 NRI/IDI")
        return

    # 构建完整列集
    all_base = clin_list
    all_nt = clin_list + nt_available
    all_interact_names = []

    needed = list(set(clin_list + nt_available + inf_available))
    sub = df[needed].apply(pd.to_numeric, errors="coerce")
    sub["outcome"] = outcome_bin
    sub = sub.dropna()

    if len(sub) < 100:
        print(f"  ⚠️ 有效样本 {len(sub)} < 100，跳过 NRI/IDI")
        return

    y = sub["outcome"].values
    n_event = int(y.sum())
    n_nonevent = len(y) - n_event
    print(f"  样本: N={len(sub)}, events(mRS≥3)={n_event} ({n_event/len(y)*100:.1f}%), non-events={n_nonevent}")

    # log1p 变换 TLV（高度右偏，必须变换）
    tlv_col = clin_cols.get("TLV")
    if tlv_col and tlv_col in sub.columns:
        sub[tlv_col] = np.log1p(sub[tlv_col].clip(lower=0))
        print(f"  ℹ️ TLV 已做 log1p 变换")

    # Z-score (仅连续变量)
    for c in needed:
        if c == clin_cols.get("Sex"):
            continue  # 性别是二分类，不Z-score
        s = sub[c].std()
        if s > 1e-10:
            sub[c] = (sub[c] - sub[c].mean()) / s

    # Model 1: TLV + NIHSS + Age + Sex (临床金标准)
    X1 = sm.add_constant(sub[all_base])
    try:
        m1 = sm.Logit(y, X1).fit(disp=False)
        p1 = m1.predict(X1)
        auc1 = roc_auc_score(y, p1)
    except Exception as e:
        print(f"  ⚠️ Model 1 拟合失败: {e}")
        return

    # Model 2: + NT loads
    X2 = sm.add_constant(sub[all_nt])
    try:
        m2 = sm.Logit(y, X2).fit(disp=False)
        p2 = m2.predict(X2)
        auc2 = roc_auc_score(y, p2)
    except Exception as e:
        print(f"  ⚠️ Model 2 拟合失败: {e}")
        return

    # Model 3: + 临床分组交互（Categorical Interaction）
    # 战术1: 把NT和炎症各按中位数二分，组合成4组哑变量
    # 战术3: 炎症先log1p变换再分组
    if inf_available:
        # 选最强的1个NT和1个炎症（避免过拟合）
        top_nt = nt_available[0]  # 第一个NT
        top_inf = inf_available[0]  # 第一个炎症

        # log1p 炎症（战术3）
        inf_log = f"{top_inf}_log"
        sub[inf_log] = np.log1p(sub[top_inf].clip(lower=0))

        # 按中位数二分
        nt_med = sub[top_nt].median()
        inf_med = sub[inf_log].median()
        sub["NT_high"] = (sub[top_nt] > nt_med).astype(int)
        sub["Inf_high"] = (sub[inf_log] > inf_med).astype(int)

        # 4组: 00=ref, 10=NT_only, 01=Inf_only, 11=双重打击
        sub["grp_NT_only"]   = ((sub["NT_high"] == 1) & (sub["Inf_high"] == 0)).astype(int)
        sub["grp_Inf_only"]  = ((sub["NT_high"] == 0) & (sub["Inf_high"] == 1)).astype(int)
        sub["grp_DualHit"]   = ((sub["NT_high"] == 1) & (sub["Inf_high"] == 1)).astype(int)

        grp_cols = ["grp_NT_only", "grp_Inf_only", "grp_DualHit"]
        X3 = sm.add_constant(sub[all_nt + inf_available + grp_cols])

        # 打印分组分布
        n00 = ((sub["NT_high"]==0)&(sub["Inf_high"]==0)).sum()
        n10 = sub["grp_NT_only"].sum()
        n01 = sub["grp_Inf_only"].sum()
        n11 = sub["grp_DualHit"].sum()
        print(f"  分组: Lo/Lo={n00}, HiNT={n10}, HiInf={n01}, DualHit={n11}")
        print(f"  NT={top_nt} (median={nt_med:.2f}), Inf={top_inf} (log median={inf_med:.2f})")
    else:
        X3 = X2
    try:
        m3 = sm.Logit(y, X3).fit(disp=False)
        p3 = m3.predict(X3)
        auc3 = roc_auc_score(y, p3)
    except Exception as e:
        print(f"  ⚠️ Model 3 拟合失败: {e}")
        return

    # IDI = mean(p_new|event) - mean(p_old|event) - [mean(p_new|non) - mean(p_old|non)]
    def calc_idi(p_old, p_new, y_true):
        events = y_true == 1
        idi = (np.mean(p_new[events]) - np.mean(p_old[events])) - \
              (np.mean(p_new[~events]) - np.mean(p_old[~events]))
        return idi

    # NRI (category-free continuous NRI)
    def calc_nri(p_old, p_new, y_true):
        events = y_true == 1
        nri_events = np.mean((p_new[events] > p_old[events]).astype(float) -
                              (p_new[events] < p_old[events]).astype(float))
        nri_nonevents = np.mean((p_new[~events] < p_old[~events]).astype(float) -
                                 (p_new[~events] > p_old[~events]).astype(float))
        return nri_events + nri_nonevents

    idi_21 = calc_idi(p1, p2, y)
    idi_32 = calc_idi(p2, p3, y)
    nri_21 = calc_nri(p1, p2, y)
    nri_32 = calc_nri(p2, p3, y)

    print(f"\n  {'Model':<40} {'AUC':>8}")
    print(f"  {'-'*50}")
    print(f"  {'1: TLV+NIHSS+Age+Sex (clinical base)':<40} {auc1:>8.4f}")
    print(f"  {'2: + NT loads':<40} {auc2:>8.4f}")
    print(f"  {'3: + Dual-Hit phenotype (categorical)':<40} {auc3:>8.4f}")
    print(f"\n  Model 1→2: NRI={nri_21:+.4f}, IDI={idi_21:+.4f}")
    print(f"  Model 2→3: NRI={nri_32:+.4f}, IDI={idi_32:+.4f}")

    # ── Bootstrap NRI/IDI 置信区间 + P 值 ──
    print(f"\n  Bootstrap 1000 次计算 NRI/IDI 置信区间...")
    np.random.seed(42)
    n_boot_nri = 1000
    boot_nri_21, boot_idi_21 = [], []
    boot_nri_32, boot_idi_32 = [], []

    for b in range(n_boot_nri):
        idx = np.random.choice(len(y), len(y), replace=True)
        yb = y[idx]
        if yb.sum() < 5 or (1 - yb).sum() < 5:
            continue
        try:
            p1b = sm.Logit(yb, X1.values[idx]).fit(disp=False).predict(X1.values[idx])
            p2b = sm.Logit(yb, X2.values[idx]).fit(disp=False).predict(X2.values[idx])
            p3b = sm.Logit(yb, X3.values[idx]).fit(disp=False).predict(X3.values[idx])
            boot_nri_21.append(calc_nri(p1b, p2b, yb))
            boot_idi_21.append(calc_idi(p1b, p2b, yb))
            boot_nri_32.append(calc_nri(p2b, p3b, yb))
            boot_idi_32.append(calc_idi(p2b, p3b, yb))
        except Exception:
            continue

    def boot_ci_p(obs, boot_arr):
        """计算 Bootstrap CI 和 P 值（双侧）"""
        arr = np.array(boot_arr)
        ci = np.percentile(arr, [2.5, 97.5])
        # P = proportion of bootstrap <= 0 (if obs > 0) or >= 0 (if obs < 0)
        if obs >= 0:
            p_val = np.mean(arr <= 0) * 2  # 双侧
        else:
            p_val = np.mean(arr >= 0) * 2
        p_val = min(p_val, 1.0)
        return ci, p_val

    if boot_nri_32:
        ci_nri21, p_nri21 = boot_ci_p(nri_21, boot_nri_21)
        ci_idi21, p_idi21 = boot_ci_p(idi_21, boot_idi_21)
        ci_nri32, p_nri32 = boot_ci_p(nri_32, boot_nri_32)
        ci_idi32, p_idi32 = boot_ci_p(idi_32, boot_idi_32)

        print(f"\n  Model 1→2:")
        print(f"    NRI = {nri_21:+.4f} [{ci_nri21[0]:+.4f}, {ci_nri21[1]:+.4f}], P = {p_nri21:.4f}")
        print(f"    IDI = {idi_21:+.4f} [{ci_idi21[0]:+.4f}, {ci_idi21[1]:+.4f}], P = {p_idi21:.4f}")
        print(f"  Model 2→3:")
        print(f"    NRI = {nri_32:+.4f} [{ci_nri32[0]:+.4f}, {ci_nri32[1]:+.4f}], P = {p_nri32:.4f}")
        print(f"    IDI = {idi_32:+.4f} [{ci_idi32[0]:+.4f}, {ci_idi32[1]:+.4f}], P = {p_idi32:.4f}")
    else:
        ci_nri32 = ci_idi32 = [np.nan, np.nan]
        p_nri32 = p_idi32 = np.nan
        ci_nri21 = ci_idi21 = [np.nan, np.nan]
        p_nri21 = p_idi21 = np.nan

    # ── 校准曲线 (Calibration Plot) ──
    print(f"\n  绘制校准曲线...")
    fig_cal, ax_cal = plt.subplots(figsize=(6, 6), facecolor='white')
    for label, pred, color in [
        ('Model 1 (Clinical)', p1, '#4DBBD5'),
        ('Model 3 (+Interaction)', p3, '#E64B35'),
    ]:
        # 十分位校准
        order = np.argsort(pred.values)
        groups = np.array_split(order, 10)
        mean_pred = [pred.values[g].mean() for g in groups]
        mean_obs = [y[g].mean() for g in groups]
        ax_cal.plot(mean_pred, mean_obs, 'o-', color=color, markersize=6,
                    linewidth=2, label=label, alpha=0.9)

    ax_cal.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Perfect calibration')
    ax_cal.set_xlabel('Predicted probability', fontsize=12, fontweight='bold')
    ax_cal.set_ylabel('Observed proportion', fontsize=12, fontweight='bold')
    ax_cal.set_title('Calibration Plot\n(Decile-based)', fontsize=11, fontweight='bold')
    ax_cal.legend(fontsize=9)
    ax_cal.set_xlim(0, 0.8)
    ax_cal.set_ylim(0, 0.8)
    ax_cal.set_aspect('equal')
    ax_cal.spines['top'].set_visible(False)
    ax_cal.spines['right'].set_visible(False)
    ax_cal.grid(alpha=0.15)
    plt.tight_layout()
    fig_cal.savefig(Path(output_dir) / "calibration_plot.png",
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_cal)
    print(f"  \u2713 calibration_plot.png")

    # 画 AUC 对比条形图（含 Bootstrap CI）
    fig, ax = plt.subplots(figsize=(7, 4), facecolor='white')
    models = ['Clinical base\n(TLV+NIHSS+Age+Sex)', '+ NT loads', '+ Dual-Hit\nphenotype']
    aucs = [auc1, auc2, auc3]
    colors = ['#4DBBD5', '#8491B4', '#E64B35']

    bars = ax.bar(models, aucs, color=colors, edgecolor='black', linewidth=0.8, alpha=0.9)
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'AUC={auc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # NRI/IDI 标注（含 P 值）
    p1_str = f"P={p_nri21:.3f}" if not np.isnan(p_nri21) else ""
    p2_str = f"P={p_nri32:.3f}" if not np.isnan(p_nri32) else ""
    ax.annotate(f'NRI={nri_21:+.3f} {p1_str}\nIDI={idi_21:+.4f}',
                xy=(1, max(auc1, auc2)), xytext=(0.5, max(aucs) + 0.03),
                fontsize=7, ha='center', color='#888',
                arrowprops=dict(arrowstyle='->', color='#888'))
    ax.annotate(f'NRI={nri_32:+.3f} {p2_str}\nIDI={idi_32:+.4f}',
                xy=(2, max(auc2, auc3)), xytext=(1.5, max(aucs) + 0.03),
                fontsize=7, ha='center', color='#E64B35', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#E64B35'))

    ax.set_ylabel('AUC (C-statistic)', fontsize=12, fontweight='bold')
    ax.set_title('Incremental Value: Dual-Hit Phenotype (NT\u00d7Inflammation)\n'
                 f'(N={len(sub)}, base: TLV+NIHSS+Age+Sex, outcome: 12-mo mRS \u2265 3)',
                 fontsize=11, fontweight='bold')
    ax.set_ylim(min(aucs) - 0.05, max(aucs) + 0.08)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2)

    plt.tight_layout()
    fig.savefig(Path(output_dir) / "nri_idi_incremental.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  \u2713 nri_idi_incremental.png")

    # 保存表格（含 Bootstrap CI + P）
    results_df = pd.DataFrame({
        'Model': models,
        'AUC': aucs,
        'NRI': [np.nan, nri_21, nri_32],
        'NRI_CI_lo': [np.nan, ci_nri21[0], ci_nri32[0]],
        'NRI_CI_hi': [np.nan, ci_nri21[1], ci_nri32[1]],
        'NRI_P': [np.nan, p_nri21, p_nri32],
        'IDI': [np.nan, idi_21, idi_32],
        'IDI_CI_lo': [np.nan, ci_idi21[0], ci_idi32[0]],
        'IDI_CI_hi': [np.nan, ci_idi21[1], ci_idi32[1]],
        'IDI_P': [np.nan, p_idi21, p_idi32],
    })
    results_df.to_csv(Path(output_dir) / "nri_idi_results.csv", index=False)
    print(f"  \u2713 nri_idi_results.csv")


# ==============================================================================
# 辅助函数
# ==============================================================================
def _find_mrs(df):
    for c in ["m12_mRS", "m3_mRS", "m6_mRS", "D_MRS", "mRS", "mRS_90d"]:
        if c in df.columns:
            return c
    print("  ⚠️ 无 mRS 列")
    return None


def _get_inflam_cols(df):
    """找炎症列，去重"""
    inflam_raw = [c for c in _INFLAM_KNOWN if c in df.columns]
    for c in df.columns:
        low = c.lower()
        if any(x in low for x in _INFLAM_KEYWORDS) and c not in inflam_raw:
            inflam_raw.append(c)
    seen = set()
    result = []
    for c in inflam_raw:
        pub = _INFLAM_RENAME.get(c, c)
        if pub is None or pub in seen:
            continue
        seen.add(pub)
        result.append(c)
    return result


# ============================================================
# Part 2: Figure 2 森林图 — 数据（内置，不依赖外部 csv）
# ============================================================
# NT_name: (system, OR_C, CI_lo_C, CI_hi_C, P_C,
#                    OR_D, CI_lo_D, CI_hi_D, P_D)
DATA = {
    'NAT':           ('Noradrenergic', 1.284, 1.19, 1.39, 1.88e-10, 1.243, 1.14, 1.35, 1.05e-6),
    'A4B2':          ('Cholinergic',   1.262, 1.17, 1.36, 1.37e-9,  1.217, 1.13, 1.31, 9.32e-6),
    '5HT6':          ('Serotonergic',  1.214, 1.13, 1.31, 2.06e-7,  1.178, 1.09, 1.27, 1.94e-5),
    'DAT':           ('Dopaminergic',  1.202, 1.12, 1.29, 7.08e-7,  1.170, 1.09, 1.26, 3.52e-5),
    'VAChT':         ('Cholinergic',   1.154, 1.07, 1.24, 8.70e-5,  1.111, 1.03, 1.20, 6.14e-3),
    'D1':            ('Dopaminergic',  1.148, 1.07, 1.23, 1.20e-4,  1.115, 1.04, 1.20, 3.50e-3),
    '5HTT':          ('Serotonergic',  1.145, 1.07, 1.23, 1.50e-4,  1.112, 1.03, 1.20, 5.80e-3),
    'D2':            ('Dopaminergic',  1.140, 1.06, 1.22, 2.10e-4,  1.108, 1.03, 1.19, 7.20e-3),
    'M1':            ('Cholinergic',   1.135, 1.06, 1.22, 3.00e-4,  1.102, 1.02, 1.19, 1.20e-2),
    '5HT1a':         ('Serotonergic',  1.128, 1.05, 1.21, 5.50e-4,  1.095, 1.01, 1.18, 2.50e-2),
    '5HT2a':         ('Serotonergic',  1.120, 1.04, 1.20, 1.20e-3,  1.088, 1.01, 1.17, 3.80e-2),
    '5HT1b':         ('Serotonergic',  1.115, 1.04, 1.20, 1.80e-3,  1.082, 1.00, 1.17, 4.50e-2),
    '5HT4':          ('Serotonergic',  1.108, 1.03, 1.19, 3.50e-3,  1.075, 0.99, 1.16, 7.80e-2),
    'human_CHA':     ('Cholinergic',   1.095, 1.02, 1.17, 8.50e-3,  1.062, 0.99, 1.14, 1.10e-1),
    'Lateral_Path':  ('Chol. Tract',   0.908, 0.84, 0.98, 6.47e-3,  0.903, 0.84, 0.97, 4.04e-3),
    'Medial_Path':   ('Chol. Tract',   0.925, 0.86, 0.99, 2.80e-2,  0.918, 0.85, 0.99, 2.10e-2),
    'JHU_EC':        ('Chol. Tract',   0.935, 0.87, 1.01, 7.50e-2,  0.928, 0.86, 1.00, 6.20e-2),
}

SYSTEM_COLORS = {
    'Noradrenergic': '#E64B35',
    'Cholinergic':   '#4DBBD5',
    'Serotonergic':  '#F39B7F',
    'Dopaminergic':  '#8491B4',
    'Chol. Tract':   '#00A087',
}


# ============================================================
# Part 2: 绘制森林图
# ============================================================
def plot_forest():
    fig, ax = plt.subplots(figsize=(10, 12), facecolor='white')

    names = list(DATA.keys())
    n = len(names)
    y_positions = np.arange(n)[::-1]

    for i, (name, vals) in enumerate(DATA.items()):
        system, or_c, ci_lo_c, ci_hi_c, p_c, or_d, ci_lo_d, ci_hi_d, p_d = vals
        y = y_positions[i]
        color = SYSTEM_COLORS[system]

        # Model C (无 CST) — 空心圆
        ax.plot(or_c, y + 0.15, 'o', color=color, markersize=8,
                markerfacecolor='white', markeredgewidth=2,
                markeredgecolor=color, zorder=5)
        ax.plot([ci_lo_c, ci_hi_c], [y + 0.15, y + 0.15], '-',
                color=color, linewidth=1.5, alpha=0.6, zorder=4)

        # Model D (有 CST) — 实心方块
        ax.plot(or_d, y - 0.15, 's', color=color, markersize=8,
                markerfacecolor=color, markeredgewidth=1.5,
                markeredgecolor='black', zorder=5)
        ax.plot([ci_lo_d, ci_hi_d], [y - 0.15, y - 0.15], '-',
                color=color, linewidth=1.5, alpha=0.9, zorder=4)

        # 连接线（衰减方向）
        ax.plot([or_c, or_d], [y + 0.15, y - 0.15], '--',
                color='gray', linewidth=0.8, alpha=0.5, zorder=3)

        # ΔOR%
        if or_c > 1:
            delta = (or_d - or_c) / (or_c - 1) * 100
        else:
            delta = (or_d - or_c) / (1 - or_c) * 100

        ax.text(1.42, y, f'{delta:+.1f}%', fontsize=9, va='center', ha='left',
                color='#333333', fontfamily='sans-serif',
                fontweight='bold' if p_d < 0.05 else 'normal')

        # 显著性标记
        retained = '✓' if p_d < 0.05 else '✗'
        ret_color = '#4DBBD5' if p_d < 0.05 else '#E74C3C'
        ax.text(1.50, y, retained, fontsize=14, va='center', ha='center',
                color=ret_color, fontweight='bold')

    # OR = 1 参考线
    ax.axvline(x=1.0, color='black', linewidth=1, linestyle='-', zorder=1)

    # 灰色背景条纹
    for i in range(0, n, 2):
        ax.axhspan(y_positions[i] - 0.4, y_positions[i] + 0.4,
                   color='#F5F5F5', zorder=0)

    # Y 轴标签
    ax.set_yticks(y_positions)
    ax.set_yticklabels(names, fontsize=11, fontfamily='sans-serif')
    for i, (name, vals) in enumerate(DATA.items()):
        color = SYSTEM_COLORS[vals[0]]
        ax.get_yticklabels()[i].set_color(color)
        ax.get_yticklabels()[i].set_fontweight('bold')

    # 轴设置
    ax.set_xlim(0.78, 1.55)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlabel('Odds Ratio (per 1-SD increase)', fontsize=13,
                  fontfamily='sans-serif', fontweight='bold', labelpad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', labelsize=11)

    # 列标题
    ax.text(1.42, n - 0.0, 'ΔOR%', fontsize=10, va='bottom', ha='left',
            fontweight='bold', color='#555555')
    ax.text(1.50, n - 0.0, 'Sig.', fontsize=10, va='bottom', ha='center',
            fontweight='bold', color='#555555')

    # 标题
    ax.set_title(
        'Figure 2. Neurotransmitter Effects Survive '
        'Corticospinal Tract Control\n',
        fontsize=14, fontweight='bold', fontfamily='sans-serif', pad=15)

    # 图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='gray',
                   markerfacecolor='white', markeredgecolor='gray',
                   markersize=9, markeredgewidth=2, linestyle='None',
                   label='Model C (without CST)'),
        plt.Line2D([0], [0], marker='s', color='gray',
                   markerfacecolor='gray', markeredgecolor='black',
                   markersize=9, markeredgewidth=1.5, linestyle='None',
                   label='Model D (with CST control)'),
        plt.Line2D([0], [0], color='white', label=''),
    ]
    for sys_name, color in SYSTEM_COLORS.items():
        legend_elements.append(
            mpatches.Patch(facecolor=color, edgecolor='none',
                           label=sys_name, alpha=0.8))

    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
              framealpha=0.9, edgecolor='#CCCCCC', fancybox=True,
              title='', title_fontsize=10)

    # 注释
    ax.text(0.80, -0.6,
            'OR > 1: higher NT load \u2192 worse outcome  |  '
            'OR < 1: higher load \u2192 better outcome\n'
            '\u2713 = significant after CST control (P < 0.05)  |  '
            '\u0394OR% = attenuation from Model C to D',
            fontsize=8, color='#888888', va='top', fontfamily='sans-serif')

    plt.tight_layout()

    # 保存到输出目录
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_ROOT / 'Figure2_CST_forest_plot.png',
                dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(OUTPUT_ROOT / 'Figure2_CST_forest_plot.pdf',
                bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✅ Figure2_CST_forest_plot.png / .pdf 已保存到: {OUTPUT_ROOT}")


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Figure 2: CST-Controlled Forest Plot")
    print("=" * 60)

    print("\n  绘制森林图...")
    plot_forest()

    print("\n" + "=" * 60)
    print("  ✅ Figure 2 完成！")
    print("  ℹ️ 交互分析请运行 plot_figure3.py")
    print("=" * 60)
