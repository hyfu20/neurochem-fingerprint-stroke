#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mRS 有序多分类逻辑回归 · 全变量深度扫描
========================================
目的：
  对合并数据中的 **所有数值型变量** 逐一进行有序逻辑回归，
  同时控制病灶体积 (TLV) + 入院 NIHSS (A_NIHSS)。

方法：
  1. mRS 分组: Good (0–2)=0, Moderate (3–4)=1, Poor (5–6)=2
  2. 逐一: mrs_cat ~ TLV + A_NIHSS + X_i  (OrderedModel, logit)
  3. 全变量 FDR 校正 (Benjamini-Hochberg)
  4. 敏感性分析: 无控制 → +TLV → +TLV+NIHSS 三层对比
  5. 显著变量自动进入多因素模型
  6. 森林图、箱线图、火山图等可视化

输入：
  /data/usersdir/liuzhengxin/merged_neuro_data.csv

用法：
  python3 ordinal_logistic_mRS.py
  python3 ordinal_logistic_mRS.py --binary           # 二分类 Good vs Poor
  python3 ordinal_logistic_mRS.py --nt-only           # 仅扫描 17 条 NT
  python3 ordinal_logistic_mRS.py --min-n 50          # 提高最小样本量
  python3 ordinal_logistic_mRS.py --longitudinal      # 纵向多时间点轨迹分析

依赖：
  pip install pandas numpy statsmodels matplotlib scipy tqdm
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

# tqdm 可选
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# ── 中文字体 & 画图风格 ─────────────────────────────────────────────
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

# ── 配色 ────────────────────────────────────────────────────────────
NT_COLORS = {
    "5HT1a": "#4DBBD5", "5HT1b": "#4DBBD5", "5HT2a": "#7DCDE5",
    "5HT4": "#A8DFF0", "5HT6": "#B0E0F0", "5HTT": "#3B9FC4",
    "A4B2": "#8491B4",
    "D1": "#E64B35", "D2": "#DC7C6B", "DAT": "#E64B35",
    "M1": "#F39B7F",
    "NAT": "#91D1C2",
    "VAChT": "#00A087",
    "human_CHA": "#2E8B57", "JHU_EC": "#6B8E23",
    "Lateral_Path": "#3CB371", "Medial_Path": "#228B22",
}

# 变量分类配色 (按前缀)
PREFIX_COLORS = {
    "NT":    "#E64B35",   # 神经递质 - 红
    "Inflam": "#4DBBD5",  # 炎症 - 蓝
    "HRV":   "#00A087",   # 心率变异 - 绿
    "Psych":  "#F39B7F",  # 精神心理 - 橙
    "Clinic": "#8491B4",  # 一般临床 - 灰蓝
    "Other":  "#7F7F7F",  # 其他 - 灰
}

# 已知 17 条 NT 通路
KNOWN_NT = {
    "5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT",
    "A4B2", "D1", "D2", "DAT", "M1", "NAT", "VAChT",
    "human_CHA", "JHU_EC", "Lateral_Path", "Medial_Path",
}


# ═══════════════════════════════════════════════════════════════════
# Koch 残差法: 解决 Load vs TLV 极端共线性 (r≈0.99, VIF>700)
# ═══════════════════════════════════════════════════════════════════
def compute_koch_residuals(df, load_cols, tlv_col):
    """
    Koch et al. (2025, Brain) 残差法:
      Load_NT = β0 + β1 × TLV + ε
      ε (残差) = "不成比例递质损伤" (Disproportionate NT Damage)

    残差与 TLV 正交 (r=0, VIF=1)，安全地与 TLV 共入模型。

    Parameters
    ----------
    df : DataFrame
    load_cols : list  — 需要做残差的 Load_ 列名
    tlv_col : str     — TLV 列名

    Returns
    -------
    df : DataFrame  — 新增 Resid_XX 列
    resid_map : dict — {原始列名: 残差列名}
    resid_report : list[dict] — 残差回归报告
    """
    resid_map = {}
    resid_report = []

    tlv = df[tlv_col].values.astype(float)

    print(f"\n{'─' * 70}")
    print(f"  ⚗️  Koch 残差法: 对 {len(load_cols)} 个 NT Load 变量回归掉 TLV")
    print(f"     模型: Load_NT = β0 + β1 × TLV + ε")
    print(f"     → 使用 ε (残差) 代替原始 Load 进入后续回归")
    print(f"     → 残差与 TLV 正交: r = 0, VIF ≈ 1")
    print(f"{'─' * 70}")
    print(f"  {'Variable':<25s} {'R²(Load~TLV)':>14s} {'β_TLV':>12s} {'P':>12s}")
    print("  " + "─" * 65)

    for col in load_cols:
        y = df[col].values.astype(float)
        valid = np.isfinite(tlv) & np.isfinite(y)

        if valid.sum() < 20:
            print(f"  {col:<25s}  skipped (N={valid.sum()})")
            continue

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            tlv[valid], y[valid]
        )

        predicted = intercept + slope * tlv
        resid = y - predicted
        resid[~valid] = np.nan

        bare_name = col.replace("Load_", "")
        resid_col_name = f"Resid_{bare_name}"
        df[resid_col_name] = resid
        resid_map[col] = resid_col_name

        r2 = r_value ** 2
        resid_report.append({
            "Original": col,
            "Residual": resid_col_name,
            "R2_Load_vs_TLV": r2,
            "Beta_TLV": slope,
            "P_value": p_value,
            "N_valid": int(valid.sum()),
        })

        flag = " ⚠️ HIGH" if r2 > 0.8 else ""
        print(f"  {col:<25s} {r2:>12.4f}   {slope:>10.4e}   {p_value:>10.2e}{flag}")

    n_high = sum(1 for r in resid_report if r["R2_Load_vs_TLV"] > 0.5)
    print(f"\n  ✓ {len(resid_map)} 个变量已转换为残差")
    print(f"  ✓ {n_high} 个变量与 TLV 高度共线 (R² > 0.5) → 必须用残差")
    print(f"  ✓ 残差与 TLV 正交: 现在可以安全地同时放入回归模型")

    return df, resid_map, resid_report


def get_var_color(name):
    """获取变量颜色 (先按 NT 精确匹配，再按前缀分类)"""
    bare = name.replace("Load_", "")
    for key, color in NT_COLORS.items():
        if key.lower() == bare.lower():
            return color
    cat = classify_variable(name)
    return PREFIX_COLORS.get(cat, "#7F7F7F")


def classify_variable(name):
    """
    将变量名分类，用于结果表的 Category 列
    """
    bare = name.replace("Load_", "")
    if bare in KNOWN_NT:
        return "NT"
    low = name.lower()
    if any(x in low for x in ["il6", "il10", "crp", "tnf", "nlr", "wbc", "inflam"]):
        return "Inflam"
    if any(x in low for x in ["hrv", "rmssd", "sdnn", "lf", "hf", "brs", "hrn"]):
        return "HRV"
    if any(x in low for x in ["hamd", "hama", "moca", "mmse", "gds", "phq"]):
        return "Psych"
    if any(x in low for x in ["age", "sex", "bmi", "sbp", "dbp", "glucose",
                                "hba1c", "cholesterol", "ldl", "hdl"]):
        return "Clinic"
    return "Other"


# ═══════════════════════════════════════════════════════════════════
# 1. 数据加载与预处理
# ═══════════════════════════════════════════════════════════════════
def load_and_prepare(csv_path, binary=False, nt_only=False, use_residuals=True):
    """
    读取合并数据，创建 mRS 分组，识别控制变量与候选自变量

    Parameters
    ----------
    use_residuals : bool
        对 NT Load 变量自动计算 Koch 残差 (默认 True)
        解决 Load vs TLV 极端共线性 (r≈0.99, VIF>700)

    Returns
    -------
    df, mrs_col, control_cols, candidate_vars, group_labels
    """
    print("=" * 70)
    print("  mRS 有序逻辑回归 · 全变量深度扫描")
    print("=" * 70)

    df = pd.read_csv(csv_path)
    print(f"\n📂 数据: {csv_path}")
    print(f"   行数: {df.shape[0]}, 列数: {df.shape[1]}")

    # ── 识别 mRS ──
    # 优先级: m3_mRS (3个月=标准主结局) > m12_mRS > mRS > mRS_90d
    mrs_col = None
    for c in ["m3_mRS", "m12_mRS", "m6_mRS", "D_MRS",
              "mRS", "mRS_90d", "mrs", "mrs_90d"]:
        if c in df.columns:
            mrs_col = c
            break
    if mrs_col is None:
        print("❌ 找不到 mRS 列！")
        mrs_candidates = [c for c in df.columns if "mrs" in c.lower() or "MRS" in c]
        print(f"   可能的列: {mrs_candidates}")
        sys.exit(1)

    # ── 识别控制变量 ──
    # 控制变量 1: TLV
    tlv_col = None
    for c in ["TLV", "TLV_mm3", "tlv"]:
        if c in df.columns:
            tlv_col = c
            break
    # 控制变量 2: A_NIHSS
    nihss_col = None
    for c in ["A_NIHSS", "NIHSS", "nihss", "a_nihss", "A_nihss"]:
        if c in df.columns:
            nihss_col = c
            break

    control_cols = []
    if tlv_col:
        control_cols.append(tlv_col)
    if nihss_col:
        control_cols.append(nihss_col)

    print(f"   mRS 列:     {mrs_col}")
    print(f"   控制变量:   {control_cols}")

    if not control_cols:
        print("⚠️  未找到 TLV / NIHSS，模型将不控制任何协变量")

    # ── 转数值 ──
    df[mrs_col] = pd.to_numeric(df[mrs_col], errors="coerce")
    for c in control_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ── 去缺失 ──
    before = len(df)
    required = [mrs_col] + control_cols
    df = df.dropna(subset=required).copy()
    print(f"   去缺失后:   {len(df)} 行 (丢弃 {before - len(df)} 行)")

    # ── mRS 分组 ──
    if binary:
        df["mrs_target"] = df[mrs_col].apply(lambda x: 0 if x <= 2 else 1)
        group_labels = {0: "Good (0-2)", 1: "Poor (3-6)"}
        print(f"\n📊 mRS 二分类:")
    else:
        def group_mrs(x):
            if x <= 2:
                return 0
            elif x <= 4:
                return 1
            elif x >= 5:
                return 2
            return np.nan
        df["mrs_target"] = df[mrs_col].apply(group_mrs)
        group_labels = {0: "Good (0-2)", 1: "Moderate (3-4)", 2: "Poor (5-6)"}
        print(f"\n📊 mRS 三分组:")

    # 去掉分组为 NaN 的行
    df = df.dropna(subset=["mrs_target"]).copy()
    df["mrs_target"] = df["mrs_target"].astype(int)

    for k, v in group_labels.items():
        n = (df["mrs_target"] == k).sum()
        print(f"   {v}: {n} 人 ({n / len(df) * 100:.1f}%)")

    group_counts = df["mrs_target"].value_counts()
    if group_counts.min() < 5:
        print(f"\n⚠️  警告: 最小组仅 {group_counts.min()} 人, 建议使用 --binary 模式")

    # ── 候选变量 ──
    # 需要排除的列: ID 类、目标变量、控制变量、非数值列
    exclude_cols = {
        "ID", "id", "code_n", "identifier_rz", "name_rz",
        mrs_col, "mrs_target", "_merge",
        # 排除所有 mRS 时间点 (避免用 mRS 预测 mRS)
        "H_MRS", "A_MRS", "D_MRS", "F3_MRS",
        "m3_mRS", "m3_mRS_36", "m3_mRS.1",
        "m6_mRS", "m12_mRS",
    }
    exclude_cols.update(control_cols)

    if nt_only:
        # 仅扫描 17 条 NT
        candidate_vars = []
        for col in df.columns:
            bare = col.replace("Load_", "")
            if bare in KNOWN_NT and col not in exclude_cols:
                candidate_vars.append(col)
        if not candidate_vars:
            candidate_vars = [c for c in df.columns if c.startswith("Load_")]
        print(f"\n🔬 NT-only 模式: 仅扫描 {len(candidate_vars)} 条通路")
    else:
        # 全扫描: 所有数值型变量
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        candidate_vars = [c for c in numeric_cols if c not in exclude_cols]
        print(f"\n🔬 全扫描模式: {len(candidate_vars)} 个数值型变量")

    # ══════════════════════════════════════════════════════════════
    # Koch 残差法: 解决 Load vs TLV 极端共线性
    # ══════════════════════════════════════════════════════════════
    resid_map = {}
    resid_report = []
    if use_residuals and tlv_col:
        # 找出候选变量中的 NT Load 变量
        load_vars_in_candidates = [
            c for c in candidate_vars
            if c.startswith("Load_") or c.replace("Load_", "") in KNOWN_NT
        ]
        if load_vars_in_candidates:
            df, resid_map, resid_report = compute_koch_residuals(
                df, load_vars_in_candidates, tlv_col
            )
            # 将候选变量列表中的原始 Load 替换为残差
            new_candidates = []
            for var in candidate_vars:
                if var in resid_map:
                    new_candidates.append(resid_map[var])
                else:
                    new_candidates.append(var)
            candidate_vars = new_candidates
            # 同时将残差列排除在控制变量之外
            exclude_cols.update(resid_map.keys())  # 排除原始 Load

            print(f"\n  📌 回归模型更新为:")
            print(f"     mrs_target ~ TLV + A_NIHSS + Resid_NT")
            print(f"     (Resid_NT 已与 TLV 正交, VIF ≈ 1)")
    elif use_residuals and not tlv_col:
        print("\n  ⚠️  无 TLV 列, 跳过 Koch 残差转换")
    elif not use_residuals and tlv_col:
        print("\n  ⚠️  --no-residuals 模式: 原始 Load 直接入模")
        print("     警告: 如果 Load 与 TLV 高度共线 (r>0.9), OR 可能不可靠!")

    return df, mrs_col, control_cols, candidate_vars, group_labels, resid_map, resid_report


# ═══════════════════════════════════════════════════════════════════
# 2. 批量有序逻辑回归 (Deep Scan)
# ═══════════════════════════════════════════════════════════════════
def deep_scan_regression(df, control_cols, candidate_vars, binary=False,
                          min_n=30):
    """
    逐一分析: mrs_target ~ TLV + A_NIHSS + X_i
    (所有自变量 Z-score 标准化)

    Parameters
    ----------
    df : DataFrame
    control_cols : list
        控制变量 (如 ['TLV', 'A_NIHSS'])
    candidate_vars : list
        候选自变量列表
    binary : bool
        是否使用二分类
    min_n : int
        最小样本量阈值

    Returns
    -------
    result_table : pd.DataFrame
    """
    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        use_ordinal = True
    except ImportError:
        use_ordinal = False

    if binary or not use_ordinal:
        import statsmodels.api as sm
        if not use_ordinal:
            print("\n⚠️  OrderedModel 不可用，改用二分类逻辑回归")

    ctrl_str = " + ".join(control_cols) if control_cols else "(无)"
    print(f"\n{'─' * 70}")
    print(f"  🚀 开始深度扫描 {len(candidate_vars)} 个变量")
    print(f"     模型: mrs_target ~ {ctrl_str} + X_i")
    print(f"     最小样本量: {min_n}")
    print(f"{'─' * 70}")

    summary_list = []
    n_skipped = 0
    n_failed = 0

    for var in tqdm(candidate_vars, desc="  回归扫描", ncols=80):
        # 构建子集
        cols_needed = ["mrs_target"] + control_cols + [var]
        sub = df[cols_needed].dropna().copy()

        # 样本量 & 方差检查
        if len(sub) < min_n:
            n_skipped += 1
            continue
        sub[var] = pd.to_numeric(sub[var], errors="coerce")
        sub = sub.dropna()
        if len(sub) < min_n or sub[var].nunique() <= 1:
            n_skipped += 1
            continue

        # Z-score 标准化
        for col in control_cols + [var]:
            col_std = sub[col].std()
            if col_std < 1e-10:
                continue
            sub[col] = (sub[col] - sub[col].mean()) / col_std

        if sub[var].std() < 1e-10:
            n_skipped += 1
            continue

        y = sub["mrs_target"]
        X = sub[control_cols + [var]]

        try:
            if use_ordinal and not binary:
                mod = OrderedModel(y, X, distr="logit")
                res = mod.fit(method="bfgs", disp=False)
            else:
                X_const = sm.add_constant(X)
                mod = sm.Logit(y, X_const)
                res = mod.fit(disp=False)

            coef = res.params[var]
            p_val = res.pvalues[var]
            ci = res.conf_int().loc[var]
            or_val = np.exp(coef)
            or_lo = np.exp(ci[0])
            or_hi = np.exp(ci[1])

            sig = ("***" if p_val < 0.001 else
                   "**" if p_val < 0.01 else
                   "*" if p_val < 0.05 else
                   "†" if p_val < 0.1 else "")

            summary_list.append({
                "Variable": var,
                "Category": classify_variable(var),
                "Beta": coef,
                "OR": or_val,
                "CI_lower": or_lo,
                "CI_upper": or_hi,
                "P_value": p_val,
                "N": len(sub),
                "AIC": res.aic,
                "Sig": sig,
            })

        except Exception:
            n_failed += 1

    print(f"\n  ✓ 成功: {len(summary_list)} | 跳过: {n_skipped} | 失败: {n_failed}")

    if not summary_list:
        print("\n❌ 没有成功拟合任何模型！请检查数据。")
        sys.exit(1)

    result_table = pd.DataFrame(summary_list).sort_values("P_value")

    # FDR 校正 (Benjamini-Hochberg)
    try:
        from statsmodels.stats.multitest import multipletests
        _, fdr_q, _, _ = multipletests(
            result_table["P_value"].values, method="fdr_bh"
        )
        result_table["FDR_q"] = fdr_q
    except ImportError:
        p_vals = result_table["P_value"].values
        n = len(p_vals)
        ranks = np.argsort(np.argsort(p_vals)) + 1
        result_table["FDR_q"] = np.minimum(1, p_vals * n / ranks)

    return result_table


# ═══════════════════════════════════════════════════════════════════
# 3. 敏感性分析: 三层控制对比
# ═══════════════════════════════════════════════════════════════════
def sensitivity_analysis(df, control_cols, sig_vars, binary=False,
                          min_n=30):
    """
    对比三种模型:
      M0: mrs_target ~ X_i                    (无控制)
      M1: mrs_target ~ TLV + X_i              (控制体积)
      M2: mrs_target ~ TLV + A_NIHSS + X_i    (控制体积+评分)

    仅对显著变量做此分析
    """
    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        use_ordinal = True
    except ImportError:
        use_ordinal = False

    if binary or not use_ordinal:
        import statsmodels.api as sm

    print(f"\n{'─' * 70}")
    print(f"  敏感性分析: 三层控制对比 ({len(sig_vars)} 个显著变量)")
    print(f"{'─' * 70}")

    # 定义模型层级
    model_levels = [("M0_raw", [])]
    if len(control_cols) >= 1:
        model_levels.append(("M1_TLV", [control_cols[0]]))
    if len(control_cols) >= 2:
        model_levels.append(("M2_TLV_NIHSS", control_cols[:2]))

    header = f"  {'Variable':<20s}"
    for label, _ in model_levels:
        header += f"  {'OR':>6s} {'p':>10s}"
    header += f"  {'结论':>12s}"
    print(header)
    print("  " + "─" * (22 + 18 * len(model_levels) + 14))

    rows = []

    for var in sig_vars:
        sub = df[["mrs_target"] + control_cols + [var]].dropna().copy()
        if len(sub) < min_n or sub[var].nunique() <= 1:
            continue

        # Z-score
        for col in control_cols + [var]:
            s = sub[col].std()
            if s > 1e-10:
                sub[col] = (sub[col] - sub[col].mean()) / s

        y = sub["mrs_target"]
        row = {"Variable": var}

        for label, ctrls in model_levels:
            X = sub[ctrls + [var]]
            try:
                if use_ordinal and not binary:
                    mod = OrderedModel(y, X, distr="logit")
                    res = mod.fit(method="bfgs", disp=False)
                else:
                    X_c = sm.add_constant(X)
                    res = sm.Logit(y, X_c).fit(disp=False)
                row[f"OR_{label}"] = np.exp(res.params[var])
                row[f"P_{label}"] = res.pvalues[var]
            except Exception:
                row[f"OR_{label}"] = np.nan
                row[f"P_{label}"] = np.nan

        # 判断结论
        p_keys = [f"P_{lbl}" for lbl, _ in model_levels]
        p_vals_list = [row.get(k, np.nan) for k in p_keys]
        if len(p_vals_list) >= 3:
            if p_vals_list[0] < 0.05 and p_vals_list[-1] < 0.05:
                row["Conclusion"] = "独立效应 ✓"
            elif p_vals_list[0] < 0.05 and p_vals_list[-1] >= 0.05:
                row["Conclusion"] = "被控制变量解释 ✗"
            elif p_vals_list[0] >= 0.05:
                row["Conclusion"] = "始终不显著 —"
            else:
                row["Conclusion"] = "—"
        elif len(p_vals_list) == 2:
            if p_vals_list[0] < 0.05 and p_vals_list[-1] < 0.05:
                row["Conclusion"] = "独立效应 ✓"
            elif p_vals_list[0] < 0.05 and p_vals_list[-1] >= 0.05:
                row["Conclusion"] = "体积驱动 ✗"
            else:
                row["Conclusion"] = "—"
        else:
            row["Conclusion"] = "—"

        rows.append(row)

        # 打印
        line = f"  {var:<20s}"
        for label, _ in model_levels:
            or_v = row.get(f"OR_{label}", np.nan)
            p_v = row.get(f"P_{label}", np.nan)
            line += f"  {or_v:6.3f} {p_v:10.4e}"
        line += f"  {row['Conclusion']:>12s}"
        print(line)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# 4. 多因素模型 (显著变量联合)
# ═══════════════════════════════════════════════════════════════════
def multivariable_model(df, control_cols, result_table, p_threshold=0.05,
                         max_vars=8, binary=False):
    """
    将 p < threshold 的变量同时放入一个模型
    """
    sig = result_table.loc[result_table["P_value"] < p_threshold, "Variable"].tolist()

    if not sig:
        print(f"\n⚠️  没有 p < {p_threshold} 的变量，跳过多因素模型")
        return None

    if len(sig) > max_vars:
        print(f"\n⚠️  显著变量 {len(sig)} 个，取 OR 偏离 1 最大的前 {max_vars} 个")
        sub_table = result_table[result_table["Variable"].isin(sig)].copy()
        sub_table["OR_distance"] = abs(np.log(sub_table["OR"]))
        sig = sub_table.nlargest(max_vars, "OR_distance")["Variable"].tolist()

    print(f"\n{'─' * 70}")
    print(f"  多因素模型 ({len(sig)} 个变量 + {len(control_cols)} 个控制变量)")
    print(f"{'─' * 70}")
    print(f"  变量: {sig}")

    cols_needed = ["mrs_target"] + control_cols + sig
    sub = df[cols_needed].dropna().copy()
    print(f"  有效样本: {len(sub)}")

    if len(sub) < 30:
        print("  ⚠️  样本不足 30，跳过")
        return None

    # Z-score
    for col in control_cols + sig:
        s = sub[col].std()
        if s > 1e-10:
            sub[col] = (sub[col] - sub[col].mean()) / s

    y = sub["mrs_target"]
    X = sub[control_cols + sig]

    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        use_ordinal = True
    except ImportError:
        use_ordinal = False

    try:
        if use_ordinal and not binary:
            mod = OrderedModel(y, X, distr="logit")
            res = mod.fit(method="bfgs", disp=False)
        else:
            import statsmodels.api as sm
            X_c = sm.add_constant(X)
            res = sm.Logit(y, X_c).fit(disp=False)

        print(f"\n{res.summary()}")

        multi_results = []
        # 只提取候选变量的结果 (不含控制变量)
        for var in sig:
            try:
                coef = res.params[var]
                p_val = res.pvalues[var]
                ci = res.conf_int().loc[var]
                multi_results.append({
                    "Variable": var,
                    "Category": classify_variable(var),
                    "Beta": coef,
                    "OR": np.exp(coef),
                    "CI_lower": np.exp(ci[0]),
                    "CI_upper": np.exp(ci[1]),
                    "P_value": p_val,
                })
            except KeyError:
                pass

        multi_df = pd.DataFrame(multi_results)

        print("\n  多因素模型结果:")
        for _, row in multi_df.iterrows():
            s = ("***" if row["P_value"] < 0.001 else
                 "**" if row["P_value"] < 0.01 else
                 "*" if row["P_value"] < 0.05 else "")
            print(f"    {row['Variable']:20s}  OR={row['OR']:.3f}  "
                  f"95%CI=[{row['CI_lower']:.3f}, {row['CI_upper']:.3f}]  "
                  f"p={row['P_value']:.4e} {s}")

        return multi_df

    except Exception as e:
        print(f"\n⚠️  多因素模型拟合失败 (可能共线性太强): {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# 5. 共线性诊断
# ═══════════════════════════════════════════════════════════════════
def collinearity_check(df, vars_to_check, top_n=30):
    """检查变量间的相关性"""
    # 只检查前 top_n 个变量的相互关系
    check_vars = vars_to_check[:top_n]

    print(f"\n{'─' * 70}")
    print(f"  共线性检查 (前 {len(check_vars)} 个变量间 Spearman 相关)")
    print(f"{'─' * 70}")

    sub = df[check_vars].dropna()
    if len(sub) < 10:
        print("  样本不足，跳过")
        return None

    corr = sub.corr(method="spearman")
    high_corr = []
    for i in range(len(check_vars)):
        for j in range(i + 1, len(check_vars)):
            r = corr.iloc[i, j]
            if abs(r) > 0.7:
                high_corr.append((check_vars[i], check_vars[j], r))

    if high_corr:
        print(f"  ⚠️  发现 {len(high_corr)} 对高相关 (|ρ| > 0.7):")
        for a, b, r in sorted(high_corr, key=lambda x: -abs(x[2]))[:20]:
            print(f"    {a} ↔ {b}: ρ = {r:.3f}")
    else:
        print("  ✓ 未发现高共线性 (|ρ| < 0.7)")

    return corr


# ═══════════════════════════════════════════════════════════════════
# 6. 森林图 (Forest Plot)
# ═══════════════════════════════════════════════════════════════════
def plot_forest(result_table, output_path, title_suffix="", max_show=30):
    """
    绘制森林图: OR ± 95% CI
    如果变量太多，只显示 p 值最小的 max_show 个
    """
    df_plot = result_table.head(max_show).sort_values("OR", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, max(6, len(df_plot) * 0.4)))

    for i, (_, row) in enumerate(df_plot.iterrows()):
        var_name = row.get("Variable", row.get("Neurotransmitter", "?"))
        color = get_var_color(var_name)
        is_sig = row["P_value"] < 0.05

        ax.errorbar(
            row["OR"], i,
            xerr=[[row["OR"] - row["CI_lower"]], [row["CI_upper"] - row["OR"]]],
            fmt="o" if is_sig else "s",
            color=color,
            markerfacecolor=color if is_sig else "white",
            markeredgecolor=color,
            markersize=8,
            capsize=4,
            linewidth=2 if is_sig else 1,
            elinewidth=2 if is_sig else 1,
        )

        # P 值标注
        p_text = (f"p={row['P_value']:.3f}" if row["P_value"] >= 0.001
                  else f"p={row['P_value']:.1e}")
        if is_sig:
            p_text += " *"
        ax.text(
            max(row["CI_upper"] + 0.05, row["OR"] + 0.15), i,
            p_text, va="center", fontsize=7,
            fontweight="bold" if is_sig else "normal",
            color="red" if is_sig else "gray",
        )

    # 基准线 OR = 1
    ax.axvline(1, color="red", linestyle="--", linewidth=1, alpha=0.7,
               label="OR = 1 (no effect)")

    labels = [row.get("Variable", row.get("Neurotransmitter", "?"))
              for _, row in df_plot.iterrows()]
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Odds Ratio (OR) with 95% CI", fontsize=12)
    ax.set_title(
        f"Ordinal Logistic Regression: Variables → mRS Group\n"
        f"(Controlled for TLV + NIHSS){title_suffix}",
        fontsize=13, fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    ax.text(0.02, -0.06,
            "Filled = p < 0.05 | Open = p ≥ 0.05 | "
            "OR > 1: ↑risk | OR < 1: ↓risk (protective)\n"
            "All predictors Z-scored; OR = effect per 1 SD increase",
            transform=ax.transAxes, fontsize=7, color="gray", style="italic")

    plt.tight_layout()
    fig.savefig(output_path)
    print(f"🎨 森林图: {output_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# 7. 火山图 (Volcano Plot)
# ═══════════════════════════════════════════════════════════════════
def plot_volcano(result_table, output_path):
    """
    x = log2(OR), y = -log10(P)
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    df_v = result_table.copy()
    df_v["log2_OR"] = np.log2(df_v["OR"])
    df_v["neg_log10_p"] = -np.log10(df_v["P_value"])

    # 分类着色
    for cat, color in PREFIX_COLORS.items():
        mask = df_v["Category"] == cat
        if mask.any():
            ax.scatter(df_v.loc[mask, "log2_OR"],
                       df_v.loc[mask, "neg_log10_p"],
                       c=color, label=cat, alpha=0.7, s=40, edgecolors="white",
                       linewidth=0.5)

    # 显著性阈线
    ax.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=0.8,
               label="p = 0.05")
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.5)

    # 标注最显著的变量名
    top_n = min(10, len(df_v))
    for _, row in df_v.head(top_n).iterrows():
        if row["P_value"] < 0.05:
            ax.annotate(
                row["Variable"],
                (row["log2_OR"], row["neg_log10_p"]),
                fontsize=7, fontweight="bold",
                xytext=(5, 5), textcoords="offset points",
            )

    ax.set_xlabel("log₂(OR)", fontsize=12)
    ax.set_ylabel("-log₁₀(P-value)", fontsize=12)
    ax.set_title("Volcano Plot: All Variables vs mRS\n(Controlled for TLV + NIHSS)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path)
    print(f"🎨 火山图: {output_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# 8. 分组箱线图
# ═══════════════════════════════════════════════════════════════════
def plot_group_boxplots(df, top_vars, group_labels, output_path, top_n=6):
    """
    画最显著的 N 个变量在各 mRS 分组中的分布
    """
    n_plot = min(top_n, len(top_vars))
    ncols = 3
    nrows = (n_plot + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    mrs_groups = sorted(df["mrs_target"].unique())
    group_names = [group_labels.get(g, str(g)) for g in mrs_groups]
    colors_list = ["#4CAF50", "#FFC107", "#F44336"]

    for idx in range(n_plot):
        ax = axes[idx]
        var = top_vars[idx]

        data_groups = [df.loc[df["mrs_target"] == g, var].dropna().values
                       for g in mrs_groups]

        bp = ax.boxplot(data_groups, labels=group_names, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], colors_list[:len(mrs_groups)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Kruskal-Wallis
        valid_groups = [g for g in data_groups if len(g) > 0]
        if len(valid_groups) >= 2:
            try:
                _, p_kw = stats.kruskal(*valid_groups)
                ax.set_title(f"{var}\n(KW p={p_kw:.4f})", fontsize=10)
            except Exception:
                ax.set_title(var, fontsize=10)
        else:
            ax.set_title(var, fontsize=10)

        ax.set_ylabel("Value")

    for idx in range(n_plot, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Top Variables Distribution by mRS Group", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path)
    print(f"🎨 箱线图: {output_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# 9. 相关矩阵热图
# ═══════════════════════════════════════════════════════════════════
def plot_correlation_heatmap(corr_matrix, output_path):
    """显著变量间 Spearman 相关矩阵热图"""
    if corr_matrix is None or len(corr_matrix) < 2:
        return

    labels = list(corr_matrix.columns)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.5), max(6, n * 0.4)))

    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=max(6, 10 - n // 5))
    ax.set_yticklabels(labels, fontsize=max(6, 10 - n // 5))

    # 标注数值 (变量少时)
    if n <= 20:
        for i in range(n):
            for j in range(n):
                val = corr_matrix.values[i, j]
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=max(5, 8 - n // 5), color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")
    ax.set_title("Significant Variables Inter-correlation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path)
    print(f"🎨 相关热图: {output_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# 10. 纵向多时间点轨迹分析
# ═══════════════════════════════════════════════════════════════════
def longitudinal_trajectory_scan(csv_path, control_cols_hint=None,
                                  candidate_vars_hint=None,
                                  binary=False, min_n=30, nt_only=False):
    """
    对 D_mRS / m3_mRS / m6_mRS / m12_mRS 四个时间点
    + 恢复值 (m12_mRS - A_mRS) 逐一跑有序逻辑回归

    Returns
    -------
    all_results : pd.DataFrame
        包含 Time_Point 列的汇总表
    """
    df_raw = pd.read_csv(csv_path)

    # 识别所有 mRS 时间点 (匹配你的实际列名)
    timepoint_map = {
        "D_MRS":   ["D_MRS", "D_mRS", "d_mRS", "mRS_discharge"],
        "m3_mRS":  ["m3_mRS", "mRS_3m", "mRS_90d", "mRS"],
        "m6_mRS":  ["m6_mRS", "mRS_6m", "mRS_180d"],
        "m12_mRS": ["m12_mRS", "mRS_12m", "mRS_1y", "mRS_365d"],
    }

    found_timepoints = {}
    for label, candidates in timepoint_map.items():
        for c in candidates:
            if c in df_raw.columns:
                found_timepoints[label] = c
                break

    # 识别基线 mRS (入院)
    baseline_mrs = None
    for c in ["A_MRS", "A_mRS", "a_mRS", "mRS_pre", "H_MRS"]:
        if c in df_raw.columns:
            baseline_mrs = c
            break

    # 如果有基线, 创建恢复值
    if baseline_mrs and "m12_mRS" in found_timepoints:
        col_12 = found_timepoints["m12_mRS"]
        df_raw["Recovery_Delta"] = (
            pd.to_numeric(df_raw[baseline_mrs], errors="coerce")
            - pd.to_numeric(df_raw[col_12], errors="coerce")
        )
        found_timepoints["Recovery_Δ"] = "Recovery_Delta"
    elif baseline_mrs and "m3_mRS" in found_timepoints:
        col_3 = found_timepoints["m3_mRS"]
        df_raw["Recovery_Delta"] = (
            pd.to_numeric(df_raw[baseline_mrs], errors="coerce")
            - pd.to_numeric(df_raw[col_3], errors="coerce")
        )
        found_timepoints["Recovery_Δ"] = "Recovery_Delta"

    print(f"\n{'=' * 70}")
    print(f"  📅 纵向轨迹分析: {len(found_timepoints)} 个时间点")
    print(f"{'=' * 70}")
    for label, col in found_timepoints.items():
        n_valid = df_raw[col].notna().sum()
        print(f"   {label:15s} → {col} (N={n_valid})")

    if not found_timepoints:
        print("  ❌ 未找到任何 mRS 时间点列！")
        return pd.DataFrame()

    # 识别控制变量
    control_cols = []
    for c in ["TLV", "TLV_mm3", "tlv"]:
        if c in df_raw.columns:
            control_cols.append(c)
            break
    for c in ["A_NIHSS", "NIHSS", "nihss"]:
        if c in df_raw.columns:
            control_cols.append(c)
            break

    # 识别候选变量
    all_outcome_cols = set(found_timepoints.values())
    if baseline_mrs:
        all_outcome_cols.add(baseline_mrs)
    exclude = {"ID", "id", "code_n", "identifier_rz", "name_rz",
               "mrs_target", "mrs_cat", "_merge", "Recovery_Delta"}
    exclude.update(all_outcome_cols)
    exclude.update(control_cols)

    if nt_only:
        candidate_vars = []
        for col in df_raw.columns:
            bare = col.replace("Load_", "")
            if bare in KNOWN_NT and col not in exclude:
                candidate_vars.append(col)
        if not candidate_vars:
            candidate_vars = [c for c in df_raw.columns if c.startswith("Load_")]
    else:
        numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        candidate_vars = [c for c in numeric_cols if c not in exclude]

    # ── Koch 残差法 (纵向分析同样需要!) ──
    tlv_col_long = control_cols[0] if control_cols else None
    load_vars_long = [
        c for c in candidate_vars
        if c.startswith("Load_") or c.replace("Load_", "") in KNOWN_NT
    ]
    if tlv_col_long and load_vars_long:
        for c in control_cols:
            df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")
        for c in load_vars_long:
            df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")
        df_raw, resid_map_long, _ = compute_koch_residuals(
            df_raw, load_vars_long, tlv_col_long
        )
        new_candidates = []
        for var in candidate_vars:
            if var in resid_map_long:
                new_candidates.append(resid_map_long[var])
            else:
                new_candidates.append(var)
        candidate_vars = new_candidates

    print(f"   控制变量:   {control_cols}")
    print(f"   候选变量:   {len(candidate_vars)} 个")

    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        use_ordinal = True
    except ImportError:
        use_ordinal = False

    if binary or not use_ordinal:
        import statsmodels.api as sm

    all_results = []

    for tp_label, tp_col in found_timepoints.items():
        print(f"\n  ── {tp_label} ──")

        df_tp = df_raw.copy()
        df_tp[tp_col] = pd.to_numeric(df_tp[tp_col], errors="coerce")
        for c in control_cols:
            df_tp[c] = pd.to_numeric(df_tp[c], errors="coerce")

        # 去缺失
        required = [tp_col] + control_cols
        df_tp = df_tp.dropna(subset=required).copy()

        # 分组 (Recovery_Delta 用连续值, 其他分组)
        if tp_label == "Recovery_Δ":
            # 连续变量, 用 OLS
            is_continuous = True
            df_tp["mrs_target"] = df_tp[tp_col]
        else:
            is_continuous = False
            if binary:
                df_tp["mrs_target"] = df_tp[tp_col].apply(
                    lambda x: 0 if x <= 2 else 1
                )
            else:
                def _group(x):
                    if x <= 2: return 0
                    elif x <= 4: return 1
                    elif x >= 5: return 2
                    return np.nan
                df_tp["mrs_target"] = df_tp[tp_col].apply(_group)

            df_tp = df_tp.dropna(subset=["mrs_target"]).copy()
            df_tp["mrs_target"] = df_tp["mrs_target"].astype(int)

        n_tp = len(df_tp)
        print(f"     有效样本: {n_tp}")
        if n_tp < min_n:
            print(f"     ⚠️  样本不足 {min_n}, 跳过")
            continue

        n_success = 0
        for var in candidate_vars:
            cols_needed = ["mrs_target"] + control_cols + [var]
            sub = df_tp[cols_needed].dropna().copy()
            sub[var] = pd.to_numeric(sub[var], errors="coerce")
            sub = sub.dropna()

            if len(sub) < min_n or sub[var].nunique() <= 1:
                continue

            # Z-score
            for col in control_cols + [var]:
                s = sub[col].std()
                if s > 1e-10:
                    sub[col] = (sub[col] - sub[col].mean()) / s
            if sub[var].std() < 1e-10:
                continue

            y = sub["mrs_target"]
            X = sub[control_cols + [var]]

            try:
                if is_continuous:
                    import statsmodels.api as sm
                    X_c = sm.add_constant(X)
                    res = sm.OLS(y, X_c).fit()
                elif use_ordinal and not binary:
                    mod = OrderedModel(y, X, distr="logit")
                    res = mod.fit(method="bfgs", disp=False)
                else:
                    import statsmodels.api as sm
                    X_c = sm.add_constant(X)
                    res = sm.Logit(y, X_c).fit(disp=False)

                coef = res.params[var]
                p_val = res.pvalues[var]
                ci = res.conf_int().loc[var]

                all_results.append({
                    "Time_Point": tp_label,
                    "Variable": var,
                    "Category": classify_variable(var),
                    "Beta": coef,
                    "OR": np.exp(coef) if not is_continuous else np.nan,
                    "CI_lower": np.exp(ci[0]) if not is_continuous else ci[0],
                    "CI_upper": np.exp(ci[1]) if not is_continuous else ci[1],
                    "P_value": p_val,
                    "N": len(sub),
                })
                n_success += 1
            except Exception:
                pass

        print(f"     成功拟合: {n_success} 个变量")

    result_df = pd.DataFrame(all_results)

    # 每个时间点内做 FDR
    if len(result_df) > 0:
        fdr_list = []
        for tp in result_df["Time_Point"].unique():
            sub = result_df[result_df["Time_Point"] == tp].copy()
            try:
                from statsmodels.stats.multitest import multipletests
                _, q, _, _ = multipletests(sub["P_value"].values, method="fdr_bh")
                sub["FDR_q"] = q
            except ImportError:
                sub["FDR_q"] = sub["P_value"] * len(sub)
            fdr_list.append(sub)
        result_df = pd.concat(fdr_list, ignore_index=True)

    return result_df


# ═══════════════════════════════════════════════════════════════════
# 11. 纵向轨迹热图
# ═══════════════════════════════════════════════════════════════════
def plot_trajectory_heatmap(traj_df, output_path, top_n=25):
    """
    热图: x = 时间点, y = 变量, 颜色 = -log10(P)
    一眼看出哪些变量在哪个时间段最显著
    """
    if traj_df is None or len(traj_df) == 0:
        return

    # 取每个变量在任一时间点 P 最小的 top_n 个
    best_p = traj_df.groupby("Variable")["P_value"].min().nsmallest(top_n)
    top_vars = best_p.index.tolist()

    sub = traj_df[traj_df["Variable"].isin(top_vars)].copy()
    timepoints = [tp for tp in ["D_MRS", "m3_mRS", "m6_mRS", "m12_mRS", "Recovery_Δ"]
                  if tp in sub["Time_Point"].unique()]

    if not timepoints:
        return

    # 创建矩阵
    matrix = np.full((len(top_vars), len(timepoints)), np.nan)
    or_matrix = np.full((len(top_vars), len(timepoints)), np.nan)

    for i, var in enumerate(top_vars):
        for j, tp in enumerate(timepoints):
            row = sub[(sub["Variable"] == var) & (sub["Time_Point"] == tp)]
            if len(row) > 0:
                matrix[i, j] = -np.log10(row["P_value"].values[0])
                or_val = row["OR"].values[0]
                if np.isfinite(or_val):
                    or_matrix[i, j] = or_val

    matrix = np.nan_to_num(matrix, nan=0)

    fig, ax = plt.subplots(figsize=(max(6, len(timepoints) * 1.5),
                                     max(8, len(top_vars) * 0.4)))

    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0,
                   vmax=max(3, np.nanmax(matrix)), aspect="auto")

    ax.set_xticks(range(len(timepoints)))
    ax.set_xticklabels(timepoints, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(top_vars)))
    ax.set_yticklabels(top_vars, fontsize=8)

    # 标注
    for i in range(len(top_vars)):
        for j in range(len(timepoints)):
            val = matrix[i, j]
            or_v = or_matrix[i, j]
            p = 10 ** (-val) if val > 0 else 1

            if val > -np.log10(0.05):  # p < 0.05
                if np.isfinite(or_v):
                    direction = "↑" if or_v > 1 else "↓"
                    txt = f"{direction}\n{or_v:.2f}"
                else:
                    txt = "*"
                fc = "white" if val > 2 else "black"
                fw = "bold"
            else:
                txt = ""
                fc = "gray"
                fw = "normal"

            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=6, fontweight=fw, color=fc)

    plt.colorbar(im, ax=ax, label="-log₁₀(P)", shrink=0.8)

    # p=0.05 参考线标注
    ax.set_title("Longitudinal Trajectory: Variable Significance Across Timepoints\n"
                 "(↑ OR>1 risk | ↓ OR<1 protect | controlled for TLV+NIHSS)",
                 fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path)
    print(f"🎨 轨迹热图: {output_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# 12. 纵向 OR 变化折线图
# ═══════════════════════════════════════════════════════════════════
def plot_trajectory_lines(traj_df, output_path, top_n=8):
    """
    折线图: 追踪 top N 个变量的 OR 值随时间的变化
    """
    if traj_df is None or len(traj_df) == 0:
        return

    # 排除连续变量 (Recovery_Δ)
    sub = traj_df[traj_df["Time_Point"] != "Recovery_Δ"].copy()
    if len(sub) == 0:
        return

    timepoints = [tp for tp in ["D_MRS", "m3_mRS", "m6_mRS", "m12_mRS"]
                  if tp in sub["Time_Point"].unique()]
    if len(timepoints) < 2:
        return

    # 选最显著的变量
    best_p = sub.groupby("Variable")["P_value"].min().nsmallest(top_n)
    top_vars = best_p.index.tolist()

    fig, ax = plt.subplots(figsize=(10, 6))

    for var in top_vars:
        var_data = sub[sub["Variable"] == var]
        ors = []
        tps = []
        for tp in timepoints:
            row = var_data[var_data["Time_Point"] == tp]
            if len(row) > 0 and np.isfinite(row["OR"].values[0]):
                ors.append(row["OR"].values[0])
                tps.append(tp)

        if len(tps) >= 2:
            color = get_var_color(var)
            ax.plot(tps, ors, "o-", label=var, color=color,
                    linewidth=2, markersize=6)

    ax.axhline(1, color="red", linestyle="--", linewidth=1, alpha=0.7,
               label="OR = 1")
    ax.set_xlabel("Timepoint", fontsize=12)
    ax.set_ylabel("Odds Ratio (OR)", fontsize=12)
    ax.set_title("OR Trajectory Across Follow-up Timepoints\n"
                 "(Controlled for TLV + NIHSS)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path)
    print(f"🎨 轨迹折线图: {output_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# 13. 时间动态分类 (Temporal Dynamics Classifier)
# ═══════════════════════════════════════════════════════════════════
def classify_temporal_dynamics(traj_df, output_path=None):
    """
    将每个变量按其在不同时间点的显著性模式分类:
      - Persistent:  所有时间点均显著 → 核心预后因子
      - Early-only:  仅出院/3月显著 → 急性保护/损伤
      - Late-emerging: 仅6/12月显著 → 长期代偿/可塑性
      - Recovery-specific: 仅 Recovery_Δ 显著 → 纯康复潜力
      - Transient:   仅某一个时间点显著 → 需谨慎解读

    Returns
    -------
    dynamics_df : pd.DataFrame
    """
    if traj_df is None or len(traj_df) == 0:
        return pd.DataFrame()

    # 排除 Recovery_Δ 做时间分类 (单独处理)
    ordinal_tps = [tp for tp in ["D_MRS", "m3_mRS", "m6_mRS", "m12_mRS"]
                   if tp in traj_df["Time_Point"].unique()]
    early_tps = {"D_MRS", "m3_mRS"}
    late_tps = {"m6_mRS", "m12_mRS"}

    all_vars = traj_df["Variable"].unique()
    rows = []

    for var in all_vars:
        var_data = traj_df[traj_df["Variable"] == var]

        # 在哪些时间点显著
        sig_tps = set(
            var_data.loc[var_data["P_value"] < 0.05, "Time_Point"].values
        )
        sig_ordinal = sig_tps & set(ordinal_tps)
        sig_early = sig_ordinal & early_tps
        sig_late = sig_ordinal & late_tps
        has_recovery = "Recovery_Δ" in sig_tps

        # 分类
        if len(sig_ordinal) == len(ordinal_tps) and len(ordinal_tps) >= 2:
            pattern = "Persistent ★"
        elif sig_early and not sig_late:
            pattern = "Early-only"
        elif sig_late and not sig_early:
            pattern = "Late-emerging"
        elif len(sig_ordinal) == 1:
            pattern = "Transient"
        elif sig_early and sig_late:
            pattern = "Persistent ★"
        elif not sig_ordinal and has_recovery:
            pattern = "Recovery-specific"
        elif not sig_ordinal:
            pattern = "Non-significant"
        else:
            pattern = "Mixed"

        # 取最佳 OR 和 P
        best_row = var_data.loc[var_data["P_value"].idxmin()]

        rows.append({
            "Variable": var,
            "Category": best_row.get("Category", classify_variable(var)),
            "Pattern": pattern,
            "N_sig_timepoints": len(sig_ordinal),
            "Sig_at": ", ".join(sorted(sig_ordinal)),
            "Recovery_sig": has_recovery,
            "Best_OR": best_row.get("OR", np.nan),
            "Best_P": best_row["P_value"],
            "Best_timepoint": best_row["Time_Point"],
        })

    dynamics_df = pd.DataFrame(rows).sort_values(
        ["N_sig_timepoints", "Best_P"], ascending=[False, True]
    )

    # 打印摘要
    print(f"\n{'─' * 70}")
    print(f"  🕐 时间动态分类")
    print(f"{'─' * 70}")
    for pattern in ["Persistent ★", "Early-only", "Late-emerging",
                     "Recovery-specific", "Transient", "Non-significant"]:
        count = (dynamics_df["Pattern"] == pattern).sum()
        if count > 0:
            print(f"   {pattern:22s}: {count} 个变量")
            if pattern != "Non-significant":
                top = dynamics_df[dynamics_df["Pattern"] == pattern].head(5)
                for _, r in top.iterrows():
                    print(f"     {r['Variable']:25s} "
                          f"OR={r['Best_OR']:.3f}  P={r['Best_P']:.4e}  "
                          f"at [{r['Sig_at']}]")

    if output_path:
        dynamics_df.to_csv(output_path, index=False)
        print(f"\n💾 时间动态: {output_path}")

    return dynamics_df


# ═══════════════════════════════════════════════════════════════════
# 14. 交互作用分析 (NT × 炎症 → mRS)
# ═══════════════════════════════════════════════════════════════════
def interaction_scan(csv_path, control_cols=None, binary=False, min_n=30):
    """
    检验: mRS ~ TLV + NIHSS + NT + Inflam + NT × Inflam
    寻找 "协同打击" 效应

    Returns
    -------
    interact_df : pd.DataFrame
    """
    df = pd.read_csv(csv_path)

    # 识别 mRS
    mrs_col = None
    for c in ["m3_mRS", "m12_mRS", "m6_mRS", "D_MRS",
              "mRS", "mRS_90d", "mrs"]:
        if c in df.columns:
            mrs_col = c
            break
    if mrs_col is None:
        print("  ⚠️  无 mRS 列, 跳过交互分析")
        return pd.DataFrame()

    # 识别控制变量
    if control_cols is None:
        control_cols = []
        for c in ["TLV", "TLV_mm3"]:
            if c in df.columns:
                control_cols.append(c)
                break
        for c in ["A_NIHSS", "NIHSS"]:
            if c in df.columns:
                control_cols.append(c)
                break

    # 识别 NT 列
    nt_cols = []
    for col in df.columns:
        bare = col.replace("Load_", "")
        if bare in KNOWN_NT:
            nt_cols.append(col)
    if not nt_cols:
        nt_cols = [c for c in df.columns if c.startswith("Load_")]

    # 识别炎症列
    inflam_keywords = ["il6", "crp", "hscrp", "il10", "tnf", "nlr", "wbc",
                       "il1", "ferritin", "procalcitonin", "saa"]
    inflam_cols = []
    for c in df.columns:
        if any(k in c.lower() for k in inflam_keywords):
            inflam_cols.append(c)

    if not nt_cols or not inflam_cols:
        print(f"  ⚠️  NT 列: {len(nt_cols)}, 炎症列: {len(inflam_cols)} — 跳过交互")
        return pd.DataFrame()

    print(f"\n{'─' * 70}")
    print(f"  🔀 交互作用分析: {len(nt_cols)} NT × {len(inflam_cols)} 炎症指标")
    print(f"{'─' * 70}")

    try:
        import statsmodels.api as sm
    except ImportError:
        print("  ⚠️  需要 statsmodels")
        return pd.DataFrame()

    results = []
    for nt in nt_cols:
        nt_label = nt.replace("Load_", "")
        for inflam in inflam_cols:
            cols = [mrs_col] + control_cols + [nt, inflam]
            sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < min_n:
                continue

            # Z-score
            for col in control_cols + [nt, inflam]:
                s = sub[col].std()
                if s > 1e-10:
                    sub[col] = (sub[col] - sub[col].mean()) / s

            if sub[nt].std() < 1e-10 or sub[inflam].std() < 1e-10:
                continue

            sub["interact"] = sub[nt] * sub[inflam]

            try:
                X = sm.add_constant(sub[control_cols + [nt, inflam, "interact"]])
                res = sm.OLS(sub[mrs_col].astype(float), X).fit()

                results.append({
                    "NT": nt_label,
                    "Inflammation": inflam,
                    "Beta_NT": res.params[nt],
                    "P_NT": res.pvalues[nt],
                    "Beta_Inflam": res.params[inflam],
                    "P_Inflam": res.pvalues[inflam],
                    "Beta_Interaction": res.params["interact"],
                    "P_Interaction": res.pvalues["interact"],
                    "Direction": "Synergistic ↑" if res.params["interact"] > 0
                                 else "Buffering ↓",
                    "N": len(sub),
                    "R2": res.rsquared,
                })
            except Exception:
                pass

    interact_df = pd.DataFrame(results)

    if len(interact_df) > 0:
        interact_df = interact_df.sort_values("P_Interaction")

        # FDR
        try:
            from statsmodels.stats.multitest import multipletests
            _, q, _, _ = multipletests(
                interact_df["P_Interaction"].values, method="fdr_bh"
            )
            interact_df["FDR_q"] = q
        except ImportError:
            interact_df["FDR_q"] = (
                interact_df["P_Interaction"] * len(interact_df)
            )

        # 打印显著交互
        sig = interact_df[interact_df["P_Interaction"] < 0.05]
        if len(sig) > 0:
            print(f"\n  ✓ 发现 {len(sig)} 个显著交互 (P < 0.05):")
            for _, r in sig.head(10).iterrows():
                print(f"    {r['NT']:12s} × {r['Inflammation']:12s}  "
                      f"β={r['Beta_Interaction']:+.4f}  "
                      f"p={r['P_Interaction']:.4e}  {r['Direction']}")
        else:
            print("  ⚠️  未发现显著交互效应")
    else:
        print("  ⚠️  未能拟合任何交互模型")

    return interact_df


# ═══════════════════════════════════════════════════════════════════
# 15. 中介效应分析 (TLV → NT → mRS)
# ═══════════════════════════════════════════════════════════════════
def mediation_scan(csv_path, control_cols=None, n_boot=5000, min_n=30):
    """
    Bootstrap 中介效应:
      TLV → NT_residual → mRS
      路径 a: TLV → NT
      路径 b: NT → mRS (控制 TLV)
      间接效应 ab: 体积通过破坏 NT 影响预后的比例

    Returns
    -------
    med_df : pd.DataFrame
    """
    df = pd.read_csv(csv_path)

    mrs_col = None
    for c in ["m3_mRS", "m12_mRS", "m6_mRS", "D_MRS",
              "mRS", "mRS_90d"]:
        if c in df.columns:
            mrs_col = c
            break
    if mrs_col is None:
        print("  ⚠️  无 mRS 列, 跳过中介分析")
        return pd.DataFrame()

    tlv_col = None
    for c in ["TLV", "TLV_mm3"]:
        if c in df.columns:
            tlv_col = c
            break
    if tlv_col is None:
        print("  ⚠️  无 TLV 列, 跳过中介分析")
        return pd.DataFrame()

    # NT 列
    nt_cols = []
    for col in df.columns:
        bare = col.replace("Load_", "")
        if bare in KNOWN_NT:
            nt_cols.append(col)
    if not nt_cols:
        nt_cols = [c for c in df.columns if c.startswith("Load_")]

    if not nt_cols:
        print("  ⚠️  无 NT 列, 跳过中介分析")
        return pd.DataFrame()

    print(f"\n{'─' * 70}")
    print(f"  🔗 中介效应分析: TLV → NT → mRS ({n_boot} bootstraps)")
    print(f"{'─' * 70}")

    try:
        import statsmodels.api as sm
    except ImportError:
        print("  ⚠️  需要 statsmodels")
        return pd.DataFrame()

    covariates = []
    for c in ["Age", "Sex", "A_NIHSS", "NIHSS"]:
        if c in df.columns:
            covariates.append(c)

    results = []
    np.random.seed(42)

    for nt in nt_cols:
        nt_label = nt.replace("Load_", "")
        cols = [tlv_col, nt, mrs_col] + covariates
        sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()

        if len(sub) < min_n:
            continue

        X_treat = sub[tlv_col].values
        M = sub[nt].values
        Y = sub[mrs_col].values
        Z = sub[covariates].values if covariates else np.empty((len(sub), 0))

        # 原始估计
        try:
            # Path a: TLV → NT
            Xa = sm.add_constant(
                np.column_stack([X_treat, Z]) if Z.shape[1] > 0
                else X_treat
            )
            res_a = sm.OLS(M, Xa).fit()
            a_hat = res_a.params[1]

            # Path b + c': [TLV, NT] → mRS
            Xb = sm.add_constant(np.column_stack(
                [X_treat, M, Z] if Z.shape[1] > 0
                else [X_treat, M]
            ))
            res_b = sm.OLS(Y, Xb).fit()
            b_hat = res_b.params[2]
            c_prime = res_b.params[1]

            # Total c: TLV → mRS
            Xc = sm.add_constant(
                np.column_stack([X_treat, Z]) if Z.shape[1] > 0
                else X_treat
            )
            res_c = sm.OLS(Y, Xc).fit()
            c_hat = res_c.params[1]

            ab_hat = a_hat * b_hat

            # Bootstrap CI
            indirect_boots = []
            for _ in range(n_boot):
                idx = np.random.randint(0, len(sub), len(sub))
                X_b = X_treat[idx]
                M_b = M[idx]
                Y_b = Y[idx]
                Z_b = Z[idx] if Z.shape[1] > 0 else None

                try:
                    Xa_b = sm.add_constant(
                        np.column_stack([X_b, Z_b]) if Z_b is not None
                        else X_b
                    )
                    a_b = sm.OLS(M_b, Xa_b).fit().params[1]

                    Xb_b = sm.add_constant(np.column_stack(
                        [X_b, M_b, Z_b] if Z_b is not None
                        else [X_b, M_b]
                    ))
                    b_b = sm.OLS(Y_b, Xb_b).fit().params[2]

                    indirect_boots.append(a_b * b_b)
                except Exception:
                    pass

            indirect_boots = np.array(indirect_boots)
            if len(indirect_boots) < 100:
                continue

            ci_lo = np.percentile(indirect_boots, 2.5)
            ci_hi = np.percentile(indirect_boots, 97.5)
            prop_med = ab_hat / c_hat if abs(c_hat) > 1e-10 else np.nan
            significant = "Yes" if (ci_lo > 0 or ci_hi < 0) else "No"

            results.append({
                "Mediator": nt_label,
                "Outcome": mrs_col,
                "Path_a (TLV→NT)": a_hat,
                "Path_b (NT→mRS|TLV)": b_hat,
                "Indirect_ab": ab_hat,
                "CI_2.5%": ci_lo,
                "CI_97.5%": ci_hi,
                "Direct_c_prime": c_prime,
                "Total_c": c_hat,
                "Proportion_mediated": prop_med,
                "Significant": significant,
                "N": len(sub),
            })

            sig_mark = "✓" if significant == "Yes" else "✗"
            print(f"  {nt_label:16s}  ab={ab_hat:.4e}  "
                  f"CI=[{ci_lo:.4e}, {ci_hi:.4e}]  "
                  f"prop={prop_med:.1%}  {sig_mark}")

        except Exception as e:
            print(f"  ⚠️  {nt_label}: {e}")

    med_df = pd.DataFrame(results)
    return med_df


# ═══════════════════════════════════════════════════════════════════
# 16. 分类汇总条形图
# ═══════════════════════════════════════════════════════════════════
def plot_category_summary(result_table, output_path):
    """
    按变量类别 (NT/Inflam/HRV/...) 汇总显著变量数量
    """
    sig = result_table[result_table["P_value"] < 0.05]
    if len(sig) == 0:
        return

    cat_counts = sig["Category"].value_counts()
    total_counts = result_table["Category"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))

    cats = sorted(total_counts.index)
    x = range(len(cats))
    total_vals = [total_counts.get(c, 0) for c in cats]
    sig_vals = [cat_counts.get(c, 0) for c in cats]
    colors = [PREFIX_COLORS.get(c, "#7F7F7F") for c in cats]

    ax.bar(x, total_vals, color=colors, alpha=0.3, label="All tested")
    ax.bar(x, sig_vals, color=colors, alpha=0.9, label="p < 0.05")

    ax.set_xticks(list(x))
    ax.set_xticklabels(cats, fontsize=11)
    ax.set_ylabel("Number of Variables", fontsize=12)
    ax.set_title("Significant Variables by Category", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    for i, (t, s) in enumerate(zip(total_vals, sig_vals)):
        ax.text(i, t + 0.3, f"{s}/{t}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path)
    print(f"🎨 分类汇总: {output_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="mRS 有序逻辑回归 · 全变量深度扫描"
    )
    parser.add_argument("--input", "-i", type=str,
                        default="/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/variable_outcom_merge_data/merged_neuro_data.csv",
                        help="合并后 CSV")
    parser.add_argument("--binary", action="store_true",
                        help="二分类 (Good 0-2 vs Poor 3-6)")
    parser.add_argument("--nt-only", action="store_true",
                        help="仅扫描 17 条神经递质通路")
    parser.add_argument("--output-dir", "-o", type=str,
                        default="/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/ordinal_results",
                        help="输出目录")
    parser.add_argument("--min-n", type=int, default=30,
                        help="最小样本量 (默认 30)")
    parser.add_argument("--p-threshold", type=float, default=0.05,
                        help="多因素模型纳入阈值 (默认 0.05)")
    parser.add_argument("--longitudinal", action="store_true",
                        help="纵向多时间点轨迹分析 (D/m3/m6/m12_mRS + Recovery)")
    parser.add_argument("--no-residuals", action="store_true",
                        help="禁用 Koch 残差法 (不推荐! 仅用于对比验证)")
    args = parser.parse_args()

    use_residuals = not args.no_residuals

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════
    # 纵向模式: 多时间点扫描
    # ══════════════════════════════════════════════════════════════
    if args.longitudinal:
        traj_df = longitudinal_trajectory_scan(
            args.input, binary=args.binary,
            min_n=args.min_n, nt_only=args.nt_only
        )

        if len(traj_df) > 0:
            mode_str = "_binary" if args.binary else "_ordinal"
            scope_str = "_nt" if args.nt_only else "_all"
            suffix = f"_longitudinal{mode_str}{scope_str}"

            # 保存
            traj_df.to_csv(out_dir / f"Multi_Timepoint_Analysis{suffix}.csv",
                           index=False)
            print(f"\n💾 纵向结果: {out_dir / f'Multi_Timepoint_Analysis{suffix}.csv'}")

            # 每个时间点的显著数
            print(f"\n{'=' * 70}")
            print(f"  📅 各时间点显著变量数 (P < 0.05)")
            print(f"{'=' * 70}")
            for tp in traj_df["Time_Point"].unique():
                sub = traj_df[traj_df["Time_Point"] == tp]
                n_sig = (sub["P_value"] < 0.05).sum()
                n_fdr = (sub["FDR_q"] < 0.05).sum() if "FDR_q" in sub.columns else 0
                print(f"   {tp:15s}: {n_sig:3d} sig (FDR: {n_fdr})  "
                      f"[共 {len(sub)} 个变量]")

            # 显示各时间点 top 5
            for tp in traj_df["Time_Point"].unique():
                sub = traj_df[traj_df["Time_Point"] == tp].nsmallest(5, "P_value")
                print(f"\n  ── {tp} Top 5 ──")
                cols = ["Variable", "OR", "P_value"]
                if "FDR_q" in sub.columns:
                    cols.append("FDR_q")
                print(sub[cols].to_string(index=False))

            # 可视化
            plot_trajectory_heatmap(
                traj_df, out_dir / f"trajectory_heatmap{suffix}.png"
            )
            plot_trajectory_lines(
                traj_df, out_dir / f"trajectory_lines{suffix}.png"
            )

            # 找 "始终显著" 的变量 (在所有时间点 p<0.05)
            all_tps = traj_df["Time_Point"].nunique()
            if all_tps >= 2:
                sig_counts = (
                    traj_df[traj_df["P_value"] < 0.05]
                    .groupby("Variable")["Time_Point"].nunique()
                )
                always_sig = sig_counts[sig_counts == all_tps].index.tolist()
                if always_sig:
                    print(f"\n  🏆 始终显著的变量 (所有 {all_tps} 个时间点 P<0.05):")
                    for v in always_sig:
                        print(f"     {v}")
                else:
                    print(f"\n  ℹ️  没有在所有时间点都显著的变量")

            # 时间动态分类
            dynamics_df = classify_temporal_dynamics(
                traj_df, out_dir / f"Temporal_Dynamics{suffix}.csv"
            )

        # 交互作用分析
        print(f"\n{'=' * 70}")
        print(f"  🔀 交互作用分析")
        print(f"{'=' * 70}")
        interact_df = interaction_scan(
            args.input, binary=args.binary, min_n=args.min_n
        )
        if len(interact_df) > 0:
            interact_df.to_csv(out_dir / f"Interaction_NT_Inflam{suffix}.csv",
                               index=False)
            print(f"💾 交互分析: {out_dir / f'Interaction_NT_Inflam{suffix}.csv'}")

        # 中介效应分析
        print(f"\n{'=' * 70}")
        print(f"  🔗 中介效应分析")
        print(f"{'=' * 70}")
        med_df = mediation_scan(args.input, min_n=args.min_n)
        if len(med_df) > 0:
            med_df.to_csv(out_dir / f"Mediation_TLV_NT_mRS{suffix}.csv",
                          index=False)
            print(f"💾 中介分析: {out_dir / f'Mediation_TLV_NT_mRS{suffix}.csv'}")

            sig_med = med_df[med_df["Significant"] == "Yes"]
            if len(sig_med) > 0:
                print(f"\n  🏆 显著中介效应 ({len(sig_med)} 个):")
                for _, r in sig_med.iterrows():
                    print(f"     TLV → {r['Mediator']:12s} → mRS  "
                          f"prop={r['Proportion_mediated']:.1%}  "
                          f"CI=[{r['CI_2.5%']:.4e}, {r['CI_97.5%']:.4e}]")

        print(f"\n{'=' * 70}")
        print(f"  ✅ 纵向轨迹分析完成！")
        print(f"{'=' * 70}")
        return

    # ══════════════════════════════════════════════════════════════
    # 单时间点模式 (原有逻辑)
    # ══════════════════════════════════════════════════════════════

    # ── Step 1: 加载数据 ──
    df, mrs_col, control_cols, candidate_vars, group_labels, resid_map, resid_report = load_and_prepare(
        args.input, binary=args.binary, nt_only=args.nt_only,
        use_residuals=use_residuals
    )

    # 保存残差转换报告
    if resid_report:
        import pandas as _pd
        resid_df = _pd.DataFrame(resid_report)
        resid_df.to_csv(out_dir / "koch_residual_report.csv", index=False)
        print(f"\n💾 Koch 残差报告: {out_dir / 'koch_residual_report.csv'}")

    # ── Step 2: 深度扫描 ──
    result_table = deep_scan_regression(
        df, control_cols, candidate_vars,
        binary=args.binary, min_n=args.min_n
    )

    # 打印结果汇总
    sig_count = (result_table["P_value"] < 0.05).sum()
    fdr_count = (result_table["FDR_q"] < 0.05).sum()

    print(f"\n{'=' * 70}")
    print(f"  📋 扫描结果汇总")
    print(f"{'=' * 70}")
    print(f"  总变量数:       {len(result_table)}")
    print(f"  P < 0.05:       {sig_count}")
    print(f"  FDR q < 0.05:   {fdr_count}")

    ctrl_str = " + ".join(control_cols)
    print(f"\n  控制变量: {ctrl_str}")

    # 显示显著结果
    sig_table = result_table[result_table["P_value"] < 0.05]
    if len(sig_table) > 0:
        print(f"\n  ── 显著变量 (P < 0.05, 共 {len(sig_table)} 个) ──")
        display_cols = ["Variable", "Category", "OR", "CI_lower", "CI_upper",
                        "P_value", "FDR_q", "N", "Sig"]
        print(sig_table[display_cols].to_string(index=False))
    else:
        print("\n  ⚠️  未发现 P < 0.05 的变量")

    # 显示 top 20
    print(f"\n  ── Top 20 (按 P 排序) ──")
    display_cols = ["Variable", "Category", "OR", "CI_lower", "CI_upper",
                    "P_value", "FDR_q", "N", "Sig"]
    print(result_table.head(20)[display_cols].to_string(index=False))

    # ── Step 3: 敏感性分析 (仅对显著变量) ──
    sig_vars = result_table.loc[result_table["P_value"] < 0.1, "Variable"].tolist()
    sens_df = None
    if sig_vars and len(control_cols) >= 1:
        sens_df = sensitivity_analysis(
            df, control_cols, sig_vars,
            binary=args.binary, min_n=args.min_n
        )

    # ── Step 4: 共线性检查 (显著变量) ──
    sig_vars_05 = result_table.loc[result_table["P_value"] < 0.05, "Variable"].tolist()
    corr_matrix = None
    if len(sig_vars_05) >= 2:
        corr_matrix = collinearity_check(df, sig_vars_05)

    # ── Step 5: 多因素模型 ──
    multi_df = multivariable_model(
        df, control_cols, result_table,
        p_threshold=args.p_threshold,
        binary=args.binary
    )

    # ── Step 6: 交互 + 中介 ──
    mode_str = "_binary" if args.binary else "_ordinal"
    scope_str = "_nt" if args.nt_only else "_all"
    suffix = mode_str + scope_str

    interact_df = interaction_scan(args.input, min_n=args.min_n)
    if len(interact_df) > 0:
        interact_df.to_csv(out_dir / f"Interaction_NT_Inflam{suffix}.csv",
                           index=False)

    med_df_result = mediation_scan(args.input, min_n=args.min_n)
    if len(med_df_result) > 0:
        med_df_result.to_csv(out_dir / f"Mediation_TLV_NT_mRS{suffix}.csv",
                             index=False)

    # ── Step 7: 可视化 ──

    # 森林图 (显著变量)
    if len(sig_table) > 0:
        plot_forest(sig_table, out_dir / f"forest_significant{suffix}.png",
                    title_suffix=f" — {len(sig_table)} significant vars")

    # 森林图 (Top 20)
    plot_forest(result_table, out_dir / f"forest_top20{suffix}.png",
                title_suffix=" — Top 20 by P-value", max_show=20)

    # 火山图
    plot_volcano(result_table, out_dir / f"volcano{suffix}.png")

    # 箱线图
    top_6_vars = result_table.head(6)["Variable"].tolist()
    if top_6_vars:
        plot_group_boxplots(df, top_6_vars, group_labels,
                            out_dir / f"boxplots_top6{suffix}.png")

    # 相关热图
    plot_correlation_heatmap(corr_matrix, out_dir / f"correlation{suffix}.png")

    # 分类汇总
    plot_category_summary(result_table, out_dir / f"category_summary{suffix}.png")

    # 多因素模型森林图
    if multi_df is not None and len(multi_df) > 0:
        plot_forest(multi_df, out_dir / f"forest_multivariable{suffix}.png",
                    title_suffix=" (Multivariable Model)")

    # ── Step 7: 保存结果 ──
    result_table.to_csv(out_dir / f"Deep_Analysis_Result{suffix}.csv", index=False)
    print(f"\n💾 完整结果: {out_dir / f'Deep_Analysis_Result{suffix}.csv'}")

    if sig_table is not None and len(sig_table) > 0:
        sig_table.to_csv(out_dir / f"Significant_Hits{suffix}.csv", index=False)
        print(f"💾 显著变量: {out_dir / f'Significant_Hits{suffix}.csv'}")

    if sens_df is not None and len(sens_df) > 0:
        sens_df.to_csv(out_dir / f"Sensitivity_Analysis{suffix}.csv", index=False)
        print(f"💾 敏感性:   {out_dir / f'Sensitivity_Analysis{suffix}.csv'}")

    if multi_df is not None:
        multi_df.to_csv(out_dir / f"Multivariable_Model{suffix}.csv", index=False)
        print(f"💾 多因素:   {out_dir / f'Multivariable_Model{suffix}.csv'}")

    # ── 结果解读 ──
    print(f"\n{'=' * 70}")
    print("  📖 结果解读")
    print(f"{'=' * 70}")

    if resid_map:
        resid_note = (
            f"\n  ⚠️  重要: NT Load 变量已通过 Koch 残差法 (Load ~ TLV) 转换!\n"
            f"     原因: Load 与 TLV 极高共线 (r≈0.99, VIF>700)\n"
            f"     Resid_NT = Load_NT 中不能被体积解释的 '超额' 递质损伤\n"
            f"     残差与 TLV 正交 (r=0, VIF≈1), 可安全同时入模\n"
            f"     共 {len(resid_map)} 个变量已转换:\n"
        )
        for orig, resid in resid_map.items():
            resid_note += f"       {orig} → {resid}\n"
    else:
        resid_note = ""

    print(f"""
  本分析控制了 {ctrl_str} 后，逐一检验每个变量对 mRS 预后分组的独立效应。
{resid_note}
  OR > 1 且 P < 0.05 → 危险因素:
    该变量越高，mRS 分组越差 (预后更差)

  OR < 1 且 P < 0.05 → 保护因素:
    该变量越高，mRS 分组越好 (预后更好)

  FDR q < 0.05 → 校正多重比较后仍然显著:
    在 {len(result_table)} 个变量中，扣除随机显著的假阳性后仍然可靠

  敏感性分析结论:
    "独立效应 ✓" = 控制 TLV + NIHSS 后仍显著 → 有独立研究价值
    "被控制变量解释 ✗" = 控制后不再显著 → 效应由病灶体积或严重程度解释

  注意: 所有自变量已 Z-score 标准化，OR 表示每增加 1 SD 的效应
    """)

    print(f"{'=' * 70}")
    print(f"  ✅ 深度扫描完成！共 {sig_count} 个显著 (FDR 校正后 {fdr_count} 个)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
