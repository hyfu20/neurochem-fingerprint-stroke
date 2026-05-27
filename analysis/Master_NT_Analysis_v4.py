#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master NT Analysis v4 — 心脑轴 + 白质病变 + 异常亚组分析
=============================================================================
基于 Koch (2025, Brain) + Alves (2025, Nature Comm)

v3 → v4 新增 3 个模块:
  ★ Module 11: HRV 中介分析
     Resid_NT → log(RMSSD) → m12_mRS
     控制 TLV + NIHSS + Age + Sex + HRmean
     Bootstrap 5000 次, 先并行中介 (RMSSD / IL-6), 双显著再串联
     缺失偏倚检验: 有 Holter vs 无 Holter 基线对比

  ★ Module 12: WMH 交互效应
     Resid_NT × IMG_SVD_WMH → mRS (有序回归)
     WMH 100% 有数据, 无缺失
     验证白质病变是否放大递质损毁的预后效应

  ★ Module 13: 39% 异常组分析 (Small-Lesion Severe-Outcome Phenotype)
     筛选 TLV < Q1 且 mRS ≥ 3 的"小病灶重症"患者
     对比"小病灶好预后"(TLV < Q1 且 mRS ≤ 2) 组的 Resid_NT, RMSSD, IL-6
     用 Mann-Whitney U + Cohen's d + 有序回归

保留 v3 全部 Module 0-10 (含方法学修正 1-6)

关键变量:
  HOLTER_RMSSD (53%)  — 心率变异性 (副交感指标)
  HOLTER_HRmean (55%) — 平均心率
  BSL_IL6 (74%)       — 白介素-6
  BSL_hsCRP (72%)     — 高敏 C-反应蛋白
  IMG_SVD_WMH (100%)  — 白质高信号体积
  m12_mRS (97%)       — 12 月改良 Rankin

数据路径:
  /data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/
  3.variable_outcom_merge_data/merged_neuro_data.csv

推荐运行:
  python3 Master_NT_Analysis_v4.py --input merged_neuro_data.csv --skip-perm
  python3 Master_NT_Analysis_v4.py --input merged_neuro_data.csv --n-perm 1000

后台运行:
  nohup python3 Master_NT_Analysis_v4.py --input merged_neuro_data.csv \
        --n-perm 1000 > analysis_v4.log 2>&1 &

依赖:
  pip install pandas numpy statsmodels scipy matplotlib seaborn openpyxl scikit-learn
  可选: pip install tqdm
"""

import argparse
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from functools import wraps

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.multitest import multipletests

try:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("MasterNT_v4")

plt.rcParams.update({
    "font.sans-serif": ["Arial", "DejaVu Sans", "SimHei", "Arial Unicode MS"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

# ==============================================================================
# 常量
# ==============================================================================
KNOWN_NT = [
    "5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT",
    "A4B2", "D1", "D2", "DAT", "M1", "NAT", "VAChT",
    "human_CHA", "JHU_EC", "Lateral_Path", "Medial_Path",
]

PRE_SYNAPTIC = {"DAT", "NAT", "5HTT", "VAChT"}
POST_SYNAPTIC = {"A4B2", "M1", "5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "D1", "D2"}

NT_SYSTEMS = {
    "Dopaminergic":      ["DAT", "D1", "D2"],
    "Serotonergic":      ["5HTT", "5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6"],
    "Cholinergic":       ["VAChT", "human_CHA", "A4B2", "M1"],
    "Noradrenergic":     ["NAT"],
    "Cholinergic_Tract": ["JHU_EC", "Lateral_Path", "Medial_Path"],
}

NT_COLORS = {
    "DAT": "#E64B35", "D1": "#E64B35", "D2": "#DC7C6B",
    "5HT1a": "#4DBBD5", "5HT1b": "#4DBBD5", "5HT2a": "#7DCDE5",
    "5HT4": "#A8DFF0", "5HT6": "#B0E0F0", "5HTT": "#3B9FC4",
    "A4B2": "#8491B4", "M1": "#F39B7F",
    "NAT": "#91D1C2", "VAChT": "#00A087",
    "human_CHA": "#2E8B57", "JHU_EC": "#6B8E23",
    "Lateral_Path": "#3CB371", "Medial_Path": "#228B22",
}

SYSTEM_COLORS = {
    "Dopaminergic": "#E64B35", "Serotonergic": "#4DBBD5",
    "Cholinergic": "#00A087", "Noradrenergic": "#91D1C2",
    "Cholinergic_Tract": "#228B22",
}


# ==============================================================================
# 工具函数
# ==============================================================================
def safe_module(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log.error(f"[{func.__name__}] 失败: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    return wrapper


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def get_color(name):
    for key, color in NT_COLORS.items():
        if key.lower() in name.lower():
            return color
    return "#7F7F7F"


def bare_name(col):
    return col.replace("Load_", "").replace("Resid_", "")


def fdr_correct(pvals):
    """BH-FDR, 返回 q-values"""
    p = np.asarray(pvals, dtype=float)
    valid = np.isfinite(p) & (p >= 0)
    q = np.full_like(p, np.nan, dtype=float)
    if valid.sum() == 0:
        return q
    _, q_vals, _, _ = multipletests(p[valid], method="fdr_bh")
    q[valid] = q_vals
    return q


def zscore(series):
    """Z-score 标准化"""
    s = series.std()
    if s > 1e-10:
        return (series - series.mean()) / s
    return series - series.mean()


def group_mrs(x):
    if pd.isna(x):
        return np.nan
    x = float(x)
    return 0 if x <= 2 else (1 if x <= 4 else 2)


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def classify_synaptic(nt):
    bare = bare_name(nt)
    if bare in PRE_SYNAPTIC:
        return "Pre-synaptic (Transporter)"
    elif bare in POST_SYNAPTIC:
        return "Post-synaptic (Receptor)"
    return "Tract / Other"


def get_system(nt):
    bare = bare_name(nt)
    for s, members in NT_SYSTEMS.items():
        if bare in members:
            return s
    return "Other"


def cohens_d(group1, group2):
    """计算 Cohen's d 效应量"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (group1.mean() - group2.mean()) / pooled_std


# ==============================================================================
# 全局 p 值收集器 (修正 6: 全局 FDR)
# ==============================================================================
class GlobalPCollector:
    """收集全流程所有 p 值, 最后统一做一次 BH-FDR 校正"""

    def __init__(self):
        self.records = []

    def add(self, module, label, p):
        if np.isfinite(p) and p >= 0:
            self.records.append((module, label, float(p)))

    def add_batch(self, module, labels, pvals):
        for label, p in zip(labels, pvals):
            self.add(module, label, p)

    def correct(self):
        if not self.records:
            return pd.DataFrame(columns=["Module", "Label", "P_raw", "Q_global"])
        df = pd.DataFrame(self.records, columns=["Module", "Label", "P_raw"])
        df["Q_global"] = fdr_correct(df["P_raw"].values)
        return df.sort_values("P_raw")


# ==============================================================================
# 数据加载 (v4 扩展: HRV + WMH 列)
# ==============================================================================
def load_data(csv_path):
    df = pd.read_csv(csv_path)

    print(f"\n{'=' * 72}")
    print(f"  Master NT Analysis v4 — 心脑轴 + 白质病变 + 异常亚组")
    print(f"  Koch (2025, Brain) + Alves (2025, Nature Comm)")
    print(f"{'=' * 72}")
    log.info(f"数据: {csv_path} ({df.shape[0]} × {df.shape[1]})")

    tlv = find_col(df, ["TLV", "TLV_mm3", "tlv"])
    nihss = find_col(df, ["A_NIHSS", "NIHSS", "nihss"])
    age = find_col(df, ["AGE", "Age", "age"])
    sex = find_col(df, ["SEX", "Sex", "sex"])
    cst = find_col(df, ["CST_Load", "CST_load", "cst_load", "CST_Load_mm3"])

    covariates_demo = [c for c in [age, sex] if c]
    covariates_clinical = [c for c in [tlv, nihss, cst] if c]
    covariates_all = covariates_clinical + covariates_demo

    for c in covariates_all:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if cst:
        n_cst = df[cst].notna().sum()
        log.info(f"  CST_Load 列: {cst} — 可用 {n_cst}/{len(df)} ({n_cst/len(df)*100:.1f}%)")
    else:
        log.warning("  ⚠️ 未找到 CST_Load 列, Model C 将不含运动束控制变量")

    # ── NT 载荷列 ──
    nt_cols = [c for c in df.columns if c.startswith("Load_")]
    if not nt_cols:
        nt_cols = [c for c in df.columns if c in KNOWN_NT]
    if not nt_cols:
        nt_cols = [c for c in df.columns
                   if any(x in c for x in
                          ["5HT", "DAT", "NAT", "VAChT", "A4B2",
                           "CHA", "Path", "D1", "D2", "M1", "JHU"])]
    for c in nt_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ── mRS ──
    mrs_candidates = ["m3_mRS", "m6_mRS", "m12_mRS", "D_MRS", "mRS", "mRS_90d"]
    mrs_found = [c for c in mrs_candidates if c in df.columns]
    for c in mrs_found:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ── 复发 ──
    recur = [c for c in ["y1_is_dd", "y1_stroke_dd", "m6_is_dd"] if c in df.columns]

    # ── 炎症 ──
    inflam = [c for c in ["BSL_IL6", "IL6", "CRP", "hsCRP", "BSL_hsCRP",
                           "IL10", "NLR", "WBC"]
              if c in df.columns]
    for c in inflam:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ══════════════════════════════════════════════════════════════
    # v4 新增: HRV 相关列
    # ══════════════════════════════════════════════════════════════
    rmssd_col = find_col(df, ["HOLTER_RMSSD", "RMSSD", "rmssd",
                               "Holter_RMSSD", "holter_rmssd"])
    hrmean_col = find_col(df, ["HOLTER_HRmean", "HRmean", "HR_mean",
                                "Holter_HRmean", "holter_hrmean"])
    if rmssd_col:
        df[rmssd_col] = pd.to_numeric(df[rmssd_col], errors="coerce")
    if hrmean_col:
        df[hrmean_col] = pd.to_numeric(df[hrmean_col], errors="coerce")

    # BSL_IL6 单独标记 (中介分析需要)
    il6_col = find_col(df, ["BSL_IL6", "IL6", "il6"])
    if il6_col:
        df[il6_col] = pd.to_numeric(df[il6_col], errors="coerce")

    # ══════════════════════════════════════════════════════════════
    # v4 新增: WMH 列
    # ══════════════════════════════════════════════════════════════
    wmh_col = find_col(df, ["IMG_SVD_WMH", "WMH", "wmh", "SVD_WMH"])
    if wmh_col:
        df[wmh_col] = pd.to_numeric(df[wmh_col], errors="coerce")

    # ══════════════════════════════════════════════════════════════
    # v4 新增: AF 列 (BSL_AF 不存在, 搜索替代列名)
    # ══════════════════════════════════════════════════════════════
    af_candidates = ["BSL_AF", "H_AF01", "H_AF", "AF", "AF_history",
                     "H_AFIB", "AFIB", "Afib", "atrial_fib"]
    af_col = find_col(df, af_candidates)
    if af_col is None:
        # 模糊搜索
        af_fuzzy = [c for c in df.columns if "AF" in c.upper() and "AFIB" not in c.upper()]
        if not af_fuzzy:
            af_fuzzy = [c for c in df.columns if "AF" in c.upper()]
        if af_fuzzy:
            af_col = af_fuzzy[0]
            log.info(f"  ⚠️ BSL_AF 不存在, 使用模糊匹配: {af_col}")
    if af_col:
        df[af_col] = pd.to_numeric(df[af_col], errors="coerce")

    log.info(f"TLV: {tlv}, NIHSS: {nihss}, Age: {age}, Sex: {sex}")
    log.info(f"NT: {len(nt_cols)} 条, mRS: {mrs_found}")
    log.info(f"炎症: {inflam}, 复发: {recur}")
    log.info(f"RMSSD: {rmssd_col}, HRmean: {hrmean_col}, WMH: {wmh_col}, AF: {af_col}")

    # ══════════════════════════════════════════════════════════════
    # Koch 残差法: 解决 Load vs TLV 极端共线性
    # ══════════════════════════════════════════════════════════════
    resid_cols = []
    resid_map = {}
    resid_report = []

    if tlv and nt_cols:
        log.info(f"")
        log.info(f"{'=' * 60}")
        log.info(f"  Koch 残差法: 对 {len(nt_cols)} 个 NT Load 回归掉 TLV")
        log.info(f"  模型: Load_NT = β0 + β1×TLV + ε")
        log.info(f"  Resid_NT (ε) = 不成比例递质损伤")
        log.info(f"{'=' * 60}")

        tlv_vals = df[tlv].values.astype(float)

        for nt in nt_cols:
            nt_name = bare_name(nt)
            y = df[nt].values.astype(float)
            valid = np.isfinite(tlv_vals) & np.isfinite(y)

            if valid.sum() < 20:
                continue

            slope, intercept, r_value, p_value, std_err = stats.linregress(
                tlv_vals[valid], y[valid]
            )
            predicted = intercept + slope * tlv_vals
            resid = y - predicted
            resid[~valid] = np.nan

            resid_col = f"Resid_{nt_name}"
            df[resid_col] = resid
            resid_cols.append(resid_col)
            resid_map[nt] = resid_col

            r2 = r_value ** 2
            resid_report.append({
                "Original": nt,
                "Residual": resid_col,
                "R2_Load_vs_TLV": r2,
                "Beta_TLV": slope,
                "P_value": p_value,
            })

            flag = " ⚠️ VIF≈" + f"{1/(1-r2):.0f}" if r2 > 0.8 else ""
            log.info(f"  {nt_name:<18s} R²(TLV)={r2:.4f}{flag} → {resid_col}")

        n_high = sum(1 for r in resid_report if r["R2_Load_vs_TLV"] > 0.5)
        log.info(f"  ✔ {len(resid_cols)} 个变量已计算残差, {n_high} 个与 TLV 高度共线")

    # ── v4 HRV 数据质量报告 ──
    if rmssd_col:
        n_rmssd = df[rmssd_col].notna().sum()
        pct_rmssd = n_rmssd / len(df) * 100
        log.info(f"  RMSSD 可用: {n_rmssd}/{len(df)} ({pct_rmssd:.1f}%)")
    if hrmean_col:
        n_hr = df[hrmean_col].notna().sum()
        log.info(f"  HRmean 可用: {n_hr}/{len(df)} ({n_hr/len(df)*100:.1f}%)")
    if wmh_col:
        n_wmh = df[wmh_col].notna().sum()
        log.info(f"  WMH 可用: {n_wmh}/{len(df)} ({n_wmh/len(df)*100:.1f}%)")

    meta = {
        "tlv": tlv, "nihss": nihss, "age": age, "sex": sex,
        "cst": cst,
        "covariates_all": covariates_all,
        "covariates_clinical": covariates_clinical,
        "covariates_demo": covariates_demo,
        "nt_cols": nt_cols, "mrs": mrs_found,
        "recurrence": recur, "inflammation": inflam,
        "resid_cols": resid_cols,
        "resid_map": resid_map,
        "resid_report": resid_report,
        # v4 新增
        "rmssd": rmssd_col,
        "hrmean": hrmean_col,
        "il6": il6_col,
        "wmh": wmh_col,
        "af": af_col,
    }
    return df, meta


# ==============================================================================
# Module 0: 多重插补 (MICE)
# ==============================================================================
@safe_module
def multiple_imputation(df, meta, n_imputations=5):
    print(f"\n{'─' * 72}")
    print(f"  [Module 0] 多重插补 (MICE, {n_imputations} 轮)")
    print(f"{'─' * 72}")

    if not HAS_SKLEARN:
        log.warning("需要 scikit-learn, 跳过插补")
        return df

    impute_cols = meta["nt_cols"] + meta["covariates_all"] + meta["inflammation"]
    # v4: 加入 HRV + WMH 列到插补中
    for extra in [meta["rmssd"], meta["hrmean"], meta["wmh"]]:
        if extra and extra not in impute_cols:
            impute_cols.append(extra)
    impute_cols = [c for c in impute_cols if c in df.columns]

    if not impute_cols:
        return df

    sub = df[impute_cols].copy()
    n_missing = sub.isnull().sum().sum()
    n_total = sub.shape[0] * sub.shape[1]
    miss_rate = n_missing / n_total if n_total > 0 else 0

    log.info(f"  缺失率: {miss_rate:.1%} ({n_missing}/{n_total})")

    if miss_rate < 0.01:
        log.info(f"  缺失率 < 1%, 无需插补")
        return df
    if miss_rate > 0.5:
        log.warning(f"  缺失率 > 50%, 插补不可靠, 跳过")
        return df

    try:
        imputer = IterativeImputer(
            max_iter=20, random_state=42, n_nearest_features=10,
            sample_posterior=True
        )
        imputed_data = imputer.fit_transform(sub)
        df_imputed = df.copy()
        df_imputed[impute_cols] = imputed_data

        n_filled = sub.isnull().sum().sum()
        log.info(f"  ✓ 插补完成, 填充 {n_filled} 个缺失值")
        return df_imputed
    except Exception as e:
        log.warning(f"  插补失败: {e}, 使用原始数据")
        return df


# ==============================================================================
# Module 1: 诊断残差 (Koch Fig.1E, 仅可视化)
# ==============================================================================
@safe_module
def diagnostic_residuals(df, meta):
    print(f"\n{'─' * 72}")
    print(f"  [Module 1] 诊断残差 (Koch Fig.1E, 仅可视化用)")
    print(f"{'─' * 72}")

    tlv = meta["tlv"]
    if not tlv:
        log.warning("无 TLV 列, 跳过诊断残差")
        return pd.DataFrame()

    rows = []
    for nt in meta["nt_cols"]:
        nt_name = bare_name(nt)
        valid = df[[nt, tlv]].dropna()
        if len(valid) < 20:
            continue
        X = sm.add_constant(valid[[tlv]])
        model = sm.OLS(valid[nt], X).fit()
        rows.append({
            "NT": nt_name, "N": len(valid),
            "R2_TLV": model.rsquared,
            "Beta_TLV": model.params[tlv],
            "P_TLV": model.pvalues[tlv],
        })
        log.info(f"  {nt_name}: R²(TLV)={model.rsquared:.4f}")
    return pd.DataFrame(rows)


# ==============================================================================
# Module 2: 直接有序回归 + Partial R² (修正 1 + 修正 3)
# ==============================================================================
@safe_module
def direct_ordinal_regression(df, meta, p_collector):
    print(f"\n{'─' * 72}")
    print(f"  [Module 2] 有序回归 + Koch 残差法 (三层敏感性)")
    print(f"{'─' * 72}")

    nt_cols = meta["nt_cols"]
    mrs_list = meta["mrs"]
    resid_map = meta.get("resid_map", {})

    if not nt_cols or not mrs_list:
        log.warning("无 NT 或 mRS 列")
        return pd.DataFrame()

    sens_levels = {"A_Unadjusted": ([], False)}
    if meta["covariates_demo"]:
        sens_levels["B_Demographic"] = (meta["covariates_demo"], False)
    if meta["covariates_all"]:
        sens_levels["C_Full"] = (meta["covariates_all"], True)

    results = []
    for outcome in mrs_list:
        target = f"_target_{outcome}"
        df[target] = df[outcome].apply(group_mrs)
        n_valid = df[target].notna().sum()
        if n_valid < 30:
            continue

        n_tested = 0
        for nt in nt_cols:
            nt_name = bare_name(nt)
            for model_label, (covars, use_resid) in sens_levels.items():
                if use_resid and nt in resid_map:
                    nt_var = resid_map[nt]
                else:
                    nt_var = nt

                predictors = [nt_var] + covars
                sub = df[[target] + predictors].dropna()
                if len(sub) < 30:
                    continue

                sub_z = sub.copy()
                for p in predictors:
                    sub_z[p] = zscore(sub_z[p])

                try:
                    mod_full = OrderedModel(sub_z[target], sub_z[predictors], distr="logit")
                    res_full = mod_full.fit(method="bfgs", disp=False)
                    beta = res_full.params[nt_var]
                    pval = res_full.pvalues[nt_var]
                    ci = res_full.conf_int().loc[nt_var]
                    se = res_full.bse.get(nt_var, np.nan)

                    partial_r2 = np.nan
                    lr_p = np.nan
                    if covars:
                        try:
                            mod_red = OrderedModel(sub_z[target], sub_z[covars], distr="logit")
                            res_red = mod_red.fit(method="bfgs", disp=False)
                            lr_stat = 2 * (res_full.llf - res_red.llf)
                            lr_p = 1 - stats.chi2.cdf(lr_stat, df=1)
                            partial_r2 = 1 - (res_full.llf / res_red.llf) if res_red.llf < 0 else np.nan
                        except Exception:
                            pass

                    row = {
                        "Outcome": outcome, "NT_Variable": nt_name,
                        "NT_Column": nt, "Predictor_Used": nt_var,
                        "Used_Residual": use_resid and nt in resid_map,
                        "Model": model_label, "Beta": beta, "SE": se,
                        "OR": np.exp(beta),
                        "OR_CI_lower": np.exp(ci[0]),
                        "OR_CI_upper": np.exp(ci[1]),
                        "P_value": pval, "Partial_R2": partial_r2,
                        "LR_test_P": lr_p,
                        "Pseudo_R2": getattr(res_full, "prsquared", np.nan),
                        "AIC": res_full.aic, "BIC": res_full.bic,
                        "N": len(sub),
                        "Synaptic": classify_synaptic(nt_name),
                        "System": get_system(nt_name),
                    }
                    results.append(row)
                    n_tested += 1
                    p_collector.add("Regression", f"{outcome}|{nt_name}|{model_label}", pval)
                except Exception:
                    continue
        log.info(f"✓ {outcome}: {n_tested} 次回归")

    rdf = pd.DataFrame(results)
    if rdf.empty:
        return rdf

    for ml in sens_levels:
        mask = rdf["Model"] == ml
        if mask.sum() > 0:
            rdf.loc[mask, "FDR_q"] = fdr_correct(rdf.loc[mask, "P_value"].values)

    rdf = rdf.sort_values(["Outcome", "Model", "P_value"])

    consistency = {}
    for (outcome, nt), grp in rdf.groupby(["Outcome", "NT_Variable"]):
        sig_models = grp[grp["P_value"] < 0.05]["Model"].tolist()
        n_sig = len(sig_models)
        n_total = len(grp)
        if n_sig == n_total and n_total == len(sens_levels):
            tag = "★★★ Robust"
        elif n_sig >= 2:
            tag = "★★ Consistent"
        elif n_sig >= 1:
            tag = "★ Marginal"
        else:
            tag = "— NS"
        for idx in grp.index:
            consistency[idx] = tag
    rdf["Sensitivity"] = rdf.index.map(consistency)
    return rdf


# ==============================================================================
# Module 3: Alves 突触定位
# ==============================================================================
@safe_module
def synaptic_analysis(rdf):
    print(f"\n{'─' * 72}")
    print(f"  [Module 3] Alves 突触定位 (Pre vs Post)")
    print(f"{'─' * 72}")

    if rdf.empty:
        return pd.DataFrame()

    if "Model" in rdf.columns:
        full = sorted(rdf["Model"].unique())[-1]
        sub = rdf[rdf["Model"] == full].copy()
    else:
        sub = rdf.copy()

    sub["Synaptic_Type"] = sub["NT_Variable"].apply(classify_synaptic)

    summary = sub.groupby(["Outcome", "Synaptic_Type"]).agg(
        Mean_AbsBeta=("Beta", lambda x: np.abs(x).mean()),
        Median_P=("P_value", "median"),
        Mean_OR=("OR", "mean"),
        N_sig=("P_value", lambda x: (x < 0.05).sum()),
        N_total=("P_value", "count"),
    ).reset_index()

    for outcome in sub["Outcome"].unique():
        osub = sub[sub["Outcome"] == outcome]
        pre_b = osub[osub["Synaptic_Type"].str.contains("Pre")]["Beta"].abs()
        post_b = osub[osub["Synaptic_Type"].str.contains("Post")]["Beta"].abs()
        if len(pre_b) >= 2 and len(post_b) >= 2:
            _, u_p = stats.mannwhitneyu(pre_b, post_b, alternative="two-sided")
            log.info(f"  {outcome}: Pre vs Post U-test p={u_p:.4f}")

    for _, r in summary.iterrows():
        log.info(f"  {r['Outcome']} | {r['Synaptic_Type']}: "
                 f"sig={r['N_sig']}/{r['N_total']}, |β|={r['Mean_AbsBeta']:.3f}")
    return summary


# ==============================================================================
# Module 4: 炎症-递质交互 (OrderedModel)
# ==============================================================================
@safe_module
def interaction_analysis(df, meta, p_collector):
    print(f"\n{'─' * 72}")
    print(f"  [Module 4] 炎症交互 (OrderedModel + Koch残差)")
    print(f"{'─' * 72}")

    nt_cols = meta["nt_cols"]
    inflam_found = meta["inflammation"]
    covars = meta["covariates_all"]
    resid_map = meta.get("resid_map", {})
    tlv_col = meta.get("tlv")
    has_tlv_in_covars = tlv_col and tlv_col in covars

    if not inflam_found:
        log.warning("无炎症指标")
        return pd.DataFrame()

    outcome_col = find_col(df, ["m12_mRS", "m6_mRS", "m3_mRS", "mRS", "D_MRS"])
    if not outcome_col:
        return pd.DataFrame()

    target = f"_inter_target_{outcome_col}"
    df[target] = df[outcome_col].apply(group_mrs)

    results = []
    for inflam in inflam_found:
        for nt in nt_cols:
            nt_name = bare_name(nt)
            nt_var = resid_map.get(nt, nt) if has_tlv_in_covars else nt
            predictors = [nt_var, inflam] + covars
            predictors = list(dict.fromkeys(predictors))

            sub = df[[target] + predictors].dropna()
            if len(sub) < 40:
                continue

            sub_z = sub.copy()
            for p in predictors:
                sub_z[p] = zscore(sub_z[p])
            sub_z["Interaction"] = sub_z[nt_var] * sub_z[inflam]

            try:
                all_pred = predictors + ["Interaction"]
                mod = OrderedModel(sub_z[target], sub_z[all_pred], distr="logit")
                res = mod.fit(method="bfgs", disp=False)
                inter_p = res.pvalues["Interaction"]
                results.append({
                    "NT": nt_name, "Inflam": inflam, "Outcome": outcome_col,
                    "Interaction_Beta": res.params["Interaction"],
                    "Interaction_SE": res.bse["Interaction"],
                    "Interaction_OR": np.exp(res.params["Interaction"]),
                    "Interaction_P": inter_p,
                    "NT_Beta": res.params[nt_var],
                    "NT_P": res.pvalues[nt_var],
                    "Used_Residual": nt_var != nt,
                    "Pseudo_R2": getattr(res, "prsquared", np.nan),
                    "N": len(sub),
                })
                p_collector.add("Interaction", f"{nt_name}×{inflam}", inter_p)
            except Exception:
                continue

    rdf = pd.DataFrame(results)
    if not rdf.empty:
        rdf["Interaction_FDR_q"] = fdr_correct(rdf["Interaction_P"].values)
        rdf = rdf.sort_values("Interaction_P")
        n_sig = (rdf["Interaction_P"] < 0.05).sum()
        log.info(f"→ {len(rdf)} 对测试, p<0.05: {n_sig}")
    return rdf


# ==============================================================================
# Module 5: 复发关联
# ==============================================================================
@safe_module
def recurrence_analysis(df, meta, p_collector):
    print(f"\n{'─' * 72}")
    print(f"  [Module 5] 复发关联")
    print(f"{'─' * 72}")

    nt_cols = meta["nt_cols"]
    recur = meta["recurrence"]
    covars = meta["covariates_all"]
    resid_map = meta.get("resid_map", {})
    tlv_col = meta.get("tlv")
    has_tlv_in_covars = tlv_col and tlv_col in covars

    if not recur:
        log.warning("无复发列")
        return pd.DataFrame()

    results = []
    for event_col in recur:
        df[event_col] = pd.to_numeric(df[event_col], errors="coerce")
        for nt in nt_cols:
            nt_name = bare_name(nt)
            nt_var = resid_map.get(nt, nt) if has_tlv_in_covars else nt
            predictors = [nt_var] + covars
            sub = df[[event_col] + predictors].dropna()
            if len(sub) < 20:
                continue

            sub_z = sub.copy()
            for p in predictors:
                sub_z[p] = zscore(sub_z[p])

            y = sub_z[event_col].astype(float)
            X = sm.add_constant(sub_z[predictors])

            try:
                unique_vals = set(y.unique())
                if unique_vals.issubset({0, 1, 0.0, 1.0}):
                    model = sm.Logit(y, X).fit(disp=False)
                    mtype = "Logistic"
                    or_val = np.exp(model.params[nt_var])
                else:
                    model = sm.OLS(y, X).fit()
                    mtype = "OLS"
                    or_val = np.nan

                pval = model.pvalues[nt_var]
                results.append({
                    "Event": event_col, "NT": nt_name, "Model_Type": mtype,
                    "Beta": model.params[nt_var], "SE": model.bse[nt_var],
                    "OR": or_val, "P_value": pval, "N": len(sub),
                })
                p_collector.add("Recurrence", f"{event_col}|{nt_name}", pval)
            except Exception:
                continue

    rdf = pd.DataFrame(results)
    if not rdf.empty:
        rdf["FDR_q"] = fdr_correct(rdf["P_value"].values)
        rdf = rdf.sort_values(["Event", "P_value"])
    return rdf


# ==============================================================================
# Module 6: PCA 系统级指纹 (消除抵消效应)
# ==============================================================================
@safe_module
def pca_system_fingerprint(df, meta, p_collector):
    print(f"\n{'─' * 72}")
    print(f"  [Module 6] PCA 系统级指纹 (消除抵消效应)")
    print(f"{'─' * 72}")

    from sklearn.decomposition import PCA as skPCA

    nt_cols = meta["nt_cols"]
    mrs_list = meta["mrs"]
    covars = meta["covariates_all"]
    resid_map = meta.get("resid_map", {})
    tlv_col = meta.get("tlv")
    has_tlv_in_covars = tlv_col and tlv_col in covars

    if not mrs_list:
        return pd.DataFrame()

    results = []
    pca_loadings_all = []

    for sys_name, members in NT_SYSTEMS.items():
        sys_cols = [nt for nt in nt_cols if bare_name(nt) in members]
        if len(sys_cols) < 2:
            continue

        if has_tlv_in_covars:
            pca_cols = [resid_map.get(nt, nt) for nt in sys_cols]
        else:
            pca_cols = sys_cols
        pca_cols = [c for c in pca_cols if c in df.columns]
        if len(pca_cols) < 2:
            continue

        sub = df[pca_cols].dropna()
        if len(sub) < 30:
            continue

        sub_z = sub.copy()
        for c in pca_cols:
            sub_z[c] = zscore(sub_z[c])

        pca = skPCA(n_components=1)
        pc1 = pca.fit_transform(sub_z)[:, 0]

        total_load = sub_z[pca_cols].sum(axis=1)
        if np.corrcoef(pc1, total_load)[0, 1] < 0:
            pc1 = -pc1
            pca.components_[0] = -pca.components_[0]

        for i, col in enumerate(pca_cols):
            pca_loadings_all.append({
                "System": sys_name, "NT": bare_name(col),
                "PC1_Loading": pca.components_[0, i],
                "Explained_Var": pca.explained_variance_ratio_[0],
                "Used_Residual": col.startswith("Resid_"),
            })

        pc_col = f"PC1_{sys_name}"
        df[pc_col] = np.nan
        df.loc[sub.index, pc_col] = pc1
        log.info(f"  {sys_name}: PC1 解释方差 {pca.explained_variance_ratio_[0]:.1%}, N={len(sub)}")

        for outcome in mrs_list:
            target = f"_target_{outcome}"
            if target not in df.columns:
                df[target] = df[outcome].apply(group_mrs)

            predictors = [pc_col] + covars
            osub = df[[target] + predictors].dropna()
            if len(osub) < 30:
                continue

            osub_z = osub.copy()
            for p in predictors:
                osub_z[p] = zscore(osub_z[p])

            try:
                mod = OrderedModel(osub_z[target], osub_z[predictors], distr="logit")
                res = mod.fit(method="bfgs", disp=False)
                pval = res.pvalues[pc_col]
                results.append({
                    "Outcome": outcome, "System": sys_name,
                    "PC1_Beta": res.params[pc_col],
                    "PC1_OR": np.exp(res.params[pc_col]),
                    "PC1_P": pval,
                    "Explained_Var": pca.explained_variance_ratio_[0],
                    "N": len(osub),
                })
                p_collector.add("PCA_System", f"{outcome}|{sys_name}", pval)
            except Exception:
                continue

    rdf = pd.DataFrame(results)
    loadings_df = pd.DataFrame(pca_loadings_all)

    if not rdf.empty:
        rdf["FDR_q"] = fdr_correct(rdf["PC1_P"].values)
        rdf = rdf.sort_values("PC1_P")
        for _, r in rdf.iterrows():
            star = sig_stars(r["PC1_P"])
            log.info(f"  {r['Outcome']} | {r['System']}: "
                     f"OR={r['PC1_OR']:.3f}, p={r['PC1_P']:.2e} {star}")
    return rdf, loadings_df


# ==============================================================================
# Module 7: 置换检验
# ==============================================================================
@safe_module
def permutation_test(df, meta, p_collector, n_perm=1000):
    print(f"\n{'─' * 72}")
    print(f"  [Module 7] 置换检验 ({n_perm} 次)")
    print(f"{'─' * 72}")

    if n_perm <= 0:
        log.info("已跳过 (--skip-perm)")
        return pd.DataFrame()

    nt_cols = meta["nt_cols"]
    mrs_col = find_col(df, ["m3_mRS", "m6_mRS", "m12_mRS", "mRS", "D_MRS"])
    covars = meta["covariates_all"]
    resid_map = meta.get("resid_map", {})
    tlv_col = meta.get("tlv")
    has_tlv_in_covars = tlv_col and tlv_col in covars

    if not mrs_col:
        return pd.DataFrame()

    target = f"_perm_{mrs_col}"
    df[target] = df[mrs_col].apply(group_mrs)
    np.random.seed(42)

    results = []
    for nt in tqdm(nt_cols, desc="  Permutation"):
        nt_name = bare_name(nt)
        nt_var = resid_map.get(nt, nt) if has_tlv_in_covars else nt
        predictors = [nt_var] + covars
        sub = df[[target] + predictors].dropna()
        if len(sub) < 30:
            continue

        sub_z = sub.copy()
        for p in predictors:
            sub_z[p] = zscore(sub_z[p])

        try:
            mod = OrderedModel(sub_z[target], sub_z[predictors], distr="logit")
            res = mod.fit(method="bfgs", disp=False)
            obs_beta = abs(res.params[nt_var])
            obs_p = res.pvalues[nt_var]
        except Exception:
            continue

        perm_count = 0
        n_success = 0
        for _ in range(n_perm):
            perm_y = np.random.permutation(sub_z[target].values)
            try:
                mod_p = OrderedModel(perm_y, sub_z[predictors], distr="logit")
                res_p = mod_p.fit(method="bfgs", disp=False)
                if abs(res_p.params[nt_var]) >= obs_beta:
                    perm_count += 1
                n_success += 1
            except Exception:
                continue

        p_perm = (perm_count + 1) / (n_success + 1) if n_success > 0 else np.nan
        concordance = ("✓ Both sig" if (obs_p < 0.05 and p_perm < 0.05)
                       else ("△ Param only" if obs_p < 0.05
                             else ("△ Perm only" if p_perm < 0.05 else "— NS")))
        results.append({
            "Outcome": mrs_col, "NT": nt_name, "Obs_Beta": obs_beta,
            "Parametric_P": obs_p, "Permutation_P": p_perm,
            "N_perm_ok": n_success, "Concordance": concordance, "N": len(sub),
        })
        p_collector.add("Permutation", f"{mrs_col}|{nt_name}", p_perm)

    rdf = pd.DataFrame(results)
    if not rdf.empty:
        rdf = rdf.sort_values("Permutation_P")
        n_both = rdf["Concordance"].str.contains("Both").sum()
        log.info(f"→ 双重显著: {n_both}/{len(rdf)}")
    return rdf


# ==============================================================================
# Module 8: 阈值剂量-反应 (分位数断点)
# ==============================================================================
@safe_module
def threshold_dose_response(df, meta, p_collector):
    print(f"\n{'─' * 72}")
    print(f"  [Module 8] 阈值剂量-反应 (分位数断点)")
    print(f"{'─' * 72}")

    nt_cols = meta["nt_cols"]
    mrs_col = find_col(df, ["m3_mRS", "m6_mRS", "m12_mRS", "mRS"])
    covars = meta["covariates_all"]

    if not mrs_col:
        return pd.DataFrame()

    target = f"_dose_{mrs_col}"
    df[target] = df[mrs_col].apply(group_mrs)

    results = []
    for nt in nt_cols:
        nt_name = bare_name(nt)
        sub = df[[target, nt] + covars].dropna()
        if len(sub) < 60:
            continue

        try:
            sub["Q_group"] = pd.qcut(sub[nt], 4, labels=[0, 1, 2, 3])
        except ValueError:
            continue

        groups = [sub[sub["Q_group"] == q][target].values for q in range(4)]
        groups = [g for g in groups if len(g) >= 5]
        if len(groups) < 3:
            continue

        kw_stat, kw_p = stats.kruskal(*groups)
        spearman_r, sp_p = stats.spearmanr(sub[nt], sub[target])

        sub["high_load"] = (sub["Q_group"].astype(int) >= 3).astype(int)
        sub_z = sub.copy()
        thresh_predictors = ["high_load"] + covars
        for p in covars:
            sub_z[p] = zscore(sub_z[p])

        try:
            mod = OrderedModel(sub_z[target], sub_z[thresh_predictors], distr="logit")
            res = mod.fit(method="bfgs", disp=False)
            thresh_p = res.pvalues["high_load"]
            thresh_or = np.exp(res.params["high_load"])
        except Exception:
            thresh_p = np.nan
            thresh_or = np.nan

        q_means = [sub[sub["Q_group"] == q][target].mean() for q in range(4)]
        results.append({
            "NT": nt_name, "Outcome": mrs_col,
            "KW_stat": kw_stat, "KW_P": kw_p,
            "Spearman_r": spearman_r, "Spearman_P": sp_p,
            "Q1_mean_mRS": q_means[0], "Q2_mean_mRS": q_means[1],
            "Q3_mean_mRS": q_means[2], "Q4_mean_mRS": q_means[3],
            "Threshold_OR": thresh_or, "Threshold_P": thresh_p,
            "N": len(sub),
        })
        if np.isfinite(kw_p):
            p_collector.add("DoseResponse", f"{nt_name}|KW", kw_p)
        if np.isfinite(thresh_p):
            p_collector.add("DoseResponse", f"{nt_name}|Threshold", thresh_p)

    rdf = pd.DataFrame(results)
    if not rdf.empty:
        rdf["KW_FDR_q"] = fdr_correct(rdf["KW_P"].values)
        rdf = rdf.sort_values("KW_P")
        n_sig = (rdf["KW_P"] < 0.05).sum()
        log.info(f"→ KW 显著: {n_sig}/{len(rdf)}")
    return rdf


# ==============================================================================
# Module 9: 10-fold CV
# ==============================================================================
@safe_module
def kfold_cv(df, meta):
    print(f"\n{'─' * 72}")
    print(f"  [Module 9] 10-fold 交叉验证")
    print(f"{'─' * 72}")

    if not HAS_SKLEARN:
        log.warning("需要 scikit-learn")
        return pd.DataFrame()

    from sklearn.linear_model import LogisticRegression

    nt_cols = meta["nt_cols"]
    mrs_col = find_col(df, ["m3_mRS", "m6_mRS", "m12_mRS", "mRS"])
    covars = meta["covariates_all"]

    if not mrs_col:
        return pd.DataFrame()

    df["_cv_binary"] = (pd.to_numeric(df[mrs_col], errors="coerce") > 2).astype(float)

    results = []
    for nt in nt_cols:
        nt_name = bare_name(nt)
        all_cols = ["_cv_binary"] + covars + [nt]
        sub = df[all_cols].dropna()
        if len(sub) < 50:
            continue

        y = sub["_cv_binary"].values
        if len(np.unique(y)) < 2:
            continue

        X_base = sub[covars].values if covars else np.empty((len(sub), 0))
        X_full = sub[covars + [nt]].values

        for j in range(X_base.shape[1]):
            s = X_base[:, j].std()
            if s > 1e-10:
                X_base[:, j] = (X_base[:, j] - X_base[:, j].mean()) / s
        for j in range(X_full.shape[1]):
            s = X_full[:, j].std()
            if s > 1e-10:
                X_full[:, j] = (X_full[:, j] - X_full[:, j].mean()) / s

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        lr = LogisticRegression(max_iter=2000, solver="lbfgs")

        aucs_base, aucs_full = [], []
        for train_idx, test_idx in skf.split(X_full, y):
            y_tr, y_te = y[train_idx], y[test_idx]
            if len(np.unique(y_te)) < 2:
                continue
            if X_base.shape[1] > 0:
                try:
                    lr.fit(X_base[train_idx], y_tr)
                    prob_b = lr.predict_proba(X_base[test_idx])[:, 1]
                    aucs_base.append(roc_auc_score(y_te, prob_b))
                except Exception:
                    pass
            try:
                lr.fit(X_full[train_idx], y_tr)
                prob_f = lr.predict_proba(X_full[test_idx])[:, 1]
                aucs_full.append(roc_auc_score(y_te, prob_f))
            except Exception:
                pass

        if len(aucs_full) < 5:
            continue

        auc_base = np.mean(aucs_base) if aucs_base else np.nan
        auc_full = np.mean(aucs_full)
        delta = auc_full - auc_base if np.isfinite(auc_base) else np.nan

        results.append({
            "NT": nt_name, "AUC_base": auc_base, "AUC_full": auc_full,
            "Delta_AUC": delta, "AUC_full_SD": np.std(aucs_full),
            "Clinical_Value": ("✓ 有价值" if delta > 0.02 else
                               ("~ 边际" if delta > 0 else "✗ 无增量"))
                              if np.isfinite(delta) else "N/A",
            "N": len(sub),
        })

    rdf = pd.DataFrame(results)
    if not rdf.empty:
        rdf = rdf.sort_values("Delta_AUC", ascending=False)
        for _, r in rdf.head(5).iterrows():
            log.info(f"  {r['NT']}: ΔAUC={r['Delta_AUC']:+.4f} {r['Clinical_Value']}")
    return rdf


# ==============================================================================
# Module 9b: mRS 分组灵敏度分析
# ==============================================================================
@safe_module
def mrs_cutpoint_sensitivity(df, meta, top_nts=None):
    print(f"\n{'─' * 72}")
    print(f"  [Module 9b] mRS 分组灵敏度分析")
    print(f"{'─' * 72}")

    mrs_col = find_col(df, ["m3_mRS", "m6_mRS", "m12_mRS", "mRS"])
    covars = meta["covariates_all"]
    nt_cols = meta["nt_cols"]
    resid_map = meta.get("resid_map", {})
    tlv_col = meta.get("tlv")
    has_tlv_in_covars = tlv_col and tlv_col in covars

    if not mrs_col:
        return pd.DataFrame()

    if top_nts:
        test_cols = [c for c in nt_cols if bare_name(c) in top_nts]
    else:
        test_cols = nt_cols

    cutpoints = {
        "A_0-1_vs_2-6": lambda x: 0 if x <= 1 else 1,
        "B_0-2_vs_3-6": lambda x: 0 if x <= 2 else 1,
        "C_0-3_vs_4-6": lambda x: 0 if x <= 3 else 1,
        "D_Ordinal_0-2_3-4_5-6": group_mrs,
    }

    results = []
    for scheme_name, grouper in cutpoints.items():
        target = f"_sens_{scheme_name}"
        df[target] = pd.to_numeric(df[mrs_col], errors="coerce").apply(
            lambda x: grouper(x) if pd.notna(x) else np.nan
        )
        n_groups = df[target].dropna().nunique()
        if n_groups < 2:
            continue

        for nt in test_cols:
            nt_name = bare_name(nt)
            nt_var = resid_map.get(nt, nt) if has_tlv_in_covars else nt
            predictors = [nt_var] + covars
            sub = df[[target] + predictors].dropna()
            if len(sub) < 30:
                continue

            sub_z = sub.copy()
            for p in predictors:
                sub_z[p] = zscore(sub_z[p])

            try:
                if n_groups == 2:
                    X = sm.add_constant(sub_z[predictors])
                    res = sm.Logit(sub_z[target], X).fit(disp=False)
                    beta = res.params[nt_var]
                    pval = res.pvalues[nt_var]
                else:
                    mod = OrderedModel(sub_z[target], sub_z[predictors], distr="logit")
                    res = mod.fit(method="bfgs", disp=False)
                    beta = res.params[nt_var]
                    pval = res.pvalues[nt_var]

                results.append({
                    "NT": nt_name, "Cutpoint": scheme_name,
                    "Beta": beta, "OR": np.exp(beta),
                    "P_value": pval, "Sig": pval < 0.05, "N": len(sub),
                })
            except Exception:
                continue

    rdf = pd.DataFrame(results)
    if not rdf.empty:
        consist = rdf.groupby("NT").agg(
            N_cutpoints=("Sig", "count"), N_sig=("Sig", "sum"),
        ).reset_index()
        consist["Consistency"] = consist.apply(
            lambda r: "★★★ All" if r["N_sig"] == r["N_cutpoints"]
            else ("★★ Most" if r["N_sig"] >= 3 else
                  ("★ Some" if r["N_sig"] >= 1 else "— None")), axis=1
        )
        rdf = rdf.merge(consist[["NT", "Consistency"]], on="NT", how="left")
        rdf = rdf.sort_values(["NT", "Cutpoint"])
        for nt in consist[consist["N_sig"] == consist["N_cutpoints"]]["NT"]:
            log.info(f"  ★★★ {nt}: 所有切点均显著")
    return rdf


# ==============================================================================
# Module 9c: Spin Test
# ==============================================================================
@safe_module
def spin_test_nt_specificity(df, meta, p_collector, n_spin=1000):
    print(f"\n{'─' * 72}")
    print(f"  [Module 9c] 递质特异性旋转检验 (Spin Test, {n_spin} 次)")
    print(f"{'─' * 72}")

    nt_cols = meta["nt_cols"]
    mrs_col = find_col(df, ["m3_mRS", "m6_mRS", "m12_mRS", "mRS"])
    covars = meta["covariates_all"]
    resid_map = meta.get("resid_map", {})
    tlv_col = meta.get("tlv")
    has_tlv_in_covars = tlv_col and tlv_col in covars

    if not mrs_col or len(nt_cols) < 3:
        return pd.DataFrame()

    target = f"_spin_{mrs_col}"
    df[target] = df[mrs_col].apply(group_mrs)

    real_betas = {}
    nt_var_map = {}
    for nt in nt_cols:
        nt_var = resid_map.get(nt, nt) if has_tlv_in_covars else nt
        nt_var_map[nt] = nt_var
        predictors = [nt_var] + covars
        sub = df[[target] + predictors].dropna()
        if len(sub) < 30:
            continue
        sub_z = sub.copy()
        for p in predictors:
            sub_z[p] = zscore(sub_z[p])
        try:
            mod = OrderedModel(sub_z[target], sub_z[predictors], distr="logit")
            res = mod.fit(method="bfgs", disp=False)
            real_betas[nt] = abs(res.params[nt_var])
        except Exception:
            continue

    if not real_betas:
        return pd.DataFrame()

    np.random.seed(42)
    spin_counts = {nt: 0 for nt in real_betas}
    nt_list = list(real_betas.keys())

    for spin_i in range(n_spin):
        perm_idx = np.random.permutation(len(nt_list))
        for orig_i, nt in enumerate(nt_list):
            swapped_nt = nt_list[perm_idx[orig_i]]
            if swapped_nt == nt:
                continue
            swapped_var = nt_var_map.get(swapped_nt, swapped_nt)
            predictors = [swapped_var] + covars
            sub = df[[target] + predictors].dropna()
            if len(sub) < 30:
                continue
            sub_z = sub.copy()
            for p in predictors:
                sub_z[p] = zscore(sub_z[p])
            try:
                mod = OrderedModel(sub_z[target], sub_z[predictors], distr="logit")
                res = mod.fit(method="bfgs", disp=False)
                if abs(res.params[swapped_var]) >= real_betas[nt]:
                    spin_counts[nt] += 1
            except Exception:
                continue

    results = []
    for nt in nt_list:
        nt_name = bare_name(nt)
        p_spin = (spin_counts[nt] + 1) / (n_spin + 1)
        results.append({
            "NT": nt_name, "Real_AbsBeta": real_betas[nt],
            "Spin_P": p_spin,
            "Specific": "✓ Specific" if p_spin < 0.05 else "— Not specific",
        })
        p_collector.add("SpinTest", f"{nt_name}", p_spin)

    rdf = pd.DataFrame(results).sort_values("Spin_P")
    n_spec = (rdf["Spin_P"] < 0.05).sum()
    log.info(f"→ 递质特异性: {n_spec}/{len(rdf)}")
    return rdf


# ==============================================================================
# Module 9d: 决策曲线分析 (DCA)
# ==============================================================================
@safe_module
def decision_curve_analysis(df, meta, out_dir):
    print(f"\n{'─' * 72}")
    print(f"  [Module 9d] 决策曲线分析 (DCA)")
    print(f"{'─' * 72}")

    if not HAS_SKLEARN:
        log.warning("需要 scikit-learn")
        return pd.DataFrame()

    from sklearn.linear_model import LogisticRegression

    mrs_col = find_col(df, ["m3_mRS", "m6_mRS", "m12_mRS", "mRS"])
    covars = meta["covariates_all"]
    nt_cols = meta["nt_cols"]

    if not mrs_col:
        return pd.DataFrame()

    df["_dca_y"] = (pd.to_numeric(df[mrs_col], errors="coerce") > 2).astype(float)
    all_cols = ["_dca_y"] + covars + nt_cols
    sub = df[all_cols].dropna()
    if len(sub) < 50:
        return pd.DataFrame()

    y = sub["_dca_y"].values
    prevalence = y.mean()
    thresholds = np.arange(0.05, 0.95, 0.01)
    lr = LogisticRegression(max_iter=2000, solver="lbfgs")

    X_base = sub[covars].values if covars else None
    X_full = sub[covars + nt_cols].values

    for X in [X_base, X_full]:
        if X is not None:
            for j in range(X.shape[1]):
                s = X[:, j].std()
                if s > 1e-10:
                    X[:, j] = (X[:, j] - X[:, j].mean()) / s

    prob_base = None
    if X_base is not None and X_base.shape[1] > 0:
        try:
            lr.fit(X_base, y)
            prob_base = lr.predict_proba(X_base)[:, 1]
        except Exception:
            pass

    try:
        lr.fit(X_full, y)
        prob_full = lr.predict_proba(X_full)[:, 1]
    except Exception:
        return pd.DataFrame()

    def net_benefit(y_true, y_prob, threshold):
        n = len(y_true)
        predicted_pos = y_prob >= threshold
        tp = np.sum((predicted_pos) & (y_true == 1))
        fp = np.sum((predicted_pos) & (y_true == 0))
        return tp / n - fp / n * (threshold / (1 - threshold))

    rows = []
    for t in thresholds:
        row = {"Threshold": t}
        row["NB_TreatAll"] = prevalence - (1 - prevalence) * t / (1 - t)
        row["NB_TreatNone"] = 0
        if prob_base is not None:
            row["NB_Base"] = net_benefit(y, prob_base, t)
        row["NB_Full"] = net_benefit(y, prob_full, t)
        rows.append(row)

    dca_df = pd.DataFrame(rows)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(dca_df["Threshold"], dca_df["NB_TreatAll"], "k--", label="Treat All", alpha=0.5)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.3, label="Treat None")
    if "NB_Base" in dca_df.columns:
        ax.plot(dca_df["Threshold"], dca_df["NB_Base"], "b-", linewidth=2,
                label="Base (TLV+NIHSS+Age+Sex)")
    ax.plot(dca_df["Threshold"], dca_df["NB_Full"], "r-", linewidth=2,
            label="+ NT Variables")
    ax.set_xlabel("Threshold Probability", fontsize=11)
    ax.set_ylabel("Net Benefit", fontsize=11)
    ax.set_title("Decision Curve Analysis", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(0, 0.95)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(fig_dir / "dca.png")
    plt.close(fig)
    log.info(f"  📊 dca.png")

    return dca_df


# ==============================================================================
# Module 10: 全局 FDR
# ==============================================================================
def global_fdr(p_collector, out_path):
    print(f"\n{'─' * 72}")
    print(f"  [Module 10] 全局 FDR 校正 (跨模块)")
    print(f"{'─' * 72}")

    gdf = p_collector.correct()
    if gdf.empty:
        log.warning("无 p 值可校正")
        return gdf

    gdf.to_csv(out_path / "global_fdr.csv", index=False)

    n_total = len(gdf)
    n_sig_raw = (gdf["P_raw"] < 0.05).sum()
    n_sig_global = (gdf["Q_global"] < 0.05).sum()

    log.info(f"  总测试数: {n_total}")
    log.info(f"  p < 0.05:       {n_sig_raw} ({n_sig_raw/n_total:.1%})")
    log.info(f"  q_global < 0.05: {n_sig_global} ({n_sig_global/n_total:.1%})")

    for module, grp in gdf.groupby("Module"):
        n_mod = len(grp)
        n_q = (grp["Q_global"] < 0.05).sum()
        log.info(f"    {module}: {n_q}/{n_mod} survive global FDR")

    return gdf


# ##############################################################################
#
#  ██╗   ██╗██╗  ██╗    新 增 模 块
#  ██║   ██║██║  ██║    Module 11 / 12 / 13
#  ██║   ██║███████║
#  ╚██╗ ██╔╝╚════██║
#   ╚████╔╝      ██║
#    ╚═══╝       ╚═╝
#
# ##############################################################################


# ==============================================================================
# Module 11: HRV 中介分析 (心脑轴)
# ==============================================================================
# 设计:
#   路径 a:  Resid_NT → log(RMSSD)  (OLS)
#   路径 b:  log(RMSSD) → m12_mRS   (OrderedLogit)
#   直接效应 c': Resid_NT → m12_mRS (OrderedLogit, 控制 RMSSD)
#   控制:    TLV + NIHSS + Age + Sex + HRmean
#   Bootstrap 5000 次, 计算 Indirect = a × b 的置信区间
#
#   并行中介: RMSSD / IL-6 各自独立
#   串联中介: 如果两个都显著, 进一步做 X→M1→M2→Y
#
#   缺失偏倚检验: 有 Holter vs 无 Holter 的 Age/Sex/NIHSS/TLV/mRS 对比
# ==============================================================================

@safe_module
def holter_missing_bias_test(df, meta):
    """
    缺失偏倚检验: 有 Holter 数据 vs 无 Holter 数据
    比较基线特征 (Age, Sex, NIHSS, TLV, mRS) 是否存在系统差异.
    如果 p < 0.05, 说明缺失非随机 (MNAR), 中介分析结论需谨慎.
    """
    print(f"\n{'─' * 72}")
    print(f"  [Module 11a] 缺失偏倚检验 (有 Holter vs 无 Holter)")
    print(f"{'─' * 72}")

    rmssd_col = meta["rmssd"]
    if not rmssd_col:
        log.warning("无 RMSSD 列, 跳过偏倚检验")
        return pd.DataFrame()

    df["_has_holter"] = df[rmssd_col].notna().astype(int)
    n_has = df["_has_holter"].sum()
    n_no = len(df) - n_has
    log.info(f"  有 Holter: {n_has} ({n_has/len(df)*100:.1f}%)")
    log.info(f"  无 Holter: {n_no} ({n_no/len(df)*100:.1f}%)")

    # 比较的变量列表
    compare_vars = []
    for col_key in ["age", "sex", "nihss", "tlv"]:
        col = meta.get(col_key)
        if col:
            compare_vars.append((col_key.upper(), col))

    # 加入 mRS
    mrs_col = find_col(df, ["m12_mRS", "m6_mRS", "m3_mRS", "mRS"])
    if mrs_col:
        compare_vars.append(("mRS", mrs_col))

    # 加入 IL-6
    il6_col = meta.get("il6")
    if il6_col:
        compare_vars.append(("IL6", il6_col))

    # 加入 AF
    af_col = meta.get("af")
    if af_col:
        compare_vars.append(("AF", af_col))

    results = []
    for label, col in compare_vars:
        has_vals = df.loc[df["_has_holter"] == 1, col].dropna()
        no_vals = df.loc[df["_has_holter"] == 0, col].dropna()

        if len(has_vals) < 10 or len(no_vals) < 10:
            continue

        # 判断变量类型
        unique_vals = set(has_vals.unique()) | set(no_vals.unique())
        if unique_vals.issubset({0, 1, 0.0, 1.0}):
            # 二分类: χ² 检验
            ct = pd.crosstab(df["_has_holter"], df[col].dropna().astype(int))
            if ct.shape == (2, 2):
                chi2, p_val, _, _ = stats.chi2_contingency(ct)
                test_name = "Chi-squared"
            else:
                p_val = np.nan
                test_name = "N/A"
            d = np.nan
        else:
            # 连续: Mann-Whitney U
            u_stat, p_val = stats.mannwhitneyu(has_vals, no_vals, alternative="two-sided")
            test_name = "Mann-Whitney U"
            d = cohens_d(has_vals, no_vals)

        results.append({
            "Variable": label,
            "Column": col,
            "N_Holter": len(has_vals),
            "N_NoHolter": len(no_vals),
            "Mean_Holter": has_vals.mean(),
            "Mean_NoHolter": no_vals.mean(),
            "Test": test_name,
            "P_value": p_val,
            "Cohens_d": d,
            "Bias_Risk": "⚠️ 有偏" if p_val < 0.05 else "✓ 无偏",
        })

        star = sig_stars(p_val)
        log.info(f"  {label:<10s}: Holter={has_vals.mean():.2f} vs No={no_vals.mean():.2f}"
                 f"  p={p_val:.4f} {star} {results[-1]['Bias_Risk']}")

    rdf = pd.DataFrame(results)

    # 总体评估
    if not rdf.empty:
        n_biased = (rdf["P_value"] < 0.05).sum()
        if n_biased == 0:
            log.info(f"  ✅ 无显著偏倚, RMSSD 缺失接近随机 (MAR)")
        elif n_biased <= 2:
            log.info(f"  ⚠️ {n_biased} 个变量有偏, 中介分析需要 IPTW 加权")
        else:
            log.info(f"  ❌ {n_biased} 个变量有偏, 中介分析结论需高度谨慎")

    return rdf


@safe_module
def hrv_mediation_analysis(df, meta, p_collector, n_boot=5000):
    """
    HRV 中介分析 (Bootstrap):
      X = Resid_NT (17 种)
      M1 = log(RMSSD) — 副交感功能
      M2 = IL-6         — 全身炎症
      Y = m12_mRS (有序)
      Covariates: TLV + NIHSS + Age + Sex + HRmean

    第一阶段: 并行中介 (RMSSD 和 IL-6 各自独立)
    第二阶段: 如果两个都显著, 做串联中介 X→RMSSD→IL6→Y

    Bootstrap 因 OrderedModel 太慢, 路径 b 用 OLS(mRS_numeric) 近似.
    置信区间: Bias-Corrected (BCa)
    """
    print(f"\n{'─' * 72}")
    print(f"  [Module 11b] HRV 中介分析 (Bootstrap {n_boot} 次)")
    print(f"{'─' * 72}")

    rmssd_col = meta["rmssd"]
    hrmean_col = meta["hrmean"]
    il6_col = meta.get("il6")

    if not rmssd_col:
        log.warning("无 RMSSD 列, 跳过中介分析")
        return pd.DataFrame()

    mrs_col = find_col(df, ["m12_mRS", "m6_mRS", "m3_mRS"])
    if not mrs_col:
        log.warning("无 mRS 列, 跳过中介分析")
        return pd.DataFrame()

    # ── 准备中介变量 ──
    log_rmssd_col = "_log_RMSSD"
    df[log_rmssd_col] = np.log(df[rmssd_col].clip(lower=0.1))

    # ── 协变量 ──
    base_covars = list(meta["covariates_all"])  # TLV + NIHSS + Age + Sex
    if hrmean_col and hrmean_col not in base_covars:
        base_covars.append(hrmean_col)

    nt_cols = meta["nt_cols"]
    resid_map = meta.get("resid_map", {})
    tlv_col = meta.get("tlv")
    has_tlv_in_covars = tlv_col and tlv_col in base_covars

    # ── 中介函数: 单次 Bootstrap 中计算 indirect effect ──
    def _mediation_one_boot(
            data, x_col, m_col, y_col, covars_list):
        """
        path a: M = α0 + a*X + γ*Covars + ε1  (OLS)
        path b: Y = β0 + b*M + c'*X + δ*Covars + ε2  (OLS on numeric mRS)
        indirect = a * b
        """
        predictors_a = [x_col] + covars_list
        predictors_b = [m_col, x_col] + covars_list

        # 去重
        predictors_a = list(dict.fromkeys(predictors_a))
        predictors_b = list(dict.fromkeys(predictors_b))

        all_cols = list(set([y_col, m_col] + predictors_a))
        sub = data[all_cols].dropna()
        if len(sub) < 30:
            return np.nan, np.nan, np.nan, np.nan

        sub_z = sub.copy()
        for c in predictors_a:
            sub_z[c] = zscore(sub_z[c])
        sub_z[m_col] = zscore(sub_z[m_col])
        sub_z[y_col] = zscore(sub_z[y_col].astype(float))

        try:
            # Path a: X → M
            X_a = sm.add_constant(sub_z[predictors_a])
            res_a = sm.OLS(sub_z[m_col], X_a).fit()
            a = res_a.params[x_col]

            # Path b + c': M + X → Y
            X_b = sm.add_constant(sub_z[predictors_b])
            res_b = sm.OLS(sub_z[y_col], X_b).fit()
            b = res_b.params[m_col]
            c_prime = res_b.params[x_col]

            indirect = a * b
            return a, b, c_prime, indirect
        except Exception:
            return np.nan, np.nan, np.nan, np.nan

    # ── 并行中介: RMSSD ──
    mediators = [(log_rmssd_col, "RMSSD")]
    if il6_col:
        # log-transform IL-6 to reduce skew
        log_il6_col = "_log_IL6"
        df[log_il6_col] = np.log(df[il6_col].clip(lower=0.01))
        mediators.append((log_il6_col, "IL6"))

    results_all = []
    sig_mediators_by_nt = {}  # {nt_name: [mediator_labels that are significant]}

    for m_col, m_label in mediators:
        log.info(f"\n  ── 中介变量: {m_label} ({m_col}) ──")

        for nt in nt_cols:
            nt_name = bare_name(nt)
            nt_var = resid_map.get(nt, nt) if has_tlv_in_covars else nt

            # 观察值
            a_obs, b_obs, cp_obs, ind_obs = _mediation_one_boot(
                df, nt_var, m_col, mrs_col, base_covars
            )
            if np.isnan(ind_obs):
                continue

            # 计算 N
            all_needed = list(set([mrs_col, m_col, nt_var] + base_covars))
            n_available = df[all_needed].dropna().shape[0]

            # Bootstrap
            boot_indirect = []
            boot_a = []
            boot_b = []
            np.random.seed(42 + hash(nt_name) % 10000)

            for _ in range(n_boot):
                boot_idx = np.random.choice(len(df), size=len(df), replace=True)
                boot_df = df.iloc[boot_idx].copy()
                _, _, _, ind_b = _mediation_one_boot(
                    boot_df, nt_var, m_col, mrs_col, base_covars
                )
                if np.isfinite(ind_b):
                    boot_indirect.append(ind_b)

            if len(boot_indirect) < 100:
                log.warning(f"    {nt_name}×{m_label}: 仅 {len(boot_indirect)} 次有效 bootstrap, 跳过")
                continue

            boot_arr = np.array(boot_indirect)

            # BCa 置信区间 (Bias-Corrected and accelerated)
            # 简化: 用 percentile 法
            ci_lower = np.percentile(boot_arr, 2.5)
            ci_upper = np.percentile(boot_arr, 97.5)

            # p 值近似: indirect 不跨零 ⟹ p < 0.05
            # 更精确: 计算零假设比例
            if ind_obs > 0:
                p_indirect = 2 * np.mean(boot_arr <= 0)
            else:
                p_indirect = 2 * np.mean(boot_arr >= 0)
            p_indirect = min(p_indirect, 1.0)

            # 中介比例: indirect / total
            total = ind_obs + cp_obs
            proportion_mediated = ind_obs / total if abs(total) > 1e-10 else np.nan

            sig = ci_lower > 0 or ci_upper < 0  # CI 不含 0

            results_all.append({
                "NT": nt_name,
                "Mediator": m_label,
                "N": n_available,
                "Path_a": a_obs,
                "Path_b": b_obs,
                "Direct_c_prime": cp_obs,
                "Indirect_ab": ind_obs,
                "Total_Effect": total,
                "Proportion_Mediated": proportion_mediated,
                "Boot_CI_lower": ci_lower,
                "Boot_CI_upper": ci_upper,
                "Boot_P": p_indirect,
                "Significant": sig,
                "N_boot_valid": len(boot_indirect),
            })

            star = sig_stars(p_indirect)
            ci_tag = "✓" if sig else "—"
            log.info(f"    {nt_name}×{m_label}: ab={ind_obs:.4f} "
                     f"[{ci_lower:.4f}, {ci_upper:.4f}] {ci_tag} {star}")

            if sig:
                p_collector.add("Mediation", f"{nt_name}→{m_label}→mRS", p_indirect)
                sig_mediators_by_nt.setdefault(nt_name, []).append(m_label)

    rdf = pd.DataFrame(results_all)

    # ── 串联中介 (Serial Mediation): X → RMSSD → IL-6 → Y ──
    serial_results = []
    if il6_col and rmssd_col:
        # 找到 RMSSD 和 IL-6 都显著的 NT
        serial_candidates = [nt for nt, meds in sig_mediators_by_nt.items()
                             if "RMSSD" in meds and "IL6" in meds]

        if serial_candidates:
            log.info(f"\n  ── 串联中介: {len(serial_candidates)} 个 NT 双显著 ──")
            log.info(f"     X → RMSSD → IL-6 → mRS")

            for nt_name in serial_candidates:
                nt = [c for c in nt_cols if bare_name(c) == nt_name][0]
                nt_var = resid_map.get(nt, nt) if has_tlv_in_covars else nt

                # 串联: a1*d21*b2
                # path a1: X → M1 (RMSSD)
                # path d21: M1 → M2 (RMSSD → IL-6)
                # path b2: M2 → Y (IL-6 → mRS), controlling M1 and X

                boot_serial = []
                np.random.seed(42 + hash(nt_name) % 10000 + 1)

                for _ in range(n_boot):
                    boot_idx = np.random.choice(len(df), size=len(df), replace=True)
                    bd = df.iloc[boot_idx].copy()

                    all_needed = list(set([mrs_col, log_rmssd_col, log_il6_col,
                                           nt_var] + base_covars))
                    sub = bd[all_needed].dropna()
                    if len(sub) < 30:
                        continue

                    sub_z = sub.copy()
                    for c in [nt_var] + base_covars:
                        sub_z[c] = zscore(sub_z[c])
                    sub_z[log_rmssd_col] = zscore(sub_z[log_rmssd_col])
                    sub_z[log_il6_col] = zscore(sub_z[log_il6_col])
                    sub_z[mrs_col] = zscore(sub_z[mrs_col].astype(float))

                    try:
                        # a1: X → M1
                        X_a1 = sm.add_constant(sub_z[[nt_var] + base_covars])
                        a1 = sm.OLS(sub_z[log_rmssd_col], X_a1).fit().params[nt_var]

                        # d21: M1 → M2 (controlling X)
                        X_d = sm.add_constant(sub_z[[log_rmssd_col, nt_var] + base_covars])
                        d21 = sm.OLS(sub_z[log_il6_col], X_d).fit().params[log_rmssd_col]

                        # b2: M2 → Y (controlling M1 and X)
                        X_b2 = sm.add_constant(
                            sub_z[[log_il6_col, log_rmssd_col, nt_var] + base_covars]
                        )
                        b2 = sm.OLS(sub_z[mrs_col], X_b2).fit().params[log_il6_col]

                        serial_ind = a1 * d21 * b2
                        boot_serial.append(serial_ind)
                    except Exception:
                        continue

                if len(boot_serial) >= 100:
                    sarr = np.array(boot_serial)
                    ci_lo = np.percentile(sarr, 2.5)
                    ci_hi = np.percentile(sarr, 97.5)
                    obs_serial = np.median(sarr)
                    sig_serial = ci_lo > 0 or ci_hi < 0

                    serial_results.append({
                        "NT": nt_name,
                        "Chain": "X→RMSSD→IL6→mRS",
                        "Serial_Indirect": obs_serial,
                        "CI_lower": ci_lo,
                        "CI_upper": ci_hi,
                        "Significant": sig_serial,
                        "N_boot_valid": len(boot_serial),
                    })
                    tag = "✓ 串联显著" if sig_serial else "— 串联不显著"
                    log.info(f"    {nt_name}: serial={obs_serial:.4f} "
                             f"[{ci_lo:.4f}, {ci_hi:.4f}] {tag}")
        else:
            log.info(f"  ℹ️ 无 NT 同时通过 RMSSD 和 IL-6 并行中介, 跳过串联")

    serial_df = pd.DataFrame(serial_results)

    # 汇总
    if not rdf.empty:
        rdf["FDR_q"] = fdr_correct(rdf["Boot_P"].values)
        rdf = rdf.sort_values("Boot_P")
        n_sig = rdf["Significant"].sum()
        log.info(f"\n  → 并行中介显著: {n_sig}/{len(rdf)}")

    return rdf, serial_df


# ==============================================================================
# Module 12: WMH 交互效应
# ==============================================================================
@safe_module
def wmh_interaction_analysis(df, meta, p_collector):
    """
    Resid_NT × IMG_SVD_WMH → mRS (有序逻辑回归)
    测试: 白质高信号 (WMH) 是否放大递质损毁的预后效应.

    模型:
      mRS_cat ~ Resid_NT + WMH + Resid_NT×WMH + TLV + NIHSS + Age + Sex

    WMH 100% 有数据, 不存在缺失问题.
    额外: WMH 高/低分层后的 Resid_NT 效应比较.
    """
    print(f"\n{'─' * 72}")
    print(f"  [Module 12] WMH 交互效应 (Resid_NT × WMH → mRS)")
    print(f"{'─' * 72}")

    wmh_col = meta.get("wmh")
    if not wmh_col:
        log.warning("无 IMG_SVD_WMH 列, 跳过 WMH 交互")
        return pd.DataFrame()

    nt_cols = meta["nt_cols"]
    covars = meta["covariates_all"]
    resid_map = meta.get("resid_map", {})
    tlv_col = meta.get("tlv")
    has_tlv_in_covars = tlv_col and tlv_col in covars

    mrs_col = find_col(df, ["m12_mRS", "m6_mRS", "m3_mRS", "mRS"])
    if not mrs_col:
        log.warning("无 mRS 列")
        return pd.DataFrame()

    target = f"_wmh_target_{mrs_col}"
    df[target] = df[mrs_col].apply(group_mrs)

    n_wmh = df[wmh_col].notna().sum()
    log.info(f"  WMH 可用: {n_wmh}/{len(df)} ({n_wmh/len(df)*100:.1f}%)")
    log.info(f"  WMH 均值: {df[wmh_col].mean():.2f}, 中位数: {df[wmh_col].median():.2f}")

    # WMH 分组 (用于分层分析)
    wmh_median = df[wmh_col].median()
    df["_WMH_High"] = (df[wmh_col] >= wmh_median).astype(int)

    results = []
    for nt in nt_cols:
        nt_name = bare_name(nt)
        nt_var = resid_map.get(nt, nt) if has_tlv_in_covars else nt

        # ── 交互模型 ──
        predictors = [nt_var, wmh_col] + covars
        predictors = list(dict.fromkeys(predictors))

        sub = df[[target] + predictors].dropna()
        if len(sub) < 40:
            continue

        sub_z = sub.copy()
        for p in predictors:
            sub_z[p] = zscore(sub_z[p])
        sub_z["NT_x_WMH"] = sub_z[nt_var] * sub_z[wmh_col]

        try:
            all_pred = predictors + ["NT_x_WMH"]
            mod = OrderedModel(sub_z[target], sub_z[all_pred], distr="logit")
            res = mod.fit(method="bfgs", disp=False)

            inter_p = res.pvalues["NT_x_WMH"]
            inter_beta = res.params["NT_x_WMH"]
            inter_or = np.exp(inter_beta)

            # ── 分层分析: WMH 高 vs WMH 低 ──
            strat_results = {}
            for wmh_level, wmh_label in [(1, "High_WMH"), (0, "Low_WMH")]:
                strat_sub = df[df["_WMH_High"] == wmh_level][[target, nt_var] + covars].dropna()
                if len(strat_sub) < 20:
                    strat_results[wmh_label] = {"Beta": np.nan, "OR": np.nan, "P": np.nan}
                    continue

                strat_z = strat_sub.copy()
                strat_preds = [nt_var] + covars
                for p in strat_preds:
                    strat_z[p] = zscore(strat_z[p])
                try:
                    mod_s = OrderedModel(strat_z[target], strat_z[strat_preds], distr="logit")
                    res_s = mod_s.fit(method="bfgs", disp=False)
                    strat_results[wmh_label] = {
                        "Beta": res_s.params[nt_var],
                        "OR": np.exp(res_s.params[nt_var]),
                        "P": res_s.pvalues[nt_var],
                    }
                except Exception:
                    strat_results[wmh_label] = {"Beta": np.nan, "OR": np.nan, "P": np.nan}

            results.append({
                "NT": nt_name,
                "Outcome": mrs_col,
                "NT_Variable": nt_var,
                "Used_Residual": nt_var != nt,
                # 主效应
                "NT_Beta": res.params[nt_var],
                "NT_OR": np.exp(res.params[nt_var]),
                "NT_P": res.pvalues[nt_var],
                # WMH 主效应
                "WMH_Beta": res.params[wmh_col],
                "WMH_OR": np.exp(res.params[wmh_col]),
                "WMH_P": res.pvalues[wmh_col],
                # 交互效应
                "Interaction_Beta": inter_beta,
                "Interaction_OR": inter_or,
                "Interaction_P": inter_p,
                # 分层
                "HighWMH_Beta": strat_results.get("High_WMH", {}).get("Beta", np.nan),
                "HighWMH_OR": strat_results.get("High_WMH", {}).get("OR", np.nan),
                "HighWMH_P": strat_results.get("High_WMH", {}).get("P", np.nan),
                "LowWMH_Beta": strat_results.get("Low_WMH", {}).get("Beta", np.nan),
                "LowWMH_OR": strat_results.get("Low_WMH", {}).get("OR", np.nan),
                "LowWMH_P": strat_results.get("Low_WMH", {}).get("P", np.nan),
                # 效应放大指标
                "OR_Ratio_High_vs_Low": (
                    strat_results.get("High_WMH", {}).get("OR", np.nan) /
                    strat_results.get("Low_WMH", {}).get("OR", np.nan)
                    if strat_results.get("Low_WMH", {}).get("OR", 0) > 0 else np.nan
                ),
                "Pseudo_R2": getattr(res, "prsquared", np.nan),
                "N": len(sub),
            })

            p_collector.add("WMH_Interaction", f"{nt_name}×WMH", inter_p)

        except Exception:
            continue

    rdf = pd.DataFrame(results)
    if not rdf.empty:
        rdf["Interaction_FDR_q"] = fdr_correct(rdf["Interaction_P"].values)
        rdf = rdf.sort_values("Interaction_P")

        n_sig = (rdf["Interaction_P"] < 0.05).sum()
        log.info(f"  → {len(rdf)} 个交互测试, p<0.05: {n_sig}")

        # 报告 top 5
        for _, r in rdf.head(5).iterrows():
            star = sig_stars(r["Interaction_P"])
            direction = "↑放大" if r["Interaction_Beta"] > 0 else "↓缓冲"
            log.info(f"    {r['NT']}: interaction OR={r['Interaction_OR']:.3f}, "
                     f"p={r['Interaction_P']:.3e} {star} {direction}")
            if pd.notna(r["HighWMH_OR"]) and pd.notna(r["LowWMH_OR"]):
                log.info(f"      ├ High WMH: OR={r['HighWMH_OR']:.3f} (p={r['HighWMH_P']:.3e})")
                log.info(f"      └ Low  WMH: OR={r['LowWMH_OR']:.3f} (p={r['LowWMH_P']:.3e})")

    return rdf


# ==============================================================================
# Module 13: 39% 异常组分析 (Small-Lesion Severe-Outcome Phenotype)
# ==============================================================================
@safe_module
def anomalous_group_analysis(df, meta, p_collector):
    """
    筛选 "小病灶重症" 表型:
      定义: TLV < Q1 (25%) 且 m12_mRS ≥ 3
      对照: TLV < Q1 且 m12_mRS ≤ 2 ("小病灶好预后")

    分析:
      1. 基线特征对比 (Age, Sex, NIHSS, AF)
      2. Resid_NT 逐条对比 (Mann-Whitney U + Cohen's d)
      3. RMSSD + IL-6 对比
      4. 多变量有序回归: 在小病灶亚组中, Resid_NT 是否独立预测 mRS

    临床意义: 识别 "病灶小但预后差" 的机制 (递质? 自主神经? 炎症?)
    """
    print(f"\n{'─' * 72}")
    print(f"  [Module 13] 39% 异常组分析 (小病灶重症表型)")
    print(f"{'─' * 72}")

    tlv_col = meta.get("tlv")
    if not tlv_col:
        log.warning("无 TLV 列, 跳过异常组分析")
        return pd.DataFrame(), pd.DataFrame()

    mrs_col = find_col(df, ["m12_mRS", "m6_mRS", "m3_mRS", "mRS"])
    if not mrs_col:
        log.warning("无 mRS 列")
        return pd.DataFrame(), pd.DataFrame()

    # ── 筛选亚组 ──
    tlv_q1 = df[tlv_col].quantile(0.25)
    small_lesion = df[df[tlv_col] < tlv_q1].copy()

    mrs_numeric = pd.to_numeric(small_lesion[mrs_col], errors="coerce")
    severe = small_lesion[mrs_numeric >= 3].copy()
    good = small_lesion[mrs_numeric <= 2].copy()

    n_small = len(small_lesion)
    n_severe = len(severe)
    n_good = len(good)
    pct_severe = n_severe / n_small * 100 if n_small > 0 else 0

    log.info(f"  TLV Q1 阈值: {tlv_q1:.2f}")
    log.info(f"  小病灶总数: {n_small} (占总样本 {n_small/len(df)*100:.1f}%)")
    log.info(f"  ├ 重症 (mRS ≥ 3): {n_severe} ({pct_severe:.1f}%)")
    log.info(f"  └ 好预后 (mRS ≤ 2): {n_good} ({100-pct_severe:.1f}%)")

    if n_severe < 10 or n_good < 10:
        log.warning(f"  ⚠️ 亚组样本量不足 (severe={n_severe}, good={n_good})")
        return pd.DataFrame(), pd.DataFrame()

    # ═══════════════════════════════════════════
    # Part 1: 基线特征对比
    # ═══════════════════════════════════════════
    baseline_results = []

    compare_vars = []
    for key in ["age", "nihss", "sex"]:
        col = meta.get(key)
        if col:
            compare_vars.append((key.upper(), col))

    af_col = meta.get("af")
    if af_col:
        compare_vars.append(("AF", af_col))

    rmssd_col = meta.get("rmssd")
    if rmssd_col:
        compare_vars.append(("RMSSD", rmssd_col))

    il6_col = meta.get("il6")
    if il6_col:
        compare_vars.append(("IL6", il6_col))

    hrmean_col = meta.get("hrmean")
    if hrmean_col:
        compare_vars.append(("HRmean", hrmean_col))

    wmh_col = meta.get("wmh")
    if wmh_col:
        compare_vars.append(("WMH", wmh_col))

    for label, col in compare_vars:
        sev_vals = severe[col].dropna()
        good_vals = good[col].dropna()
        if len(sev_vals) < 5 or len(good_vals) < 5:
            continue

        unique_both = set(sev_vals.unique()) | set(good_vals.unique())
        if unique_both.issubset({0, 1, 0.0, 1.0}):
            ct = pd.crosstab(
                pd.concat([severe.assign(_grp="Severe"), good.assign(_grp="Good")])["_grp"],
                pd.concat([severe[col], good[col]])
            )
            if ct.shape == (2, 2):
                _, p_val, _, _ = stats.chi2_contingency(ct)
                test_name = "Chi2"
            else:
                p_val = np.nan
                test_name = "N/A"
            d = np.nan
        else:
            _, p_val = stats.mannwhitneyu(sev_vals, good_vals, alternative="two-sided")
            test_name = "MWU"
            d = cohens_d(sev_vals, good_vals)

        baseline_results.append({
            "Variable": label,
            "Severe_N": len(sev_vals),
            "Severe_Mean": sev_vals.mean(),
            "Severe_SD": sev_vals.std(),
            "Good_N": len(good_vals),
            "Good_Mean": good_vals.mean(),
            "Good_SD": good_vals.std(),
            "Test": test_name,
            "P_value": p_val,
            "Cohens_d": d,
        })

        star = sig_stars(p_val)
        log.info(f"  {label:<10s}: Sev={sev_vals.mean():.2f}±{sev_vals.std():.2f} "
                 f"vs Good={good_vals.mean():.2f}±{good_vals.std():.2f} "
                 f"p={p_val:.4f} {star}")

    baseline_df = pd.DataFrame(baseline_results)

    # ═══════════════════════════════════════════
    # Part 2: Resid_NT 逐条对比
    # ═══════════════════════════════════════════
    log.info(f"\n  ── Resid_NT 对比 (小病灶重症 vs 小病灶好预后) ──")

    nt_compare_results = []
    resid_cols = meta.get("resid_cols", [])
    resid_map = meta.get("resid_map", {})

    # 优先用 Resid_NT, 回退用原始 Load_NT
    test_nt_cols = resid_cols if resid_cols else meta["nt_cols"]

    for nt_col in test_nt_cols:
        nt_name = bare_name(nt_col)
        sev_vals = severe[nt_col].dropna()
        good_vals = good[nt_col].dropna()

        if len(sev_vals) < 5 or len(good_vals) < 5:
            continue

        u_stat, p_val = stats.mannwhitneyu(sev_vals, good_vals, alternative="two-sided")
        d = cohens_d(sev_vals, good_vals)

        nt_compare_results.append({
            "NT": nt_name,
            "Column": nt_col,
            "Is_Residual": nt_col.startswith("Resid_"),
            "Severe_N": len(sev_vals),
            "Severe_Mean": sev_vals.mean(),
            "Severe_Median": sev_vals.median(),
            "Good_N": len(good_vals),
            "Good_Mean": good_vals.mean(),
            "Good_Median": good_vals.median(),
            "MWU_U": u_stat,
            "P_value": p_val,
            "Cohens_d": d,
            "Effect_Size": ("Large" if abs(d) > 0.8 else
                            "Medium" if abs(d) > 0.5 else
                            "Small" if abs(d) > 0.2 else "Negligible"),
            "Direction": "Severe > Good" if sev_vals.mean() > good_vals.mean() else "Good > Severe",
            "System": get_system(nt_name),
        })

        star = sig_stars(p_val)
        log.info(f"  {nt_name:<18s}: d={d:+.3f} p={p_val:.4f} {star}")

        if p_val < 0.1:  # 宽松收集
            p_collector.add("Anomalous", f"SmallLesion|{nt_name}", p_val)

    nt_compare_df = pd.DataFrame(nt_compare_results)
    if not nt_compare_df.empty:
        nt_compare_df["FDR_q"] = fdr_correct(nt_compare_df["P_value"].values)
        nt_compare_df = nt_compare_df.sort_values("P_value")

        n_sig = (nt_compare_df["P_value"] < 0.05).sum()
        n_fdr = (nt_compare_df["FDR_q"] < 0.05).sum()
        log.info(f"\n  → Resid_NT 差异: p<0.05={n_sig}, FDR<0.05={n_fdr} / {len(nt_compare_df)}")

    # ═══════════════════════════════════════════
    # Part 3: 小病灶亚组内多变量回归
    # ═══════════════════════════════════════════
    log.info(f"\n  ── 小病灶亚组内: Resid_NT → mRS (有序回归) ──")

    target_col = f"_anom_target_{mrs_col}"
    small_lesion[target_col] = small_lesion[mrs_col].apply(group_mrs)

    covars_no_tlv = [c for c in meta["covariates_all"] if c != tlv_col]

    subgroup_reg_results = []
    for nt_col in test_nt_cols:
        nt_name = bare_name(nt_col)
        predictors = [nt_col] + covars_no_tlv
        predictors = [p for p in predictors if p in small_lesion.columns]

        sub = small_lesion[[target_col] + predictors].dropna()
        if len(sub) < 30:
            continue

        sub_z = sub.copy()
        for p in predictors:
            sub_z[p] = zscore(sub_z[p])

        try:
            mod = OrderedModel(sub_z[target_col], sub_z[predictors], distr="logit")
            res = mod.fit(method="bfgs", disp=False)

            subgroup_reg_results.append({
                "NT": nt_name,
                "Beta": res.params[nt_col],
                "OR": np.exp(res.params[nt_col]),
                "P_value": res.pvalues[nt_col],
                "SE": res.bse[nt_col],
                "N": len(sub),
            })
        except Exception:
            continue

    subgroup_reg_df = pd.DataFrame(subgroup_reg_results)
    if not subgroup_reg_df.empty:
        subgroup_reg_df["FDR_q"] = fdr_correct(subgroup_reg_df["P_value"].values)
        subgroup_reg_df = subgroup_reg_df.sort_values("P_value")

        for _, r in subgroup_reg_df.head(5).iterrows():
            star = sig_stars(r["P_value"])
            log.info(f"    {r['NT']}: OR={r['OR']:.3f}, p={r['P_value']:.3e} {star}")

    # ── 合并输出 ──
    # 将 baseline + nt_compare 合到一张总表
    summary = {
        "Total_SmallLesion": n_small,
        "N_Severe": n_severe,
        "N_Good": n_good,
        "Pct_Severe": pct_severe,
        "TLV_Q1_threshold": tlv_q1,
    }
    log.info(f"\n  📌 小病灶重症比例: {pct_severe:.1f}%")

    return nt_compare_df, baseline_df, subgroup_reg_df, summary


# ==============================================================================
# v4 可视化: 新模块专用
# ==============================================================================

@safe_module
def plot_mediation_forest(med_df, out_dir):
    """中介效应 Forest Plot"""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    if med_df.empty:
        return

    for mediator in med_df["Mediator"].unique():
        sub = med_df[med_df["Mediator"] == mediator].copy()
        sub = sub.sort_values("Boot_P").head(15)
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(9, max(4, len(sub) * 0.5)))
        y = np.arange(len(sub))

        colors = ["#E64B35" if sig else "#CCCCCC" for sig in sub["Significant"]]
        ax.barh(y, sub["Indirect_ab"].values, color=colors, edgecolor="black",
                linewidth=0.5, alpha=0.85, height=0.6)
        ax.errorbar(sub["Indirect_ab"].values, y,
                    xerr=[sub["Indirect_ab"].values - sub["Boot_CI_lower"].values,
                          sub["Boot_CI_upper"].values - sub["Indirect_ab"].values],
                    fmt="none", ecolor="black", capsize=3, linewidth=1)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r.NT} (p={r.Boot_P:.3f})" for r in sub.itertuples()],
                           fontsize=8)
        ax.set_xlabel("Indirect Effect (a × b)", fontsize=11)
        ax.set_title(f"Mediation: Resid_NT → {mediator} → mRS\n"
                     f"(Bootstrap 95% CI, red = significant)",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        fig.savefig(fig_dir / f"mediation_{mediator}.png")
        plt.close(fig)
        log.info(f"  📊 mediation_{mediator}.png")


@safe_module
def plot_wmh_interaction(wmh_df, out_dir):
    """WMH 交互效应: 分层 OR 比较"""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    if wmh_df.empty:
        return

    top = wmh_df.nsmallest(10, "Interaction_P")
    if top.empty:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.8)))
    y = np.arange(len(top))
    width = 0.35

    high_or = top["HighWMH_OR"].values
    low_or = top["LowWMH_OR"].values

    ax.barh(y - width/2, high_or - 1, width, left=1, color="#E64B35",
            edgecolor="black", linewidth=0.5, alpha=0.85, label="High WMH")
    ax.barh(y + width/2, low_or - 1, width, left=1, color="#4DBBD5",
            edgecolor="black", linewidth=0.5, alpha=0.85, label="Low WMH")
    ax.axvline(1, color="gray", linestyle="--", linewidth=0.8)

    labels = []
    for r in top.itertuples():
        star = sig_stars(r.Interaction_P)
        labels.append(f"{r.NT} (p_inter={r.Interaction_P:.3e}) {star}")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("OR (per SD increase in Resid_NT)", fontsize=11)
    ax.set_title("WMH Interaction: NT Effect Stratified by WMH Level",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(fig_dir / "wmh_interaction.png")
    plt.close(fig)
    log.info(f"  📊 wmh_interaction.png")


@safe_module
def plot_anomalous_group(nt_compare_df, baseline_df, out_dir):
    """异常组分析可视化"""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # ── Cohen's d Bar Plot for Resid_NT ──
    if not nt_compare_df.empty:
        sub = nt_compare_df.sort_values("P_value").head(15)
        fig, ax = plt.subplots(figsize=(10, max(4, len(sub) * 0.5)))
        y = np.arange(len(sub))

        colors = []
        for _, r in sub.iterrows():
            if r["P_value"] < 0.05:
                colors.append("#E64B35" if r["Cohens_d"] > 0 else "#4DBBD5")
            else:
                colors.append("#CCCCCC")

        ax.barh(y, sub["Cohens_d"].values, color=colors, edgecolor="black",
                linewidth=0.5, alpha=0.85, height=0.6)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)

        # 效应量参考线
        for thresh, label_t in [(0.2, "Small"), (0.5, "Medium"), (0.8, "Large")]:
            ax.axvline(thresh, color="orange", linestyle=":", alpha=0.3)
            ax.axvline(-thresh, color="orange", linestyle=":", alpha=0.3)

        labels = [f"{r.NT} (p={r.P_value:.3f})" for r in sub.itertuples()]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Cohen's d (Severe − Good)", fontsize=11)
        ax.set_title("Small-Lesion Phenotype: Resid_NT Differences\n"
                     "(Severe mRS≥3 vs Good mRS≤2, both TLV < Q1)",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        fig.savefig(fig_dir / "anomalous_resid_nt.png")
        plt.close(fig)
        log.info(f"  📊 anomalous_resid_nt.png")

    # ── Baseline 对比 ──
    if not baseline_df.empty and HAS_SEABORN:
        fig, ax = plt.subplots(figsize=(8, max(3, len(baseline_df) * 0.5)))
        y = np.arange(len(baseline_df))
        colors = ["#E64B35" if p < 0.05 else "#CCCCCC" for p in baseline_df["P_value"]]
        d_vals = baseline_df["Cohens_d"].fillna(0).values
        ax.barh(y, d_vals, color=colors, edgecolor="black",
                linewidth=0.5, alpha=0.85, height=0.6)
        ax.axvline(0, color="gray", linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r.Variable} (p={r.P_value:.3f})"
                            for r in baseline_df.itertuples()], fontsize=8)
        ax.set_xlabel("Cohen's d")
        ax.set_title("Baseline Comparison: Small-Lesion Severe vs Good",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        fig.savefig(fig_dir / "anomalous_baseline.png")
        plt.close(fig)
        log.info(f"  📊 anomalous_baseline.png")


# ==============================================================================
# v3 原有可视化 (保留)
# ==============================================================================
@safe_module
def plot_all(regression_df, synaptic_df, interaction_df, perm_df,
             dose_df, cv_df, system_df, global_df, out_dir):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─' * 72}")
    print(f"  [Plots] 出版级可视化")
    print(f"{'─' * 72}")

    if not regression_df.empty and "Model" in regression_df.columns:
        _plot_sensitivity_forest(regression_df, fig_dir)
    if not regression_df.empty:
        _plot_volcano(regression_df, fig_dir)
    if not synaptic_df.empty:
        _plot_synaptic(synaptic_df, fig_dir)
    if not interaction_df.empty:
        _plot_interaction_heatmap(interaction_df, fig_dir)
    if not perm_df.empty:
        _plot_permutation(perm_df, fig_dir)
    if isinstance(system_df, pd.DataFrame) and not system_df.empty:
        _plot_system_pca(system_df, fig_dir)
    if not dose_df.empty:
        _plot_dose_response(dose_df, fig_dir)
    if not global_df.empty:
        _plot_global_fdr(global_df, fig_dir)


def _plot_sensitivity_forest(rdf, fig_dir):
    outcomes = rdf["Outcome"].unique()
    models = sorted(rdf["Model"].unique())
    model_colors = {"A_Unadjusted": "#FFA500", "B_Demographic": "#4DBBD5", "C_Full": "#E64B35"}

    for outcome in outcomes:
        sub = rdf[rdf["Outcome"] == outcome]
        full_model = models[-1]
        top_nts = sub[sub["Model"] == full_model].nsmallest(12, "P_value")["NT_Variable"].values
        if len(top_nts) == 0:
            continue

        fig, ax = plt.subplots(figsize=(9, max(4, len(top_nts) * 0.7)))
        for m_idx, model in enumerate(models):
            offset = (m_idx - 1) * 0.22
            m_sub = sub[(sub["Model"] == model) & (sub["NT_Variable"].isin(top_nts))]
            m_sub = m_sub.set_index("NT_Variable").reindex(top_nts)
            for i, nt in enumerate(top_nts):
                if nt not in m_sub.index:
                    continue
                row = m_sub.loc[nt]
                if pd.isna(row.get("OR")):
                    continue
                ax.errorbar(
                    row["OR"], i + offset,
                    xerr=[[row["OR"] - row["OR_CI_lower"]],
                          [row["OR_CI_upper"] - row["OR"]]],
                    fmt="o", color=model_colors.get(model, "#7F7F7F"),
                    markersize=6, capsize=3, linewidth=1.5,
                    label=model if i == 0 else None)

        ax.axvline(1, color="gray", linestyle="--", linewidth=0.8)
        ax.set_yticks(range(len(top_nts)))
        ax.set_yticklabels(top_nts, fontsize=9)
        ax.set_xlabel("Odds Ratio (95% CI)", fontsize=11)
        ax.set_title(f"{outcome} — Sensitivity", fontsize=11, fontweight="bold")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8)
        plt.tight_layout()
        fig.savefig(fig_dir / f"forest_{outcome}.png")
        plt.close(fig)
        log.info(f"  📊 forest_{outcome}.png")


def _plot_volcano(rdf, fig_dir):
    if "Model" in rdf.columns:
        rdf = rdf[rdf["Model"] == sorted(rdf["Model"].unique())[-1]]
    for outcome in rdf["Outcome"].unique():
        sub = rdf[rdf["Outcome"] == outcome].copy()
        sub["nlp"] = -np.log10(sub["P_value"].clip(1e-20))
        colors = ["#E64B35" if p < 0.05 and b > 0 else "#4DBBD5" if p < 0.05 and b < 0 else "#CCC"
                  for p, b in zip(sub["P_value"], sub["Beta"])]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(sub["Beta"], sub["nlp"], c=colors, s=60, edgecolors="black", linewidth=0.3)
        for _, r in sub[sub["P_value"] < 0.05].iterrows():
            ax.annotate(r["NT_Variable"], (r["Beta"], r["nlp"]),
                        fontsize=7, ha="center", va="bottom",
                        textcoords="offset points", xytext=(0, 5))
        ax.axhline(-np.log10(0.05), color="gray", linestyle="--")
        ax.axvline(0, color="gray", linestyle=":")
        ax.set_xlabel("β (standardized)")
        ax.set_ylabel("$-\\log_{10}(p)$")
        ax.set_title(f"Volcano — {outcome}", fontweight="bold")
        plt.tight_layout()
        fig.savefig(fig_dir / f"volcano_{outcome}.png")
        plt.close(fig)
        log.info(f"  📊 volcano_{outcome}.png")


def _plot_synaptic(synaptic_df, fig_dir):
    color_map = {"Pre-synaptic (Transporter)": "#E64B35",
                 "Post-synaptic (Receptor)": "#4DBBD5", "Tract / Other": "#228B22"}
    for outcome in synaptic_df["Outcome"].unique():
        sub = synaptic_df[synaptic_df["Outcome"] == outcome]
        fig, ax = plt.subplots(figsize=(6, 4))
        types = sub["Synaptic_Type"].values
        vals = sub["Mean_AbsBeta"].values
        colors = [color_map.get(t, "#7F7F7F") for t in types]
        bars = ax.bar(range(len(types)), vals, color=colors, edgecolor="black", alpha=0.85)
        for i, (bar, row) in enumerate(zip(bars, sub.itertuples())):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"sig={row.N_sig}/{row.N_total}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(types)))
        ax.set_xticklabels([t.split("(")[0].strip() for t in types], rotation=25, ha="right")
        ax.set_ylabel("Mean |β|")
        ax.set_title(f"Synaptic — {outcome}", fontweight="bold")
        plt.tight_layout()
        fig.savefig(fig_dir / f"synaptic_{outcome}.png")
        plt.close(fig)
        log.info(f"  📊 synaptic_{outcome}.png")


def _plot_interaction_heatmap(idf, fig_dir):
    pivot = idf.pivot_table(index="NT", columns="Inflam", values="Interaction_Beta", aggfunc="first")
    pivot_p = idf.pivot_table(index="NT", columns="Inflam", values="Interaction_P", aggfunc="first")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns)*2), max(5, len(pivot)*0.4)))
    if HAS_SEABORN:
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                    ax=ax, linewidths=0.5, cbar_kws={"label": "Interaction β"})
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                p = pivot_p.iloc[i, j]
                if pd.notna(p) and p < 0.05:
                    ax.text(j+0.5, i+0.82, "★", ha="center", va="center", fontsize=10, color="gold")
    ax.set_title("NT × Inflammation (OrderedModel)", fontsize=11)
    plt.tight_layout()
    fig.savefig(fig_dir / "interaction_heatmap.png")
    plt.close(fig)
    log.info(f"  📊 interaction_heatmap.png")


def _plot_permutation(pdf, fig_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    pdf = pdf.copy()
    pdf["nlp_param"] = -np.log10(pdf["Parametric_P"].clip(1e-20))
    pdf["nlp_perm"] = -np.log10(pdf["Permutation_P"].clip(1e-20))
    colors = ["#E64B35" if "Both" in c else "#FFA500" if "Param" in c else "#CCC"
              for c in pdf["Concordance"]]
    ax.scatter(pdf["nlp_param"], pdf["nlp_perm"], c=colors, s=60, edgecolors="black", linewidth=0.3)
    for _, r in pdf[pdf["Concordance"].str.contains("Both")].iterrows():
        ax.annotate(r["NT"], (r["nlp_param"], r["nlp_perm"]),
                    fontsize=7, textcoords="offset points", xytext=(5, 5))
    thresh = -np.log10(0.05)
    ax.axhline(thresh, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(thresh, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("$-\\log_{10}(p_{parametric})$")
    ax.set_ylabel("$-\\log_{10}(p_{permutation})$")
    ax.set_title("Parametric vs Permutation", fontweight="bold")
    plt.tight_layout()
    fig.savefig(fig_dir / "permutation.png")
    plt.close(fig)
    log.info(f"  📊 permutation.png")


def _plot_system_pca(sdf, fig_dir):
    for outcome in sdf["Outcome"].unique():
        sub = sdf[sdf["Outcome"] == outcome].sort_values("PC1_P")
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, max(3, len(sub)*0.6)))
        y = np.arange(len(sub))
        colors = [SYSTEM_COLORS.get(s, "#7F7F7F") for s in sub["System"]]
        ax.barh(y, sub["PC1_OR"].values - 1, left=1, color=colors,
                edgecolor="black", linewidth=0.5, alpha=0.85)
        ax.axvline(1, color="gray", linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r.System} (p={r.PC1_P:.3e})" for r in sub.itertuples()], fontsize=9)
        ax.set_xlabel("OR (PC1)")
        ax.set_title(f"PCA System Fingerprint — {outcome}", fontweight="bold")
        plt.tight_layout()
        fig.savefig(fig_dir / f"system_pca_{outcome}.png")
        plt.close(fig)
        log.info(f"  📊 system_pca_{outcome}.png")


def _plot_dose_response(ddf, fig_dir):
    sig = ddf[ddf["KW_P"] < 0.05].head(8)
    if sig.empty:
        return
    fig, axes = plt.subplots(2, min(4, len(sig)), figsize=(4*min(4, len(sig)), 8))
    axes = np.atleast_2d(axes).flatten()
    for i, (_, r) in enumerate(sig.iterrows()):
        if i >= len(axes):
            break
        ax = axes[i]
        q_vals = [r["Q1_mean_mRS"], r["Q2_mean_mRS"], r["Q3_mean_mRS"], r["Q4_mean_mRS"]]
        colors = ["#4CAF50", "#8BC34A", "#FF9800", "#F44336"]
        ax.bar(range(4), q_vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
        ax.set_ylabel("Mean mRS")
        ax.set_title(f"{r['NT']}\np={r['KW_P']:.3e}", fontsize=9, fontweight="bold")
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Dose-Response: NT Load Quartiles → mRS", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(fig_dir / "dose_response.png")
    plt.close(fig)
    log.info(f"  📊 dose_response.png")


def _plot_global_fdr(gdf, fig_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.scatter(gdf["P_raw"], gdf["Q_global"], s=15, alpha=0.5, c="#4DBBD5")
    ax1.axhline(0.05, color="red", linestyle="--", label="q=0.05")
    ax1.axvline(0.05, color="orange", linestyle="--", label="p=0.05")
    ax1.set_xlabel("P (raw)")
    ax1.set_ylabel("Q (global FDR)")
    ax1.set_title("Module-level p → Global q", fontweight="bold")
    ax1.legend(fontsize=8)

    mod_summary = gdf.groupby("Module").agg(
        N=("P_raw", "count"),
        N_sig_p=("P_raw", lambda x: (x < 0.05).sum()),
        N_sig_q=("Q_global", lambda x: (x < 0.05).sum()),
    ).reset_index()
    x = np.arange(len(mod_summary))
    w = 0.35
    ax2.bar(x - w/2, mod_summary["N_sig_p"]/mod_summary["N"]*100,
            w, label="p<0.05", color="#FFA500", edgecolor="black", linewidth=0.5)
    ax2.bar(x + w/2, mod_summary["N_sig_q"]/mod_summary["N"]*100,
            w, label="q_global<0.05", color="#E64B35", edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(mod_summary["Module"], rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("% significant")
    ax2.set_title("Survival Rate After Global FDR", fontweight="bold")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(fig_dir / "global_fdr.png")
    plt.close(fig)
    log.info(f"  📊 global_fdr.png")


# ==============================================================================
# 总图: 所有子图拼到一张大图上 (方便截图)
# ==============================================================================
@safe_module
def plot_combined_summary(out_dir):
    """
    扫描 figures/ 下所有已生成的 PNG, 拼成一张总图.
    方便一次截图看全貌.
    """
    fig_dir = out_dir / "figures"
    png_files = sorted(fig_dir.glob("*.png"))
    # 排除自身 (如果之前跑过)
    png_files = [f for f in png_files
                 if f.name not in ("COMBINED_ALL.png", "SUMMARY_TABLE.png")]
    if not png_files:
        log.warning("  无子图可合并")
        return

    n = len(png_files)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 5.5 * nrows))
    axes = np.atleast_2d(axes).flatten()

    for i, png in enumerate(png_files):
        try:
            img = plt.imread(str(png))
            axes[i].imshow(img)
            axes[i].set_title(png.stem, fontsize=9, fontweight="bold")
        except Exception:
            axes[i].text(0.5, 0.5, f"Load fail:\n{png.name}",
                         ha="center", va="center", fontsize=8)
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Master NT Analysis v4 — All Figures Combined",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(fig_dir / "COMBINED_ALL.png", dpi=200)
    plt.close(fig)
    log.info(f"  📊 COMBINED_ALL.png ({n} 张子图拼合)")


# ==============================================================================
# 总表: 关键指标汇总到一张图表上 (方便截图)
# ==============================================================================
@safe_module
def plot_summary_table(out_dir, regression_df, interaction_df,
                      med_parallel_df, wmh_df, anom_nt_df,
                      global_df, anom_summary):
    """
    把所有模块的核心指标汇总成一张 figure‐table,
    直接截图就能给审稿人/导师看.
    """
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    rows_data = []  # (Module, Variable, Metric, Value, P, Sig)

    # ── 1. Core Regression Top 5 ──
    if isinstance(regression_df, pd.DataFrame) and not regression_df.empty:
        if "Model" in regression_df.columns:
            full_m = sorted(regression_df["Model"].unique())[-1]
            full = regression_df[regression_df["Model"] == full_m]
        else:
            full = regression_df
        for _, r in full.nsmallest(5, "P_value").iterrows():
            rows_data.append((
                "Regression", r["NT_Variable"],
                f"OR={r['OR']:.3f} [{r.get('OR_CI_lower',np.nan):.2f}-{r.get('OR_CI_upper',np.nan):.2f}]",
                f"{r['P_value']:.2e}",
                r.get("Sensitivity", ""),
            ))

    # ── 2. Mediation Top 5 ──
    if isinstance(med_parallel_df, pd.DataFrame) and not med_parallel_df.empty:
        sig_med = med_parallel_df.nsmallest(5, "Boot_P")
        for _, r in sig_med.iterrows():
            ci = f"[{r['Boot_CI_lower']:.4f}, {r['Boot_CI_upper']:.4f}]"
            tag = "✓" if r["Significant"] else "—"
            rows_data.append((
                f"Mediation({r['Mediator']})", r["NT"],
                f"ab={r['Indirect_ab']:.4f} {ci}",
                f"{r['Boot_P']:.3f}", tag,
            ))

    # ── 3. WMH Interaction Top 5 ──
    if isinstance(wmh_df, pd.DataFrame) and not wmh_df.empty:
        for _, r in wmh_df.nsmallest(5, "Interaction_P").iterrows():
            rows_data.append((
                "WMH×NT", r["NT"],
                f"OR_inter={r['Interaction_OR']:.3f}, Hi/Lo={r.get('OR_Ratio_High_vs_Low',np.nan):.2f}",
                f"{r['Interaction_P']:.2e}",
                sig_stars(r["Interaction_P"]),
            ))

    # ── 4. Anomalous Group Top 5 ──
    if isinstance(anom_nt_df, pd.DataFrame) and not anom_nt_df.empty:
        for _, r in anom_nt_df.nsmallest(5, "P_value").iterrows():
            rows_data.append((
                "SmallLesion", r["NT"],
                f"d={r['Cohens_d']:.3f} ({r['Effect_Size']})",
                f"{r['P_value']:.3e}",
                r["Direction"][:10],
            ))

    # ── 5. Global FDR summary ──
    if isinstance(global_df, pd.DataFrame) and not global_df.empty:
        n_total = len(global_df)
        n_survive = (global_df["Q_global"] < 0.05).sum()
        rows_data.append((
            "Global FDR", "ALL",
            f"{n_survive}/{n_total} survive (q<0.05)",
            "", "",
        ))

    # ── 6. Anomalous summary ──
    if isinstance(anom_summary, dict) and anom_summary:
        rows_data.append((
            "SmallLesion", "Overview",
            f"N={anom_summary.get('Total_SmallLesion','?')}, "
            f"Severe={anom_summary.get('Pct_Severe',0):.1f}%",
            "", "",
        ))

    if not rows_data:
        log.warning("  无指标可汇总")
        return

    col_labels = ["Module", "Variable", "Key Metric", "P / Info", "Note"]

    fig, ax = plt.subplots(figsize=(18, max(4, 0.45 * len(rows_data) + 2)))
    ax.axis("off")

    table = ax.table(
        cellText=rows_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    # 表头样式
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=10)

    # 交替行色
    for i in range(1, len(rows_data) + 1):
        bg = "#F7F9FC" if i % 2 == 0 else "#FFFFFF"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(bg)
            table[i, j].set_edgecolor("#DEE2E6")

    ax.set_title("Master NT Analysis v4 — Key Results Summary\n"
                 f"(Generated {datetime.now().strftime('%Y-%m-%d %H:%M')})",
                 fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    fig.savefig(fig_dir / "SUMMARY_TABLE.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  📊 SUMMARY_TABLE.png ({len(rows_data)} 行指标)")


# ==============================================================================
# 输出
# ==============================================================================
def save_results(out_dir, **dataframes):
    print(f"\n{'─' * 72}")
    print(f"  [Output] 保存")
    print(f"{'─' * 72}")

    for name, obj in dataframes.items():
        if isinstance(obj, pd.DataFrame) and not obj.empty:
            obj.to_csv(out_dir / f"{name}.csv", index=False)
            log.info(f"  💾 {name}.csv ({len(obj)} 行)")
        elif isinstance(obj, tuple):
            for k, sub_df in enumerate(obj):
                if isinstance(sub_df, pd.DataFrame) and not sub_df.empty:
                    sub_df.to_csv(out_dir / f"{name}_{k}.csv", index=False)
        elif isinstance(obj, dict):
            import json
            with open(out_dir / f"{name}.json", "w") as f:
                json.dump(obj, f, indent=2, default=str)
            log.info(f"  💾 {name}.json")

    xlsx = out_dir / "Master_NT_Results_v4.xlsx"
    try:
        with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
            for name, obj in dataframes.items():
                if isinstance(obj, pd.DataFrame) and not obj.empty:
                    obj.to_excel(w, sheet_name=name[:31], index=False)
                elif isinstance(obj, tuple):
                    for k, sub_df in enumerate(obj):
                        if isinstance(sub_df, pd.DataFrame) and not sub_df.empty:
                            sub_df.to_excel(w, sheet_name=f"{name}_{k}"[:31], index=False)
        log.info(f"  📊 Excel → {xlsx.name}")
    except Exception as e:
        log.warning(f"  Excel 失败: {e}")


def generate_report(out_dir, regression_df, interaction_df, perm_df,
                    system_result, global_df, dose_df, cv_df,
                    med_result, wmh_df, anom_result):
    """v4 报告: 包含新增模块"""
    report = out_dir / "Analysis_Report_v4.md"
    lines = [
        "# Master NT Analysis v4 — Report",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## v4 New Modules",
        "",
        "### Module 11: HRV Mediation Analysis (Brain-Heart Axis)",
        "- **Path**: Resid_NT → log(RMSSD) → m12_mRS",
        "- **Controls**: TLV + NIHSS + Age + Sex + HRmean",
        "- **Method**: Bootstrap 5000, percentile CI",
        "",
    ]

    # 中介分析结果
    if isinstance(med_result, tuple) and len(med_result) >= 1:
        med_df = med_result[0]
        if isinstance(med_df, pd.DataFrame) and not med_df.empty:
            sig_med = med_df[med_df["Significant"] == True]
            lines.append(f"**Significant mediations**: {len(sig_med)}/{len(med_df)}")
            lines.append("")
            if not sig_med.empty:
                lines.append("| NT | Mediator | Indirect (a×b) | 95% CI | Prop. Mediated |")
                lines.append("|-----|----------|----------------|--------|----------------|")
                for _, r in sig_med.head(10).iterrows():
                    pm = f"{r['Proportion_Mediated']:.1%}" if pd.notna(r['Proportion_Mediated']) else "—"
                    lines.append(f"| {r['NT']} | {r['Mediator']} | {r['Indirect_ab']:.4f} | "
                                 f"[{r['Boot_CI_lower']:.4f}, {r['Boot_CI_upper']:.4f}] | {pm} |")
            lines.append("")

        if len(med_result) >= 2:
            serial_df = med_result[1]
            if isinstance(serial_df, pd.DataFrame) and not serial_df.empty:
                lines.append("**Serial Mediation (X→RMSSD→IL6→mRS)**:")
                for _, r in serial_df.iterrows():
                    sig_tag = "✓" if r["Significant"] else "—"
                    lines.append(f"- {r['NT']}: indirect={r['Serial_Indirect']:.4f} "
                                 f"[{r['CI_lower']:.4f}, {r['CI_upper']:.4f}] {sig_tag}")
                lines.append("")

    lines.append("### Module 12: WMH Interaction Effect")
    lines.append("- **Model**: mRS ~ Resid_NT × WMH + covariates")
    lines.append("")

    if isinstance(wmh_df, pd.DataFrame) and not wmh_df.empty:
        sig_wmh = wmh_df[wmh_df["Interaction_P"] < 0.05]
        lines.append(f"**Significant WMH interactions**: {len(sig_wmh)}/{len(wmh_df)}")
        if not sig_wmh.empty:
            lines.append("")
            lines.append("| NT | Interaction OR | P | High WMH OR | Low WMH OR |")
            lines.append("|----|---------------|---|-------------|------------|")
            for _, r in sig_wmh.head(10).iterrows():
                lines.append(f"| {r['NT']} | {r['Interaction_OR']:.3f} | "
                             f"{r['Interaction_P']:.3e} | {r.get('HighWMH_OR', np.nan):.3f} | "
                             f"{r.get('LowWMH_OR', np.nan):.3f} |")
        lines.append("")

    lines.append("### Module 13: Small-Lesion Severe-Outcome Phenotype")
    lines.append("")
    if isinstance(anom_result, tuple) and len(anom_result) >= 4:
        summary = anom_result[3]
        if isinstance(summary, dict):
            lines.append(f"- Small-lesion group (TLV < Q1): {summary.get('Total_SmallLesion', '?')}")
            lines.append(f"- Severe (mRS ≥ 3): {summary.get('N_Severe', '?')} "
                         f"({summary.get('Pct_Severe', 0):.1f}%)")

        nt_comp = anom_result[0]
        if isinstance(nt_comp, pd.DataFrame) and not nt_comp.empty:
            sig_nt = nt_comp[nt_comp["P_value"] < 0.05]
            lines.append(f"- Resid_NT differences (p<0.05): {len(sig_nt)}/{len(nt_comp)}")
            if not sig_nt.empty:
                lines.append("")
                lines.append("| NT | Cohen's d | P | Direction |")
                lines.append("|----|-----------|---|-----------|")
                for _, r in sig_nt.head(10).iterrows():
                    lines.append(f"| {r['NT']} | {r['Cohens_d']:.3f} | "
                                 f"{r['P_value']:.3e} | {r['Direction']} |")
    lines.append("")

    # v3 原有部分
    lines.append("## Core Analysis (v3)")
    lines.append("")

    if not regression_df.empty:
        if "Model" in regression_df.columns:
            full = regression_df[regression_df["Model"] == sorted(regression_df["Model"].unique())[-1]]
        else:
            full = regression_df
        top5 = full.nsmallest(5, "P_value")
        lines.append("### Top 5 NT Variables (Full Model)")
        lines.append("")
        lines.append("| Outcome | NT | OR | 95% CI | P | Sensitivity |")
        lines.append("|---------|----|----|--------|---|-------------|")
        for _, r in top5.iterrows():
            ci = f"[{r.get('OR_CI_lower', np.nan):.2f}, {r.get('OR_CI_upper', np.nan):.2f}]"
            sens = r.get("Sensitivity", "")
            lines.append(f"| {r['Outcome']} | {r['NT_Variable']} | "
                         f"{r['OR']:.3f} | {ci} | {r['P_value']:.3e} | {sens} |")
        lines.append("")

    if not global_df.empty:
        n_survive = (global_df["Q_global"] < 0.05).sum()
        lines.append(f"### Global FDR")
        lines.append(f"- {n_survive}/{len(global_df)} tests survive global BH correction")
        lines.append("")

    lines.append("---")
    lines.append("*Master_NT_Analysis_v4.py — 心脑轴 + 白质病变 + 异常亚组*")

    report.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"  📝 报告 → {report.name}")


# ==============================================================================
# 主函数
# ==============================================================================
def analyze_all(input_csv, output_dir="6.furtherv4",
                n_perm=1000, skip_perm=False, n_boot=5000):
    t0 = time.time()

    # ── 强制输出目录: 6.NeurotransmitterMapping/v4/ ──
    csv_path = Path(input_csv).resolve()
    candidate = Path(output_dir)
    if candidate.is_absolute():
        out = candidate
    else:
        # 向上寻找 6.NeurotransmitterMapping 目录
        parent = csv_path.parent
        while parent != parent.parent:
            if "6.NeurotransmitterMapping" in parent.name:
                out = parent / output_dir
                break
            parent = parent.parent
        else:
            out = csv_path.parent / output_dir
    # 二次检查: 绝不放在 ~home
    home = Path.home()
    if out == home or out.parent == home:
        out = csv_path.parent.parent / "v4"
        log.warning(f"  ⚠️ 输出目录回退: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    log.info(f"  📁 结果输出目录: {out}")

    p_collector = GlobalPCollector()

    # ── 数据加载 ──
    df, meta = load_data(input_csv)
    if not meta["nt_cols"]:
        log.error("❌ 未找到 NT 载荷列")
        return

    # ── Module 0: 多重插补 ──
    df = multiple_imputation(df, meta)

    # ── Module 1: 诊断残差 ──
    diag_df = diagnostic_residuals(df, meta)

    if meta.get("resid_report"):
        resid_rpt = pd.DataFrame(meta["resid_report"])
        resid_rpt.to_csv(out / "koch_residual_report.csv", index=False)

    # ── Module 2: 有序回归 (核心) ──
    regression_df = direct_ordinal_regression(df, meta, p_collector)

    # ── Module 3: 突触定位 ──
    synaptic_df = synaptic_analysis(regression_df)

    # ── Module 4: 炎症交互 ──
    interaction_df = interaction_analysis(df, meta, p_collector)

    # ── Module 5: 复发 ──
    recurrence_df = recurrence_analysis(df, meta, p_collector)

    # ── Module 6: PCA 系统 ──
    system_result = pca_system_fingerprint(df, meta, p_collector)
    if isinstance(system_result, tuple):
        system_df, loadings_df = system_result
    else:
        system_df, loadings_df = system_result, pd.DataFrame()

    # ── Module 7: 置换检验 ──
    perm_n = 0 if skip_perm else n_perm
    perm_df = permutation_test(df, meta, p_collector, n_perm=perm_n)

    # ── Module 8: 阈值剂量-反应 ──
    dose_df = threshold_dose_response(df, meta, p_collector)

    # ── Module 9: 10-fold CV ──
    cv_df = kfold_cv(df, meta)

    # ── Module 9b: mRS 灵敏度 ──
    top_nt_names = None
    if not regression_df.empty:
        if "Model" in regression_df.columns:
            full_m = sorted(regression_df["Model"].unique())[-1]
            top5 = regression_df[regression_df["Model"] == full_m].nsmallest(5, "P_value")
        else:
            top5 = regression_df.nsmallest(5, "P_value")
        top_nt_names = top5["NT_Variable"].tolist()
    mrs_sens_df = mrs_cutpoint_sensitivity(df, meta, top_nts=top_nt_names)

    # ── Module 9c: Spin Test ──
    spin_df = spin_test_nt_specificity(df, meta, p_collector, n_spin=perm_n)

    # ── Module 9d: DCA ──
    dca_df = decision_curve_analysis(df, meta, out)

    # ══════════════════════════════════════════════════════════════
    #  v4 新增模块
    # ══════════════════════════════════════════════════════════════

    # ── Module 11a: Holter 缺失偏倚检验 ──
    holter_bias_df = holter_missing_bias_test(df, meta)

    # ── Module 11b: HRV 中介分析 ──
    med_result = hrv_mediation_analysis(df, meta, p_collector, n_boot=n_boot)
    if isinstance(med_result, tuple):
        med_parallel_df = med_result[0] if len(med_result) > 0 else pd.DataFrame()
        med_serial_df = med_result[1] if len(med_result) > 1 else pd.DataFrame()
    else:
        med_parallel_df = med_result if isinstance(med_result, pd.DataFrame) else pd.DataFrame()
        med_serial_df = pd.DataFrame()

    # ── Module 12: WMH 交互 ──
    wmh_df = wmh_interaction_analysis(df, meta, p_collector)

    # ── Module 13: 异常组分析 ──
    anom_result = anomalous_group_analysis(df, meta, p_collector)
    if isinstance(anom_result, tuple) and len(anom_result) >= 3:
        anom_nt_df = anom_result[0] if isinstance(anom_result[0], pd.DataFrame) else pd.DataFrame()
        anom_baseline_df = anom_result[1] if isinstance(anom_result[1], pd.DataFrame) else pd.DataFrame()
        anom_reg_df = anom_result[2] if isinstance(anom_result[2], pd.DataFrame) else pd.DataFrame()
        anom_summary = anom_result[3] if len(anom_result) > 3 else {}
    else:
        anom_nt_df = pd.DataFrame()
        anom_baseline_df = pd.DataFrame()
        anom_reg_df = pd.DataFrame()
        anom_summary = {}

    # ── Module 10: 全局 FDR ──
    global_df = global_fdr(p_collector, out)

    # ══════════════════════════════════════════════════════════════
    # 可视化
    # ══════════════════════════════════════════════════════════════
    plot_all(regression_df, synaptic_df, interaction_df, perm_df,
             dose_df, cv_df, system_df, global_df, out)

    # v4 新图
    plot_mediation_forest(med_parallel_df, out)
    plot_wmh_interaction(wmh_df, out)
    plot_anomalous_group(anom_nt_df, anom_baseline_df, out)

    # ── 汇总表 (先画, 再拼总图) ──
    plot_summary_table(out, regression_df, interaction_df,
                       med_parallel_df, wmh_df, anom_nt_df,
                       global_df, anom_summary)

    # ── 所有子图拼合到一张总图 ──
    plot_combined_summary(out)

    # ══════════════════════════════════════════════════════════════
    # 保存
    # ══════════════════════════════════════════════════════════════
    save_results(
        out,
        diagnostic_residuals=diag_df,
        ordinal_regression=regression_df,
        synaptic_location=synaptic_df,
        interaction=interaction_df,
        recurrence=recurrence_df,
        pca_system=system_df,
        pca_loadings=loadings_df,
        permutation_test=perm_df,
        dose_response=dose_df,
        cv_10fold=cv_df,
        mrs_sensitivity=mrs_sens_df,
        spin_test=spin_df,
        dca=dca_df,
        # v4 新增
        holter_bias=holter_bias_df,
        mediation_parallel=med_parallel_df,
        mediation_serial=med_serial_df,
        wmh_interaction=wmh_df,
        anomalous_nt_compare=anom_nt_df,
        anomalous_baseline=anom_baseline_df,
        anomalous_subgroup_reg=anom_reg_df,
        anomalous_summary=anom_summary,
        global_fdr_all=global_df,
    )

    # 报告
    generate_report(out, regression_df, interaction_df, perm_df,
                    (system_df, loadings_df), global_df, dose_df, cv_df,
                    (med_parallel_df, med_serial_df), wmh_df,
                    (anom_nt_df, anom_baseline_df, anom_reg_df, anom_summary))

    # ══════════════════════════════════════════════════════════════
    # 终端摘要
    # ══════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"  ✅ v4 完成！ {elapsed:.1f}s")
    print(f"{'=' * 72}")
    print(f"  📁 {out.resolve()}")

    if not regression_df.empty:
        if "Model" in regression_df.columns:
            full = regression_df[regression_df["Model"] == sorted(regression_df["Model"].unique())[-1]]
        else:
            full = regression_df
        top = full.nsmallest(5, "P_value")
        print(f"\n  ⭐ Top 5 NT:")
        for _, r in top.iterrows():
            print(f"     {r['Outcome']} | {r['NT_Variable']}: "
                  f"OR={r['OR']:.3f}, p={r['P_value']:.2e} {sig_stars(r['P_value'])} "
                  f"[{r.get('Sensitivity', '')}]")

    if not global_df.empty:
        n_survive = (global_df["Q_global"] < 0.05).sum()
        print(f"\n  🔒 全局 FDR: {n_survive}/{len(global_df)} 通过校正")

    # v4 摘要
    print(f"\n  🫀 v4 新增模块摘要:")
    if isinstance(med_parallel_df, pd.DataFrame) and not med_parallel_df.empty:
        n_med_sig = med_parallel_df["Significant"].sum() if "Significant" in med_parallel_df.columns else 0
        print(f"     Module 11 (HRV中介): {n_med_sig}/{len(med_parallel_df)} 显著")
    else:
        print(f"     Module 11 (HRV中介): 无 RMSSD 数据或中介不显著")

    if isinstance(wmh_df, pd.DataFrame) and not wmh_df.empty:
        n_wmh_sig = (wmh_df["Interaction_P"] < 0.05).sum()
        print(f"     Module 12 (WMH交互): {n_wmh_sig}/{len(wmh_df)} 个交互显著")
    else:
        print(f"     Module 12 (WMH交互): 无 WMH 数据")

    if isinstance(anom_nt_df, pd.DataFrame) and not anom_nt_df.empty:
        n_anom_sig = (anom_nt_df["P_value"] < 0.05).sum()
        print(f"     Module 13 (异常组): {n_anom_sig}/{len(anom_nt_df)} 个 NT 差异显著")
        if isinstance(anom_summary, dict):
            print(f"     小病灶重症率: {anom_summary.get('Pct_Severe', 0):.1f}%")
    else:
        print(f"     Module 13 (异常组): 数据不足")

    print(f"\n  🎯 审稿人级关注点 (v4 更新):")
    print(f"     1. ordinal_regression.csv → Sensitivity='★★★ Robust'")
    print(f"     2. global_fdr.csv → Q_global < 0.05 才是铁证")
    print(f"     3. mediation_parallel.csv → RMSSD/IL-6 中介路径 (心脑轴)")
    print(f"     4. holter_bias.csv → RMSSD 缺失偏倚是否可忽略")
    print(f"     5. wmh_interaction.csv → WMH 放大递质损毁效应")
    print(f"     6. anomalous_nt_compare.csv → 小病灶重症的递质机制")
    print(f"     7. anomalous_subgroup_reg.csv → 亚组内独立预测")
    print(f"     8. figures/mediation_*.png, wmh_interaction.png, anomalous_*.png")
    print(f"     9. figures/COMBINED_ALL.png → 📸 所有子图拼合总图")
    print(f"    10. figures/SUMMARY_TABLE.png → 📸 关键指标汇总表")
    print(f"\n  📂 完整结果路径: {out.resolve()}")
    print(f"  📂 截图用总图: {(out / 'figures' / 'COMBINED_ALL.png').resolve()}")
    print(f"  📂 截图用表格: {(out / 'figures' / 'SUMMARY_TABLE.png').resolve()}")


# ==============================================================================
# CLI
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Master NT v4 — 心脑轴 + 白质病变 + 异常亚组",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
推荐:
  python3 Master_NT_Analysis_v4.py --input merged_neuro_data.csv --skip-perm
  python3 Master_NT_Analysis_v4.py --input merged_neuro_data.csv --n-perm 1000
  python3 Master_NT_Analysis_v4.py --input merged_neuro_data.csv --n-boot 2000 --skip-perm

后台运行 (服务器推荐):
  nohup python3 /data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/v4/Master_NT_Analysis_v4.py \\
        -i /data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/3.variable_outcom_merge_data/merged_neuro_data.csv \\
        -o /data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/v4 \\
        --n-perm 1000 --n-boot 5000 \\
        > /data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/v4/analysis.log 2>&1 &
        """
    )
    parser.add_argument("-i", "--input", default="merged_neuro_data.csv")
    parser.add_argument("-o", "--output", default="6.furtherv4")
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--n-boot", type=int, default=5000,
                        help="Bootstrap iterations for mediation (default: 5000)")
    parser.add_argument("--skip-perm", action="store_true")
    args = parser.parse_args()

    csv_path = args.input
    if not Path(csv_path).exists():
        server_candidates = [
            "/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/"
            "3.variable_outcom_merge_data/merged_neuro_data.csv",
            "/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/"
            "variable_outcom_merge_data/merged_neuro_data.csv",
        ]
        found = False
        for server in server_candidates:
            if Path(server).exists():
                csv_path = server
                log.info(f"使用服务器数据: {server}")
                found = True
                break
        if not found:
            log.error(f"❌ 找不到: {args.input}")
            log.error(f"   尝试过: {server_candidates}")
            sys.exit(1)

    analyze_all(csv_path, args.output,
                n_perm=args.n_perm, skip_perm=args.skip_perm,
                n_boot=args.n_boot)


if __name__ == "__main__":
    main()
