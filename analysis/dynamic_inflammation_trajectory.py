#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Inflammation Trajectory Analysis
=============================================================================
回应同行评议建议："胆碱能网络受损的病人，不仅基线炎症高，而且他们的
炎症'刹不住车'（持续高亢），而网络完好的病人炎症能迅速回落。"

核心思路:
  既往主分析只用了入院基线 IL-6 / hsCRP，CNSR-III 队列 hsCRP 实际有三个
  时间点 (baseline / 3 个月 / 12 个月)；IL-6 只在基线测量，无法做轨迹。
  本脚本利用 hsCRP 三时间点子队列检验:

    H1) "刹不住车"表型与 12 月不良结局相关 (Sustained > 3 mg/L at M03)
    H2) 胆碱能损伤程度与 hsCRP 不能消退 (slope ≥ 0) 有关
    H3) Resid_CHA × hsCRP slope 对 12 月 mRS 存在交互效应
    H4) 四细胞加性交互的升级版 (CHA × sustained-hsCRP)

数据列 (基于 merged_neuro_data.csv 实际探测):
  BSL_hsCRP_multic   n=2890 (80.7%)
  M03_hsCRP_multic   n=1436 (40.1%)
  M12_hsCRP          n=1099 (30.7%)
  三时间点 + m12_mRS 全有: N=820
  BSL+M03 双时间点 + m12_mRS: N=1334  ← 主分析子队列

输出 (out/dynamic_inflammation/):
  trajectory_descriptive_by_CHA_tertile.csv
  slope_by_CHA_tertile.csv
  sustained_phenotype_2x2.csv
  dynamic_interaction_ordinal.csv
  additive_interaction_sustained.csv
  FigS7_trajectory.png (3 panels)
  log.txt

用法:
  cd /data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/3.variable_outcom_merge_data
  python3 /path/to/dynamic_inflammation_trajectory.py
  # 或自定义路径
  python3 dynamic_inflammation_trajectory.py --input <path> --outdir <out>

依赖: pandas numpy scipy statsmodels matplotlib seaborn
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

warnings.filterwarnings("ignore")

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("DynInflam")

# ==============================================================================
# 常量
# ==============================================================================
DEFAULT_INPUT = (
    "/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/"
    "3.variable_outcom_merge_data/merged_neuro_data.csv"
)
DEFAULT_OUTDIR = "/data/usersdir/liuzhengxin/Stepbystep/7.figure/S7/out"

# 列名 (基于实际数据探测)
COL_BSL = "BSL_hsCRP_multic"
COL_M03 = "M03_hsCRP_multic"
COL_M12 = "M12_hsCRP"           # 无 multic 版本

# AHA 高心血管风险阈值 (Ridker et al.)
SUSTAINED_THRESHOLD = 3.0  # mg/L

# 主交互对子 (与 Fig. 3 最强对子一致)
CHA_COL_RESID = "Resid_human_CHA"
CHA_COL_RAW   = "human_CHA"

# ==============================================================================
# 工具函数
# ==============================================================================
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def fmt_n_pct(n, total):
    return f"{n} ({n/total*100:.1f}%)"


def log1p_safe(s):
    s = pd.to_numeric(s, errors="coerce")
    return np.log1p(s.clip(lower=0))


def wilson_ci(k, n, alpha=0.05):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


# ==============================================================================
# 数据装载 + 构造子队列
# ==============================================================================
def load_and_build_subcohort(input_path):
    log.info(f"加载数据: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    log.info(f"  总样本 N = {len(df)}, 总列数 = {df.shape[1]}")

    # 必需列检查
    needed = [COL_BSL, COL_M03, COL_M12, "m12_mRS"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        log.error(f"  缺失关键列: {missing}")
        sys.exit(1)

    # CHA 列容错
    cha_col = find_col(df, [CHA_COL_RESID, "Resid_human_CHA"])
    cha_raw = find_col(df, [CHA_COL_RAW, "human_CHA", "Load_human_CHA"])
    if cha_col is None and cha_raw is None:
        log.error("  未找到 human_CHA / Resid_human_CHA 列")
        sys.exit(1)

    # 协变量
    covariates = {
        "TLV":   find_col(df, ["TLV", "TLV_mm3", "tlv"]),
        "NIHSS": find_col(df, ["A_NIHSS", "NIHSS", "nihss", "BSL_NIHSS"]),
        "Age":   find_col(df, ["AGE", "Age", "age"]),
        "Sex":   find_col(df, ["SEX", "Sex", "sex"]),
    }
    log.info(f"  CHA 残差列: {cha_col} | CHA 原始列: {cha_raw}")
    log.info(f"  协变量: {covariates}")

    # 类型转换
    for c in [COL_BSL, COL_M03, COL_M12, "m12_mRS", cha_col, cha_raw] + list(covariates.values()):
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 派生变量
    df["log_BSL"] = log1p_safe(df[COL_BSL])
    df["log_M03"] = log1p_safe(df[COL_M03])
    df["log_M12"] = log1p_safe(df[COL_M12])

    # 衰减斜率 (每月 log 单位)
    df["slope_BSL_M03"]  = (df["log_M03"] - df["log_BSL"]) / 3.0    # /月
    df["slope_BSL_M12"]  = (df["log_M12"] - df["log_BSL"]) / 12.0

    # 表型标签
    df["sustained_M03"] = (df[COL_M03] >= SUSTAINED_THRESHOLD).astype("Int64")
    df["sustained_M12"] = (df[COL_M12] >= SUSTAINED_THRESHOLD).astype("Int64")

    # 子队列定义
    mask_dual = df[[COL_BSL, COL_M03, "m12_mRS"]].notna().all(axis=1)
    mask_tri  = df[[COL_BSL, COL_M03, COL_M12, "m12_mRS"]].notna().all(axis=1)

    log.info(f"  双时间点子队列 (BSL+M03+mRS): N = {mask_dual.sum()}")
    log.info(f"  三时间点子队列 (BSL+M03+M12+mRS): N = {mask_tri.sum()}")

    return df, mask_dual, mask_tri, cha_col, cha_raw, covariates


# ==============================================================================
# Analysis 1: 三时间点描述性 (by CHA tertile)
# ==============================================================================
def descriptive_by_cha_tertile(df, mask, cha_col, outdir):
    log.info("─" * 70)
    log.info("Analysis 1: 三时间点描述性统计 (by CHA tertile)")
    sub = df[mask].copy()

    # CHA 三分位
    sub["CHA_tertile"] = pd.qcut(sub[cha_col], 3, labels=["T1_low", "T2_mid", "T3_high"])

    rows = []
    for tertile in ["T1_low", "T2_mid", "T3_high"]:
        g = sub[sub["CHA_tertile"] == tertile]
        for tp, col in [("BSL", COL_BSL), ("M03", COL_M03), ("M12", COL_M12)]:
            vals = g[col].dropna()
            if len(vals) > 0:
                rows.append({
                    "CHA_tertile": tertile,
                    "timepoint":   tp,
                    "N":           len(vals),
                    "median":      vals.median(),
                    "Q1":          vals.quantile(0.25),
                    "Q3":          vals.quantile(0.75),
                    "mean":        vals.mean(),
                    "SD":          vals.std(),
                })

    res = pd.DataFrame(rows)
    fp = outdir / "trajectory_descriptive_by_CHA_tertile.csv"
    res.to_csv(fp, index=False)
    log.info(f"  → {fp.name}")
    log.info(f"\n{res.to_string(index=False)}")
    return res


# ==============================================================================
# Analysis 2: hsCRP slope ~ Resid_CHA 关联
# ==============================================================================
def slope_vs_cha(df, mask, cha_col, covariates, outdir, slope_col="slope_BSL_M03"):
    log.info("─" * 70)
    log.info(f"Analysis 2: hsCRP 衰减斜率 ({slope_col}) ~ Resid_CHA + 协变量")
    sub = df[mask].copy()

    # CHA 三分位上的 slope 分布
    sub["CHA_tertile"] = pd.qcut(sub[cha_col], 3, labels=["T1_low", "T2_mid", "T3_high"])
    by_t = sub.groupby("CHA_tertile")[slope_col].describe()
    log.info(f"\nslope ({slope_col}, 每月 log 单位) by CHA tertile:\n{by_t}")

    # Kruskal-Wallis 三组比较
    groups = [sub[sub["CHA_tertile"] == t][slope_col].dropna()
              for t in ["T1_low", "T2_mid", "T3_high"]]
    H, p_kw = stats.kruskal(*groups)

    # Spearman 单调相关
    pair = sub[[cha_col, slope_col]].dropna()
    rho, p_sp = stats.spearmanr(pair[cha_col], pair[slope_col])

    # OLS 回归 (调整协变量)
    cov_cols = [c for c in covariates.values() if c]
    ols_cols = [cha_col, slope_col] + cov_cols
    ols_df = sub[ols_cols].dropna()
    X = sm.add_constant(ols_df[[cha_col] + cov_cols])
    y = ols_df[slope_col]
    fit = sm.OLS(y, X).fit()
    beta = fit.params[cha_col]
    se = fit.bse[cha_col]
    p_ols = fit.pvalues[cha_col]
    ci_lo, ci_hi = fit.conf_int().loc[cha_col]

    log.info(f"\nKruskal-Wallis: H={H:.3f}, p={p_kw:.4g}")
    log.info(f"Spearman ρ (CHA, slope): {rho:+.3f}, p={p_sp:.4g}")
    log.info(f"OLS β(CHA → slope) adj. for {cov_cols}: "
             f"{beta:+.4f} (SE {se:.4f}, 95% CI {ci_lo:+.4f} to {ci_hi:+.4f}), p={p_ols:.4g}")
    log.info(f"  N = {len(ols_df)}")

    # 输出
    rows = [
        {"test": "Kruskal-Wallis (3 tertiles)", "statistic": H, "p_value": p_kw, "N": len(pair)},
        {"test": "Spearman rho",                "statistic": rho, "p_value": p_sp, "N": len(pair)},
        {"test": "OLS β(CHA→slope, adj.)",      "statistic": beta, "p_value": p_ols, "N": len(ols_df)},
    ]
    res = pd.DataFrame(rows)
    fp = outdir / "slope_by_CHA_tertile.csv"
    res.to_csv(fp, index=False)
    by_t.to_csv(outdir / "slope_descriptive_by_CHA_tertile.csv")
    log.info(f"  → {fp.name}")
    return res, sub


# ==============================================================================
# Analysis 3: Sustained-hsCRP 表型 vs 12-mo mRS
# ==============================================================================
def sustained_vs_outcome(df, mask, covariates, outdir):
    log.info("─" * 70)
    log.info("Analysis 3: Sustained-hsCRP 表型 (M03 ≥ 3 mg/L) vs 12-month mRS")
    sub = df[mask].copy()
    sub["poor"] = (sub["m12_mRS"] > 2).astype(int)
    sub["sustained"] = (sub[COL_M03] >= SUSTAINED_THRESHOLD).astype(int)

    # 2x2 计数
    ct = pd.crosstab(sub["sustained"], sub["poor"], margins=True)
    log.info(f"\n2x2 表 (Sustained × Poor outcome):\n{ct}")

    # 各组不良结局比例
    rows = []
    for grp_label, grp in [("Resolver (M03<3)", sub[sub["sustained"] == 0]),
                            ("Sustained (M03≥3)", sub[sub["sustained"] == 1])]:
        n = len(grp); k = grp["poor"].sum()
        lo, hi = wilson_ci(k, n)
        rows.append({"group": grp_label, "N": n, "poor_outcome_n": k,
                     "poor_outcome_pct": 100 * k / n,
                     "Wilson_95CI_lo": 100 * lo, "Wilson_95CI_hi": 100 * hi})
    res = pd.DataFrame(rows)
    log.info(f"\n{res.to_string(index=False)}")

    # 卡方 + Fisher (双侧)
    table22 = pd.crosstab(sub["sustained"], sub["poor"]).values
    chi2, p_chi, dof, _ = stats.chi2_contingency(table22, correction=False)
    odds, p_fisher = stats.fisher_exact(table22)

    # 调整后 logistic
    cov_cols = [c for c in covariates.values() if c]
    fit_cols = ["sustained", "log_BSL", "poor"] + cov_cols
    fit_df = sub[fit_cols].dropna()
    X = sm.add_constant(fit_df[["sustained", "log_BSL"] + cov_cols])
    y = fit_df["poor"]
    logit = sm.Logit(y, X).fit(disp=0)
    OR_sustained = np.exp(logit.params["sustained"])
    ci_lo, ci_hi = np.exp(logit.conf_int().loc["sustained"])
    p_adj = logit.pvalues["sustained"]

    log.info(f"\nχ² unadjusted: χ²={chi2:.3f}, p={p_chi:.4g}")
    log.info(f"Fisher unadjusted: OR={odds:.3f}, p={p_fisher:.4g}")
    log.info(f"Logistic adjusted (controlling BSL_hsCRP + TLV + NIHSS + Age + Sex):")
    log.info(f"  OR(sustained) = {OR_sustained:.3f} [95% CI {ci_lo:.3f}, {ci_hi:.3f}], p={p_adj:.4g}")
    log.info(f"  N = {len(fit_df)}")

    summary = pd.DataFrame([
        {"test": "Chi-square unadjusted", "stat": chi2, "p": p_chi},
        {"test": "Fisher exact (OR)",     "stat": odds, "p": p_fisher},
        {"test": "Logistic adj. OR(sustained | BSL,TLV,NIHSS,Age,Sex)",
         "stat": OR_sustained, "p": p_adj,
         "CI_lo": ci_lo, "CI_hi": ci_hi, "N": len(fit_df)},
    ])
    res.to_csv(outdir / "sustained_phenotype_2x2.csv", index=False)
    summary.to_csv(outdir / "sustained_phenotype_tests.csv", index=False)
    log.info(f"  → sustained_phenotype_2x2.csv + sustained_phenotype_tests.csv")
    return res, summary, sub


# ==============================================================================
# Analysis 4: 动态交互 — m12_mRS ~ Resid_CHA × slope (有序 logit)
# ==============================================================================
def dynamic_interaction_ordinal(df, mask, cha_col, covariates, outdir,
                                  slope_col="slope_BSL_M03"):
    log.info("─" * 70)
    log.info(f"Analysis 4: 动态交互 — m12_mRS ~ Resid_CHA × {slope_col} (Ordinal logit)")

    sub = df[mask].copy()
    cov_cols = [c for c in covariates.values() if c]
    fit_cols = [cha_col, slope_col, "log_BSL", "m12_mRS"] + cov_cols
    fit_df = sub[fit_cols].dropna().copy()
    fit_df["m12_mRS_int"] = fit_df["m12_mRS"].astype(int)

    # z-score
    for c in [cha_col, slope_col, "log_BSL"]:
        fit_df[f"z_{c}"] = (fit_df[c] - fit_df[c].mean()) / fit_df[c].std()

    fit_df["interact"] = fit_df[f"z_{cha_col}"] * fit_df[f"z_{slope_col}"]

    predictors = [f"z_{cha_col}", f"z_{slope_col}", "interact", "z_log_BSL"] + cov_cols
    log.info(f"  N = {len(fit_df)}, 预测变量 = {predictors}")

    try:
        model = OrderedModel(fit_df["m12_mRS_int"], fit_df[predictors], distr="logit")
        fit = model.fit(method="bfgs", disp=0, maxiter=200)
        rows = []
        for p in predictors:
            beta = fit.params[p]
            se = fit.bse[p]
            pval = fit.pvalues[p]
            OR = np.exp(beta)
            ci_lo = np.exp(beta - 1.96 * se)
            ci_hi = np.exp(beta + 1.96 * se)
            rows.append({"term": p, "beta": beta, "SE": se, "OR": OR,
                         "CI95_lo": ci_lo, "CI95_hi": ci_hi, "p_value": pval})
        res = pd.DataFrame(rows)
        log.info(f"\n{res.to_string(index=False)}")

        # LR test for the interaction term
        model_null = OrderedModel(fit_df["m12_mRS_int"],
                                  fit_df[[c for c in predictors if c != "interact"]],
                                  distr="logit")
        fit_null = model_null.fit(method="bfgs", disp=0, maxiter=200)
        LR = 2 * (fit.llf - fit_null.llf)
        p_LR = 1 - stats.chi2.cdf(LR, df=1)
        log.info(f"\nLR test for interaction (1 df): χ² = {LR:.3f}, p = {p_LR:.4g}")

        fp = outdir / "dynamic_interaction_ordinal.csv"
        res.to_csv(fp, index=False)
        with open(outdir / "dynamic_interaction_LR.txt", "w") as f:
            f.write(f"LR test for Resid_CHA × {slope_col} interaction (1 df)\n")
            f.write(f"  N = {len(fit_df)}\n")
            f.write(f"  χ² = {LR:.4f}\n")
            f.write(f"  p  = {p_LR:.4g}\n")
        log.info(f"  → {fp.name}, dynamic_interaction_LR.txt")
        return res, fit
    except Exception as e:
        log.error(f"  ordinal logit 拟合失败: {e}")
        return None, None


# ==============================================================================
# Analysis 5: 四细胞加性交互 (CHA × sustained) — Knol & VanderWeele 升级版
# ==============================================================================
def additive_interaction_sustained(df, mask, cha_col, covariates, outdir,
                                    n_boot=1000):
    log.info("─" * 70)
    log.info("Analysis 5: 四细胞加性交互 — Resid_CHA × sustained-hsCRP")

    sub = df[mask].copy()
    sub["poor"] = (sub["m12_mRS"] > 2).astype(int)
    sub["sustained"] = (sub[COL_M03] >= SUSTAINED_THRESHOLD).astype(int)

    # 中位切分 CHA → high/low
    cha_med = sub[cha_col].median()
    sub["CHA_high"] = (sub[cha_col] >= cha_med).astype(int)

    # 四细胞: LL=00, LH=01, HL=10, HH=11
    cell_map = {(0, 0): "LL", (0, 1): "LH", (1, 0): "HL", (1, 1): "HH"}
    sub["cell"] = sub.apply(lambda r: cell_map[(r["CHA_high"], r["sustained"])], axis=1)

    # 协变量
    cov_cols = [c for c in covariates.values() if c]
    fit_cols = ["poor", "cell"] + cov_cols
    fit_df = sub[fit_cols].dropna().copy()
    log.info(f"  N (additive-interaction sub-cohort) = {len(fit_df)}")

    # 描述: 每细胞 N + poor%
    cells = ["LL", "LH", "HL", "HH"]
    cell_summary = []
    for c in cells:
        g = fit_df[fit_df["cell"] == c]
        n = len(g); k = g["poor"].sum()
        lo, hi = wilson_ci(k, n)
        cell_summary.append({"cell": c, "N": n,
                             "poor_n": k, "poor_pct": 100 * k / n if n > 0 else np.nan,
                             "Wilson_95CI_lo": 100 * lo, "Wilson_95CI_hi": 100 * hi})
    summary_df = pd.DataFrame(cell_summary)
    log.info(f"\n{summary_df.to_string(index=False)}")

    # Logistic with 3 dummy cells (LL = reference)
    def fit_additive(dfin):
        X = pd.get_dummies(dfin["cell"], prefix="cell").drop(columns=["cell_LL"], errors="ignore")
        X = pd.concat([X, dfin[cov_cols].reset_index(drop=True)], axis=1)
        X = sm.add_constant(X.astype(float))
        y = dfin["poor"].astype(int).reset_index(drop=True)
        return sm.Logit(y, X).fit(disp=0)

    fit_main = fit_additive(fit_df.reset_index(drop=True))
    OR = {c: np.exp(fit_main.params.get(f"cell_{c}", 0.0)) for c in ["LH", "HL", "HH"]}
    OR["LL"] = 1.0
    RERI = OR["HH"] - OR["LH"] - OR["HL"] + 1
    AP   = RERI / OR["HH"] if OR["HH"] > 0 else np.nan
    S    = (OR["HH"] - 1) / ((OR["LH"] - 1) + (OR["HL"] - 1)) \
            if (OR["LH"] - 1) + (OR["HL"] - 1) != 0 else np.nan

    log.info(f"\nORs (vs LL): LH={OR['LH']:.3f}, HL={OR['HL']:.3f}, HH={OR['HH']:.3f}")
    log.info(f"RERI = {RERI:+.3f}, AP = {AP:+.3f}, S = {S:.3f}")

    # Bootstrap CI
    log.info(f"\nBootstrap CI (n_boot = {n_boot})...")
    rng = np.random.default_rng(20260525)
    boot_RERI, boot_AP, boot_S = [], [], []
    boot_OR = {c: [] for c in ["LH", "HL", "HH"]}
    n = len(fit_df)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bs = fit_df.iloc[idx].reset_index(drop=True)
        if bs["cell"].nunique() < 4:
            continue
        try:
            f = fit_additive(bs)
            ors = {c: np.exp(f.params.get(f"cell_{c}", 0.0)) for c in ["LH", "HL", "HH"]}
            r = ors["HH"] - ors["LH"] - ors["HL"] + 1
            ap = r / ors["HH"] if ors["HH"] > 0 else np.nan
            s = (ors["HH"] - 1) / ((ors["LH"] - 1) + (ors["HL"] - 1)) \
                if (ors["LH"] - 1) + (ors["HL"] - 1) != 0 else np.nan
            boot_RERI.append(r); boot_AP.append(ap); boot_S.append(s)
            for c in ["LH", "HL", "HH"]: boot_OR[c].append(ors[c])
        except Exception:
            continue
        if (b + 1) % 200 == 0:
            log.info(f"  bootstrap {b+1}/{n_boot}")

    def ci(arr):
        a = np.asarray(arr)
        a = a[np.isfinite(a)]
        if len(a) < 10:
            return (np.nan, np.nan)
        return np.percentile(a, [2.5, 97.5])

    RERI_lo, RERI_hi = ci(boot_RERI)
    AP_lo, AP_hi     = ci(boot_AP)
    S_lo, S_hi       = ci(boot_S)

    add_rows = [
        {"measure": "OR_LH", "value": OR["LH"],
         "CI_lo": np.percentile(boot_OR["LH"], 2.5),
         "CI_hi": np.percentile(boot_OR["LH"], 97.5)},
        {"measure": "OR_HL", "value": OR["HL"],
         "CI_lo": np.percentile(boot_OR["HL"], 2.5),
         "CI_hi": np.percentile(boot_OR["HL"], 97.5)},
        {"measure": "OR_HH", "value": OR["HH"],
         "CI_lo": np.percentile(boot_OR["HH"], 2.5),
         "CI_hi": np.percentile(boot_OR["HH"], 97.5)},
        {"measure": "RERI",  "value": RERI, "CI_lo": RERI_lo, "CI_hi": RERI_hi},
        {"measure": "AP",    "value": AP,   "CI_lo": AP_lo,   "CI_hi": AP_hi},
        {"measure": "S",     "value": S,    "CI_lo": S_lo,    "CI_hi": S_hi},
    ]
    add_df = pd.DataFrame(add_rows)
    log.info(f"\n{add_df.to_string(index=False)}")

    summary_df.to_csv(outdir / "additive_interaction_sustained_cells.csv", index=False)
    add_df.to_csv(outdir / "additive_interaction_sustained.csv", index=False)
    log.info(f"  → additive_interaction_sustained_cells.csv + .csv")
    return summary_df, add_df, sub


# ==============================================================================
# Supplementary Figure S7
# ==============================================================================
def make_figure_S7(df, mask_dual, mask_tri, cha_col, outdir,
                    slope_col="slope_BSL_M03", fig_suffix=""):
    log.info("─" * 70)
    log.info(f"绘制 Supplementary Fig. S7 (slope={slope_col})")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    plt.subplots_adjust(left=0.05, right=0.99, top=0.90, bottom=0.13, wspace=0.30)

    palette = {"T1_low": "#4575B4", "T2_mid": "#ABD9E9", "T3_high": "#D73027"}

    # ── Panel A: 三时间点轨迹 by CHA tertile ──
    sub_tri = df[mask_tri].copy()
    sub_tri["CHA_tertile"] = pd.qcut(sub_tri[cha_col], 3,
                                      labels=["T1_low", "T2_mid", "T3_high"])
    ax = axes[0]
    tps_lbl = ["Baseline", "3 mo", "12 mo"]
    tps_col = [COL_BSL, COL_M03, COL_M12]
    x = np.arange(3)
    for t, c in palette.items():
        g = sub_tri[sub_tri["CHA_tertile"] == t]
        medians = [g[col].median() for col in tps_col]
        q1s     = [g[col].quantile(0.25) for col in tps_col]
        q3s     = [g[col].quantile(0.75) for col in tps_col]
        ax.plot(x, medians, "o-", color=c, label=f"CHA {t} (N={len(g)})",
                linewidth=2.0, markersize=7)
        ax.fill_between(x, q1s, q3s, color=c, alpha=0.15)
    ax.set_xticks(x); ax.set_xticklabels(tps_lbl)
    ax.set_ylabel("hsCRP (mg/L, median ± IQR)")
    ax.set_xlabel("Timepoint")
    ax.axhline(SUSTAINED_THRESHOLD, ls="--", color="grey", lw=0.8,
               label=f"Sustained threshold ({SUSTAINED_THRESHOLD} mg/L)")
    ax.set_title("A | hsCRP trajectory by Resid_CHA tertile", loc="left")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_yscale("log")

    # ── Panel B: slope 分布 by CHA tertile (boxplot) ──
    sub_dual = df[mask_dual].copy()
    sub_dual["CHA_tertile"] = pd.qcut(sub_dual[cha_col], 3,
                                      labels=["T1_low", "T2_mid", "T3_high"])
    ax = axes[1]
    box_data = [sub_dual[sub_dual["CHA_tertile"] == t][slope_col].dropna().values
                for t in ["T1_low", "T2_mid", "T3_high"]]
    bp = ax.boxplot(box_data, patch_artist=True,
                    labels=["T1_low", "T2_mid", "T3_high"],
                    widths=0.55, showfliers=False)
    for patch, t in zip(bp["boxes"], ["T1_low", "T2_mid", "T3_high"]):
        patch.set_facecolor(palette[t]); patch.set_alpha(0.6)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    if slope_col == "slope_BSL_M12":
        ax.set_ylabel("log hsCRP slope per month\n(M12 \u2212 BSL) / 12")
        ax.set_title("B | hsCRP resolution slope (BSL\u2192M12) by CHA tertile",
                     loc="left")
    else:
        ax.set_ylabel("log hsCRP slope per month\n(M03 \u2212 BSL) / 3")
        ax.set_title("B | hsCRP resolution slope (BSL\u2192M03) by CHA tertile",
                     loc="left")

    # Kruskal-Wallis 注释
    H, p_kw = stats.kruskal(*[b for b in box_data if len(b) > 0])
    ax.text(0.02, 0.97, f"Kruskal\u2013Wallis  P = {p_kw:.3g}",
            transform=ax.transAxes, fontsize=9, va="top",
            usetex=False,
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="none", alpha=0.95))

    # ── Panel C: Sustained vs Resolver — 12-month mRS distribution ──
    ax = axes[2]
    sub_d = sub_dual.copy()
    sub_d["sustained"] = (sub_d[COL_M03] >= SUSTAINED_THRESHOLD).astype(int)
    sub_d = sub_d[sub_d["m12_mRS"].notna()].copy()
    sub_d["mRS_cat"] = pd.cut(sub_d["m12_mRS"],
                               bins=[-0.1, 2, 4, 6],
                               labels=["Good (0–2)", "Moderate (3–4)", "Poor (5–6)"])
    ct = pd.crosstab(sub_d["sustained"], sub_d["mRS_cat"], normalize="index") * 100
    ct.index = ["Resolver\n(M03<3)", "Sustained\n(M03≥3)"]
    ct.plot(kind="bar", stacked=True, ax=ax,
            color=["#2C7FB8", "#FDAE61", "#D7191C"], width=0.6)
    ax.set_ylabel("Patients (%)")
    ax.set_xlabel("")
    ax.set_title("C | 12-mo mRS distribution by sustained-hsCRP phenotype",
                 loc="left")
    ax.legend(title="12-mo mRS", fontsize=8, loc="lower right")
    plt.setp(ax.get_xticklabels(), rotation=0)

    # 注释 poor% 差异
    n_res = (sub_d["sustained"] == 0).sum()
    n_sus = (sub_d["sustained"] == 1).sum()
    poor_res = ((sub_d["sustained"] == 0) & (sub_d["m12_mRS"] > 2)).sum()
    poor_sus = ((sub_d["sustained"] == 1) & (sub_d["m12_mRS"] > 2)).sum()
    ax.text(0.02, 0.97,
            f"Resolver (N={n_res}): poor={100*poor_res/n_res:.1f}%\n"
            f"Sustained (N={n_sus}): poor={100*poor_sus/n_sus:.1f}%",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fp_png = outdir / f"FigS7_dynamic_inflammation_trajectory{fig_suffix}.png"
    fp_pdf = outdir / f"FigS7_dynamic_inflammation_trajectory{fig_suffix}.pdf"
    fig.savefig(fp_png, dpi=300)
    fig.savefig(fp_pdf)
    plt.close(fig)
    log.info(f"  → {fp_png.name} + .pdf")


# ==============================================================================
# 主入口
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="Bootstrap iterations for RERI CI (default 1000)")
    ap.add_argument("--triplet-only", action="store_true",
                    help="Sensitivity mode: run all inferential analyses on the "
                         "three-timepoint sub-cohort (N=820, BSL+M03+M12+m12_mRS) "
                         "using the BSL\u2192M12 per-month log-slope. Default mode "
                         "(N=1334, BSL\u2192M03) is unchanged. Outputs are written "
                         "to <outdir>_triplet/ to avoid overwriting the main run.")
    args = ap.parse_args()

    input_path = Path(args.input)
    if args.triplet_only:
        outdir = Path(args.outdir).with_name(Path(args.outdir).name + "_triplet")
    else:
        outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 同时把日志写到文件
    fh = logging.FileHandler(outdir / "log.txt", mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                       datefmt="%H:%M:%S"))
    log.addHandler(fh)

    log.info("=" * 70)
    log.info("Dynamic Inflammation Trajectory Analysis")
    log.info("=" * 70)

    df, mask_dual, mask_tri, cha_col, cha_raw, covariates = \
        load_and_build_subcohort(input_path)

    if args.triplet_only:
        log.info("")
        log.info("*** TRIPLET-ONLY SENSITIVITY MODE ***")
        log.info(f"  Sub-cohort: N = {mask_tri.sum()} (BSL+M03+M12+m12_mRS)")
        log.info("  Slope:      [log1p(M12_hsCRP) - log1p(BSL_hsCRP)] / 12")
        log.info(f"  Output dir: {outdir}")
        log.info("")
        inf_mask = mask_tri
        slope_col = "slope_BSL_M12"
        fig_suffix = "_triplet"
    else:
        inf_mask = mask_dual
        slope_col = "slope_BSL_M03"
        fig_suffix = ""

    # 五个分析模块
    descriptive_by_cha_tertile(df, mask_tri, cha_col, outdir)
    slope_vs_cha(df, inf_mask, cha_col, covariates, outdir, slope_col=slope_col)
    sustained_vs_outcome(df, inf_mask, covariates, outdir)
    dynamic_interaction_ordinal(df, inf_mask, cha_col, covariates, outdir,
                                  slope_col=slope_col)
    additive_interaction_sustained(df, inf_mask, cha_col, covariates, outdir,
                                    n_boot=args.n_boot)

    # 主图
    make_figure_S7(df, inf_mask, mask_tri, cha_col, outdir,
                    slope_col=slope_col, fig_suffix=fig_suffix)

    log.info("=" * 70)
    log.info(f"✅ 完成. 输出位于: {outdir}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
