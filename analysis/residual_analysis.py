#!/usr/bin/env python3
"""
残差分析与临床关联 — Koch et al. (2025, Brain) 方法学实现

核心思路:
  1. 对每种递质的加权受损负荷 (Load) 与总病灶体积 (TLV) 做线性回归
  2. 提取残差 → "不成比例递质损伤" (Disproportionate NT Damage)
  3. 用残差作为自变量预测临床结局 (mRS, 炎症指标, 自主神经指标)
  4. 中介效应分析: 病灶 → 递质损伤(残差) → 炎症/神经 → 预后

用法:
  python3 residual_analysis.py <neurochem_loads.csv> [clinical_data.csv] [output_dir]

输入:
  neurochem_loads.csv: extract_weighted_load_batch.sh 的输出
                       列: ID, TLV_mm3, Load_DAT, Load_5HT1a, ...
  clinical_data.csv:   临床数据 (可选)
                       列: ID, mRS, Age, Sex, NIHSS, IL6, HRn, RMSSD, ...

输出:
  output_dir/
    residuals.csv              — 残差数据表
    regression_summary.csv     — 回归模型摘要
    ordinal_logistic.txt       — 序数逻辑回归结果
    mediation_results.csv      — 中介效应分析结果
    figures/                   — 所有可视化图形
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# ==============================================================================
# 辅助函数
# ==============================================================================
def safe_float(x):
    """安全转换为浮点数"""
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (ValueError, TypeError):
        return np.nan


def identify_load_columns(df):
    """自动识别 Load_ 开头的列"""
    return [c for c in df.columns if c.startswith("Load_")]


def clean_nt_name(col):
    """从列名提取递质简称: Load_DAT → DAT"""
    return col.replace("Load_", "").replace("_", " ")


# ==============================================================================
# 第一步: 残差计算 (Koch 2025, Fig.1E)
# ==============================================================================
def compute_residuals(df, load_cols, volume_col="TLV_mm3"):
    """
    对每种递质做线性回归: Load_NT ~ TLV
    提取残差 = 不成比例递质损伤

    Parameters
    ----------
    df : DataFrame
        包含 ID, TLV, Load_xx 列
    load_cols : list
        Load_ 开头的列名
    volume_col : str
        总病灶体积列名

    Returns
    -------
    residual_df : DataFrame
        残差数据
    regression_summary : DataFrame
        回归模型摘要
    """
    residuals = {"ID": df["ID"].values}
    reg_summary = []

    tlv = df[volume_col].values.astype(float)

    for col in load_cols:
        nt_name = clean_nt_name(col)
        y = df[col].values.astype(float)

        # 去除 NaN
        valid = np.isfinite(tlv) & np.isfinite(y)
        if valid.sum() < 10:
            print(f"  ⚠️  {nt_name}: 有效样本不足 ({valid.sum()}), 跳过")
            residuals[f"Resid_{nt_name}"] = np.full(len(df), np.nan)
            continue

        # OLS 回归: Load = β0 + β1 × TLV + ε
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            tlv[valid], y[valid]
        )

        # 计算所有样本的残差 (含原本有 NaN 的)
        predicted = intercept + slope * tlv
        resid = y - predicted
        resid[~valid] = np.nan

        residuals[f"Resid_{nt_name}"] = resid

        reg_summary.append({
            "Neurotransmitter": nt_name,
            "N_valid": int(valid.sum()),
            "Beta_TLV": slope,
            "Intercept": intercept,
            "R_squared": r_value ** 2,
            "P_value": p_value,
            "Std_Error": std_err,
        })

        print(f"  ✓ {nt_name}: β={slope:.4e}, R²={r_value**2:.4f}, p={p_value:.2e}")

    residual_df = pd.DataFrame(residuals)
    regression_summary = pd.DataFrame(reg_summary)

    return residual_df, regression_summary


# ==============================================================================
# 第二步: 序数逻辑回归 (Koch 2025, Table 3)
# ==============================================================================
def ordinal_logistic_regression(merged_df, resid_cols, outcome="mRS",
                                 covariates=None):
    """
    序数逻辑回归: mRS ~ Residual_NT + Age + Sex + NIHSS + TLV

    如果 statsmodels 不可用, 退回使用 OLS 近似
    """
    if covariates is None:
        covariates = ["Age", "Sex", "NIHSS", "TLV_mm3"]

    results = []

    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        use_ordinal = True
    except ImportError:
        print("  ⚠️  statsmodels OrderedModel 不可用, 使用 OLS 近似")
        import statsmodels.api as sm
        use_ordinal = False

    for rcol in resid_cols:
        nt_name = rcol.replace("Resid_", "")

        # 构建设计矩阵
        predictors = [rcol] + [c for c in covariates if c in merged_df.columns]
        all_cols = [outcome] + predictors
        sub = merged_df[all_cols].dropna()

        if len(sub) < 20:
            print(f"  ⚠️  {nt_name}: 样本不足 ({len(sub)}), 跳过")
            continue

        y = sub[outcome]
        X = sub[predictors]

        try:
            if use_ordinal:
                model = OrderedModel(y, X, distr="logit")
                res = model.fit(method="bfgs", disp=False)

                # 提取残差变量的系数
                coef = res.params[rcol]
                pval = res.pvalues[rcol]
                ci = res.conf_int().loc[rcol]

                results.append({
                    "Neurotransmitter": nt_name,
                    "Model": "Ordinal Logistic",
                    "Coefficient": coef,
                    "P_value": pval,
                    "CI_lower": ci[0],
                    "CI_upper": ci[1],
                    "N": len(sub),
                    "AIC": res.aic,
                    "Pseudo_R2": getattr(res, "prsquared", np.nan),
                })
            else:
                import statsmodels.api as sm
                X_const = sm.add_constant(X)
                model = sm.OLS(y, X_const).fit()

                coef = model.params[rcol]
                pval = model.pvalues[rcol]
                ci = model.conf_int().loc[rcol]

                results.append({
                    "Neurotransmitter": nt_name,
                    "Model": "OLS (approx)",
                    "Coefficient": coef,
                    "P_value": pval,
                    "CI_lower": ci[0],
                    "CI_upper": ci[1],
                    "N": len(sub),
                    "AIC": model.aic,
                    "Pseudo_R2": model.rsquared,
                })

            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"  {nt_name}: β={coef:.4f}, p={pval:.4e} {sig}")

        except Exception as e:
            print(f"  ⚠️  {nt_name}: 模型拟合失败 - {e}")

    return pd.DataFrame(results)


# ==============================================================================
# 第三步: 炎症/自主神经关联 (你的特色假设)
# ==============================================================================
def clinical_association_analysis(merged_df, resid_cols):
    """
    分析递质残差与炎症/自主神经指标的关联

    假设检验:
    1. Resid_VAChT → IL-6 升高 (胆碱能抗炎通路)
    2. Resid_DAT → HRn/RMSSD 失衡 (多巴胺-自主神经)
    """
    results = []

    # 定义你的假设驱动的关联对
    hypothesis_pairs = [
        # (递质残差列, 临床指标列, 假设说明)
        ("Resid_VAChT", "IL6", "胆碱能损伤 → 炎症升高"),
        ("Resid_VAChT", "CRP", "胆碱能损伤 → CRP升高"),
        ("Resid_VAChT", "IL10", "胆碱能损伤 → 抗炎减弱"),
        ("Resid_DAT", "HRn", "多巴胺损伤 → 心率变异"),
        ("Resid_DAT", "RMSSD", "多巴胺损伤 → 迷走张力"),
        ("Resid_DAT", "SDNN", "多巴胺损伤 → 自主神经总变异"),
        ("Resid_5HT1a", "HAMD", "5-HT1a损伤 → 抑郁"),
        ("Resid_5HTT", "HAMD", "5-HTT损伤 → 抑郁"),
        ("Resid_NAT", "RMSSD", "去甲肾上腺素损伤 → 迷走张力"),
        ("Resid_GABAa", "mRS", "GABA损伤 → 功能预后"),
    ]

    # 探索性分析: 所有递质残差 × 所有临床指标
    clinical_cols = [c for c in merged_df.columns
                     if c not in ["ID", "TLV_mm3"] and not c.startswith("Load_")
                     and not c.startswith("Resid_")]

    print("\n  --- 假设驱动分析 ---")
    for resid_col, clin_col, hypothesis in hypothesis_pairs:
        if resid_col not in merged_df.columns or clin_col not in merged_df.columns:
            continue

        sub = merged_df[[resid_col, clin_col]].dropna()
        if len(sub) < 10:
            continue

        r, p = stats.pearsonr(sub[resid_col], sub[clin_col])
        rho, p_spear = stats.spearmanr(sub[resid_col], sub[clin_col])

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {hypothesis}: r={r:.3f} (p={p:.4e}{sig}), ρ={rho:.3f}")

        results.append({
            "Hypothesis": hypothesis,
            "Residual": resid_col,
            "Clinical": clin_col,
            "Pearson_r": r,
            "Pearson_p": p,
            "Spearman_rho": rho,
            "Spearman_p": p_spear,
            "N": len(sub),
            "Type": "hypothesis-driven",
        })

    print("\n  --- 探索性分析 (所有递质 × 所有指标) ---")
    for rcol in resid_cols:
        for ccol in clinical_cols:
            sub = merged_df[[rcol, ccol]].dropna()
            if len(sub) < 10:
                continue
            try:
                r, p = stats.pearsonr(
                    sub[rcol].astype(float), sub[ccol].astype(float)
                )
                results.append({
                    "Hypothesis": "exploratory",
                    "Residual": rcol,
                    "Clinical": ccol,
                    "Pearson_r": r,
                    "Pearson_p": p,
                    "Spearman_rho": np.nan,
                    "Spearman_p": np.nan,
                    "N": len(sub),
                    "Type": "exploratory",
                })
            except Exception:
                pass

    results_df = pd.DataFrame(results)

    # FDR 校正
    if len(results_df) > 0 and "Pearson_p" in results_df.columns:
        from scipy.stats import false_discovery_control  # Python 3.12+ / scipy 1.11+
        try:
            p_vals = results_df["Pearson_p"].values
            valid_p = np.isfinite(p_vals)
            fdr = np.full(len(p_vals), np.nan)
            if valid_p.sum() > 0:
                # Benjamini-Hochberg
                from statsmodels.stats.multitest import multipletests
                _, fdr_vals, _, _ = multipletests(p_vals[valid_p], method="fdr_bh")
                fdr[valid_p] = fdr_vals
            results_df["FDR_q"] = fdr
        except ImportError:
            # 手动 BH 校正
            p_vals = results_df["Pearson_p"].values
            n = len(p_vals)
            sorted_idx = np.argsort(p_vals)
            fdr = np.full(n, np.nan)
            for rank, idx in enumerate(sorted_idx, 1):
                fdr[idx] = p_vals[idx] * n / rank
            fdr = np.minimum.accumulate(fdr[np.argsort(sorted_idx)][::-1])[::-1]
            results_df["FDR_q"] = np.clip(fdr, 0, 1)

    return results_df


# ==============================================================================
# 第四步: 中介效应分析
# ==============================================================================
def mediation_analysis(merged_df, mediator_col, outcome_col,
                       treatment_col="TLV_mm3", covariates=None):
    """
    中介效应分析: 病灶体积 → 递质残差 → 临床预后

    路径:
      c  = TLV → mRS (总效应)
      a  = TLV → Mediator (递质残差)
      b  = Mediator → mRS (控制TLV后)
      c' = TLV → mRS (控制Mediator后, 直接效应)
      ab = a × b (间接效应/中介效应)

    使用 Bootstrap 检验间接效应显著性
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        print("  需要 statsmodels 包")
        return None

    if covariates is None:
        covariates = []

    all_cols = [treatment_col, mediator_col, outcome_col] + covariates
    sub = merged_df[all_cols].dropna()

    if len(sub) < 30:
        return None

    X_treat = sub[treatment_col].values
    M = sub[mediator_col].values
    Y = sub[outcome_col].values

    # 构建协变量矩阵
    if covariates:
        Z = sub[covariates].values
    else:
        Z = np.empty((len(sub), 0))

    n_boot = 5000
    indirect_effects = []

    np.random.seed(42)
    for _ in range(n_boot):
        idx = np.random.randint(0, len(sub), len(sub))
        X_b, M_b, Y_b = X_treat[idx], M[idx], Y[idx]

        # Path a: X → M
        Xa = sm.add_constant(np.column_stack([X_b, Z[idx]]) if Z.shape[1] > 0
                             else sm.add_constant(X_b))
        try:
            res_a = sm.OLS(M_b, Xa).fit()
            a = res_a.params[1]

            # Path b: M → Y (控制 X)
            Xb = sm.add_constant(
                np.column_stack([X_b, M_b, Z[idx]]) if Z.shape[1] > 0
                else np.column_stack([X_b, M_b])
            )
            Xb = sm.add_constant(Xb[:, 1:])  # 确保有截距
            res_b = sm.OLS(Y_b, Xb).fit()
            b = res_b.params[2]  # M 的系数

            indirect_effects.append(a * b)
        except Exception:
            pass

    indirect_effects = np.array(indirect_effects)

    # 原始数据上的估计
    Xa_orig = sm.add_constant(X_treat)
    res_a_orig = sm.OLS(M, Xa_orig).fit()
    a_hat = res_a_orig.params[1]

    Xb_orig = sm.add_constant(np.column_stack([X_treat, M]))
    res_b_orig = sm.OLS(Y, Xb_orig).fit()
    b_hat = res_b_orig.params[2]
    c_prime = res_b_orig.params[1]  # 直接效应

    # 总效应
    Xc_orig = sm.add_constant(X_treat)
    res_c = sm.OLS(Y, Xc_orig).fit()
    c_hat = res_c.params[1]

    ab_hat = a_hat * b_hat
    ci_lower = np.percentile(indirect_effects, 2.5)
    ci_upper = np.percentile(indirect_effects, 97.5)
    p_indirect = 2 * min(
        np.mean(indirect_effects < 0), np.mean(indirect_effects > 0)
    )

    # 中介比例
    proportion_mediated = ab_hat / c_hat if abs(c_hat) > 1e-10 else np.nan

    result = {
        "Mediator": mediator_col,
        "Outcome": outcome_col,
        "Path_a (TLV→M)": a_hat,
        "Path_b (M→Y|TLV)": b_hat,
        "Indirect_ab": ab_hat,
        "CI_2.5%": ci_lower,
        "CI_97.5%": ci_upper,
        "P_indirect": p_indirect,
        "Direct_c_prime": c_prime,
        "Total_c": c_hat,
        "Proportion_mediated": proportion_mediated,
        "Significant": "Yes" if (ci_lower > 0 or ci_upper < 0) else "No",
        "N": len(sub),
        "N_bootstrap": n_boot,
    }

    sig = "✓" if result["Significant"] == "Yes" else "✗"
    print(f"  {mediator_col} → {outcome_col}: "
          f"ab={ab_hat:.4e}, 95%CI=[{ci_lower:.4e}, {ci_upper:.4e}] {sig}, "
          f"比例={proportion_mediated:.1%}")

    return result


# ==============================================================================
# 主函数
# ==============================================================================
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    neurochem_csv = Path(sys.argv[1])
    clinical_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("results_koch2025")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("  递质特异性受损残差分析")
    print("  Koch et al. (2025, Brain) | Hansen et al. (2022, Nat Neurosci)")
    print("=" * 70)

    # --- 加载数据 ---
    print("\n[1] 加载数据...")
    df = pd.read_csv(neurochem_csv)
    print(f"  神经化学数据: {len(df)} 行, {len(df.columns)} 列")
    print(f"  列名: {list(df.columns)}")

    # 转换数值列
    for col in df.columns:
        if col != "ID":
            df[col] = df[col].apply(safe_float)

    load_cols = identify_load_columns(df)
    print(f"  识别到 {len(load_cols)} 个递质负荷列: {load_cols}")

    # 基本统计
    print(f"\n  TLV 统计:")
    print(f"    N = {df['TLV_mm3'].notna().sum()}")
    print(f"    Mean = {df['TLV_mm3'].mean():.1f} mm³")
    print(f"    Median = {df['TLV_mm3'].median():.1f} mm³")
    print(f"    Range = [{df['TLV_mm3'].min():.1f}, {df['TLV_mm3'].max():.1f}]")

    # --- 第一步: 残差分析 ---
    print("\n" + "=" * 70)
    print("[2] 残差分析: Load_NT ~ TLV (Koch 2025, Fig.1E)")
    print("=" * 70)

    residual_df, reg_summary = compute_residuals(df, load_cols)

    # 保存
    residual_df.to_csv(output_dir / "residuals.csv", index=False)
    reg_summary.to_csv(output_dir / "regression_summary.csv", index=False)
    print(f"\n  残差已保存: {output_dir / 'residuals.csv'}")
    print(f"  回归摘要: {output_dir / 'regression_summary.csv'}")

    # 合并残差到原始数据
    merged = df.merge(residual_df, on="ID", how="left")

    resid_cols = [c for c in residual_df.columns if c.startswith("Resid_")]

    # --- 如果有临床数据, 继续分析 ---
    if clinical_csv and clinical_csv.exists():
        print("\n" + "=" * 70)
        print("[3] 加载临床数据并合并...")
        print("=" * 70)

        clinical = pd.read_csv(clinical_csv)
        print(f"  临床数据: {len(clinical)} 行, {len(clinical.columns)} 列")

        # 合并
        merged = merged.merge(clinical, on="ID", how="inner")
        print(f"  合并后: {len(merged)} 行")

        # 序数逻辑回归
        if "mRS" in merged.columns:
            print("\n" + "=" * 70)
            print("[4] 序数逻辑回归: mRS ~ Residual_NT + 协变量")
            print("    (Koch 2025, Table 3)")
            print("=" * 70)

            olr_results = ordinal_logistic_regression(
                merged, resid_cols, outcome="mRS",
                covariates=["Age", "Sex", "NIHSS", "TLV_mm3"]
            )
            if len(olr_results) > 0:
                olr_results.to_csv(output_dir / "ordinal_logistic.csv", index=False)
                print(f"\n  结果: {output_dir / 'ordinal_logistic.csv'}")

        # 临床关联分析
        print("\n" + "=" * 70)
        print("[5] 递质残差 × 临床指标关联分析")
        print("=" * 70)

        assoc_results = clinical_association_analysis(merged, resid_cols)
        if len(assoc_results) > 0:
            assoc_results.to_csv(output_dir / "clinical_associations.csv", index=False)

            # 显示显著结果
            sig = assoc_results[assoc_results["Pearson_p"] < 0.05]
            if len(sig) > 0:
                print(f"\n  显著关联 (p<0.05): {len(sig)} 个")
                for _, row in sig.iterrows():
                    print(f"    {row['Residual']} ↔ {row['Clinical']}: "
                          f"r={row['Pearson_r']:.3f}, p={row['Pearson_p']:.4e}")

        # 中介效应分析
        print("\n" + "=" * 70)
        print("[6] 中介效应分析: TLV → 递质残差 → 临床预后")
        print("=" * 70)

        mediation_results = []

        # 核心中介模型
        mediation_pairs = [
            # (mediator, outcome, description)
            ("Resid_VAChT", "IL6", "胆碱能 → 炎症"),
            ("Resid_DAT", "RMSSD", "多巴胺 → 迷走张力"),
            ("Resid_DAT", "mRS", "多巴胺 → 预后"),
            ("Resid_VAChT", "mRS", "胆碱能 → 预后"),
            ("Resid_5HT1a", "HAMD", "5-HT1a → 抑郁"),
            ("Resid_NAT", "mRS", "去甲肾上腺素 → 预后"),
        ]

        for mediator, outcome, desc in mediation_pairs:
            if mediator in merged.columns and outcome in merged.columns:
                print(f"\n  [{desc}]")
                result = mediation_analysis(
                    merged, mediator, outcome,
                    covariates=["Age", "Sex"] if "Age" in merged.columns else []
                )
                if result:
                    result["Description"] = desc
                    mediation_results.append(result)

        if mediation_results:
            med_df = pd.DataFrame(mediation_results)
            med_df.to_csv(output_dir / "mediation_results.csv", index=False)
            print(f"\n  中介效应结果: {output_dir / 'mediation_results.csv'}")
    else:
        print("\n" + "=" * 70)
        print("[!] 未提供临床数据, 跳过临床关联分析")
        print("    请准备 clinical_data.csv (列: ID, mRS, Age, Sex, NIHSS, IL6, ...)")
        print("    然后重新运行:")
        print(f"    python3 {sys.argv[0]} {neurochem_csv} clinical_data.csv")
        print("=" * 70)

    # --- 保存合并数据 ---
    merged.to_csv(output_dir / "merged_data.csv", index=False)
    print(f"\n  合并数据: {output_dir / 'merged_data.csv'}")

    print("\n" + "=" * 70)
    print("  分析完成！")
    print(f"  所有输出在: {output_dir}/")
    print("=" * 70)

    return merged, residual_df, reg_summary


if __name__ == "__main__":
    main()
