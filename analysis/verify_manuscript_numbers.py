#!/usr/bin/env python3
"""
verify_manuscript_numbers.py
============================
Single-shot verification of every numeric statement in the manuscript that
currently has multiple values in different sections.

Run:
    python analysis/verify_manuscript_numbers.py

Outputs ONE block per claim in the form:

    [CLAIM #N] <description>
        Manuscript says : <values currently in text>
        Source file     : <path>
        Computed value  : <ground truth from CSV>
        VERDICT         : MATCH / MISMATCH / AMBIGUOUS

Copy the whole stdout back so we can lock every number to a single ground
truth before submission.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- paths
SRC = Path(
    "/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/6.furtherv4"
)
PUB = SRC / "use" / "manuscript_outputs" / "publication_ready"
NIVA_DIR = SRC.parent.parent / "7.figure" / "vulnerability_map" / "validation"
DATA_CSV = (
    SRC.parent.parent
    / "3.variable_outcom_merge_data"
    / "merged_neuro_data.csv"
)


def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(f" {title}")
    print("=" * 78)


def claim(n: int, desc: str) -> None:
    print(f"\n[CLAIM {n:>2}] {desc}")


def show(label: str, value) -> None:
    print(f"           {label:<24}{value}")


def safe_read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"           ⚠️  MISSING: {path}")
        return None
    return pd.read_csv(path)


# ============================================================================
# 0. Sanity: directory exists
# ============================================================================
header("Path sanity")
print(f"SRC      = {SRC}        exists={SRC.exists()}")
print(f"PUB      = {PUB}        exists={PUB.exists()}")
print(f"NIVA_DIR = {NIVA_DIR}   exists={NIVA_DIR.exists()}")
print(f"DATA_CSV = {DATA_CSV}   exists={DATA_CSV.exists()}")

# ============================================================================
# 1. Cohort size & Discovery / Validation split
# ============================================================================
header("1. Cohort size & follow-up rates")
df_main = safe_read(DATA_CSV) if DATA_CSV.exists() else None
if df_main is not None:
    n = len(df_main)
    claim(1, "Total N (manuscript: 3,582)")
    show("Computed N", n)
    show("VERDICT", "MATCH" if n == 3582 else f"MISMATCH (got {n})")

    for c in ("D_MRS", "m3_mRS", "m6_mRS", "m12_mRS"):
        if c in df_main.columns:
            n_avail = df_main[c].notna().sum()
            show(f"{c} non-null", f"{n_avail} ({n_avail/n*100:.1f}%)")

    for c in ("BSL_IL6", "BSL_HSCRP", "RMSSD"):
        if c in df_main.columns:
            n_avail = df_main[c].notna().sum()
            show(f"{c} non-null", f"{n_avail} ({n_avail/n*100:.1f}%)")

    claim(2, "70/30 split: Discovery N=2,507 / Validation N=1,075")
    show("If stratified by m12_mRS>2, expect", "≈2507 / 1075")
    show("Computed (no split file)", "no split CSV found in 6.furtherv4/")
    show(
        "VERDICT",
        "AMBIGUOUS — 该 split 在 manuscript Methods 里被声称，但 6.furtherv4/ "
        "没有 split.csv，且 Fig 3C/4B 实际 in-sample/10-fold CV 不依赖该 split. "
        "建议要么生成 split CSV、要么删除 Methods L73 中的 70/30 split 句子.",
    )

# ============================================================================
# 2. ordinal_regression.csv: Model C / D significance counts
# ============================================================================
header("2. NT prognostic significance counts")
ord_df = safe_read(SRC / "ordinal_regression.csv")
if ord_df is not None:
    claim(3, "Discharge mRS — Model C 显著 NT 数 (manuscript: 6 at P<1e-4)")
    sub = ord_df[(ord_df["Outcome"] == "D_MRS") & (ord_df["Model"] == "C_Full")]
    n_strict = (sub["P_value"] < 1e-4).sum()
    n_p05 = (sub["P_value"] < 0.05).sum()
    n_q05 = (
        (sub["FDR_q"] < 0.05).sum() if "FDR_q" in sub.columns else "no FDR_q col"
    )
    show("P < 1e-4", n_strict)
    show("P < 0.05", n_p05)
    show("FDR_q < 0.05", n_q05)
    show("Top 6 by P", sub.nsmallest(6, "P_value")[["NT_Variable", "OR", "P_value"]].to_string(index=False))

    claim(4, "12-month mRS — Model C: any NT 显著 (manuscript: 0)")
    sub12 = ord_df[(ord_df["Outcome"] == "m12_mRS") & (ord_df["Model"] == "C_Full")]
    n_p05_12 = (sub12["P_value"] < 0.05).sum()
    n_q05_12 = (
        (sub12["FDR_q"] < 0.05).sum() if "FDR_q" in sub12.columns else "no FDR_q col"
    )
    show("P < 0.05", n_p05_12)
    show("FDR_q < 0.05", n_q05_12)
    show("VERDICT", "MATCH" if n_q05_12 == 0 else f"MISMATCH ({n_q05_12} > 0)")

# ============================================================================
# 3. cst_nt_comparison.csv: 12 vs 13 of 17 surviving CST
# ============================================================================
header("3. CST control — 12 vs 13 of 17")
cst = safe_read(SRC / "cst_nt_comparison.csv")
if cst is not None:
    print(f"           cst columns: {list(cst.columns)}")
    print(cst.head(3).to_string())
    if "Control" in cst.columns:
        for ctrl, grp in cst.groupby("Control"):
            n_p05 = (grp["P"] < 0.05).sum()
            n_q05 = (
                (grp["FDR_q"] < 0.05).sum() if "FDR_q" in grp.columns else "no FDR_q"
            )
            n_q05_global = (
                (grp["FDR_q_global"] < 0.05).sum()
                if "FDR_q_global" in grp.columns
                else "no FDR_q_global"
            )
            print(
                f"           Control={ctrl}: P<0.05 → {n_p05}/{len(grp)}; "
                f"FDR_q<0.05 → {n_q05}; FDR_q_global<0.05 → {n_q05_global}"
            )

    claim(
        5,
        "Manuscript: 'Results=12 of 17, Discussion=13/17 systems surviving CST'. "
        "Pick whichever matches Model D global FDR.",
    )
    show("→ Lock manuscript to one of these counts.", "")

# ----------------------------------------------------------------------------
# 5b. global_fdr.csv: how many NT survive at q<0.05 in Model D (CST-adjusted)?
# ----------------------------------------------------------------------------
header("5b. global_fdr.csv — Model D (CST adjusted) survivors")
gfdr_all = safe_read(SRC / "global_fdr.csv")
if gfdr_all is not None:
    print(f"           global_fdr columns: {list(gfdr_all.columns)}")
    print(f"           Module values: {sorted(gfdr_all['Module'].dropna().unique().tolist())}")
    if "Model" in gfdr_all.columns:
        print(f"           Model values:  {sorted(gfdr_all['Model'].dropna().unique().tolist())}")
    # try to isolate Model D / CST-adjusted ordinal rows
    candidates = []
    for col in ["Module", "Model", "Analysis"]:
        if col in gfdr_all.columns:
            for v in gfdr_all[col].dropna().unique():
                if any(tag in str(v) for tag in ("Model_D", "ModelD", "Model D", "CST", "cst")):
                    candidates.append((col, v))
    print(f"           CST/Model-D candidate slices: {candidates}")
    for col, v in candidates:
        sub = gfdr_all[gfdr_all[col] == v]
        if "FDR_q_global" in sub.columns:
            n_sig = (sub["FDR_q_global"] < 0.05).sum()
            print(f"           {col}={v}: N={len(sub)}, q_global<0.05 → {n_sig}")
        elif "FDR_q" in sub.columns:
            n_sig = (sub["FDR_q"] < 0.05).sum()
            print(f"           {col}={v}: N={len(sub)}, q<0.05 → {n_sig}")
    show(
        "→ If a slice gives 12, the 'after global FDR correction' wording is correct.",
        "",
    )
    show(
        "→ If only 11 or 13, must either re-pick the slice or change wording to 'at P<0.05'.",
        "",
    )

# ============================================================================
# 4. interaction.csv / global_fdr.csv: 7 sig pairs and which?
# ============================================================================
header("4. NT × Inflammation interactions")
inter = safe_read(SRC / "interaction.csv")
gfdr = safe_read(SRC / "global_fdr.csv")
if inter is not None and gfdr is not None:
    gi = gfdr[gfdr["Module"] == "Interaction"].copy()
    print(f"           gfdr Interaction module rows: {len(gi)}")
    if "Label" in gi.columns:
        gi[["NT", "Inflam"]] = gi["Label"].str.split("×", n=1, expand=True)
        gi["NT"] = gi["NT"].str.strip()
        gi["Inflam"] = gi["Inflam"].str.strip()

    claim(6, "Total NT × Inflam pairs (manuscript: 34 = 17×2)")
    show("Computed", len(gi))

    claim(7, "FDR-significant pairs (global FDR q < 0.05) (manuscript: 7)")
    sig = gi[gi["Q_global"] < 0.05].sort_values("Q_global")
    show("Computed n_sig", len(sig))
    show("Pairs (NT × Inflam, Q_global)", "")
    print(
        sig[["NT", "Inflam", "Q_global"]]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    show(
        "VERDICT",
        "MATCH" if len(sig) == 7 else f"MISMATCH (got {len(sig)} sig pairs)",
    )

    claim(8, "Manuscript-named hits & Q values (q values must match)")
    expected = {
        "Medial_Path × hsCRP": 0.011,
        "VAChT × IL-6 / VAChT × BSL_IL6": 0.019,
        "NAT × IL-6": 0.046,
        "DAT × IL-6": 0.011,
        "5HT6 × IL-6": 0.024,
        "D2 × IL-6": 0.037,
        "D1 × IL-6": 0.048,
    }
    for k, v in expected.items():
        show(f"{k} (manuscript)", f"q={v}")
    show("→ Cross-check above against the printed sig table.", "")

# ============================================================================
# 5. Table_S2_NRI_IDI_AUC.csv: NRI = +0.297 vs +0.351
# ============================================================================
header("5. NRI / IDI / AUC for dual-hit (CHA × IL-6)")
nri = safe_read(PUB / "Table_S2_NRI_IDI_AUC.csv")
if nri is not None:
    print(nri.to_string(index=False))
    claim(
        9,
        "NRI manuscript values: Results=+0.297, Discussion=+0.30, your-summary=+0.351",
    )
    nri_row = nri[nri["Metric"].str.contains("NRI", case=False, na=False)]
    if len(nri_row):
        nri_val = nri_row["Estimate"].iloc[0]
        ci_lo = nri_row.get("CI_low", pd.Series([np.nan])).iloc[0]
        ci_hi = nri_row.get("CI_high", pd.Series([np.nan])).iloc[0]
        p_val = nri_row.get("P", pd.Series([np.nan])).iloc[0]
        show("Computed NRI", f"{nri_val:+.4f}")
        show("Computed 95% CI", f"[{ci_lo:+.4f}, {ci_hi:+.4f}]")
        show("Computed P", f"{p_val:.4f}")
        show(
            "VERDICT (lock all manuscript text to this value, 3 dp)",
            f"NRI = {nri_val:+.3f} [{ci_lo:+.3f}, {ci_hi:+.3f}], P = {p_val:.3f}",
        )

    claim(10, "AUC base / dual-hit (manuscript: 0.718 / 0.728~0.729)")
    auc_b = nri[nri["Metric"].str.contains("AUC_base", na=False)]
    auc_d = nri[nri["Metric"].str.contains("AUC_dh", na=False)]
    if len(auc_b) and len(auc_d):
        a_b = auc_b["Estimate"].iloc[0]
        a_d = auc_d["Estimate"].iloc[0]
        show("AUC base", f"{a_b:.4f}")
        show("AUC dual-hit", f"{a_d:.4f}")
        show("ΔAUC", f"{a_d - a_b:+.4f}")

    claim(11, "IDI (manuscript: +0.009~+0.010 [+0.003, +0.017], P<0.001)")
    idi_row = nri[nri["Metric"].str.contains("IDI", case=False, na=False)]
    if len(idi_row):
        idi_val = idi_row["Estimate"].iloc[0]
        ci_lo = idi_row.get("CI_low", pd.Series([np.nan])).iloc[0]
        ci_hi = idi_row.get("CI_high", pd.Series([np.nan])).iloc[0]
        p_val = idi_row.get("P", pd.Series([np.nan])).iloc[0]
        show("Computed IDI", f"{idi_val:+.4f}")
        show("Computed 95% CI", f"[{ci_lo:+.4f}, {ci_hi:+.4f}]")
        show("Computed P", f"{p_val:.4f}")

# ============================================================================
# 6. cv_10fold.csv: per-NT max ΔAUC; dual-hit CV (Discussion claims +0.004)
# ============================================================================
header("6. 10-fold cross-validation")
cv = safe_read(SRC / "cv_10fold.csv")
if cv is not None:
    print(cv.head(20).to_string(index=False))
    claim(12, "Per-NT max ΔAUC (manuscript: α4β2 = +0.0015)")
    if {"NT", "Delta_AUC"}.issubset(cv.columns):
        top = cv.sort_values("Delta_AUC", ascending=False).head(3)
        show("Top 3 NT", top.to_string(index=False))
        show("MAX ΔAUC", f"{cv['Delta_AUC'].max():+.4f}")
        show("# NT with ΔAUC > +0.001", (cv["Delta_AUC"] > 0.001).sum())
        show("# NT with ΔAUC > +0.002", (cv["Delta_AUC"] > 0.002).sum())

    claim(
        13,
        "Discussion claims dual-hit ΔAUC = +0.004, P = 0.134 (Fig 4B). "
        "Locate the actual source: dual-hit row in cv_10fold.csv?",
    )
    if "NT" in cv.columns:
        dh_rows = cv[cv["NT"].str.contains("dual|DH|hit|CHA.*IL6", case=False, na=False)]
        if len(dh_rows):
            print(dh_rows.to_string(index=False))
        else:
            show(
                "Match",
                "no dual-hit row in cv_10fold.csv → Discussion ΔAUC=+0.004 来源待验",
            )

# ============================================================================
# 7. mediation: 5 of 8 null, 3 of 8 sig
# ============================================================================
header("7. Mediation (NT → IL-6 → mRS)")
med = safe_read(SRC / "mediation_parallel.csv")
extra_p = SRC / "mediation_top_NT_IL6.csv"
if med is not None:
    if extra_p.exists():
        extra = pd.read_csv(extra_p)
        med = pd.concat([med, extra], ignore_index=True).drop_duplicates(
            subset=["NT", "Mediator"], keep="last"
        )
    il6 = med[med["Mediator"].str.contains("IL", case=False, na=False)]
    print(il6.to_string(index=False))
    claim(14, "Mediation: 5/8 null, 3/8 (A4B2/DAT/5HT6) sig (manuscript)")
    sig_count = (
        (il6["Boot_CI_lower"] > 0) | (il6["Boot_CI_upper"] < 0)
    ).sum()
    show("Total IL-6 mediation rows", len(il6))
    show("Significant (CI excludes 0)", sig_count)
    sig_rows = il6[(il6["Boot_CI_lower"] > 0) | (il6["Boot_CI_upper"] < 0)]
    show("Sig NT names", ", ".join(sig_rows["NT"].astype(str).tolist()))

# ============================================================================
# 8. Small-lesion: N=894, severe % = 17.7 vs 18.1
# ============================================================================
header("8. Small-lesion phenotype")
nt_cmp = safe_read(SRC / "anomalous_nt_compare.csv")
base = safe_read(SRC / "anomalous_baseline.csv")
if df_main is not None and "TLV" in df_main.columns:
    q1 = df_main["TLV"].quantile(0.25)
    sm = df_main[df_main["TLV"] < q1]
    claim(15, "Small-lesion N (manuscript: 894)")
    show("Computed N (TLV<Q1)", len(sm))
    show("Q1 of TLV (mL)", f"{q1/1000:.3f}")

    claim(
        16,
        "Manuscript: discharge severe% = 26.8% (D_MRS), 12-mo severe% = 18.1% "
        "(m12_mRS); Fig 6 legend: 17.7%",
    )
    if "D_MRS" in sm.columns:
        sev_d = (sm["D_MRS"] >= 3).mean() * 100
        show("Discharge mRS≥3 %", f"{sev_d:.2f}%")
    if "m12_mRS" in sm.columns:
        sev_12 = (sm["m12_mRS"] >= 3).mean() * 100
        show("12-month mRS≥3 %", f"{sev_12:.2f}%")
        show(
            "VERDICT",
            "lock Fig 6 legend & Results to whichever matches actual panel.",
        )

if nt_cmp is not None:
    print(nt_cmp.to_string(index=False))
if base is not None:
    print(base.to_string(index=False))

# ============================================================================
# 9. NIVA split-half r (Fig 5B: 0.984 vs 0.997 vs >0.85)
# ============================================================================
header("9. NIVA split-half cross-validation r")
niva_csv_candidates = [
    NIVA_DIR / "niva_spatial_crossvalidation.csv",
    NIVA_DIR / "niva_split_half.csv",
    NIVA_DIR / "split_half_correlation.csv",
    NIVA_DIR / "validation_summary.csv",
]
found = False
for p in niva_csv_candidates:
    if p.exists():
        d = pd.read_csv(p)
        print(f"           {p.name}:")
        print(d.to_string(index=False))
        found = True
if not found:
    print(
        "           ⚠️  No NIVA validation CSV found. The r value is currently "
        "embedded only in niva_spatial_crossvalidation.png title. "
        "Open that PNG and read the number written on the panel."
    )

# ============================================================================
# 10. Hansen correlation r range (manuscript: 0.40 / 0.41 to 0.86)
# ============================================================================
hansen_csv_candidates = [
    NIVA_DIR / "niva_hansen_correlation.csv",
    NIVA_DIR / "hansen_correlation.csv",
]
header("10. NIVA × Hansen PET correlation range")
for p in hansen_csv_candidates:
    if p.exists():
        d = pd.read_csv(p)
        print(f"           {p.name}:")
        print(d.to_string(index=False))
        if "r" in d.columns:
            show("min r", f"{d['r'].min():.3f}")
            show("max r", f"{d['r'].max():.3f}")
        break
else:
    print("           ⚠️  No Hansen correlation CSV found")

# ============================================================================
# 11. Global FDR total tests = 373
# ============================================================================
header("11. Global FDR test count (manuscript: 373)")
if gfdr is not None:
    show("Computed total tests", len(gfdr))
    if "Module" in gfdr.columns:
        for mod, grp in gfdr.groupby("Module"):
            show(f"  Module={mod}", len(grp))
    show(
        "VERDICT",
        "MATCH" if len(gfdr) == 373 else f"MISMATCH (got {len(gfdr)})",
    )

# ============================================================================
# 12. PCA system aggregation (manuscript: cholinergic PC1 OR=1.18, P=9.4e-6)
# ============================================================================
header("12. PCA system-level aggregation")
pca = safe_read(SRC / "pca_system.csv")
if pca is not None:
    if "Outcome" in pca.columns:
        pca_d = pca[pca["Outcome"] == "D_MRS"]
    else:
        pca_d = pca
    print(pca_d.to_string(index=False))
    claim(17, "Cholinergic PC1 OR (manuscript: 1.18, P=9.4e-6)")
    chol = pca_d[pca_d["System"].astype(str).str.contains("hol", case=False, na=False)]
    if len(chol):
        show(
            "Computed",
            f"OR = {chol['PC1_OR'].iloc[0]:.3f}, "
            f"P = {chol.get('PC1_P', chol.get('P_value', pd.Series([np.nan]))).iloc[0]:.2e}",
        )

# ============================================================================
# 13. Pre/Post-synaptic effect sizes (manuscript: |β| 0.18 vs 0.10)
# ============================================================================
header("13. Pre- vs post-synaptic effect sizes")
syn = safe_read(SRC / "synaptic_location.csv")
if syn is not None:
    if "Outcome" in syn.columns:
        syn_d = syn[syn["Outcome"] == "D_MRS"]
    else:
        syn_d = syn
    print(syn_d.to_string(index=False))
    claim(18, "Mean |β| pre vs post (manuscript: 0.18 vs 0.10)")
    if "Mean_AbsBeta" in syn_d.columns:
        for _, row in syn_d.iterrows():
            show(f"{row['Synaptic_Type']}", f"|β| = {row['Mean_AbsBeta']:.3f}")

# ============================================================================
# 14. Permutation 4 of 17 sig
# ============================================================================
header("14. Permutation test (manuscript: 4 of 17)")
perm = safe_read(SRC / "permutation_test.csv")
if perm is not None:
    print(perm.head(20).to_string(index=False))
    claim(19, "Permutation N significant (manuscript: 4 of 17)")
    if "Perm_P" in perm.columns:
        n_perm = (perm["Perm_P"] < 0.05).sum()
        show("Computed", n_perm)

# ============================================================================
# 15. CHA atlas source: read the CHA / human_CHA column metadata
# ============================================================================
header("15. CHA atlas source — currently 'source citation to be confirmed'")
print(
    "           ⚠️  This is a documentation gap not a numeric mismatch. "
    "Methods L33 has placeholder '(CHA; source citation to be confirmed)'. "
    "Required: name + DOI of the cholinergic terminal density atlas used."
)

# ============================================================================
# DONE
# ============================================================================
header("DONE — copy whole stdout to chat for reconciliation.")
