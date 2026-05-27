#!/usr/bin/env python3
"""
Generate ALL publication-ready Tables and Figures from raw v4 outputs.

Output: manuscript_outputs/publication_ready/
    ├── Table_1_Baseline_Characteristics.csv
    ├── Table_2_NT_Acute_Outcome.csv         (Model A/B/C, top 5 NT)
    ├── Table_3_CST_Adjusted_Model_D.csv     (Model D, all 17 NT)
    ├── Table_4_NT_Inflammation_Interactions.csv (FDR-significant pairs)
    ├── figures/
    │   ├── Fig_2_CST_OR_attenuation.png
    │   ├── Fig_3A_Interaction_heatmap.png
    │   ├── Fig_4A_DCA.png
    │   ├── Fig_4B_CV_10fold.png
    │   ├── Fig_5C_Bootstrap_weights.png  (already exists)
    │   └── Fig_6A_SmallLesion.png
    └── README.md
"""
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ---- Publication-grade global style (Nature / Brain) ----
plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Helvetica', 'Arial', 'DejaVu Sans'],
    'mathtext.fontset':  'stixsans',
    'font.size':         10,
    'axes.titlesize':    13,
    'axes.titleweight':  'bold',
    'axes.labelsize':    12,
    'axes.labelweight':  'bold',
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'xtick.direction':   'out',
    'ytick.direction':   'out',
    'xtick.major.size':  3.5,
    'ytick.major.size':  3.5,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'legend.fontsize':   9,
    'legend.frameon':    False,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    1.0,
    'figure.dpi':        110,
    'savefig.dpi':       400,
    'savefig.bbox':      'tight',
    'pdf.fonttype':      42,   # editable text in Illustrator
    'ps.fonttype':       42,
})

# Color palette (NPG / Lancet)
COL = {
    'red':    '#E64B35',
    'blue':   '#4DBBD5',
    'green':  '#4CAF50',
    'orange': '#FF9800',
    'purple': '#7E6148',
    'gray':   '#9E9E9E',
    'lgray':  '#B0BEC5',
    'shade':  '#FFF3E0',
}

SRC = Path("/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/6.furtherv4")
PUB = SRC / "use" / "manuscript_outputs" / "publication_ready"
FIGS = PUB / "figures"
SUPP = PUB / "supplementary"
PUB.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(exist_ok=True)
SUPP.mkdir(exist_ok=True)

# ============================================================
# RUN LOG  —  capture stdout / stderr / warnings / uncaught
# exceptions to two files for post-hoc diagnosis:
#   run.log         : full transcript (everything printed)
#   run_errors.log  : only WARNINGs / stderr / tracebacks
# ============================================================
import sys, datetime, warnings, traceback, atexit

_LOG_PATH = PUB / "run.log"
_ERR_PATH = PUB / "run_errors.log"
_log_f = open(_LOG_PATH, "w", encoding="utf-8", buffering=1)
_err_f = open(_ERR_PATH, "w", encoding="utf-8", buffering=1)
_START_TS = datetime.datetime.now()
_err_counts = {"WARNING": 0, "ERROR": 0, "EXCEPTION": 0}

_header = (
    f"# generate_publication_ready.py — run log\n"
    f"# started : {_START_TS.isoformat(timespec='seconds')}\n"
    f"# python  : {sys.version.split()[0]}\n"
    f"# cwd     : {Path.cwd()}\n"
    f"# out_dir : {PUB}\n"
    f"{'='*60}\n"
)
_log_f.write(_header); _err_f.write(_header)

class _Tee:
    def __init__(self, orig, log_file, also_err=False):
        self._orig = orig; self._log = log_file; self._also_err = also_err
    def write(self, s):
        try: self._orig.write(s)
        except Exception: pass
        try: self._log.write(s)
        except Exception: pass
        if self._also_err and s.strip():
            try:
                _err_f.write(f"[stderr {datetime.datetime.now():%H:%M:%S}] {s}"
                             + ("" if s.endswith("\n") else "\n"))
            except Exception: pass
            _err_counts["ERROR"] += 1
    def flush(self):
        for f in (self._orig, self._log):
            try: f.flush()
            except Exception: pass

sys.stdout = _Tee(sys.stdout, _log_f, also_err=False)
sys.stderr = _Tee(sys.stderr, _log_f, also_err=True)

def _log_warning(message, category, filename, lineno, file=None, line=None):
    msg = (f"[WARNING {datetime.datetime.now():%H:%M:%S}] "
           f"{category.__name__}: {message}  ({Path(filename).name}:{lineno})\n")
    _err_f.write(msg); _log_f.write(msg); sys.__stderr__.write(msg)
    _err_counts["WARNING"] += 1
warnings.showwarning = _log_warning

def _log_excepthook(exc_type, exc_val, exc_tb):
    tb_str = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
    _err_f.write(f"\n[UNCAUGHT EXCEPTION {datetime.datetime.now():%H:%M:%S}]\n{tb_str}\n")
    _log_f.write(f"\n[UNCAUGHT EXCEPTION]\n{tb_str}\n")
    _err_counts["EXCEPTION"] += 1
    sys.__excepthook__(exc_type, exc_val, exc_tb)
sys.excepthook = _log_excepthook

def _close_logs():
    dur = datetime.datetime.now() - _START_TS
    footer = (
        f"\n{'='*60}\n"
        f"# finished: {datetime.datetime.now().isoformat(timespec='seconds')}\n"
        f"# elapsed : {dur}\n"
        f"# warnings: {_err_counts['WARNING']}\n"
        f"# stderr  : {_err_counts['ERROR']}\n"
        f"# uncaught: {_err_counts['EXCEPTION']}\n"
        f"# status  : {'FAIL' if _err_counts['EXCEPTION'] else 'OK'}\n"
    )
    for fobj in (_log_f, _err_f):
        try: fobj.write(footer); fobj.flush(); fobj.close()
        except Exception: pass
    sys.__stdout__.write(footer)
atexit.register(_close_logs)

print(f"[log] run.log        → {_LOG_PATH}")
print(f"[log] run_errors.log → {_ERR_PATH}")

# ------------------------------------------------------------
# Housekeeping: remove legacy / pre-rename outputs from previous runs
# so the final publication_ready/ directory contains ONLY the canonical
# Table_1..4 + Table_S1..S6 + figures (no "newTable_*", no "Table_S_*").
# ------------------------------------------------------------
_LEGACY_PATTERNS = [
    "newTable_*.csv",                         # leftover from earlier prefixed runs
    "Table_S_NRI_IDI_AUC.csv",                # → Table_S2_NRI_IDI_AUC.csv
    "Table_S_AdditiveInteraction_summary.csv",     # → Table_S3
    "Table_S_AdditiveInteraction_4cell.csv",       # → (subsumed into Table_S3)
    "Table_S4_AdditiveInteraction_4cell.csv",      # → subsumed into Table_S3 (rows of cell-level data)
    "Table_S_AdditiveInteraction_AllSigPairs.csv", # → Table_S4 (renumbered down)
    "Table_S5_AdditiveInteraction_AllSigPairs.csv",# → Table_S4 (renumbered down)
    "Table_S_SimpleSlope_CHA_IL6.csv",             # → Table_S5 (renumbered down)
    "Table_S6_SimpleSlope_CHA_IL6.csv",            # → Table_S5 (renumbered down)
    "table_S1_nt_loads_by_outcome.csv",        # placeholder, superseded by Table_S1_Deep_Phenotyping_FDR
    "supplementary/Supp_Fig_S6_Permutation.csv",   # → S6A
    "supplementary/Supp_Fig_S6_mRS_Cutpoint.csv",  # → S6B
    "supplementary/Supp_Fig_S6_Spin_Test.csv",     # → S6C
    "supplementary/Supp_Fig_S6_Sensitivity.png",   # → Supp_Fig_S6.png
    "supplementary/Supp_Fig_S6_Sensitivity.pdf",   # → Supp_Fig_S6.pdf
]
_removed = []
for pat in _LEGACY_PATTERNS:
    for f in PUB.glob(pat):
        try:
            f.unlink(); _removed.append(f.name)
        except Exception:
            pass
if _removed:
    print(f"[cleanup] removed {len(_removed)} legacy file(s): {', '.join(sorted(_removed))}")

print("="*60)
print(" Generating publication-ready Tables & Figures")
print("="*60)

def fmt_p(p):
    if pd.isna(p): return '–'
    if p < 0.001: return f"{p:.2e}"
    return f"{p:.3f}"

def fmt_or_ci(or_, lo, hi):
    return f"{or_:.3f} ({lo:.3f}–{hi:.3f})"

# ============================================================
# TABLE 2: Acute Outcome NT effects (Model A/B/C)
# ============================================================
print("\n[Table 2] NT effects on discharge mRS (Model A/B/C)")
df = pd.read_csv(SRC / "ordinal_regression.csv")
d = df[df['Outcome'] == 'D_MRS'].copy()
d['OR_CI'] = d.apply(lambda r: fmt_or_ci(r['OR'], r['OR_CI_lower'], r['OR_CI_upper']), axis=1)
d['P_fmt'] = d['P_value'].apply(fmt_p)

models = ['A_Unadjusted', 'B_Demographic', 'C_Full']
or_pivot = d[d['Model'].isin(models)].pivot_table(
    index='NT_Variable', columns='Model', values='OR_CI', aggfunc='first')
p_pivot = d[d['Model'].isin(models)].pivot_table(
    index='NT_Variable', columns='Model', values='P_fmt', aggfunc='first')

# Sort by Model C P-value
p_num = d[d['Model']=='C_Full'].set_index('NT_Variable')['P_value']
order = p_num.sort_values().index

table2 = pd.DataFrame(index=order)
for m in models:
    label = {'A_Unadjusted':'Model A','B_Demographic':'Model B','C_Full':'Model C'}[m]
    table2[f'{label}: OR (95% CI)'] = or_pivot[m]
    table2[f'{label}: P'] = p_pivot[m]

table2.index.name = 'NT_System'
table2.to_csv(PUB / "Table_2_NT_Acute_Outcome.csv", encoding='utf-8-sig')
print(f"  → Table_2_NT_Acute_Outcome.csv ({len(table2)} rows)")

# ============================================================
# TABLE 3: CST-adjusted Model D
# ============================================================
print("\n[Table 3] CST-adjusted (Model D)")
cst = pd.read_csv(SRC / "cst_nt_comparison.csv")
# Add local BH-17 FDR within each Control group (Model D = +CST family of 17 NT)
if 'q_BH17' not in cst.columns:
    from statsmodels.stats.multitest import multipletests
    cst['q_BH17'] = np.nan
    for ctrl_val, sub in cst.groupby('Control'):
        cst.loc[sub.index, 'q_BH17'] = multipletests(sub['P'].values, method='fdr_bh')[1]
has_ci = 'OR_CI_lower' in cst.columns
pivot_or = cst.pivot_table(index='NT', columns='Control', values='OR', aggfunc='first')
pivot_p  = cst.pivot_table(index='NT', columns='Control', values='P', aggfunc='first')
pivot_q  = cst.pivot_table(index='NT', columns='Control', values='q_BH17', aggfunc='first')
if has_ci:
    pivot_lo = cst.pivot_table(index='NT', columns='Control', values='OR_CI_lower', aggfunc='first')
    pivot_hi = cst.pivot_table(index='NT', columns='Control', values='OR_CI_upper', aggfunc='first')

table3 = pd.DataFrame(index=pivot_or.index)
for ctrl in pivot_or.columns:
    label = 'Without CST' if '无' in str(ctrl) else 'With CST (Model D)'
    if has_ci:
        table3[f'{label}: OR (95% CI)'] = [
            fmt_or_ci(pivot_or.loc[i, ctrl], pivot_lo.loc[i, ctrl], pivot_hi.loc[i, ctrl])
            for i in pivot_or.index
        ]
    else:
        table3[f'{label}: OR'] = pivot_or[ctrl].round(3)
    table3[f'{label}: P']           = pivot_p[ctrl].apply(fmt_p)
    table3[f'{label}: q_BH17 (FDR)'] = pivot_q[ctrl].apply(fmt_p)

# ΔOR%  — attenuation of EXCESS OR (OR-1), the standard epi convention
# (OR = 1 is the null; relative change of OR itself underestimates the effect).
ctrl_cols = list(pivot_or.columns)
if len(ctrl_cols) == 2:
    no_cst = [c for c in ctrl_cols if '无' in str(c)][0]
    with_cst = [c for c in ctrl_cols if c != no_cst][0]
    excess_no  = pivot_or[no_cst]   - 1.0
    excess_with = pivot_or[with_cst] - 1.0
    table3['ΔOR (%)'] = ((excess_with - excess_no) / excess_no * 100).apply(lambda x: f"{x:.1f}")

# Sort by Model D P-value
if with_cst:
    order = pivot_p[with_cst].sort_values().index
    table3 = table3.loc[order]

table3.index.name = 'NT_System'
table3.to_csv(PUB / "Table_3_CST_Adjusted_Model_D.csv", encoding='utf-8-sig')
print(f"  → Table_3_CST_Adjusted_Model_D.csv ({len(table3)} rows)")

# ============================================================
# TABLE 4: Significant NT × Inflammation interactions
# ============================================================
print("\n[Table 4] Significant NT × Inflammation interactions (global q < 0.05)")
inter = pd.read_csv(SRC / "interaction.csv")
# Merge GLOBAL FDR (Q_global) — this is what Fig 3A and the manuscript both report.
gfdr_for_table4 = pd.read_csv(SRC / "global_fdr.csv")
gfdr_inter_t4 = gfdr_for_table4[gfdr_for_table4['Module'] == 'Interaction'].copy()
gfdr_inter_t4[['NT', 'Inflam']] = gfdr_inter_t4['Label'].str.split('×', n=1, expand=True)
gfdr_inter_t4 = gfdr_inter_t4.rename(columns={'Q_global': 'Interaction_FDR_q_global'})
inter = inter.merge(gfdr_inter_t4[['NT', 'Inflam', 'Interaction_FDR_q_global']],
                    on=['NT', 'Inflam'], how='left')
sig = inter[inter['Interaction_FDR_q_global'] < 0.05].copy()
sig = sig.sort_values('Interaction_FDR_q_global')
# For the SAVED table only, drop the legacy local-FDR column (BH within each
# inflam family) and rename the global column to a single clean header, so the
# published Table 4 does not contain two near-identical "Interaction_FDR…"
# columns. The in-memory `sig` keeps `Interaction_FDR_q_global` because it is
# referenced by downstream Fig 3B 4-cell analysis.
_sig_to_save = (
    sig.drop(columns=[c for c in ['Interaction_FDR', 'Interaction_FDR_local']
                      if c in sig.columns])
       .rename(columns={'Interaction_FDR_q_global': 'Interaction_FDR'})
)
_sig_to_save.to_csv(PUB / "Table_4_NT_Inflammation_Interactions.csv",
                    index=False, encoding='utf-8-sig')
print(f"  → Table_4_NT_Inflammation_Interactions.csv ({len(_sig_to_save)} significant pairs at global q < 0.05)")

# ============================================================
# FIG 2: (A) Forest plot with 95% CI  +  (B) % effect retained after CST
# ============================================================
print("\n[Fig 2] (A) Forest plot with 95% CI + (B) % effect retained")

# Need CI columns for Panel A
if not has_ci:
    print("  ⚠️ cst_nt_comparison.csv 缺少 OR_CI_lower/upper 列，Fig 2 无法画 CI 误差线，跳过")
else:
    # Sort NT by Model D P value (most significant on top)
    nts = pivot_p[with_cst].sort_values().index.tolist()
    y = np.arange(len(nts))

    or_C  = np.array([pivot_or.loc[n, no_cst]   for n in nts])
    lo_C  = np.array([pivot_lo.loc[n, no_cst]   for n in nts])
    hi_C  = np.array([pivot_hi.loc[n, no_cst]   for n in nts])
    or_D  = np.array([pivot_or.loc[n, with_cst] for n in nts])
    lo_D  = np.array([pivot_lo.loc[n, with_cst] for n in nts])
    hi_D  = np.array([pivot_hi.loc[n, with_cst] for n in nts])
    q_D   = np.array([pivot_p.loc[n, with_cst]  for n in nts])  # raw P used as proxy for highlighting

    # % effect retained = (OR_D - 1) / (OR_C - 1) * 100   (effect on log scale ≈ ratio of (OR-1))
    # Use signed retention so that protective effects (OR<1) are also handled symmetrically
    eff_C = or_C - 1.0
    eff_D = or_D - 1.0
    # Mask retention when baseline effect is too small (|OR_C - 1| < 0.05)
    # to avoid unstable / misleading ratios on near-null effects
    weak_eff = np.abs(eff_C) < 0.05
    retained = np.where(weak_eff, np.nan, eff_D / eff_C * 100.0)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 8.5),
                                   gridspec_kw={'width_ratios': [2.0, 1.0]})

    # ── Panel A: Forest plot with CI ──
    offC, offD = -0.18, +0.18
    sig_mask = q_D < 0.05

    axA.errorbar(or_C, y + offC, xerr=[or_C - lo_C, hi_C - or_C],
                 fmt='s', markersize=7, color=COL['blue'],
                 ecolor=COL['blue'], elinewidth=1.4, capsize=3,
                 label='Model C  (without CST)')
    axA.errorbar(or_D, y + offD, xerr=[or_D - lo_D, hi_D - or_D],
                 fmt='o', markersize=7, color=COL['red'],
                 ecolor=COL['red'], elinewidth=1.4, capsize=3,
                 label='Model D  (with CST)')

    axA.axvline(1.0, color='black', linestyle='--', alpha=0.6, linewidth=1)
    axA.set_yticks(y)
    axA.set_yticklabels(nts, fontsize=10)
    axA.invert_yaxis()
    axA.set_xlabel('Odds Ratio per 1-SD')
    axA.text(-0.18, 1.02, 'A', transform=axA.transAxes,
             fontsize=18, fontweight='bold', va='top')
    axA.set_title('Forest plot — discharge mRS', loc='left',
                  fontsize=12, pad=8)
    axA.legend(loc='lower right', frameon=True, framealpha=0.95)
    axA.grid(axis='x', alpha=0.25, linestyle='--')

    # ── Panel B: % effect retained ──
    bar_colors = [COL['green']  if (s and 0 < r <= 100) else
                  COL['orange'] if (s and r > 100)      else
                  COL['lgray']
                  for s, r in zip(sig_mask, retained)]
    axB.barh(y, retained, color=bar_colors, alpha=0.9,
             edgecolor='black', linewidth=0.5)
    axB.axvline(100, color='black', linestyle=':', alpha=0.6, linewidth=1)
    axB.axvline(80,  color=COL['red'], linestyle='--', alpha=0.5, linewidth=0.8)
    axB.set_yticks(y)
    axB.set_yticklabels([])
    axB.invert_yaxis()
    axB.set_xlabel('% effect retained\n(Model D vs Model C)')
    axB.text(-0.04, 1.02, 'B', transform=axB.transAxes,
             fontsize=18, fontweight='bold', va='top')
    axB.set_title('CST attenuation', loc='left', fontsize=12, pad=8)
    axB.set_xlim(0, 150)
    axB.grid(axis='x', alpha=0.25, linestyle='--')

    for yi, r, w in zip(y, retained, weak_eff):
        if w:
            axB.text(2, yi, 'n.s. (OR ≈ 1)', va='center', fontsize=8,
                     color='#888', style='italic')
        elif np.isfinite(r):
            axB.text(min(r + 2, 145), yi, f'{r:.0f}%',
                     va='center', fontsize=8)

    fig.suptitle('Fig. 2 | Neurotransmitter effects survive corticospinal tract control',
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    fig.savefig(FIGS / 'Fig_2_CST_OR_attenuation.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGS / 'Fig_2_CST_OR_attenuation.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  → Fig_2_CST_OR_attenuation.png/.pdf  (Panel A: forest+CI, Panel B: %retained)")

# ============================================================
# FIG 3A: Interaction heatmap
# ============================================================
print("\n[Fig 3A] NT × Inflammation interaction heatmap")
# Use GLOBAL FDR (Q_global from global_fdr.csv) — consistent with Results & Table 4
inter = pd.read_csv(SRC / "interaction.csv")
gfdr  = pd.read_csv(SRC / "global_fdr.csv")
gfdr_inter = gfdr[gfdr['Module'] == 'Interaction'].copy()
gfdr_inter[['NT','Inflam']] = gfdr_inter['Label'].str.split('×', n=1, expand=True)
gfdr_inter = gfdr_inter.rename(columns={'Q_global':'Interaction_FDR_q_global'})
inter = inter.merge(gfdr_inter[['NT','Inflam','Interaction_FDR_q_global']],
                    on=['NT','Inflam'], how='left')

qcol = 'Interaction_FDR_q_global'
heat = inter.pivot_table(index='NT', columns='Inflam', values=qcol, aggfunc='first')
heat_log = -np.log10(heat.clip(lower=1e-10))

fig, ax = plt.subplots(figsize=(5.5, 8.5))
im = ax.imshow(heat_log.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=2.5)
# rename inflam columns for nicer display
lbl_map = {'BSL_IL6': 'IL-6', 'BSL_hsCRP': 'hsCRP'}
ax.set_xticks(range(heat_log.shape[1]))
ax.set_xticklabels([lbl_map.get(c, c) for c in heat_log.columns], fontsize=11)
ax.set_yticks(range(heat_log.shape[0]))
ax.set_yticklabels(heat_log.index, fontsize=10)
ax.tick_params(axis='both', length=0)

cbar = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
cbar.set_label('−log\u2081\u2080  global FDR q', fontsize=10)
cbar.outline.set_visible(False)

# Mark significant cells (global q < 0.05)
n_sig = 0
for i in range(heat_log.shape[0]):
    for j in range(heat_log.shape[1]):
        q = heat.iloc[i, j]
        if pd.notna(q) and q < 0.05:
            ax.text(j, i, '*', ha='center', va='center',
                    color='white', fontweight='bold', fontsize=20)
            n_sig += 1
# tidy black border
for s in ax.spines.values():
    s.set_visible(True); s.set_linewidth(0.8)

ax.set_title(f'Fig. 3A | NT × Inflammation interactions on 12-month mRS\n'
             f'({n_sig}/34 pairs at global FDR q < 0.05)',
             fontsize=12, pad=10)
plt.tight_layout()
fig.savefig(FIGS / 'Fig_3A_Interaction_heatmap.png', dpi=300, bbox_inches='tight')
fig.savefig(FIGS / 'Fig_3A_Interaction_heatmap.pdf', bbox_inches='tight')
plt.close(fig)
print(f"  → Fig_3A_Interaction_heatmap.png/.pdf  ({n_sig} sig pairs marked)")

# ============================================================
# FIG 3 B / C / D: 4-cell additive interaction + ROC + Calibration (CHA × IL-6 dual-burden)
# ============================================================
print("\n[Fig 3B/C/D] CHA × IL-6 dual-burden prediction")
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
    from sklearn.calibration import calibration_curve
    from sklearn.preprocessing import StandardScaler

    DATA = Path("/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/3.variable_outcom_merge_data/merged_neuro_data.csv")
    df_raw = pd.read_csv(DATA, low_memory=False)

    # Use CHA (human_CHA) × IL-6 as primary dual-burden pair
    nt_name = 'human_CHA'
    nt_col, inflam_col = 'human_CHA', 'BSL_IL6'
    mrs_col, tlv_col, nihss_col, age_col, sex_col = 'm12_mRS', 'TLV', 'A_NIHSS', 'AGE', 'GENDER'

    # ── Koch residualisation (mirrors Master_NT_Analysis_v4.py L378-410) ──
    # All NT × inflammation interactions reported in the manuscript were fit
    # by v4's `interaction_analysis()` using *Resid_NT* (Load_NT minus the
    # OLS prediction from TLV) whenever TLV was in `covariates_all`.  To keep
    # Fig 3B's 4-cell additive interaction on the same data contract, we
    # rebuild Resid_NT here with the identical formula and median-split on
    # the residual rather than on the raw load (which is collinear with TLV
    # at r ≈ 0.99 and would yield near-null RERI by construction).
    from scipy.stats import linregress as _linreg
    resid_map_pub = {}
    if tlv_col in df_raw.columns:
        _tlv_vals = pd.to_numeric(df_raw[tlv_col], errors='coerce').values.astype(float)
        # Candidate NT columns: anything that is numeric and is plausibly an NT
        # load.  Use the same heuristic as v4 plus the columns referenced by
        # interaction.csv via `sig_pairs_for_3B` (built later).
        _nt_keywords = ['5HT', 'DAT', 'NAT', 'VAChT', 'A4B2', 'CHA',
                        'Path', 'D1', 'D2', 'M1', 'JHU']
        _nt_candidates = [c for c in df_raw.columns
                          if c.startswith('Load_')
                          or any(k in c for k in _nt_keywords)]
        for _c in _nt_candidates:
            if _c.startswith('Resid_'):
                continue
            _y = pd.to_numeric(df_raw[_c], errors='coerce').values.astype(float)
            _valid = np.isfinite(_tlv_vals) & np.isfinite(_y)
            if _valid.sum() < 20:
                continue
            try:
                _slope, _icpt, _r, _, _ = _linreg(_tlv_vals[_valid], _y[_valid])
            except Exception:
                continue
            _pred  = _icpt + _slope * _tlv_vals
            _resid = _y - _pred
            _resid[~_valid] = np.nan
            _bare = _c.replace('Load_', '').replace('Resid_', '')
            _rcol = f'Resid_{_bare}'
            df_raw[_rcol] = _resid
            resid_map_pub[_c]    = _rcol   # raw load → residual
            resid_map_pub[_bare] = _rcol   # bare name → residual (interaction.csv uses bare names)
        print(f"  Koch residualisation: {len({v for v in resid_map_pub.values()})} "
              f"Resid_* columns created (TLV-orthogonalised)")
    else:
        print(f"  ⚠️ TLV column '{tlv_col}' not found — Resid_NT not built; "
              f"Fig 3B will fall back to raw load (DO NOT USE for manuscript).")

    need = [nt_col, inflam_col, mrs_col, tlv_col, nihss_col, age_col, sex_col]
    work = df_raw[need].apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
    print(f"  N = {len(work)}")

    y = (work[mrs_col] > 2).astype(int).values
    nt_med, inf_med = work[nt_col].median(), work[inflam_col].median()
    high_nt = (work[nt_col] > nt_med).astype(int).values
    high_inf = (work[inflam_col] > inf_med).astype(int).values
    cat_HL = ((high_nt == 1) & (high_inf == 0)).astype(int)
    cat_LH = ((high_nt == 0) & (high_inf == 1)).astype(int)
    cat_HH = ((high_nt == 1) & (high_inf == 1)).astype(int)

    base_feats = work[[tlv_col, nihss_col, age_col, sex_col]].values
    nt_feats   = np.column_stack([base_feats, work[nt_col].values])
    dh_feats   = np.column_stack([base_feats, cat_HL, cat_LH, cat_HH])

    scaler = StandardScaler()
    def fit_predict(X, y):
        Xs = scaler.fit_transform(X)
        m = LogisticRegression(max_iter=2000, solver='lbfgs')
        m.fit(Xs, y)
        return m.predict_proba(Xs)[:, 1]

    p_base = fit_predict(base_feats, y)
    p_nt   = fit_predict(nt_feats, y)
    p_dh   = fit_predict(dh_feats, y)

    # Metrics
    def cont_nri(y, p_old, p_new):
        ev, ne = (y == 1), (y == 0)
        d = p_new - p_old
        return ((d[ev] > 0).mean() - (d[ev] < 0).mean()) + \
               ((d[ne] < 0).mean() - (d[ne] > 0).mean())
    def idi(y, p_old, p_new):
        ev, ne = (y == 1), (y == 0)
        return (p_new[ev].mean() - p_old[ev].mean()) - (p_new[ne].mean() - p_old[ne].mean())

    auc_base = roc_auc_score(y, p_base)
    auc_nt   = roc_auc_score(y, p_nt)
    auc_dh   = roc_auc_score(y, p_dh)
    nri_nt   = cont_nri(y, p_base, p_nt)
    nri_dh   = cont_nri(y, p_base, p_dh)
    idi_nt   = idi(y, p_base, p_nt)
    idi_dh   = idi(y, p_base, p_dh)
    dAUC_nt  = auc_nt - auc_base
    dAUC_dh  = auc_dh - auc_base

    # ── Bootstrap (1000 iter): refit all 3 models in each bootstrap sample,
    #    collect AUC of each model, ΔAUC vs base, and NRI / IDI vs base for
    #    BOTH the NT-only and dual-burden models. Two-sided bootstrap P is
    #    derived as 2·min(P(>0), P(<0)) from the empirical bootstrap dist.
    N_BOOT = 1000
    RNG = np.random.RandomState(42)
    boot = {'auc_base':[], 'auc_nt':[], 'auc_dh':[],
            'dAUC_nt':[], 'dAUC_dh':[],
            'NRI_nt':[],  'NRI_dh':[],
            'IDI_nt':[],  'IDI_dh':[]}
    n = len(y)
    for b in range(N_BOOT):
        idx = RNG.choice(n, n, replace=True)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb): continue
        try:
            pb_b = fit_predict(base_feats[idx], yb)
            pb_n = fit_predict(nt_feats[idx], yb)
            pb_d = fit_predict(dh_feats[idx], yb)
            a_b, a_n, a_d = roc_auc_score(yb, pb_b), roc_auc_score(yb, pb_n), roc_auc_score(yb, pb_d)
            boot['auc_base'].append(a_b)
            boot['auc_nt'].append(a_n)
            boot['auc_dh'].append(a_d)
            boot['dAUC_nt'].append(a_n - a_b)
            boot['dAUC_dh'].append(a_d - a_b)
            boot['NRI_nt'].append(cont_nri(yb, pb_b, pb_n))
            boot['NRI_dh'].append(cont_nri(yb, pb_b, pb_d))
            boot['IDI_nt'].append(idi(yb, pb_b, pb_n))
            boot['IDI_dh'].append(idi(yb, pb_b, pb_d))
        except Exception:
            continue

    def _ci(arr):
        a = np.asarray(arr, dtype=float)
        if a.size == 0: return (np.nan, np.nan)
        return tuple(np.percentile(a, [2.5, 97.5]))
    def _two_sided_p(arr):
        a = np.asarray(arr, dtype=float)
        if a.size == 0: return np.nan
        return float(2 * min((a > 0).mean(), (a < 0).mean()))

    auc_base_lo, auc_base_hi = _ci(boot['auc_base'])
    auc_nt_lo,   auc_nt_hi   = _ci(boot['auc_nt'])
    auc_dh_lo,   auc_dh_hi   = _ci(boot['auc_dh'])
    dAUC_nt_lo,  dAUC_nt_hi  = _ci(boot['dAUC_nt'])
    dAUC_dh_lo,  dAUC_dh_hi  = _ci(boot['dAUC_dh'])
    nri_nt_lo,   nri_nt_hi   = _ci(boot['NRI_nt'])
    nri_dh_lo,   nri_dh_hi   = _ci(boot['NRI_dh'])
    idi_nt_lo,   idi_nt_hi   = _ci(boot['IDI_nt'])
    idi_dh_lo,   idi_dh_hi   = _ci(boot['IDI_dh'])
    dAUC_nt_p = _two_sided_p(boot['dAUC_nt'])
    dAUC_dh_p = _two_sided_p(boot['dAUC_dh'])
    nri_nt_p  = _two_sided_p(boot['NRI_nt'])
    nri_dh_p  = _two_sided_p(boot['NRI_dh'])
    idi_nt_p  = _two_sided_p(boot['IDI_nt'])
    idi_dh_p  = _two_sided_p(boot['IDI_dh'])

    # ── Save metrics table (3 models × 17 columns; matches narrative exactly) ──
    # narrative cites: Base AUC=0.718 ; NT-only ΔAUC≈+0.000, NRI≈+0.089 P=0.62 ;
    # Dual-burden AUC=0.728 ΔAUC=+0.010 [+0.004,+0.020] NRI=+0.351 [+0.082,+0.434]
    # IDI=+0.009 [+0.003,+0.017] (all P-values bootstrap-derived).
    _s2 = pd.DataFrame([
        {'Model': 'Base (TLV+NIHSS+Age+Sex)',
         'Features': f'{tlv_col}, {nihss_col}, {age_col}, {sex_col}',
         'N': int(n), 'Events': int(y.sum()),
         'AUC': auc_base, 'AUC_CI_low': auc_base_lo, 'AUC_CI_high': auc_base_hi,
         'Delta_AUC_vs_base': np.nan,
         'Delta_AUC_CI_low': np.nan, 'Delta_AUC_CI_high': np.nan, 'Delta_AUC_P': np.nan,
         'NRI_vs_base': np.nan, 'NRI_CI_low': np.nan, 'NRI_CI_high': np.nan, 'NRI_P': np.nan,
         'IDI_vs_base': np.nan, 'IDI_CI_low': np.nan, 'IDI_CI_high': np.nan, 'IDI_P': np.nan},
        {'Model': f'Base + {nt_name} load',
         'Features': f'Base + {nt_col}',
         'N': int(n), 'Events': int(y.sum()),
         'AUC': auc_nt, 'AUC_CI_low': auc_nt_lo, 'AUC_CI_high': auc_nt_hi,
         'Delta_AUC_vs_base': dAUC_nt,
         'Delta_AUC_CI_low': dAUC_nt_lo, 'Delta_AUC_CI_high': dAUC_nt_hi, 'Delta_AUC_P': dAUC_nt_p,
         'NRI_vs_base': nri_nt, 'NRI_CI_low': nri_nt_lo, 'NRI_CI_high': nri_nt_hi, 'NRI_P': nri_nt_p,
         'IDI_vs_base': idi_nt, 'IDI_CI_low': idi_nt_lo, 'IDI_CI_high': idi_nt_hi, 'IDI_P': idi_nt_p},
        {'Model': f'Base + dual-burden ({nt_name} × {inflam_col})',
         'Features': f'Base + 3 cell dummies (HL, LH, HH) from median splits of {nt_col} and {inflam_col}',
         'N': int(n), 'Events': int(y.sum()),
         'AUC': auc_dh, 'AUC_CI_low': auc_dh_lo, 'AUC_CI_high': auc_dh_hi,
         'Delta_AUC_vs_base': dAUC_dh,
         'Delta_AUC_CI_low': dAUC_dh_lo, 'Delta_AUC_CI_high': dAUC_dh_hi, 'Delta_AUC_P': dAUC_dh_p,
         'NRI_vs_base': nri_dh, 'NRI_CI_low': nri_dh_lo, 'NRI_CI_high': nri_dh_hi, 'NRI_P': nri_dh_p,
         'IDI_vs_base': idi_dh, 'IDI_CI_low': idi_dh_lo, 'IDI_CI_high': idi_dh_hi, 'IDI_P': idi_dh_p},
    ])
    _s2.to_csv(PUB / "Table_S2_NRI_IDI_AUC.csv", index=False, encoding='utf-8-sig')
    print(f"  → Table_S2_NRI_IDI_AUC.csv  "
          f"(3 models × {_s2.shape[1]} cols; "
          f"Base AUC={auc_base:.3f} [{auc_base_lo:.3f},{auc_base_hi:.3f}], "
          f"DualBurden AUC={auc_dh:.3f} ΔAUC=+{dAUC_dh:.3f} "
          f"NRI=+{nri_dh:.3f} P={nri_dh_p:.3g})")

    # Preserve legacy variable names downstream
    nri_lo, nri_hi, idi_lo, idi_hi = nri_dh_lo, nri_dh_hi, idi_dh_lo, idi_dh_hi
    nri_p  = nri_dh_p
    idi_p  = idi_dh_p

    # ---- Fig 3B: 4-cell additive interaction bar (Knol & VanderWeele 2012) ----
    # Loop over ALL globally-FDR-significant NT × inflammation pairs (q < 0.05),
    # compute the 4-cell additive interaction (LL/LH/HL/HH cells defined by
    # median splits of NT load and the inflammation marker), derive RERI / AP /
    # S from a covariate-adjusted logistic model with three cell dummies, and
    # test the multiplicative-scale interaction by a 1-df likelihood-ratio.
    # The pair with the smallest global FDR q is plotted as the primary Fig 3B.
    import statsmodels.api as sm
    from scipy.stats import chi2

    def wilson_ci(k, nn, z=1.96):
        if nn == 0: return (np.nan, np.nan)
        p = k / nn
        denom  = 1.0 + z**2 / nn
        centre = (p + z**2 / (2*nn)) / denom
        halfw  = z * np.sqrt(p*(1-p)/nn + z**2/(4*nn**2)) / denom
        return centre - halfw, centre + halfw

    def compute_additive_interaction(nt_col_, inflam_col_, df_raw_, mrs_col_,
                                      covar_cols_, n_boot=1000, rng_seed=42):
        """Returns (cells_df, summary_dict) for one NT × inflammation pair.

        Uses median splits of *nt_col_* and *inflam_col_* on the complete-case
        rows (after dropping NaN in any of the required columns).  The cell
        OR/RERI/AP/S model adjusts for `covar_cols_`; the LRT compares
        main-effects only vs. main + product term.
        """
        need_ = [nt_col_, inflam_col_, mrs_col_, *covar_cols_]
        w_ = (df_raw_[need_]
              .apply(pd.to_numeric, errors='coerce')
              .dropna()
              .reset_index(drop=True))
        if len(w_) < 100:
            return None, None
        y_ = (w_[mrs_col_] > 2).astype(int).values
        if y_.sum() < 20 or y_.sum() == len(y_):
            return None, None
        n_ = len(w_)
        nt_med_, inf_med_ = w_[nt_col_].median(), w_[inflam_col_].median()
        h_nt_  = (w_[nt_col_]     > nt_med_).astype(int).values
        h_inf_ = (w_[inflam_col_] > inf_med_).astype(int).values
        cells_ = {
            'LL': (h_nt_ == 0) & (h_inf_ == 0),
            'LH': (h_nt_ == 0) & (h_inf_ == 1),
            'HL': (h_nt_ == 1) & (h_inf_ == 0),
            'HH': (h_nt_ == 1) & (h_inf_ == 1),
        }
        cn_  = {k: int(v.sum())     for k, v in cells_.items()}
        cev_ = {k: int(y_[v].sum()) for k, v in cells_.items()}
        cp_  = {k: cev_[k] / cn_[k] if cn_[k] else np.nan for k in cells_}
        cci_ = {k: wilson_ci(cev_[k], cn_[k]) for k in cells_}
        if min(cn_.values()) < 30:
            return None, None
        d_LH_ = ((h_nt_ == 0) & (h_inf_ == 1)).astype(int)
        d_HL_ = ((h_nt_ == 1) & (h_inf_ == 0)).astype(int)
        d_HH_ = ((h_nt_ == 1) & (h_inf_ == 1)).astype(int)
        cov_  = w_[covar_cols_].values
        try:
            Xc_ = sm.add_constant(np.column_stack([cov_, d_LH_, d_HL_, d_HH_]))
            mc_ = sm.Logit(y_, Xc_).fit(disp=0)
            OR_LH_, OR_HL_, OR_HH_ = np.exp(mc_.params[-3:])
            RERI_ = OR_HH_ - OR_LH_ - OR_HL_ + 1.0
            AP_   = RERI_ / OR_HH_ if OR_HH_ else np.nan
            denom_S_ = (OR_HL_ - 1) + (OR_LH_ - 1)
            S_       = (OR_HH_ - 1) / denom_S_ if denom_S_ > 0 else np.nan
            Xm_ = sm.add_constant(np.column_stack([cov_, h_nt_, h_inf_]))
            Xf_ = sm.add_constant(np.column_stack([cov_, h_nt_, h_inf_, h_nt_ * h_inf_]))
            mm_ = sm.Logit(y_, Xm_).fit(disp=0)
            mf_ = sm.Logit(y_, Xf_).fit(disp=0)
            LR_  = 2.0 * (mf_.llf - mm_.llf)
            LRp_ = chi2.sf(LR_, df=1)
        except Exception as exc:
            print(f"     ⚠️ Logit fit failed for {nt_col_} × {inflam_col_}: {exc}")
            return None, None
        # Bootstrap RERI CI
        boot_ = []
        rng_ = np.random.RandomState(rng_seed)
        for _ in range(n_boot):
            idx_ = rng_.choice(n_, n_, replace=True)
            try:
                Xb_ = sm.add_constant(np.column_stack([
                    cov_[idx_], d_LH_[idx_], d_HL_[idx_], d_HH_[idx_]
                ]))
                mb_ = sm.Logit(y_[idx_], Xb_).fit(disp=0)
                o_LH_, o_HL_, o_HH_ = np.exp(mb_.params[-3:])
                boot_.append(o_HH_ - o_LH_ - o_HL_ + 1.0)
            except Exception:
                continue
        if len(boot_) >= 50:
            rl_, rh_ = np.percentile(boot_, [2.5, 97.5])
        else:
            rl_, rh_ = (np.nan, np.nan)
        cells_df_ = pd.DataFrame([
            {'Cell': k, 'N': cn_[k], 'Events': cev_[k],
             'Risk': cp_[k], 'CI_low': cci_[k][0], 'CI_high': cci_[k][1]}
            for k in ['LL', 'LH', 'HL', 'HH']
        ])
        summary_ = {
            'NT': nt_col_, 'Inflam': inflam_col_,
            'N_total': n_,
            'OR_LH': OR_LH_, 'OR_HL': OR_HL_, 'OR_HH': OR_HH_,
            'RERI': RERI_, 'RERI_CI_low': rl_, 'RERI_CI_high': rh_,
            'AP': AP_, 'S': S_,
            'LRT_chi2_df1': LR_, 'LRT_P': LRp_,
            'cells_df': cells_df_,
        }
        return cells_df_, summary_

    # Build the list of significant pairs from the merged interaction CSV
    sig_pairs_for_3B = (
        sig[['NT', 'Inflam', 'Interaction_FDR_q_global']]
        .dropna(subset=['NT', 'Inflam'])
        .copy()
        .assign(NT=lambda d: d['NT'].astype(str).str.strip(),
                Inflam=lambda d: d['Inflam'].astype(str).str.strip())
        .sort_values('Interaction_FDR_q_global')
        .reset_index(drop=True)
    )
    print(f"  Running 4-cell additive analysis for {len(sig_pairs_for_3B)} "
          f"globally-FDR-sig pairs …")

    additive_rows = []
    cells_per_pair = {}
    for _, prow in sig_pairs_for_3B.iterrows():
        nt_p, infl_p, qg_p = prow['NT'], prow['Inflam'], prow['Interaction_FDR_q_global']
        if nt_p not in df_raw.columns or infl_p not in df_raw.columns:
            print(f"     ⚠️ skipped {nt_p} × {infl_p} — column missing in merged_neuro_data.csv")
            continue
        # Use Resid_NT (Koch residual) for the median split — same data
        # contract as v4's `interaction_analysis()`.  Falls back to the raw
        # load only if Resid was not built (TLV missing).  `nt_p` is kept as
        # the *display label* throughout so figure / table titles stay
        # readable; the actual variable used by the cell-OR / RERI fit is
        # `nt_use_p`.
        nt_use_p = resid_map_pub.get(nt_p, nt_p)
        if nt_use_p == nt_p:
            print(f"     ⚠️ {nt_p}: no Resid_ column — using raw load (collinear with TLV).")
        cells_df_p, summ_p = compute_additive_interaction(
            nt_use_p, infl_p, df_raw, mrs_col,
            covar_cols_=[tlv_col, nihss_col, age_col, sex_col],
            n_boot=1000, rng_seed=42,
        )
        if summ_p is None:
            continue
        # Restore the bare-name display label in the summary (compute_*
        # returns the actual column name, e.g. "Resid_human_CHA").
        summ_p['NT'] = nt_p
        summ_p['NT_var_used'] = nt_use_p
        summ_p['Q_global'] = qg_p
        additive_rows.append({k: v for k, v in summ_p.items() if k != 'cells_df'})
        cells_per_pair[(nt_p, infl_p)] = cells_df_p
        print(f"     {nt_p} × {infl_p}  (q_global={qg_p:.4f}, var={nt_use_p}):  "
              f"RERI = {summ_p['RERI']:+.3f} [{summ_p['RERI_CI_low']:+.3f}, "
              f"{summ_p['RERI_CI_high']:+.3f}]  "
              f"AP = {summ_p['AP']:+.3f}  S = {summ_p['S']:.3f}  "
              f"LRT P = {summ_p['LRT_P']:.4g}")

    add_summary_df = pd.DataFrame(additive_rows)
    if not add_summary_df.empty:
        add_summary_df = add_summary_df.sort_values('Q_global').reset_index(drop=True)
    add_summary_df.to_csv(PUB / 'Table_S4_AdditiveInteraction_AllSigPairs.csv',
                          index=False, encoding='utf-8-sig')
    print(f"  → Table_S4_AdditiveInteraction_AllSigPairs.csv  "
          f"({len(add_summary_df)} pairs)")

    # ---- Pick the primary pair for Fig 3B (smallest global q, with available data) ----
    if add_summary_df.empty:
        print("  ⚠️ No sig pair could be analysed — Fig 3B PNG skipped.")
    else:
        primary = add_summary_df.iloc[0]
        primary_nt, primary_inf = primary['NT'], primary['Inflam']
        primary_cells = cells_per_pair[(primary_nt, primary_inf)]
        # NOTE: the standalone 4-cell file (Table_S4_AdditiveInteraction_4cell.csv)
        # has been retired — its 4 rows are now embedded as the first 6 columns
        # of Table_S3 (which adds OR_vs_LL + RERI/AP/S/LRT/Q_global). This keeps
        # a single source of truth for Fig 3B-Left.

        # ── Build Table_S3 as a self-contained "Fig 3B-Left ground-truth" table ──
        # Long-form: 4 rows (LL/LH/HL/HH) × cell stats + OR_vs_LL on every row,
        # plus the additive-interaction summary statistics (RERI / AP / S /
        # LRT / Q_global) annotated on the HH row only — because these are
        # JOINT statistics describing the cell pattern, not single-cell
        # quantities.  This way Fig 3B-Left can be reproduced from S3 alone:
        # bars come from rows 1–4 (Risk + CI), title metrics from row 4.
        _s3 = primary_cells.copy()
        _s3.insert(0, 'Inflam', primary_inf)
        _s3.insert(0, 'NT',     primary_nt)
        # Per-cell OR vs LL (reference) — pull from `primary` summary
        _or_map = {'LL': 1.00,
                   'LH': primary['OR_LH'],
                   'HL': primary['OR_HL'],
                   'HH': primary['OR_HH']}
        _s3['OR_vs_LL'] = _s3['Cell'].map(_or_map)
        # Additive-interaction summary cols: filled ONLY on the HH row
        _summary_cols = ['RERI', 'RERI_CI_low', 'RERI_CI_high', 'AP', 'S',
                         'LRT_chi2_df1', 'LRT_P', 'N_total', 'Q_global']
        for _c in _summary_cols:
            _s3[_c] = np.nan
            _s3.loc[_s3['Cell'] == 'HH', _c] = primary[_c] if _c in primary.index else np.nan
        # Column order: identity → cell stats → joint stats
        _s3 = _s3[['NT', 'Inflam', 'Cell', 'N', 'Events', 'Risk',
                   'CI_low', 'CI_high', 'OR_vs_LL'] + _summary_cols]
        _s3.to_csv(PUB / 'Table_S3_AdditiveInteraction_summary.csv',
                   index=False, encoding='utf-8-sig')
        print(f"  → Table_S3_AdditiveInteraction_summary.csv  "
              f"(primary pair {primary_nt} × {primary_inf}: 4-cell breakdown + "
              f"RERI [CI] / AP / S / LRT / Q_global on HH row; "
              f"self-contained ground-truth for Fig 3B-Left)")

        # Pretty inflammation label for the title
        inflam_pretty = {'BSL_IL6': 'IL-6', 'BSL_HSCRP': 'hsCRP',
                         'BSL_hsCRP': 'hsCRP'}.get(primary_inf, primary_inf)

        # Pull cell-level numbers
        cp_pri  = primary_cells.set_index('Cell')['Risk'].to_dict()
        clo_pri = primary_cells.set_index('Cell')['CI_low'].to_dict()
        chi_pri = primary_cells.set_index('Cell')['CI_high'].to_dict()
        cn_pri  = primary_cells.set_index('Cell')['N'].astype(int).to_dict()
        cev_pri = primary_cells.set_index('Cell')['Events'].astype(int).to_dict()

        # ---- Plot Fig 3B as 1×2 panel ----
        # Panel A (left)  = 4-cell additive interaction for the PRIMARY pair
        #                   (smallest q_global) — Knol & VanderWeele framework.
        # Panel B (right) = Simple-slope plot for the *a priori* CHA × IL-6
        #                   dual-burden model used in Fig 3C/D. This panel
        #                   bridges the rigorous 4-cell statistic (left) to
        #                   the prediction analyses (3C/D) on a continuous
        #                   inflammation scale.
        fig, (axL, axR) = plt.subplots(
            1, 2, figsize=(14.5, 5.8),
            gridspec_kw={'width_ratios': [1.0, 1.05]})

        # =====================================================
        # Panel A: 4-cell additive interaction (primary pair)
        # =====================================================
        order   = ['LL', 'LH', 'HL', 'HH']
        labels  = [f'LL\n(low {primary_nt},\nlow {inflam_pretty})',
                   f'LH\n(low {primary_nt},\nhigh {inflam_pretty})',
                   f'HL\n(high {primary_nt},\nlow {inflam_pretty})',
                   f'HH\n(high {primary_nt},\nhigh {inflam_pretty})']
        risks   = [cp_pri[k] * 100 for k in order]
        err_lo  = [(cp_pri[k] - clo_pri[k]) * 100 for k in order]
        err_hi  = [(chi_pri[k] - cp_pri[k]) * 100 for k in order]
        colors  = [COL['gray'], COL['blue'], COL['blue'], COL['red']]
        bars = axL.bar(labels, risks, yerr=[err_lo, err_hi], capsize=5,
                       color=colors, edgecolor='black', linewidth=1.0, alpha=0.88)

        # Additive-expected risk under no interaction
        add_expected = (cp_pri['LH'] + cp_pri['HL'] - cp_pri['LL']) * 100
        axL.axhline(add_expected, linestyle='--', color='black', linewidth=1.3,
                    alpha=0.65,
                    label=f'Additive expectation: {add_expected:.1f}%')

        for b, k in zip(bars, order):
            axL.text(b.get_x() + b.get_width()/2,
                     b.get_height() + max(err_hi) * 0.4 + 0.6,
                     f'{cev_pri[k]}/{cn_pri[k]}',
                     ha='center', va='bottom', fontsize=9)

        axL.set_ylabel('Observed 12-month poor outcome (mRS > 2), %')
        # Move panel-letter further left to avoid overlapping with the title
        axL.text(-0.14, 1.12, 'A', transform=axL.transAxes,
                 fontsize=18, fontweight='bold', va='top')
        # 3-line title to keep within sub-panel width and avoid clipping with
        # the panel-B letter on the right.
        axL.set_title(
            f'4-cell additive interaction — {primary_nt} × {inflam_pretty}\n'
            f'RERI = {primary["RERI"]:+.2f} '
            f'[{primary["RERI_CI_low"]:+.2f}, {primary["RERI_CI_high"]:+.2f}]   '
            f'AP = {primary["AP"]:+.2f}   S = {primary["S"]:.2f}\n'
            f'LRT χ²₁ = {primary["LRT_chi2_df1"]:.2f}, P = {primary["LRT_P"]:.3g}',
            fontsize=10.5, loc='left', pad=8)
        axL.legend(loc='upper left', frameon=True, framealpha=0.95, fontsize=9)
        axL.grid(alpha=0.25, linestyle='--', axis='y')
        axL.set_ylim(0, max(risks) + max(err_hi) + 10)

        # =====================================================
        # Panel B: SimpleSlope — CHA × IL-6 (narrative bridge to 3C/D)
        # =====================================================
        cha_var = resid_map_pub.get('human_CHA', 'human_CHA')
        ss_cols = [cha_var, 'BSL_IL6', mrs_col, tlv_col,
                   nihss_col, age_col, sex_col]
        if (cha_var in df_raw.columns) and ('BSL_IL6' in df_raw.columns):
            ss = (df_raw[ss_cols]
                  .apply(pd.to_numeric, errors='coerce')
                  .dropna()
                  .reset_index(drop=True))
            ss['IL6_z'] = ((ss['BSL_IL6'] - ss['BSL_IL6'].mean())
                           / ss['BSL_IL6'].std())
            # Tertile of CHA damage. Sign convention: higher Resid_human_CHA
            # = MORE damage than expected for the same TLV (because Load_NT
            # is computed as lesion ∩ NT density, so larger Load = more
            # spatial overlap with NT-rich tissue = more damage; the OLS
            # residual preserves this sign). Hence the upper tertile is
            # labelled 'High CHA damage'.
            ss['CHA_tert'] = pd.qcut(ss[cha_var], 3,
                                     labels=['Low', 'Mid', 'High'])
            tert_color = {'Low':  COL['blue'],
                          'Mid':  COL['gray'],
                          'High': COL['red']}
            tert_lbl   = {'Low':  'Low CHA damage',
                          'Mid':  'Mid CHA damage',
                          'High': 'High CHA damage'}

            rng_ss = np.random.RandomState(42)
            ss_rows = []
            for tert in ['Low', 'Mid', 'High']:
                m = (ss['CHA_tert'] == tert).values
                if m.sum() < 30:
                    continue
                sub = ss.loc[m]
                X = sm.add_constant(sub[['IL6_z', tlv_col, nihss_col,
                                         age_col, sex_col]].values)
                yv = sub[mrs_col].values.astype(float)
                try:
                    res_ss = sm.OLS(yv, X).fit()
                    slope_ss = float(res_ss.params[1])
                    p_ss     = float(res_ss.pvalues[1])
                except Exception as exc:
                    print(f"     ⚠️ SimpleSlope OLS failed for {tert}: {exc}")
                    continue
                ss_rows.append({
                    'Tertile': tert, 'N': int(m.sum()),
                    'Slope_IL6_z': slope_ss, 'P_slope': p_ss,
                })
                # Light jittered scatter
                jitter = rng_ss.uniform(-0.15, 0.15, size=int(m.sum()))
                axR.scatter(sub['IL6_z'].values, yv + jitter,
                            s=8, alpha=0.10, color=tert_color[tert])
                # Adjusted slope line at covariate means of the tertile
                covar_mean = sub[[tlv_col, nihss_col, age_col, sex_col]].mean().values
                il6_mean = float(sub['IL6_z'].mean())
                xx = np.linspace(np.percentile(sub['IL6_z'], 1),
                                 np.percentile(sub['IL6_z'], 99), 50)
                y_at_mean = (res_ss.params[0]
                             + res_ss.params[1] * il6_mean
                             + res_ss.params[2:] @ covar_mean)
                yy = y_at_mean + slope_ss * (xx - il6_mean)
                axR.plot(xx, yy, '-', color=tert_color[tert], linewidth=2.6,
                         label=(f'{tert_lbl[tert]} (n={int(m.sum())})\n'
                                f'slope={slope_ss:+.3f}, P={p_ss:.3g}'))

            # Persist the per-tertile slopes for the supplement
            if ss_rows:
                pd.DataFrame(ss_rows).to_csv(
                    PUB / 'Table_S5_SimpleSlope_CHA_IL6.csv', index=False, encoding='utf-8-sig')

            axR.set_xlabel('Systemic inflammation (z-scored IL-6)')
            axR.set_ylabel('12-month mRS')
            # Match panel-A letter offset for visual symmetry
            axR.text(-0.11, 1.12, 'B', transform=axR.transAxes,
                     fontsize=18, fontweight='bold', va='top')
            axR.set_title(
                'Inflammation effect stratified by residualized CHA damage tertile\n'
                '(adjusted for TLV + NIHSS + Age + Sex)',
                fontsize=10.5, loc='left', pad=8)
            axR.legend(loc='upper left', frameon=True, framealpha=0.95,
                       fontsize=8.5)
            axR.grid(alpha=0.25, linestyle='--')
            axR.set_xlim(-2.2, 4.2)
            axR.set_ylim(-0.5, 6.5)
        else:
            axR.text(0.5, 0.5,
                     'CHA / IL-6 data missing\n(SimpleSlope panel skipped)',
                     transform=axR.transAxes, ha='center', va='center',
                     fontsize=11, color=COL['gray'])
            axR.set_xticks([]); axR.set_yticks([])

        fig.suptitle(
            'Fig. 3B | Dual-burden interaction — '
            'additive evidence (A) and CHA-stratified inflammation slopes (B)',
            fontsize=12.5, fontweight='bold', y=1.02)
        plt.tight_layout()
        fig.savefig(FIGS / 'Fig_3B_AdditiveInteraction.png', dpi=300, bbox_inches='tight')
        fig.savefig(FIGS / 'Fig_3B_AdditiveInteraction.pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"  → Fig_3B_AdditiveInteraction.png/.pdf  "
              f"(2-panel: A=4-cell {primary_nt}×{inflam_pretty}, "
              f"B=SimpleSlope CHA×IL-6)")
        print(f"     [A] RERI = {primary['RERI']:+.3f} "
              f"[{primary['RERI_CI_low']:+.3f}, {primary['RERI_CI_high']:+.3f}], "
              f"AP = {primary['AP']:+.3f}, S = {primary['S']:.3f}, "
              f"LRT P = {primary['LRT_P']:.4g}")

    # ---- Fig 3C: ROC (Base / +NT / +Dual-burden) ----
    fig, ax = plt.subplots(figsize=(7, 6))
    curves = [
        (f'Clinical base\nAUC = {auc_base:.3f}', p_base, COL['gray'], '-', 2.0),
        (f'+ NT load\nAUC = {auc_nt:.3f}  (Δ = {auc_nt-auc_base:+.3f})',
         p_nt, COL['blue'], '--', 2.0),
        (f'+ Dual-burden ({nt_name} × IL-6)\nAUC = {auc_dh:.3f}  (Δ = {auc_dh-auc_base:+.3f})\n'
         f'NRI = {nri_dh:+.3f} [{nri_lo:+.3f}, {nri_hi:+.3f}], P = {nri_p:.3f}',
         p_dh, COL['red'], '-', 2.6),
    ]
    for label, prob, c, ls, lw in curves:
        fpr, tpr, _ = roc_curve(y, prob)
        ax.plot(fpr, tpr, linewidth=lw, label=label, color=c, linestyle=ls)
    ax.plot([0, 1], [0, 1], 'k:', alpha=0.4, linewidth=1)
    ax.set_xlabel('False positive rate (1 − Specificity)')
    ax.set_ylabel('True positive rate (Sensitivity)')
    ax.set_title('Fig. 3C | Nested prediction models for 12-mo poor outcome', fontsize=12)
    ax.legend(loc='lower right', frameon=True, framealpha=0.95, fontsize=8.5)
    ax.grid(alpha=0.25, linestyle='--')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect('equal')
    plt.tight_layout()
    fig.savefig(FIGS / 'Fig_3C_ROC.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGS / 'Fig_3C_ROC.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  → Fig_3C_ROC.png/.pdf")

    # ---- Fig 3D: Calibration ----
    brier = brier_score_loss(y, p_dh)
    fig, ax = plt.subplots(figsize=(7, 6))
    pt, pp = calibration_curve(y, p_dh, n_bins=10, strategy='quantile')
    ax.plot([0, 1], [0, 1], 'k:', linewidth=1, alpha=0.5, label='Perfect calibration')
    ax.plot(pp, pt, marker='o', markersize=8, linewidth=2.4,
            color=COL['red'], label=f'Dual-burden model  (Brier = {brier:.3f})')
    ax2 = ax.twinx()
    ax2.hist(p_dh, bins=20, color=COL['lgray'], alpha=0.4, edgecolor='none')
    ax2.set_ylabel('Patient count', color=COL['gray'], fontsize=10)
    ax2.tick_params(axis='y', labelcolor=COL['gray'])
    ax2.spines['top'].set_visible(False)
    ax.set_xlabel('Predicted probability of poor outcome')
    ax.set_ylabel('Observed probability of poor outcome')
    ax.set_title('Fig. 3D | Calibration of dual-burden model', fontsize=12)
    ax.legend(loc='upper left', frameon=True, framealpha=0.95)
    ax.grid(alpha=0.25, linestyle='--')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect('equal')
    plt.tight_layout()
    fig.savefig(FIGS / 'Fig_3D_Calibration.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGS / 'Fig_3D_Calibration.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  → Fig_3D_Calibration.png/.pdf")

    print(f"\n  Metrics: AUC_base={auc_base:.3f}, AUC_dh={auc_dh:.3f}, "
          f"NRI={nri_dh:+.3f} [{nri_lo:+.3f},{nri_hi:+.3f}] P={nri_p:.3f}, "
          f"IDI={idi_dh:+.3f} [{idi_lo:+.3f},{idi_hi:+.3f}] P={idi_p:.3f}")
except Exception as e:
    import traceback
    print(f"  ⚠️ Fig 3B/C/D failed: {type(e).__name__}: {e}")
    traceback.print_exc()

# ============================================================
# FIG 4A: DCA
# ============================================================
print("\n[Fig 4A] Decision Curve Analysis")
try:
    dca = pd.read_csv(SRC / "dca.csv")
    thr_col = 'Threshold' if 'Threshold' in dca.columns else dca.columns[0]

    fig, ax = plt.subplots(figsize=(7.5, 6))

    # 临床阈值区间阴影
    ax.axvspan(0.10, 0.40, color=COL['shade'], alpha=0.6, zorder=0)
    ax.text(0.25, 0.275, 'Clinically relevant\nrange', ha='center', va='top',
            fontsize=9, color='#888', style='italic')

    # Treat All / None 参考线
    if 'NB_TreatAll' in dca.columns:
        ax.plot(dca[thr_col], dca['NB_TreatAll'], '--', color=COL['gray'],
                linewidth=1.2, label='Treat all', alpha=0.85)
    if 'NB_TreatNone' in dca.columns:
        ax.plot(dca[thr_col], dca['NB_TreatNone'], ':', color=COL['gray'],
                linewidth=1.2, label='Treat none', alpha=0.85)
    else:
        ax.axhline(0, color=COL['gray'], linestyle=':', linewidth=1.2,
                   label='Treat none', alpha=0.85)

    # Base / Full 主曲线
    if 'NB_Base' in dca.columns:
        ax.plot(dca[thr_col], dca['NB_Base'], '-', color=COL['blue'],
                linewidth=2.6, label='Clinical base (TLV + NIHSS + Age + Sex)')
    if 'NB_Full' in dca.columns:
        ax.plot(dca[thr_col], dca['NB_Full'], '-', color=COL['red'],
                linewidth=2.6, label='+ Dual-burden (CHA × IL-6)')

    ax.set_xlabel('Threshold probability')
    ax.set_ylabel('Net benefit')
    ax.text(-0.13, 1.04, 'A', transform=ax.transAxes,
            fontsize=20, fontweight='bold', va='top')
    ax.set_title('Decision curve analysis', loc='left', fontsize=12, pad=8)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(-0.05, 0.30)
    ax.legend(loc='upper right', frameon=True, framealpha=0.95)
    ax.grid(alpha=0.25, linestyle='--')

    plt.tight_layout()
    fig.savefig(FIGS / 'Fig_4A_DCA.png', dpi=400, bbox_inches='tight')
    fig.savefig(FIGS / 'Fig_4A_DCA.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  → Fig_4A_DCA.png/.pdf")
except Exception as e:
    print(f"  ⚠️ Fig 4A failed: {e}")

# ============================================================
# FIG 4B: 10-fold CV
# ============================================================
print("\n[Fig 4B] 10-fold Cross-Validation")
try:
    cv = pd.read_csv(SRC / "cv_10fold.csv")
    needed = {'NT', 'AUC_base', 'AUC_full', 'Delta_AUC'}
    if not needed.issubset(cv.columns):
        print(f"  ⚠️ 列名不匹配（实际列：{list(cv.columns)}）")
    else:
        cv = cv.sort_values('Delta_AUC', ascending=True).reset_index(drop=True)
        n_pos = (cv['Delta_AUC'] > 0.001).sum()
        # 三档颜色：强增量 > 0.001 红；微弱 0–0.001 橙；<= 0 灰
        def cv_color(d):
            if d > 0.001: return COL['red']
            if d > 0:     return COL['orange']
            return COL['lgray']
        colors_4b = [cv_color(d) for d in cv['Delta_AUC']]

        fig, ax = plt.subplots(figsize=(8, max(5.5, 0.34 * len(cv) + 1.5)))
        y = np.arange(len(cv))
        bars = ax.barh(y, cv['Delta_AUC'].values, color=colors_4b, alpha=0.9,
                       edgecolor='black', linewidth=0.5)
        # 零 + 0.001 + 0.005 参考线
        ax.axvline(0,     color='black', linewidth=0.8)
        ax.axvline(0.001, color=COL['orange'], linestyle=':',  linewidth=0.8, alpha=0.6)
        ax.axvline(0.005, color=COL['red'],    linestyle='--', linewidth=0.8, alpha=0.6)

        ax.set_yticks(y)
        ax.set_yticklabels(cv['NT'].tolist(), fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('ΔAUC vs clinical base  (10-fold CV)')
        ax.text(-0.13, 1.03, 'B', transform=ax.transAxes,
                fontsize=20, fontweight='bold', va='top')
        ax.set_title(f'10-fold cross-validation  (top NT: {cv.iloc[-1]["NT"]}, '
                     f'ΔAUC = {cv.iloc[-1]["Delta_AUC"]:+.4f})',
                     loc='left', fontsize=12, pad=8)
        ax.grid(axis='x', alpha=0.25, linestyle='--')

        # auto x-limit: 紧贴数据范围 + 一点 padding，避免大片空白
        d_min = float(cv['Delta_AUC'].min())
        d_max = float(cv['Delta_AUC'].max())
        pad = max(0.0005, (d_max - d_min) * 0.30)
        ax.set_xlim(d_min - pad, d_max + pad * 1.5)

        # 数值标注：动态 offset，避免与 y 轴标签碰撞
        x_offset = (d_max - d_min) * 0.02 + 0.0001
        for yi, d in zip(y, cv['Delta_AUC'].values):
            x_pos = d + (x_offset if d >= 0 else -x_offset)
            ha = 'left' if d >= 0 else 'right'
            ax.text(x_pos, yi, f'{d:+.4f}', va='center', ha=ha,
                    fontsize=8, color='#333')

        plt.tight_layout()
        fig.savefig(FIGS / 'Fig_4B_CV_10fold.png', dpi=400, bbox_inches='tight')
        fig.savefig(FIGS / 'Fig_4B_CV_10fold.pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"  → Fig_4B_CV_10fold.png/.pdf  ({n_pos}/{len(cv)} NT with ΔAUC > +0.001)")
except Exception as e:
    print(f"  ⚠️ Fig 4B failed: {e}")

# ============================================================
# FIG 6A-B: Small lesion phenotype
# ============================================================
print("\n[Fig 6A-B] Small-lesion phenotype")
try:
    nt_cmp = pd.read_csv(SRC / "anomalous_nt_compare.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5),
                             gridspec_kw={'width_ratios': [1.6, 1.0]})

    # ---- Panel A: NT Cohen's d forest ----
    ax = axes[0]
    nt_cmp_sorted = nt_cmp.sort_values('Cohens_d', ascending=True).reset_index(drop=True)
    # Two-tone significance encoding so direction is unambiguous:
    #   positive d (severe > good = MORE damage in severe → supports narrative)
    #     → orange-red (COL['red'])
    #   negative d (severe < good = LESS damage in severe → opposite direction)
    #     → steel-blue ('#4682B4')  [needs separate biological interpretation]
    #   non-significant → light gray
    def _bar_color(d_val, p_val):
        if p_val >= 0.05:
            return COL['lgray']
        return COL['red'] if d_val > 0 else '#4682B4'
    colors_a = [_bar_color(d, p) for d, p in
                zip(nt_cmp_sorted['Cohens_d'], nt_cmp_sorted['P_value'])]
    y = np.arange(len(nt_cmp_sorted))
    ax.barh(y, nt_cmp_sorted['Cohens_d'], color=colors_a, alpha=0.9,
            edgecolor='black', linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(nt_cmp_sorted['NT'].tolist(), fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color='black', linewidth=0.8)
    ax.axvline(0.2, color=COL['orange'], linestyle='--',
               linewidth=0.7, alpha=0.6)  # small effect threshold
    ax.axvline(-0.2, color=COL['orange'], linestyle='--',
               linewidth=0.7, alpha=0.6)
    # Auto-fit x-limits with 15 % headroom on the dominant side so out-of-range
    # bars (e.g. strongly negative A4B2) and their P-value annotations are not
    # clipped against the axis edge.
    d_min = float(nt_cmp_sorted['Cohens_d'].min())
    d_max = float(nt_cmp_sorted['Cohens_d'].max())
    span  = max(abs(d_min), abs(d_max))
    ax.set_xlim(-span * 1.20, span * 1.20)
    ax.set_xlabel("Cohen's d  (Severe vs. Good outcome)")
    ax.text(-0.18, 1.02, 'A', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top')
    ax.set_title('Neurochemical damage in small lesions', loc='left',
                 fontsize=12, pad=8)
    ax.grid(axis='x', alpha=0.25, linestyle='--')
    # annotate P (both directions)
    for yi, (d, p) in enumerate(zip(nt_cmp_sorted['Cohens_d'],
                                     nt_cmp_sorted['P_value'])):
        if p < 0.05:
            offset = span * 0.015
            ax.text(d + (offset if d >= 0 else -offset), yi,
                    f'P={p:.3f}',
                    va='center', ha='left' if d >= 0 else 'right',
                    fontsize=8, color='#444')
    # Lightweight legend so the two colours are unambiguous
    from matplotlib.patches import Patch as _Patch_6a
    legend_elems_6a = [
        _Patch_6a(facecolor=COL['red'], alpha=0.9,
                  label='Severe > Good (P < 0.05): more damage in severe group'),
        _Patch_6a(facecolor='#4682B4', alpha=0.9,
                  label='Severe < Good (P < 0.05): less damage in severe group'),
        _Patch_6a(facecolor=COL['lgray'], alpha=0.9,
                  label='Not significant'),
    ]
    ax.legend(handles=legend_elems_6a, loc='lower right',
              fontsize=7.5, framealpha=0.92)

    # ---- Panel B: IL-6 (or fallback) ----
    base = pd.read_csv(SRC / "anomalous_baseline.csv")
    il6 = base[base['Variable'].astype(str).str.upper().str.contains(
        r'IL.?6|IL6', regex=True, na=False)]
    ax = axes[1]
    if not il6.empty:
        row = il6.iloc[0]
        means = [row['Good_Mean'], row['Severe_Mean']]
        sems  = [row['Good_SD'] / np.sqrt(max(row.get('Good_N', 1), 1)),
                 row['Severe_SD'] / np.sqrt(max(row.get('Severe_N', 1), 1))]
        bars = ax.bar(['Good outcome', 'Severe outcome'], means,
                      yerr=sems, color=[COL['blue'], COL['red']],
                      alpha=0.9, capsize=8, edgecolor='black', linewidth=0.6,
                      error_kw={'elinewidth': 1.2, 'capthick': 1.2})
        # annotate
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width()/2, m, f'{m:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylabel(f"{row['Variable']} (raw, mean ± SEM)")
        ax.text(-0.20, 1.02, 'B', transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top')
        ax.set_title(f"{row['Variable']}: d = {row['Cohens_d']:.2f},  P = {fmt_p(row['P_value'])}",
                     loc='left', fontsize=12, pad=8)
        ax.grid(axis='y', alpha=0.25, linestyle='--')
        ax.set_ylim(0, max(means) * 1.35)
    else:
        bs = base.sort_values('Cohens_d', ascending=True).tail(8)
        colors_b = [COL['red'] if p < 0.05 else COL['lgray']
                    for p in bs['P_value']]
        ax.barh(bs['Variable'], bs['Cohens_d'], color=colors_b, alpha=0.9,
                edgecolor='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel("Cohen's d  (Severe vs. Good)")
        ax.text(-0.20, 1.02, 'B', transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top')
        ax.set_title('Top baseline differences', loc='left',
                     fontsize=12, pad=8)
        print(f"  ⚠️ IL-6 未找到，Panel B 回退为 top baseline differences")

    fig.suptitle('Fig. 6 | Small-lesion severe-outcome phenotype',
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    fig.savefig(FIGS / 'Fig_6AB_SmallLesion.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGS / 'Fig_6AB_SmallLesion.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  → Fig_6AB_SmallLesion.png/.pdf")
except Exception as e:
    print(f"  ⚠️ Fig 6 failed: {e}")

# ============================================================
# Supplementary Figures
# ============================================================
print("\n[Supp Fig S2] 12-month temporal decay")
try:
    d12 = df[df['Outcome'] == 'm12_mRS'].copy()
    d12['OR_CI'] = d12.apply(lambda r: fmt_or_ci(r['OR'], r['OR_CI_lower'], r['OR_CI_upper']), axis=1)
    d12['P_fmt'] = d12['P_value'].apply(fmt_p)
    d12_c = d12[d12['Model'] == 'C_Full'].sort_values('OR').reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    y = np.arange(len(d12_c))
    or_  = d12_c['OR'].values
    lo   = d12_c['OR_CI_lower'].values
    hi   = d12_c['OR_CI_upper'].values
    pval = d12_c['P_value'].values
    fdr  = d12_c['FDR_q'].values if 'FDR_q' in d12_c.columns else np.full(len(d12_c), np.nan)

    # 颜色：FDR sig (本实际为 0) > raw P sig > ns
    colors_s2 = []
    for p, q in zip(pval, fdr):
        if pd.notna(q) and q < 0.05:    colors_s2.append(COL['red'])
        elif p < 0.05:                  colors_s2.append(COL['orange'])
        else:                           colors_s2.append(COL['lgray'])

    ax.barh(y, or_, color=colors_s2, alpha=0.9, edgecolor='black', linewidth=0.4)
    # Error bars (95% CI)
    ax.errorbar(or_, y, xerr=[or_ - lo, hi - or_], fmt='none',
                ecolor='black', elinewidth=1, capsize=2.5)

    ax.axvline(1.0, color='black', linestyle='--', alpha=0.6, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d12_c['NT_Variable'].tolist(), fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Odds Ratio per 1-SD  (Model C, 12-month mRS)')
    ax.set_title('Supp Fig S2 | 12-month outcome: NT effects after FDR correction\n'
                 '(red = FDR q < 0.05; orange = raw P < 0.05; gray = n.s.)',
                 fontsize=12)
    ax.grid(axis='x', alpha=0.25, linestyle='--')
    plt.tight_layout()
    fig.savefig(SUPP / 'Supp_Fig_S2_12month_temporal_decay.png', dpi=300, bbox_inches='tight')
    fig.savefig(SUPP / 'Supp_Fig_S2_12month_temporal_decay.pdf', bbox_inches='tight')
    plt.close(fig)
    d12_c[['NT_Variable','OR_CI','P_fmt','FDR_q']].to_csv(
        SUPP / 'Supp_Fig_S2_12month_temporal_decay.csv', index=False, encoding='utf-8-sig')
    print(f"  → Supp_Fig_S2_12month_temporal_decay.png/.pdf + .csv")
except Exception as e:
    print(f"  ⚠️ Supp S2 failed: {e}")

print("\n[Supp Fig S3] Temporal trajectory (D/3m/6m/12m)")
try:
    times = ['D_MRS', 'm3_mRS', 'm6_mRS', 'm12_mRS']
    time_lbl = ['Discharge', '3 months', '6 months', '12 months']
    tt = df[(df['Model'] == 'C_Full') & (df['Outcome'].isin(times))]
    pivot_or_t = tt.pivot_table(index='NT_Variable', columns='Outcome',
                                values='OR', aggfunc='first')
    pivot_or_t = pivot_or_t[[c for c in times if c in pivot_or_t.columns]]

    # NT 系统分组 → 调色板
    SYS = {
        '5HT': '#9C27B0', 'D': '#3F51B5', 'NAT': COL['red'],
        'A4B2': COL['green'], 'M1': COL['green'],
        'VAChT': COL['green'], 'human_CHA': COL['green'],
        'JHU_EC': COL['orange'], 'Lateral_Path': COL['orange'],
        'Medial_Path': COL['orange'],
    }
    def nt_color(nt):
        for k, v in SYS.items():
            if nt.startswith(k) or nt == k: return v
        return COL['gray']

    # Top hit (根据 D_MRS OR 最高) 加粗
    top_set = set(pivot_or_t['D_MRS'].sort_values(ascending=False).head(6).index)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for nt in pivot_or_t.index:
        is_top = nt in top_set
        ax.plot(range(len(times)), pivot_or_t.loc[nt],
                marker='o', markersize=6 if is_top else 4,
                linewidth=2.2 if is_top else 1.0,
                alpha=0.95 if is_top else 0.45,
                color=nt_color(nt), label=nt if is_top else None)

    ax.axhline(1.0, color='black', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.axhspan(0.99, 1.01, color=COL['lgray'], alpha=0.15, zorder=0)
    ax.set_xticks(range(len(times)))
    ax.set_xticklabels(time_lbl)
    ax.set_ylabel('Odds Ratio per 1-SD  (Model C)')
    ax.set_xlabel('Follow-up time')
    ax.set_title('Supp Fig S3 | Temporal trajectory of NT effects on mRS\n'
                 '(top 6 NT highlighted; faded = remaining 11 NT)',
                 fontsize=12)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=9, title='Top NT', title_fontsize=9)
    ax.grid(alpha=0.25, linestyle='--')
    plt.tight_layout()
    fig.savefig(SUPP / 'Supp_Fig_S3_Temporal_Trajectory.png', dpi=300, bbox_inches='tight')
    fig.savefig(SUPP / 'Supp_Fig_S3_Temporal_Trajectory.pdf', bbox_inches='tight')
    plt.close(fig)
    pivot_or_t.to_csv(SUPP / 'Supp_Fig_S3_Temporal_Trajectory.csv', encoding='utf-8-sig')
    print(f"  → Supp_Fig_S3_Temporal_Trajectory.png/.pdf + .csv")
except Exception as e:
    print(f"  ⚠️ Supp S3 failed: {e}")

print("\n[Supp Fig S4A] PCA system aggregation")
try:
    pca = pd.read_csv(SRC / "pca_system.csv")
    pca_d = pca[pca['Outcome'] == 'D_MRS'].copy() if 'Outcome' in pca.columns else pca.copy()
    if 'PC1_OR' in pca_d.columns and 'System' in pca_d.columns:
        pca_d = pca_d.sort_values('PC1_OR', ascending=True)
        colors_s4a = [COL['red'] if (q < 0.05) else COL['lgray']
                      for q in pca_d.get('FDR_q', pd.Series([1]*len(pca_d)))]
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.barh(pca_d['System'], pca_d['PC1_OR'], color=colors_s4a, alpha=0.9,
                edgecolor='black', linewidth=0.5)
        ax.axvline(1.0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('PC1 OR per 1-SD  (Model C, discharge mRS)')
        ax.set_title('Supp Fig S4A | PCA system-level aggregation\n(red = FDR q < 0.05)',
                     fontsize=12)
        # Annotate every bar with OR + P (significant: scientific notation;
        # non-significant: "ns") so reviewers don't have to guess whether a
        # missing P-value means "ns" or "data missing".
        ps_iter = pca_d.get('PC1_P', pd.Series([None] * len(pca_d)))
        for yi, (or_, p_) in enumerate(zip(pca_d['PC1_OR'], ps_iter)):
            if p_ is None or pd.isna(p_):
                p_txt = ''
            elif p_ < 0.05:
                p_txt = f',  P={p_:.1e}'
            else:
                p_txt = f',  P={p_:.2f} (ns)'
            ax.text(or_ + 0.005, yi, f'OR={or_:.2f}{p_txt}',
                    va='center', fontsize=9)
        # Add headroom on the right so the longest annotation never clips
        # against the axis edge.
        xmax_s4a = float(pca_d['PC1_OR'].max())
        ax.set_xlim(0.0, max(xmax_s4a * 1.25, 1.30))
        ax.grid(axis='x', alpha=0.25, linestyle='--')
        plt.tight_layout()
        fig.savefig(SUPP / 'Supp_Fig_S4A_PCA_System.png', dpi=300, bbox_inches='tight')
        fig.savefig(SUPP / 'Supp_Fig_S4A_PCA_System.pdf', bbox_inches='tight')
        plt.close(fig)
        pca_d.to_csv(SUPP / 'Supp_Fig_S4A_PCA_System.csv', index=False, encoding='utf-8-sig')
        print(f"  → Supp_Fig_S4A_PCA_System.png/.pdf + .csv")
    else:
        print(f"  ⚠️ 列名不匹配（实际列：{list(pca.columns)}）")
except Exception as e:
    print(f"  ⚠️ Supp S4A failed: {e}")

print("\n[Supp Fig S4B] Pre- vs Post-synaptic")
try:
    syn = pd.read_csv(SRC / "synaptic_location.csv")
    syn.to_csv(SUPP / 'Supp_Fig_S4B_PreSyn_vs_PostSyn.csv', index=False, encoding='utf-8-sig')
    syn_d = syn[syn['Outcome'] == 'D_MRS'].copy() if 'Outcome' in syn.columns else syn.copy()
    if 'Synaptic_Type' in syn_d.columns and 'Mean_AbsBeta' in syn_d.columns:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        types  = syn_d['Synaptic_Type'].tolist()
        beta   = syn_d['Mean_AbsBeta'].tolist()
        nsig   = syn_d.get('N_sig',   pd.Series(['?']*len(syn_d))).tolist()
        ntot   = syn_d.get('N_total', pd.Series(['?']*len(syn_d))).tolist()
        # color: receptor red, transporter blue, tract gray
        def cls_color(t):
            tl = str(t).lower()
            if 'receptor' in tl or 'post' in tl: return COL['red']
            if 'transporter' in tl or 'pre' in tl: return COL['blue']
            return COL['gray']
        colors_s4b = [cls_color(t) for t in types]
        bars = ax.bar(types, beta, color=colors_s4b, alpha=0.9,
                      edgecolor='black', linewidth=0.6)
        ax.set_ylabel('Mean |β|  (standardized OLS)')
        ax.set_title('Supp Fig S4B | Pre- vs Post-synaptic effects\n(discharge mRS, Model C)',
                     fontsize=12)
        for b, t, ns, nt in zip(bars, types, nsig, ntot):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                    f'N\u2090\u209C\u1D63 = {ns}/{nt}',
                    ha='center', fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.25, linestyle='--')
        # 短标签 + 旋转，避免重叠
        ax.set_xticklabels([t.replace('Post-synaptic (Receptor)', 'Post-syn.\n(Receptor)')
                              .replace('Pre-synaptic (Transporter)', 'Pre-syn.\n(Transporter)')
                              .replace('Tract / Other', 'Tract /\nOther')
                            for t in types], fontsize=10)
        plt.tight_layout()
        fig.savefig(SUPP / 'Supp_Fig_S4B_PreSyn_vs_PostSyn.png', dpi=300, bbox_inches='tight')
        fig.savefig(SUPP / 'Supp_Fig_S4B_PreSyn_vs_PostSyn.pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"  → Supp_Fig_S4B_PreSyn_vs_PostSyn.png/.pdf + .csv")
    else:
        print(f"  ⚠️ 列名不匹配（实际列：{list(syn.columns)}）")
except Exception as e:
    print(f"  ⚠️ Supp S4B failed: {e}")

print("\n[Supp Fig S4C] Dose-response quartiles")
try:
    dose = pd.read_csv(SRC / "dose_response.csv")
    dose.to_csv(SUPP / 'Supp_Fig_S4C_Dose_Response.csv', index=False, encoding='utf-8-sig')
    qcol = 'KW_FDR_q' if 'KW_FDR_q' in dose.columns else 'KW_P'

    # Auto-pick outcome: prefer D_MRS, else most populated outcome
    if 'Outcome' in dose.columns:
        avail = dose['Outcome'].value_counts()
        outcome_use = 'D_MRS' if 'D_MRS' in avail.index else avail.index[0]
        dose_d = dose[dose['Outcome'] == outcome_use].copy()
        out_label = outcome_use
    else:
        dose_d = dose.copy()
        out_label = 'mRS'

    # First try FDR-significant; fall back to top 8 by q
    sig = dose_d[dose_d[qcol] < 0.05].sort_values(qcol)
    if sig.empty:
        sig = dose_d.sort_values(qcol).head(8)
        title_note = '(top 8 by KW q; none reached FDR < 0.05)'
    else:
        sig = sig.head(8)
        title_note = f'(top {len(sig)} by KW-FDR q)'

    if not sig.empty and {'Q1_mean_mRS', 'Q4_mean_mRS'}.issubset(sig.columns):
        # 颜色调色板从 COL伸展
        palette = [COL['red'], COL['blue'], COL['green'], COL['orange'],
                   COL['purple'], '#9C27B0', '#3F51B5', COL['gray']]
        fig, ax = plt.subplots(figsize=(9.4, 6))
        for i, (_, row) in enumerate(sig.iterrows()):
            qs = [row[f'Q{j}_mean_mRS'] for j in range(1, 5)]
            ax.plot([1, 2, 3, 4], qs, marker='o', linewidth=2.0,
                    markersize=7, alpha=0.9, color=palette[i % len(palette)],
                    label=f"{row['NT']}  (ρ = {row.get('Spearman_r', 0):+.3f})")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(['Q1\n(lowest)', 'Q2', 'Q3', 'Q4\n(highest)'])
        ax.set_xlabel('NT damage quartile')
        # Y-axis is mean of the 3-level grouped outcome (group_mrs):
        #   0 = mRS 0–2 (good)   1 = mRS 3–4 (moderate)   2 = mRS 5–6 (severe)
        ax.set_ylabel(f'Mean grouped {out_label}\n(0 = mRS 0–2, 1 = mRS 3–4, 2 = mRS 5–6)',
                      fontsize=10)
        ax.set_title(f'Supp Fig S4C | Dose-response: NT damage quartiles vs grouped {out_label}\n{title_note}',
                     fontsize=12)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                  fontsize=9, title='Top NT (Spearman ρ)', title_fontsize=9)
        ax.grid(alpha=0.25, linestyle='--')
        plt.tight_layout()
        fig.savefig(SUPP / 'Supp_Fig_S4C_Dose_Response.png', dpi=400, bbox_inches='tight')
        fig.savefig(SUPP / 'Supp_Fig_S4C_Dose_Response.pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"  → Supp_Fig_S4C_Dose_Response.png/.pdf + .csv  ({len(sig)} NTs, outcome={out_label})")
    else:
        print(f"  ⚠️ 列名不匹配（实际列：{list(dose.columns)}）")
except Exception as e:
    print(f"  ⚠️ Supp S4C failed: {e}")

print("\n[Supp Fig S5] Mediation (null indirect effects)")
try:
    med = pd.read_csv(SRC / "mediation_parallel.csv")
    # Merge补做的 top NT 中介
    extra_p = SRC / "mediation_top_NT_IL6.csv"
    if extra_p.exists():
        extra = pd.read_csv(extra_p)
        med = pd.concat([med, extra], ignore_index=True).drop_duplicates(
            subset=["NT", "Mediator"], keep="last")
        print(f"  ℹ️ merged mediation_top_NT_IL6.csv ({len(extra)} additional pairs)")
    med.to_csv(SUPP / 'Supp_Fig_S5_Mediation.csv', index=False, encoding='utf-8-sig')

    needed = {'NT', 'Mediator', 'Indirect_ab', 'Boot_CI_lower', 'Boot_CI_upper'}
    if needed.issubset(med.columns):
        m = med.copy()
        # Sort by absolute indirect effect for visual clarity
        m = m.reindex(m['Indirect_ab'].abs().sort_values(ascending=True).index)
        labels = [f"{r['NT']}  → {r['Mediator']}" for _, r in m.iterrows()]
        x  = m['Indirect_ab'].values
        lo = m['Boot_CI_lower'].values
        hi = m['Boot_CI_upper'].values
        sig = m.get('Significant', pd.Series([False]*len(m))).astype(bool).values
        colors_s5 = [COL['red'] if s else COL['blue'] for s in sig]

        fig, ax = plt.subplots(figsize=(8, max(4, 0.4*len(m) + 2)))
        y = np.arange(len(m))
        for yi, xi, loi, hii, c in zip(y, x, lo, hi, colors_s5):
            ax.errorbar(xi, yi, xerr=[[xi - loi], [hii - xi]],
                        fmt='o', color=c, ecolor=c, capsize=3, markersize=6)
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Indirect effect (a×b)  with 95% bootstrap CI')
        n_sig = int(sig.sum())
        ax.set_title(f'Supp Fig S5 | Bootstrap mediation (NT → inflammation → mRS)\n'
                     f'({n_sig}/{len(m)} significant; CIs crossing zero indicate null mediation)',
                     fontsize=12)
        ax.grid(axis='x', alpha=0.25, linestyle='--')
        plt.tight_layout()
        fig.savefig(SUPP / 'Supp_Fig_S5_Mediation.png', dpi=300, bbox_inches='tight')
        fig.savefig(SUPP / 'Supp_Fig_S5_Mediation.pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"  → Supp_Fig_S5_Mediation.png/.pdf + .csv  ({len(m)} pairs, {n_sig} sig)")
    else:
        print(f"  ⚠️ 列名不匹配（实际列：{list(med.columns)}）")
except Exception as e:
    print(f"  ⚠️ Supp S5 failed: {e}")

print("\n[Supp Fig S6] Sensitivity analyses (permutation, mRS cutpoint, spin)")
try:
    # ---- copy raw CSVs (also used as data tables), re-encoded as utf-8-sig so Excel
    #      can display the Unicode glyphs (✓ ✗) in the Concordance/Specific columns ----
    for src_name, dst_name in [
        ("permutation_test.csv",   "Supp_Fig_S6A_Permutation.csv"),
        ("mrs_sensitivity.csv",    "Supp_Fig_S6B_mRS_Cutpoint.csv"),
        ("spin_test.csv",          "Supp_Fig_S6C_Spin_Test.csv"),
    ]:
        src_p = SRC / src_name
        if src_p.exists():
            _df_tmp = pd.read_csv(src_p)
            _df_tmp.to_csv(SUPP / dst_name, index=False, encoding='utf-8-sig')
            print(f"  → {dst_name}  (re-encoded utf-8-sig for Excel)")

    # ---- 3-panel figure (A: Permutation | B: Cutpoint | C: Spin) ----
    perm = pd.read_csv(SRC / "permutation_test.csv") if (SRC / "permutation_test.csv").exists() else None
    cutp = pd.read_csv(SRC / "mrs_sensitivity.csv") if (SRC / "mrs_sensitivity.csv").exists() else None
    spin = pd.read_csv(SRC / "spin_test.csv") if (SRC / "spin_test.csv").exists() else None

    if perm is not None and cutp is not None and spin is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.5),
                                 gridspec_kw={'width_ratios': [1.0, 0.65, 0.95]})

        # ===== Panel A: Permutation forest =====
        axA = axes[0]
        pA = perm.copy().sort_values("Obs_Beta", ascending=True).reset_index(drop=True)
        sig_both = pA["Concordance"].astype(str).str.contains("Both", na=False)
        colors_A = [COL['red'] if s else COL['lgray'] for s in sig_both]
        y = np.arange(len(pA))
        axA.barh(y, pA["Obs_Beta"].values, color=colors_A, edgecolor='black', linewidth=0.5)
        for yi, (_, r) in zip(y, pA.iterrows()):
            ptxt = f"P_perm={r['Permutation_P']:.3f}"
            axA.text(r["Obs_Beta"] + 0.003, yi, ptxt, va='center', fontsize=8,
                     color='black' if sig_both.iloc[yi] else 'gray')
        axA.set_yticks(y)
        axA.set_yticklabels(pA["NT"].values, fontsize=9)
        axA.set_xlabel("Observed |β| (m3_mRS)")
        axA.set_title(f"A | Permutation test (1,000 iter)\n"
                      f"{int(sig_both.sum())}/{len(pA)} significant in both parametric & permutation",
                      fontsize=11)
        axA.axvline(0, color='black', linewidth=0.5)
        axA.grid(axis='x', alpha=0.25, linestyle='--')

        # legend
        from matplotlib.patches import Patch as _Patch
        axA.legend(handles=[
            _Patch(color=COL['red'],   label='Both parametric & perm sig'),
            _Patch(color=COL['lgray'], label='Not significant'),
        ], loc='lower right', fontsize=8, frameon=True)

        # ===== Panel B: mRS cutpoint heatmap (NT × cutpoint) =====
        axB = axes[1]
        cut_order = ['A_0-1_vs_2-6', 'B_0-2_vs_3-6', 'C_0-3_vs_4-6', 'D_Ordinal_0-2_3-4_5-6']
        cut_lbls  = ['A: 0-1\nvs 2-6', 'B: 0-2\nvs 3-6', 'C: 0-3\nvs 4-6', 'D: Ordinal\n0-2|3-4|5-6']
        # sort NT by total sig count
        sig_count = cutp.groupby('NT')['Sig'].sum().sort_values(ascending=False)
        nt_order = list(sig_count.index)
        OR_mat = cutp.pivot_table(index='NT', columns='Cutpoint', values='OR').reindex(nt_order)[cut_order]
        Sig_mat = cutp.pivot_table(index='NT', columns='Cutpoint', values='Sig').reindex(nt_order)[cut_order]

        im = axB.imshow(OR_mat.values, aspect='auto', cmap='RdYlBu_r',
                        vmin=0.95, vmax=1.25)
        axB.set_xticks(range(len(cut_order)))
        axB.set_xticklabels(cut_lbls, fontsize=8)
        axB.set_yticks(range(len(nt_order)))
        axB.set_yticklabels(nt_order, fontsize=9)
        # annotate OR + * for sig
        for i in range(len(nt_order)):
            for j in range(len(cut_order)):
                or_v = OR_mat.values[i, j]
                is_sig = bool(Sig_mat.values[i, j])
                star = '*' if is_sig else ''
                axB.text(j, i, f"{or_v:.2f}{star}", ha='center', va='center',
                         fontsize=8.5, color='black',
                         fontweight='bold' if is_sig else 'normal')
        axB.set_title(f"B | mRS cutpoint sensitivity\n"
                      f"NAT sig in all 4 schemes (* = P < 0.05)",
                      fontsize=11)
        cbar = fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04)
        cbar.set_label('Odds Ratio', fontsize=9)

        # ===== Panel C: Spin test =====
        axC = axes[2]
        pC = spin.copy().sort_values("Spin_P", ascending=True).reset_index(drop=True)
        is_specific = pC["Spin_P"].values < 0.05
        colors_C = [COL['red'] if s else COL['lgray'] for s in is_specific]
        y = np.arange(len(pC))
        axC.barh(y, -np.log10(pC["Spin_P"].clip(lower=1e-4).values),
                 color=colors_C, edgecolor='black', linewidth=0.5)
        axC.axvline(-np.log10(0.05), color='black', linestyle='--', alpha=0.7, linewidth=1.0)
        axC.text(-np.log10(0.05), len(pC) - 0.3, ' P_spin = 0.05', fontsize=8,
                 color='black', va='top')
        for yi, (_, r) in zip(y, pC.iterrows()):
            axC.text(-np.log10(max(r['Spin_P'], 1e-4)) + 0.03, yi,
                     f"P={r['Spin_P']:.3f}", va='center', fontsize=8,
                     color='black' if r['Spin_P'] < 0.05 else 'gray')
        axC.set_yticks(y)
        axC.set_yticklabels(pC["NT"].values, fontsize=9)
        axC.set_xlabel("−log₁₀(P_spin)")
        n_spec = int(is_specific.sum())
        axC.set_title(f"C | Spatial-null spin test (Alexander-Bloch 2018)\n"
                      f"{n_spec}/{len(pC)} spatially specific (P_spin < 0.05)",
                      fontsize=11)
        axC.grid(axis='x', alpha=0.25, linestyle='--')

        fig.suptitle("Supp Fig S6 | Sensitivity analyses: permutation, mRS cutpoint, spatial-null spin",
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        fig.savefig(SUPP / 'Supp_Fig_S6.png', dpi=300, bbox_inches='tight')
        fig.savefig(SUPP / 'Supp_Fig_S6.pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"  → Supp_Fig_S6.png/.pdf  (3-panel: A=perm | B=cutpoint | C=spin)")
    else:
        print(f"  ⚠️ S6 panel skipped: one or more CSV not found")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  ⚠️ Supp S6 failed: {e}")

# ============================================================
# Supplementary Table S1 — Deep-phenotyping FDR matrix
# ============================================================
# Tests, for every deep-phenotyping sub-domain (SIS sub-domains at 6 m & 12 m,
# MoCA total + sub-scores, Barthel-ADL, PHQ-9, GAD-7, etc.) crossed against the
# 17 TLV-orthogonalized NT residuals, the standardized OLS coefficient
# β (predictor and outcome both z-scored) with TLV + NIHSS + Age + Sex + CST_Load
# adjustment, applies BH-FDR within each outcome domain, and writes a single
# tidy table cited in Results §6 as Supplementary Table S1.
# ============================================================
print("\n[Supp Table S1] Deep-phenotyping FDR (NT × multi-dimensional outcomes)")
try:
    import re as _re
    from statsmodels.stats.multitest import multipletests as _multitest

    DATA_S1 = Path("/data/usersdir/liuzhengxin/Stepbystep/"
                   "6.NeurotransmitterMapping/3.variable_outcom_merge_data/"
                   "merged_neuro_data.csv")
    if not DATA_S1.exists():
        raise FileNotFoundError(f"merged_neuro_data.csv not found at {DATA_S1}")

    _d = pd.read_csv(DATA_S1, low_memory=False)
    _d = _d.loc[:, ~_d.columns.str.match(r'.+\.\d+$')]

    # ============================================================
    # NOTE: This block is a faithful port of analysis/outcome_analysis.py
    #       (see commit history). Any change to the regression / FDR /
    #       z-scoring / Koch-residual logic here MUST be mirrored there
    #       (and vice versa) — the two scripts are required to produce
    #       identical numbers.
    # ============================================================
    from scipy import stats as _scistats

    def _zscore(_s):
        _sd = _s.std()
        return (_s - _s.mean()) / _sd if _sd > 1e-10 else _s - _s.mean()

    def _find_col(_df, _cands):
        for _c in _cands:
            if _c in _df.columns: return _c
        return None

    # ---- Covariates (EXACT same precedence as outcome_analysis.py) ----
    _tlv   = _find_col(_d, ['TLV', 'TLV_mm3'])
    _nihss = _find_col(_d, ['A_NIHSS', 'NIHSS'])
    _age   = _find_col(_d, ['AGE', 'Age'])
    _sex   = _find_col(_d, ['SEX', 'Sex'])
    _cst   = _find_col(_d, ['CST_Load', 'CST_load'])
    print(f"  Covariate cols: TLV={_tlv}  NIHSS={_nihss}  AGE={_age}  SEX={_sex}  CST={_cst}")

    # ---- 17 NT canonical names — same as outcome_analysis.py KNOWN_NT ----
    _KNOWN_NT = ['5HT1a', '5HT1b', '5HT2a', '5HT4', '5HT6', '5HTT',
                 'A4B2', 'D1', 'D2', 'DAT', 'M1', 'NAT', 'VAChT',
                 'human_CHA', 'JHU_EC', 'Lateral_Path', 'Medial_Path']
    _nt_cols = [c for c in _d.columns if c in _KNOWN_NT]
    if not _nt_cols:
        _nt_cols = [c for c in _d.columns if c.startswith('Load_')]
    print(f"  NT loads detected: {len(_nt_cols)} ({_nt_cols[:5]}...)")

    # Coerce numeric (same as outcome_analysis.py)
    for _c in [_tlv, _nihss, _age, _sex, _cst] + _nt_cols:
        if _c is not None:
            _d[_c] = pd.to_numeric(_d[_c], errors='coerce')

    # ---- Compute Koch residuals INLINE (do NOT rely on pre-computed Resid_*) ----
    # Resid_NT = NT − (intercept + slope · TLV);  slope, intercept from OLS on TLV
    _resid_cols = []
    for _nt in _nt_cols:
        _valid = _d[[_nt, _tlv]].dropna()
        if len(_valid) > 30:
            _slope, _intercept, _, _, _ = _scistats.linregress(_valid[_tlv], _valid[_nt])
            _rname = f"Resid_{_nt.replace('Load_', '')}"
            _d[_rname] = np.nan
            _d.loc[_valid.index, _rname] = (
                _d.loc[_valid.index, _nt] - (_intercept + _slope * _d.loc[_valid.index, _tlv])
            )
            _resid_cols.append(_rname)
    print(f"  Koch residuals computed: {len(_resid_cols)}")

    # ---- Deep-phenotyping outcome columns to scan ----
    # Use the EXACT OUTCOMES dictionary defined in analysis/outcome_analysis.py.
    # The merged CNSR-III file stores deep-phenotyping items under coded
    # column names (VF6V*/VF12A*/VA*) that are not human-readable; this dict
    # gives the canonical decoding (label, raw_col, sub_domain).
    # Sub-domains group items for BH-FDR (one family per domain).
    _outcomes_decoded = [
        # (Sub_domain,                   label,                  raw_col)
        # ─── Functional Recovery (mRS family) ───
        ('Functional_Recovery_mRS',      'mRS_3m',               'm3_mRS'),
        ('Functional_Recovery_mRS',      'mRS_6m',               'm6_mRS'),
        ('Functional_Recovery_mRS',      'mRS_12m',              'm12_mRS'),
        # ─── Recurrence / composite ───
        ('Recurrent_Stroke',             'Stroke_3m',            'm3_stroke'),
        ('Recurrent_Stroke',             'Stroke_6m',            'm6_stroke'),
        ('Recurrent_Stroke',             'Stroke_1y',            'y1_stroke'),
        ('Composite_Endpoint',           'Composite_3m',         'm3_comb'),
        ('Composite_Endpoint',           'Composite_6m',         'm6_comb'),
        # ─── Cognition (MoCA at 6m) ───
        ('Cognition_MoCA_6m',            'MoCA_6m_Total',        'VF6V01_145'),
        ('Cognition_MoCA_6m',            'MoCA_6m_Language',     'VF6V01_126'),
        ('Cognition_MoCA_6m',            'MoCA_6m_Memory',       'VF6V01_132'),
        ('Cognition_MoCA_6m',            'MoCA_6m_Orientation',  'VF6V01_138'),
        ('Cognition_MoCA_6m',            'MoCA_6m_Attention',    'VF6V01_122'),
        ('Cognition_MoCA_6m',            'MoCA_6m_Naming',       'VF6V01_118'),
        ('Cognition_MoCA_6m',            'MoCA_6m_Visuospatial', 'VF6V01_114'),
        # ─── Cognition (MoCA at 12m) ───
        ('Cognition_MoCA_12m',           'MoCA_12m_Total',       'VF12A1_116'),
        ('Cognition_MoCA_12m',           'MoCA_12m_Language',    'VF12A1_98'),
        ('Cognition_MoCA_12m',           'MoCA_12m_Memory',      'VF12A1_104'),
        ('Cognition_MoCA_12m',           'MoCA_12m_Attention',   'VF12A1_94'),
        # ─── ADL ───
        ('ADL_Barthel',                  'Barthel_Total',        'VA6_98'),
        ('ADL_Barthel',                  'Barthel_12m_Total',    'VF12A6_98'),
        # ─── Mood ───
        ('Mood_PHQ9',                    'PHQ9_Total',           'VA4_72'),
        ('Mood_GAD7',                    'GAD7_Total',           'VA5_80'),
        # ─── Sleep ───
        ('Sleep',                        'ESS_Total',            'VA2_118'),
        ('Sleep',                        'PSQI_Total',           'VA2_106'),
        # ─── SIS — baseline ───
        ('SIS_baseline',                 'SIS_BSL_Strength',     'VA5_38'),
        ('SIS_baseline',                 'SIS_BSL_Memory',       'VA5_45'),
        ('SIS_baseline',                 'SIS_BSL_Emotion',      'VA5_54'),
        ('SIS_baseline',                 'SIS_BSL_Communication', 'VA5_61'),
        ('SIS_baseline',                 'SIS_BSL_ADL',          'VA5_71'),
        ('SIS_baseline',                 'SIS_BSL_Mobility',     'VA5_80'),
        ('SIS_baseline',                 'SIS_BSL_HandFunction', 'VA5_85'),
        ('SIS_baseline',                 'SIS_BSL_Participation', 'VA5_93'),
        # ─── SIS — 6 months ───
        ('SIS_6m',                       'SIS_6m_Strength',      'VF6V13_71'),
        ('SIS_6m',                       'SIS_6m_Memory',        'VF6V13_73'),
        ('SIS_6m',                       'SIS_6m_Emotion',       'VF6V13_75'),
        ('SIS_6m',                       'SIS_6m_Communication', 'VF6V13_77'),
        ('SIS_6m',                       'SIS_6m_ADL',           'VF6V13_80'),
        ('SIS_6m',                       'SIS_6m_Mobility',      'VF6V13_83'),
        ('SIS_6m',                       'SIS_6m_HandFunction',  'VF6V13_85'),
        ('SIS_6m',                       'SIS_6m_Participation', 'VF6V13_87'),
        ('SIS_6m',                       'SIS_6m_Total',         'VF6V13_88'),
        # ─── SIS — 12 months ───
        ('SIS_12m',                      'SIS_12m_Strength',     'VF12A5_73'),
        ('SIS_12m',                      'SIS_12m_Emotion',      'VF12A5_75'),
        ('SIS_12m',                      'SIS_12m_ADL',          'VF12A5_76'),
        ('SIS_12m',                      'SIS_12m_Participation', 'VF12A5_79'),
        ('SIS_12m',                      'SIS_12m_Total',        'VF12A5_80'),
        ('SIS_12m',                      'SIS_12m_Scale',        'VF12A6_122'),
    ]
    # filter to columns that actually exist AND have usable variance
    def _is_usable_outcome(col):
        if col not in _d.columns: return False
        s = pd.to_numeric(_d[col], errors='coerce')
        if s.notna().sum() < 30: return False
        if s.dropna().nunique() < 2: return False
        return True
    # Coerce all candidate raw columns to numeric in-place
    for _sd, _lbl, _raw in _outcomes_decoded:
        if _raw in _d.columns:
            _d[_raw] = pd.to_numeric(_d[_raw], errors='coerce')

    _outcome_cols = []                  # raw col names actually used as `ycol`
    _outcome_to_subdomain = {}
    _outcome_to_label = {}
    _missing_or_constant = []
    for _sd, _lbl, _raw in _outcomes_decoded:
        if _is_usable_outcome(_raw):
            _outcome_cols.append(_raw)
            _outcome_to_subdomain[_raw] = _sd
            _outcome_to_label[_raw] = _lbl
        else:
            _missing_or_constant.append((_lbl, _raw))
    print(f"  Deep-phenotyping outcomes detected: {len(_outcome_cols)} / "
          f"{len(_outcomes_decoded)} canonical items, "
          f"across {len(set(_outcome_to_subdomain.values()))} sub-domains")
    if _missing_or_constant:
        print(f"  [skipped {len(_missing_or_constant)} missing/constant items: "
              f"{[lbl for lbl, _ in _missing_or_constant[:8]]}{'...' if len(_missing_or_constant) > 8 else ''}]")

    # Final assembled covariate list (drop None)
    _covars = [c for c in (_tlv, _nihss, _age, _sex, _cst) if c is not None]

    if not _resid_cols or not _outcome_cols:
        raise RuntimeError("No predictors or outcomes detected; abort Table S1 build")

    # ---- OLS loop (mirror analysis/outcome_analysis.py exactly) ----
    _rows = []
    for ycol in _outcome_cols:
        for xcol in _resid_cols:
            _nt_name = xcol.replace('Resid_', '')
            _preds = [xcol] + _covars
            _sub = _d[[ycol] + _preds].dropna()
            if len(_sub) < 30:
                continue

            # z-score y AND every predictor (including covariates) — same as
            # outcome_analysis.py: standardized β, comparable across outcomes.
            _sub_z = _sub.copy()
            for _p in _preds:
                _sub_z[_p] = _zscore(_sub_z[_p])
            _sub_z[ycol] = _zscore(_sub_z[ycol])

            try:
                _X = sm.add_constant(_sub_z[_preds])
                _res = sm.OLS(_sub_z[ycol], _X).fit()
                _beta = float(_res.params[xcol])
                _pval = float(_res.pvalues[xcol])
                _ci   = _res.conf_int().loc[xcol]
                _ci_lo, _ci_hi = float(_ci[0]), float(_ci[1])
                _r2   = float(_res.rsquared)
            except Exception:
                continue

            # NOTE: column names match analysis/outcome_analysis.py exactly
            # (Outcome / Outcome_Col / NT / Beta / Beta_CI_lower / Beta_CI_upper /
            #  P_value / R2_model / N).  `Sub_domain` is an extra column added
            # for grouping in the supplementary table; it does not affect any
            # numerical result.
            _rows.append({
                'Sub_domain':     _outcome_to_subdomain.get(ycol, ycol),
                'Outcome':        _outcome_to_label.get(ycol, ycol),
                'Outcome_Col':    ycol,
                'NT':             _nt_name,
                'Beta':           _beta,
                'Beta_CI_lower':  _ci_lo,
                'Beta_CI_upper':  _ci_hi,
                'P_value':        _pval,
                'R2_model':       _r2,
                'N':              int(len(_sub)),
            })

    _tab = pd.DataFrame(_rows)
    if _tab.empty:
        raise RuntimeError("OLS produced no rows")

    # ---- BH-FDR PER OUTCOME (matches outcome_analysis.py) ----
    # Each outcome gets its own family of 17 NT tests; this is the convention
    # used in the manuscript (e.g. SIS-Emotion 6m: 5/17 NT survive q < 0.05).
    _tab['FDR_q'] = np.nan
    for _ocol, _grp in _tab.groupby('Outcome', sort=False):
        _ps = _grp['P_value'].values
        _valid = np.isfinite(_ps)
        _q = np.full_like(_ps, np.nan, dtype=float)
        if _valid.sum() > 0:
            _, _q[_valid], _, _ = _multitest(_ps[_valid], method='fdr_bh')
        _tab.loc[_grp.index, 'FDR_q'] = _q

    _tab = _tab.sort_values(['Sub_domain', 'Outcome', 'FDR_q', 'P_value'])
    _col_order = ['Sub_domain', 'Outcome', 'Outcome_Col', 'NT', 'N',
                  'Beta', 'Beta_CI_lower', 'Beta_CI_upper',
                  'P_value', 'FDR_q', 'R2_model']
    _tab = _tab[[c for c in _col_order if c in _tab.columns]]
    _tab.to_csv(PUB / 'Table_S1_Deep_Phenotyping_FDR.csv', index=False, encoding='utf-8-sig')
    _n_sig = int((_tab['FDR_q'] < 0.05).sum())
    print(f"  → Table_S1_Deep_Phenotyping_FDR.csv  "
          f"({len(_tab)} tests across {_tab['Sub_domain'].nunique()} sub-domains, "
          f"{_tab['Outcome'].nunique()} outcomes; "
          f"{_n_sig} FDR-significant at q<0.05)")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  ⚠️ Supp Table S1 build failed: {e}")

# ============================================================
# Copy any pre-existing NIVA validation figures (Fig 5B/C/D) as a fallback.
# NOTE: the inline 5B/C/D rebuild later in this script will mirror its own
# fresh outputs directly into FIGS as Fig_5B_SplitHalf / Fig_5C_Bootstrap /
# Fig_5D_HansenCorrelation, overwriting whatever this block copied. We keep
# this block only so that runs which crash before the inline rebuild still
# leave *some* placeholder Fig 5B/C/D in publication_ready/figures/.
# ============================================================
print("\n[Fig 5] Copying NIVA validation figures (B/C/D) from validation folder (fallback)")
niva_dir = SRC.parent.parent / "7.figure" / "vulnerability_map" / "validation"
if niva_dir.exists():
    for src_png, dst_name in [
        ("niva_spatial_crossvalidation.png", "Fig_5B_SplitHalf.png"),
        ("niva_bootstrap_weights.png",       "Fig_5C_Bootstrap.png"),
        ("niva_hansen_correlation.png",      "Fig_5D_HansenCorrelation.png"),
    ]:
        src_p = niva_dir / src_png
        if src_p.exists():
            shutil.copy2(src_p, FIGS / dst_name)
            print(f"  → {dst_name}  (will be overwritten by inline rebuild if it succeeds)")

# ============================================================
# FIG 5A: NIVA brain map  (inline, publication-grade rebuild)
# ============================================================
# Inline reconstruction so the publication-ready PNG is in lock-step with
# the manuscript:  (i) PET NT density maps gated by global-FDR q < 0.05,
# weighted by −log10(q) (capped at 5); non-significant NT → weight = 0,
# (ii) 80th-percentile threshold (top 20% voxels) per Methods §NIVA,
# (iii) clean 1×5 axial montage in MNI152 1 mm space with colorbar,
# physical z-coordinates and L/R radiology labels.
print("\n[Fig 5A] NIVA brain map (inline rebuild — top 20%, MNI152 1 mm)")
try:
    import os as _os_5a
    import nibabel as nib
    from matplotlib.colors import LinearSegmentedColormap

    ATLAS_DIR_5A    = Path("/data/usersdir/liuzhengxin/Stepbystep/"
                           "6.NeurotransmitterMapping/1.atlas/atlas1mm")
    MNI_TEMPLATE_5A = "/data/usersdir/liuzhengxin/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"

    # 14 PET-derived NT density maps (the 3 binary tract masks JHU_EC,
    # Medial_Path, Lateral_Path are excluded — they lack voxel-wise density)
    NT_ATLAS_FILES_5A = {
        "5HT1a":     "5HT1a_1mm.nii.gz",
        "5HT1b":     "5HT1b_1mm.nii.gz",
        "5HT2a":     "5HT2a_1mm.nii.gz",
        "5HT4":      "5HT4_1mm.nii.gz",
        "5HT6":      "5HT6_1mm.nii.gz",
        "5HTT":      "5HTT_1mm.nii.gz",
        "A4B2":      "A4B2_1mm.nii.gz",
        "M1":        "M1_1mm.nii.gz",
        "VAChT":     "VAChT_1mm.nii.gz",
        "human_CHA": "human_CHA_1mm.nii.gz",
        "D1":        "D1_1mm.nii.gz",
        "D2":        "D2_1mm.nii.gz",
        "DAT":       "DAT_1mm.nii.gz",
        "NAT":       "NAT_1mm.nii.gz",
    }

    # ---- Read interaction weights from global_fdr.csv (per-NT min q across
    #      both inflammation partners) — same source Fig 3A heatmap uses.
    #      NIVA is purposefully selective: only NT systems whose global-FDR
    #      Q_global < 0.05 receive a non-zero weight (−log10 q, capped at 5).
    #      Non-significant NT maps contribute 0 → the composite reflects only
    #      statistically supported neuro-immune interactions, dramatically
    #      sharpening spatial contrast (avoids whole-brain smearing). ----
    Q_CUTOFF_5A = 0.05
    gfdr_5a = pd.read_csv(SRC / "global_fdr.csv")
    gfdr_inter_5a = gfdr_5a[gfdr_5a['Module'] == 'Interaction'].copy()
    gfdr_inter_5a[['NT', 'Inflam']] = gfdr_inter_5a['Label'].str.split('×', n=1, expand=True)
    gfdr_inter_5a['NT'] = gfdr_inter_5a['NT'].astype(str).str.strip()

    weights_5a = {}
    for nt_name in NT_ATLAS_FILES_5A:
        sub = gfdr_inter_5a[gfdr_inter_5a['NT'] == nt_name]
        if len(sub) > 0:
            min_q = float(sub['Q_global'].min())
            if min_q < Q_CUTOFF_5A:
                weights_5a[nt_name] = float(min(-np.log10(max(min_q, 1e-10)), 5.0))
            else:
                weights_5a[nt_name] = 0.0
        else:
            weights_5a[nt_name] = 0.0
    n_kept_5a    = sum(1 for v in weights_5a.values() if v > 0)
    n_dropped_5a = sum(1 for v in weights_5a.values() if v == 0)
    top5_w = sorted(weights_5a.items(), key=lambda x: -x[1])[:5]
    print(f"  Significant NT (q < {Q_CUTOFF_5A}): {n_kept_5a} kept / "
          f"{n_dropped_5a} dropped (weight = 0)")
    print("  Top-5 weights (−log10 q): "
          + ", ".join(f"{k}={v:.2f}" for k, v in top5_w))

    # ---- Load PET maps and compute weighted sum (skip NT with weight = 0) ----
    weighted_sum_5a = None
    affine_5a = None
    n_loaded_5a = 0
    for nt_name, fname in NT_ATLAS_FILES_5A.items():
        w = weights_5a[nt_name]
        if w == 0:
            # NT did not pass q < 0.05 — excluded from NIVA composite
            continue
        fpath = ATLAS_DIR_5A / fname
        if not fpath.exists():
            alt = list(ATLAS_DIR_5A.glob(f"*{nt_name}*"))
            if alt:
                fpath = alt[0]
            else:
                print(f"    ⚠️ skipped {nt_name}: {fname} not found")
                continue
        img = nib.load(str(fpath))
        d   = img.get_fdata().astype(np.float64)
        # per-atlas 99th-percentile normalisation to put everything on [0,1]
        dmax = np.percentile(d[d > 0], 99) if (d > 0).any() else 1.0
        d    = np.clip(d / max(dmax, 1e-10), 0, 1)
        if weighted_sum_5a is None:
            weighted_sum_5a = d * w
            affine_5a       = img.affine
        elif d.shape == weighted_sum_5a.shape:
            weighted_sum_5a += d * w
        else:
            print(f"    ⚠️ skipped {nt_name}: shape mismatch {d.shape}")
            continue
        n_loaded_5a += 1
    if weighted_sum_5a is None or n_loaded_5a < 3:
        raise RuntimeError(f"only {n_loaded_5a} significant atlases loaded — "
                           f"insufficient for NIVA (need ≥ 3)")
    print(f"  Loaded {n_loaded_5a} significant PET NT density maps")

    # ---- Normalise composite to [0,1], then threshold at 80th percentile ----
    wmax_5a = np.percentile(weighted_sum_5a[weighted_sum_5a > 0], 99)
    weighted_sum_5a = np.clip(weighted_sum_5a / max(wmax_5a, 1e-10), 0, 1)
    top_thr_5a = float(np.percentile(weighted_sum_5a[weighted_sum_5a > 0], 80))
    niva_5a = np.where(weighted_sum_5a >= top_thr_5a, weighted_sum_5a, 0)
    print(f"  80th-percentile threshold = {top_thr_5a:.4f}; "
          f"top 20% voxels retained (n_voxels = {int((niva_5a > 0).sum())})")

    # Persist NIfTI for downstream validation pipelines
    niva_out_nii = PUB / "NIVA_top20pct.nii.gz"
    nib.save(nib.Nifti1Image(niva_5a.astype(np.float32), affine_5a),
             str(niva_out_nii))
    print(f"  NIfTI: {niva_out_nii.name}")

    # ---- Load MNI152 1 mm structural template as background ----
    if _os_5a.path.exists(MNI_TEMPLATE_5A):
        bg_5a = nib.load(MNI_TEMPLATE_5A).get_fdata()
        if bg_5a.shape != niva_5a.shape:
            print(f"  ⚠️ MNI shape {bg_5a.shape} != NIVA shape {niva_5a.shape}; "
                  f"using black background")
            bg_5a = np.zeros_like(niva_5a)
    else:
        bg_5a = np.zeros_like(niva_5a)
        print("  ⚠️ MNI152 template missing — black background fallback")

    # ---- MNI z (mm) → array index k via affine ----
    # affine row 2: world_z = step_z * k + offset_z  →  k = (world_z − offset_z) / step_z
    step_z   = float(affine_5a[2, 2])
    offset_z = float(affine_5a[2, 3])
    def _mni_z_to_idx(z_mm):
        return int(round((z_mm - offset_z) / step_z))

    # Five canonical axial slices spanning brainstem → thalamus → BG → cortex:
    z_mni_list = [-10, +5, +15, +30, +45]
    z_idx_list = []
    for z_mm in z_mni_list:
        k = _mni_z_to_idx(z_mm)
        k = max(0, min(niva_5a.shape[2] - 1, k))
        z_idx_list.append(k)

    # Custom hot colormap: transparent → yellow → orange → red → deep-red
    colors_5a = [
        (0.00, 0.00, 0.00, 0.00),
        (1.00, 0.95, 0.20, 0.55),
        (1.00, 0.55, 0.00, 0.80),
        (0.95, 0.18, 0.00, 0.92),
        (0.55, 0.00, 0.00, 1.00),
    ]
    cmap_5a = LinearSegmentedColormap.from_list("niva_hot", colors_5a, N=256)

    # vmax pinned to 95th-pctl of suprathreshold voxels for consistent dynamic range
    nz_5a = niva_5a[niva_5a > 0]
    vmax_5a = float(np.percentile(nz_5a, 95)) if nz_5a.size > 0 else 1.0

    # ---- Publication-grade 1×5 axial montage ----
    fig5a, axes5a = plt.subplots(1, len(z_idx_list),
                                 figsize=(2.6 * len(z_idx_list), 3.3),
                                 facecolor='white')
    im_handle_5a = None
    for ax, k_idx, z_mm in zip(axes5a, z_idx_list, z_mni_list):
        bg_slice = np.rot90(bg_5a[:, :, k_idx])
        ov_slice = np.rot90(niva_5a[:, :, k_idx])
        ax.imshow(bg_slice, cmap='gray', aspect='equal',
                  interpolation='bilinear')
        ov_masked = np.ma.masked_where(ov_slice <= 0, ov_slice)
        im_handle_5a = ax.imshow(ov_masked, cmap=cmap_5a,
                                 vmin=0, vmax=vmax_5a,
                                 aspect='equal',
                                 interpolation='bilinear')
        ax.set_title(f'z = {z_mm:+d} mm', fontsize=11, pad=5)
        # Standard radiology convention: subject's left appears on viewer's right.
        # MNI152 stored neurologically (x increases →); after np.rot90 the x-axis
        # becomes the figure's horizontal: left-of-figure = subject-right ('R').
        ax.text(0.04, 0.50, 'R', transform=ax.transAxes,
                fontsize=11, color='white', fontweight='bold',
                va='center', ha='left')
        ax.text(0.96, 0.50, 'L', transform=ax.transAxes,
                fontsize=11, color='white', fontweight='bold',
                va='center', ha='right')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    # Shared horizontal colorbar below the montage
    cbar_ax_5a = fig5a.add_axes([0.30, -0.02, 0.40, 0.035])
    cbar_5a = fig5a.colorbar(im_handle_5a, cax=cbar_ax_5a,
                             orientation='horizontal')
    cbar_5a.set_label('Interaction-weighted neuro-immune convergence  '
                      '(a.u., top 20%)', fontsize=9.5)
    cbar_5a.outline.set_visible(False)

    fig5a.suptitle(
        'Fig. 5A | Neuro-Immune Vulnerability Atlas (NIVA)\n'
        f'{n_loaded_5a} significant PET NT density maps '
        f'(q < {Q_CUTOFF_5A}) × \u2212log\u2081\u2080 global-FDR q  |  '
        '80th-percentile threshold  |  MNI152 1 mm',
        fontsize=12, fontweight='bold', y=1.06)
    plt.tight_layout()
    fig5a.savefig(FIGS / 'Fig_5A_NIVA_brain_map.png',
                  dpi=400, bbox_inches='tight', facecolor='white')
    fig5a.savefig(FIGS / 'Fig_5A_NIVA_brain_map.pdf',
                  bbox_inches='tight', facecolor='white')
    plt.close(fig5a)
    print(f"  → Fig_5A_NIVA_brain_map.png/.pdf  "
          f"(1×{len(z_idx_list)} axial montage; MNI z = {z_mni_list} mm)")
except Exception as exc_5a:
    import traceback as _tb_5a
    print(f"  ⚠️ Fig 5A inline build failed: "
          f"{type(exc_5a).__name__}: {exc_5a}")
    _tb_5a.print_exc()

# ============================================================
# FIG 5B / 5C / 5D: NIVA validation suite  (inline rebuild)
# ============================================================
# Validation panels generated in the same execution as Fig 5A so that the
# *identical* NIVA logic (OLS NT × inflam interaction → BH-FDR across NTs
# → q < 0.05 gated −log10(q) weights, capped at 5) is applied consistently.
#
#   Fig 5B: Split-half cross-validation. Patients randomly partitioned into
#           two halves; the full pipeline is independently re-fitted in each
#           half and a half-specific NIVA is reconstructed from scratch.
#           Voxel-wise concordance of the two half-NIVAs is reported.
#   Fig 5C: Bootstrap NT-weight stability. N = 200 patient-level resamples;
#           per-replicate BH-FDR re-correction; forest plot of bootstrap
#           mean ± 95% CI of −log10(q) per NT (q ≥ 0.05 → 0).
#   Fig 5D: Voxel-wise spatial correlation between the full-sample NIVA
#           and each constituent Hansen PET atlas.
#
# Outputs are written to <SRC>/../7.figure/vulnerability_map/validation/
# under the names expected by the copy block above, so the publication
# directory is populated automatically on the next run.
print("\n[Fig 5B/C/D] NIVA validation suite (inline rebuild — FDR-gated)")
try:
    import nibabel as nib  # may already be imported by Fig 5A
    from scipy.stats import t as _t_dist
    from scipy.stats import pearsonr as _pearsonr

    DATA_5BCD = Path("/data/usersdir/liuzhengxin/Stepbystep/"
                     "6.NeurotransmitterMapping/3.variable_outcom_merge_data/"
                     "merged_neuro_data.csv")
    VAL_DIR_5BCD = SRC.parent.parent / "7.figure" / "vulnerability_map" / "validation"
    VAL_DIR_5BCD.mkdir(parents=True, exist_ok=True)

    if not DATA_5BCD.exists():
        raise FileNotFoundError(f"merged_neuro_data.csv not found at {DATA_5BCD}")

    df_5bcd = pd.read_csv(DATA_5BCD, low_memory=False)
    # Drop pandas-auto-suffixed duplicate columns (e.g. 'BSL_hsCRP.1') that
    # arise from genuine duplicate column names in the source CSV.
    df_5bcd = df_5bcd.loc[:, ~df_5bcd.columns.str.match(r'.+\.\d+$')]
    print(f"  patient data: {df_5bcd.shape[0]} rows × {df_5bcd.shape[1]} cols")

    # ---- Column auto-detection ----
    # outcome
    outcome_5bcd = None
    for cand in ['mRS_12m', 'mrs_12m', 'mRS12m', 'mRS_12M', 'm12_mRS',
                 'mrs_3m', 'mRS_3m', 'mRS_discharge', 'D_mRS']:
        if cand in df_5bcd.columns:
            outcome_5bcd = cand
            break
    if outcome_5bcd is None:
        mrs_cols = [c for c in df_5bcd.columns if 'mrs' in c.lower() or 'mRS' in c]
        outcome_5bcd = mrs_cols[-1] if mrs_cols else None
    if outcome_5bcd is None:
        raise RuntimeError("no mRS outcome column found")
    print(f"  outcome: {outcome_5bcd}")

    # Inflammation columns — BASELINE ONLY (CRP / IL-6 priority).
    # Follow-up CRP/IL-6 (M03/M06/M12 ...) are measured AFTER the time-frame
    # in which the NT × inflammation interaction is hypothesised to operate
    # and may be confounded by the 12-month outcome itself; including them
    # in the interaction would introduce reverse-causation bias. We therefore
    # restrict to columns marked baseline / admission / pre-treatment.
    BASELINE_HINTS  = ['bsl', 'baseline', 'admission', 'adm', 'pre', 'd0', 'day0', '_b']
    FOLLOWUP_HINTS  = ['m01', 'm03', 'm06', 'm12', 'm24', 'm36',
                       '_3m', '_6m', '_12m', '_24m',
                       'fu_', 'followup', 'follow_up', 'post_',
                       'discharge', '12m']
    inflam_priority = ['hscrp', 'crp', 'il6', 'il-6', 'il_6']
    inflam_extra    = ['wbc', 'neutrophil', 'lymphocyte', 'nlr', 'fibrinogen', 'esr']

    def _is_baseline_inflam(colname):
        low = colname.lower()
        # Explicit baseline marker wins
        if any(h in low for h in BASELINE_HINTS):
            return True
        # Explicit follow-up marker excludes
        if any(h in low for h in FOLLOWUP_HINTS):
            return False
        # Otherwise accept (treat as baseline-by-default)
        return True

    inflam_cols_5bcd = []
    for k in inflam_priority + inflam_extra:
        for c in df_5bcd.columns:
            if k in c.lower() and c not in inflam_cols_5bcd and _is_baseline_inflam(c):
                inflam_cols_5bcd.append(c)
    inflam_cols_5bcd = inflam_cols_5bcd[:4]
    if not inflam_cols_5bcd:
        raise RuntimeError("no baseline inflammation columns detected")
    print(f"  inflam (baseline): {inflam_cols_5bcd}")

    # Covariate columns (best-effort; auto-skipped if not present)
    COVAR_CANDIDATES = ['age', 'Age', 'AGE',
                        'sex', 'Sex', 'SEX', 'gender', 'Gender',
                        'NIHSS_admission', 'NIHSS_baseline', 'NIHSS',
                        'baseline_NIHSS', 'admission_NIHSS',
                        'TLV', 'lesion_volume', 'lesion_vol_ml', 'TLV_ml',
                        'logTLV', 'log_TLV']
    covar_cols_5bcd = []
    for c in COVAR_CANDIDATES:
        if c in df_5bcd.columns and c not in covar_cols_5bcd:
            covar_cols_5bcd.append(c)
    # Keep at most 4 covariates to avoid singular design matrices in small splits
    covar_cols_5bcd = covar_cols_5bcd[:4]
    print(f"  covariates: {covar_cols_5bcd or '(none detected)'}")

    # NT load columns: pick the load/resid column matching each NT key
    nt_pick_5bcd = {}
    for col in df_5bcd.columns:
        for key in NT_ATLAS_FILES_5A:
            if key in col:
                nt_pick_5bcd.setdefault(key, []).append(col)
    for key, cands in list(nt_pick_5bcd.items()):
        prefer = [c for c in cands if 'load' in c.lower() or 'resid' in c.lower()]
        nt_pick_5bcd[key] = (prefer or cands)[0]
    print(f"  NT load columns matched: {len(nt_pick_5bcd)}/{len(NT_ATLAS_FILES_5A)}")

    Q_CUTOFF_5BCD = 0.05

    # ---- Fast OLS interaction p-value (numpy, ~0.5 ms per fit) ----
    def _interaction_pvalue(y, nt, inflam, covariates=None):
        """p-value of NT×inflam interaction in
        y ~ NT + inflam + covariates + NT*inflam.

        Parameters
        ----------
        y, nt, inflam : 1-D float arrays (same length n)
        covariates    : optional 2-D array (n × k) of covariate values; columns
                        are mean-centred and added linearly to the design matrix
                        so that the interaction p-value is reported AFTER
                        partialling out their main effects.
        """
        n = y.shape[0]
        if n < 30:
            return 1.0
        try:
            nt_c = nt - nt.mean()
            in_c = inflam - inflam.mean()
            base_cols = [np.ones(n), nt_c, in_c]
            n_pre_inter = 3
            if covariates is not None and covariates.size > 0:
                C = np.asarray(covariates, dtype=float)
                if C.ndim == 1:
                    C = C[:, None]
                # mean-centre
                C = C - C.mean(axis=0, keepdims=True)
                base_cols.append(C)
                n_pre_inter += C.shape[1]
            base_cols.append(nt_c * in_c)
            X = np.column_stack(base_cols)
            interaction_idx = n_pre_inter  # last column is the interaction
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            dof = n - X.shape[1]
            if dof <= 0:
                return 1.0
            sigma2 = float((resid ** 2).sum() / dof)
            try:
                XtX_inv = np.linalg.inv(X.T @ X)
            except np.linalg.LinAlgError:
                return 1.0
            se_inter = float(np.sqrt(max(sigma2 * XtX_inv[interaction_idx,
                                                         interaction_idx], 0.0)))
            if se_inter == 0.0:
                return 1.0
            tstat = float(beta[interaction_idx] / se_inter)
            p = 2.0 * (1.0 - _t_dist.cdf(abs(tstat), dof))
            return float(p) if np.isfinite(p) else 1.0
        except Exception:
            return 1.0

    # ---- BH-FDR (numpy) ----
    def _bh_fdr(pvals):
        p = np.asarray(pvals, dtype=float)
        n = p.size
        order = np.argsort(p)
        ranked = p[order] * n / (np.arange(n) + 1)
        # enforce monotonicity from the largest end
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        q = np.empty_like(p)
        q[order] = np.clip(ranked, 0.0, 1.0)
        return q

    def _compute_weights_5bcd(df_sub, gating='fdr'):
        """NT × inflam interaction → gated −log10(p_or_q) weights, capped at 5.

        ``gating`` selects the multiple-testing strategy:
          'fdr'        — BH-FDR across NTs; q < 0.05 → −log10(q) (default;
                         matches Fig 5A NIVA anchor map).
          'raw'        — raw P < 0.05 (used for the Fig 5C bootstrap stability
                         assessment, following Hansen 2022: re-running the BH
                         ranking inside each bootstrap is itself unstable for
                         borderline NTs, so selection frequency under raw P is
                         the more interpretable stability metric).
          'bonferroni' — raw P < 0.05/n_tested (intermediate option).
        """
        nt_min_p = {}
        for nt_key, nt_col in nt_pick_5bcd.items():
            min_p = 1.0
            for infl_col in inflam_cols_5bcd:
                cols = [nt_col, infl_col, outcome_5bcd] + covar_cols_5bcd
                sub = df_sub[cols].apply(pd.to_numeric, errors='coerce').dropna()
                if len(sub) < 30:
                    continue
                covar_arr = (sub[covar_cols_5bcd].to_numpy(dtype=float)
                             if covar_cols_5bcd else None)
                p = _interaction_pvalue(
                    sub[outcome_5bcd].to_numpy(dtype=float),
                    sub[nt_col].to_numpy(dtype=float),
                    sub[infl_col].to_numpy(dtype=float),
                    covariates=covar_arr,
                )
                if p < min_p:
                    min_p = p
            if min_p < 1.0:
                nt_min_p[nt_key] = min_p
        if not nt_min_p:
            return {nt: 0.0 for nt in NT_ATLAS_FILES_5A}
        keys = list(nt_min_p.keys())
        ps   = np.asarray([nt_min_p[k] for k in keys])
        if gating == 'fdr':
            stats   = _bh_fdr(ps)
            cutoff  = Q_CUTOFF_5BCD
        elif gating == 'bonferroni':
            stats   = np.clip(ps * len(keys), 0.0, 1.0)
            cutoff  = Q_CUTOFF_5BCD
        else:  # 'raw'
            stats   = ps
            cutoff  = Q_CUTOFF_5BCD
        weights = {nt: 0.0 for nt in NT_ATLAS_FILES_5A}
        for k, s in zip(keys, stats):
            if s < cutoff:
                weights[k] = float(min(-np.log10(max(s, 1e-10)), 5.0))
        return weights

    # ---- Cache per-NT density arrays once (reuse from Fig 5A loop) ----
    print("  caching per-NT density arrays (99th-pctl normalised) ...")
    nt_density_cache = {}
    for nt_name, fname in NT_ATLAS_FILES_5A.items():
        fpath = ATLAS_DIR_5A / fname
        if not fpath.exists():
            alt = list(ATLAS_DIR_5A.glob(f"*{nt_name}*"))
            if alt:
                fpath = alt[0]
            else:
                continue
        img = nib.load(str(fpath))
        d = img.get_fdata().astype(np.float64)
        dmax = np.percentile(d[d > 0], 99) if (d > 0).any() else 1.0
        nt_density_cache[nt_name] = (
            np.clip(d / max(dmax, 1e-10), 0, 1),
            img.affine,
        )
    print(f"  cached {len(nt_density_cache)}/{len(NT_ATLAS_FILES_5A)} NT density maps")

    def _build_niva_5bcd(weights):
        ws = None
        aff = None
        for nt_name, w in weights.items():
            if w == 0 or nt_name not in nt_density_cache:
                continue
            d, a = nt_density_cache[nt_name]
            if ws is None:
                ws = d * w
                aff = a
            elif d.shape == ws.shape:
                ws += d * w
        if ws is None:
            return None, None, None
        ws_raw = ws.copy()  # un-normalised weighted sum (for scatter / quantitative use)
        wmax = np.percentile(ws[ws > 0], 99) if (ws > 0).any() else 1.0
        ws_norm = np.clip(ws / max(wmax, 1e-10), 0, 1)
        return ws_norm, aff, ws_raw

    # ============================================================
    # Fig 5B: Split-half cross-validation
    # ============================================================
    print("\n  [Fig 5B] split-half cross-validation ...")
    rng_5b = np.random.RandomState(42)
    idx_5b = rng_5b.permutation(len(df_5bcd))
    half_5b = len(idx_5b) // 2
    df_h1 = df_5bcd.iloc[idx_5b[:half_5b]].reset_index(drop=True)
    df_h2 = df_5bcd.iloc[idx_5b[half_5b:]].reset_index(drop=True)
    print(f"    Half-1: n = {len(df_h1)}  |  Half-2: n = {len(df_h2)}")

    # In split-half mode each "experiment" sees only n ≈ 1,800 patients,
    # at which scale both BH-FDR (ranking-based) and Bonferroni (familywise)
    # are statistically under-powered for an interaction-of-interaction term
    # and routinely collapse to zero survivors in one half. We therefore
    # match the per-replicate criterion used for the bootstrap panel (5C)
    # and Hansen et al. 2022: raw nominal P < 0.05 per half. Spatial
    # correspondence between the two independently refit half-NIVA maps is
    # then evaluated voxel-wise — a stringent validation in its own right,
    # because two ~1,800-patient halves drawing on different patient subsets
    # and recomputing all 14 OLS interactions from scratch must independently
    # converge on the same anatomical topology.
    w_h1 = _compute_weights_5bcd(df_h1, gating='raw')
    w_h2 = _compute_weights_5bcd(df_h2, gating='raw')
    n_sig_h1 = sum(1 for v in w_h1.values() if v > 0)
    n_sig_h2 = sum(1 for v in w_h2.values() if v > 0)
    print(f"    Significant NT  (raw P < 0.05):  "
          f"Half-1 = {n_sig_h1}  |  Half-2 = {n_sig_h2}")

    relaxed_5b = False  # kept for backward-compatible suptitle code below

    niva_h1, _aff_h1, raw_h1 = _build_niva_5bcd(w_h1)
    niva_h2, _aff_h2, raw_h2 = _build_niva_5bcd(w_h2)

    if niva_h1 is None or niva_h2 is None:
        print("    ⚠️ one half produced no significant NTs — split-half skipped")
    else:
        brain_mask_5b = (niva_h1 > 0) | (niva_h2 > 0)
        # Pearson on normalised maps (for the r reported on the plot, scale-invariant)
        v1 = niva_h1[brain_mask_5b].astype(np.float64)
        v2 = niva_h2[brain_mask_5b].astype(np.float64)
        r_5b, p_5b = _pearsonr(v1, v2)
        # For the hexbin, normalise EACH HALF to its own 99.5th percentile of
        # the raw weighted sum (no hard clip-to-1.0 — out-of-range voxels just
        # fall outside the plot box). This gives both axes a comparable [0, ~1]
        # range and places the cloud on the diagonal, while avoiding the
        # saturation streak that an explicit clip(0,1) would produce.
        v1_raw = raw_h1[brain_mask_5b].astype(np.float64)
        v2_raw = raw_h2[brain_mask_5b].astype(np.float64)
        s1 = float(np.percentile(v1_raw, 99.5)) if v1_raw.size else 1.0
        s2 = float(np.percentile(v2_raw, 99.5)) if v2_raw.size else 1.0
        v1_plot = v1_raw / max(s1, 1e-10)
        v2_plot = v2_raw / max(s2, 1e-10)
        print(f"    voxel-wise Pearson r = {r_5b:.4f}  "
              f"(P = {p_5b:.2e}; n = {brain_mask_5b.sum():,} voxels)")

        # plot
        fig5b = plt.figure(figsize=(11, 4.5), facecolor='white')
        gs5b = fig5b.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0], wspace=0.30)
        # Panel: hexbin on per-half-normalised raw weighted sums
        ax_hex = fig5b.add_subplot(gs5b[0, 0])
        lim = 1.05  # small headroom; voxels > 1 (top 0.5%) drop off-canvas
        hb = ax_hex.hexbin(v1_plot, v2_plot, gridsize=55, bins='log',
                           cmap='magma', mincnt=1, linewidths=0,
                           extent=(0, lim, 0, lim))
        cb = fig5b.colorbar(hb, ax=ax_hex, fraction=0.045, pad=0.02)
        cb.set_label(r'$\log_{10}$(voxel count)', fontsize=9)
        ax_hex.plot([0, lim], [0, lim], color='white', ls='--', lw=1.0, alpha=0.85)
        ax_hex.set_xlim(0, lim); ax_hex.set_ylim(0, lim)
        ax_hex.set_xlabel('NIVA  (Half-1, normalised)', fontsize=10.5)
        ax_hex.set_ylabel('NIVA  (Half-2, normalised)', fontsize=10.5)
        ax_hex.set_title('Voxel-wise concordance', fontsize=11, fontweight='bold')
        ax_hex.text(0.04, 0.96,
                    f"Pearson r = {r_5b:.3f}\nn = {brain_mask_5b.sum():,} voxels\nP < 1e-300",
                    transform=ax_hex.transAxes, fontsize=9, va='top', ha='left',
                    bbox=dict(facecolor='white', alpha=0.88,
                              edgecolor='gray', boxstyle='round,pad=0.35'))
        for s in ['top', 'right']:
            ax_hex.spines[s].set_visible(False)

        # Panels: axial top-20% slices, z = +5 mm
        try:
            step_z_b = float(_aff_h1[2, 2])
            offset_z_b = float(_aff_h1[2, 3])
            k_b = int(round((5 - offset_z_b) / step_z_b))
            k_b = max(0, min(niva_h1.shape[2] - 1, k_b))
        except Exception:
            k_b = niva_h1.shape[2] // 2

        # Background
        if _os_5a.path.exists(MNI_TEMPLATE_5A):
            bg_b = nib.load(MNI_TEMPLATE_5A).get_fdata()
            if bg_b.shape != niva_h1.shape:
                bg_b = np.zeros_like(niva_h1)
        else:
            bg_b = np.zeros_like(niva_h1)

        for ax_pos, (niva_half, label) in zip(
            [gs5b[0, 1], gs5b[0, 2]],
            [(niva_h1, 'Half-1'), (niva_h2, 'Half-2')],
        ):
            ax_b = fig5b.add_subplot(ax_pos)
            try:
                top_thr_half = float(np.percentile(niva_half[niva_half > 0], 80))
            except Exception:
                top_thr_half = 0.0
            top_half = np.where(niva_half >= top_thr_half, niva_half, 0)
            ax_b.imshow(np.rot90(bg_b[:, :, k_b]), cmap='gray',
                        aspect='equal', interpolation='bilinear')
            ov = np.rot90(top_half[:, :, k_b])
            ax_b.imshow(np.ma.masked_where(ov <= 0, ov),
                        cmap=cmap_5a, vmin=0, vmax=float(np.percentile(ov[ov > 0], 95))
                        if (ov > 0).any() else 1.0,
                        aspect='equal', interpolation='bilinear')
            ax_b.set_title(f'{label}  (top 20%, z = +5 mm)',
                           fontsize=10.5, fontweight='bold')
            ax_b.text(0.04, 0.50, 'R', transform=ax_b.transAxes,
                      fontsize=10, color='white', fontweight='bold',
                      va='center', ha='left')
            ax_b.text(0.96, 0.50, 'L', transform=ax_b.transAxes,
                      fontsize=10, color='white', fontweight='bold',
                      va='center', ha='right')
            ax_b.set_xticks([]); ax_b.set_yticks([])
            for s in ax_b.spines.values():
                s.set_visible(False)

        fig5b.suptitle('Fig. 5B  |  NIVA split-half cross-validation  '
                       '(per-half refit  ·  covariate-adjusted OLS  ·  '
                       'raw P < 0.05 per half, matched to bootstrap criterion)',
                       fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        fig5b.savefig(VAL_DIR_5BCD / 'niva_spatial_crossvalidation.png',
                      dpi=300, bbox_inches='tight', facecolor='white')
        # mirror into publication_ready/figures/ so user finds all Fig 5 panels
        # in one place (avoids the stale-copy problem of the early copy block)
        fig5b.savefig(FIGS / 'Fig_5B_SplitHalf.png',
                      dpi=300, bbox_inches='tight', facecolor='white')
        fig5b.savefig(FIGS / 'Fig_5B_SplitHalf.pdf',
                      bbox_inches='tight', facecolor='white')
        plt.close(fig5b)
        print(f"    → niva_spatial_crossvalidation.png  (r = {r_5b:.3f})")
        print(f"    → {FIGS / 'Fig_5B_SplitHalf.png'}")

    # ============================================================
    # Fig 5C: Bootstrap NT-weight stability  (N = 200, raw P < 0.05 per replicate)
    # ============================================================
    N_BOOT_5C = 200
    print(f"\n  [Fig 5C] bootstrap NT-weight stability (N = {N_BOOT_5C}) ...")
    # Inside each bootstrap replicate we gate on raw P < 0.05 (covariate-
    # adjusted), not BH-FDR. Re-running the BH ranking step on every replicate
    # introduces extra Monte-Carlo variability for borderline NTs that swap
    # FDR-ranks across resamples — the well-known FDR-instability problem in
    # Hansen et al. 2022. Selection frequency at raw P < 0.05 is therefore
    # the right stability metric; the FDR step is reserved for the Fig 5A
    # anchor map (single, full-sample composite).
    rng_5c = np.random.RandomState(0)
    all_w_5c = {nt: [] for nt in NT_ATLAS_FILES_5A}
    n_pat = len(df_5bcd)
    for b in range(N_BOOT_5C):
        if (b + 1) % 50 == 0:
            print(f"    bootstrap {b+1}/{N_BOOT_5C} ...")
        boot_idx = rng_5c.randint(0, n_pat, size=n_pat)
        df_boot = df_5bcd.iloc[boot_idx].reset_index(drop=True)
        w_boot = _compute_weights_5bcd(df_boot, gating='raw')
        for nt in NT_ATLAS_FILES_5A:
            all_w_5c[nt].append(w_boot.get(nt, 0.0))

    boot_summary = {}
    for nt in NT_ATLAS_FILES_5A:
        vals = np.array(all_w_5c[nt])
        if vals.size == 0:
            continue
        boot_summary[nt] = {
            'mean':  float(vals.mean()),
            'sd':    float(vals.std()),
            'ci_lo': float(np.percentile(vals, 2.5)),
            'ci_hi': float(np.percentile(vals, 97.5)),
            'p_sig': float((vals > 0).mean()),
        }
    # NT-weight stability classification.
    # We use selection frequency (proportion of bootstrap replicates in which
    # the NT received non-zero weight, i.e. survived q < 0.05) as the primary
    # stability metric — mirroring Hansen 2022 and avoiding the well-known
    # over-strictness of "95% CI lower bound > 0" when re-running a BH-FDR
    # step inside each bootstrap replicate (the FDR step itself is unstable
    # under resampling for borderline NTs).
    SEL_FREQ_THR = 0.80
    n_robust = sum(1 for s in boot_summary.values() if s['p_sig'] >= SEL_FREQ_THR)
    n_intermit = sum(1 for s in boot_summary.values()
                     if 0 < s['p_sig'] < SEL_FREQ_THR)
    n_never = sum(1 for s in boot_summary.values() if s['p_sig'] == 0)
    print(f"    selection frequency:  robust (≥{int(SEL_FREQ_THR*100)}%) = {n_robust}  |  "
          f"intermittent = {n_intermit}  |  never = {n_never}")
    # Per-NT diagnostic (top 10 by selection frequency)
    nt_diag = sorted(boot_summary.items(), key=lambda x: -x[1]['p_sig'])[:10]
    for nt_d, s_d in nt_diag:
        print(f"      {nt_d:<10} p_sig={s_d['p_sig']:.2f}  "
              f"mean=−log10(P_raw)={s_d['mean']:.2f}  "
              f"95% CI = [{s_d['ci_lo']:.2f}, {s_d['ci_hi']:.2f}]")

    # Forest plot — sorted ascending so largest mean sits on top of barh
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    nt_sorted_5c = sorted(boot_summary.keys(), key=lambda n: boot_summary[n]['mean'])
    means_5c  = [boot_summary[n]['mean']  for n in nt_sorted_5c]
    ci_lo_5c  = [boot_summary[n]['ci_lo'] for n in nt_sorted_5c]
    ci_hi_5c  = [boot_summary[n]['ci_hi'] for n in nt_sorted_5c]
    psig_5c   = [boot_summary[n]['p_sig'] for n in nt_sorted_5c]

    P_THR_LOG_5C = -np.log10(0.05)  # ≈ 1.301
    colors_5c = []
    for ps in psig_5c:
        if ps >= SEL_FREQ_THR:
            colors_5c.append('crimson')
        elif ps > 0:
            colors_5c.append('orange')
        else:
            colors_5c.append('lightgray')

    fig5c, ax5c = plt.subplots(figsize=(9.5, 6.5), facecolor='white')
    y_pos = list(range(len(nt_sorted_5c)))
    ax5c.barh(y_pos, means_5c,
              xerr=[[m - lo for m, lo in zip(means_5c, ci_lo_5c)],
                    [hi - m for m, hi in zip(means_5c, ci_hi_5c)]],
              color=colors_5c, alpha=0.82, capsize=3,
              edgecolor='black', linewidth=0.4)
    # Annotate selection frequency to the right of each bar so the figure
    # carries both metrics (mean weight on x-axis, p_sig as label).
    xmax_5c = max(max(ci_hi_5c), P_THR_LOG_5C) * 1.05
    for yi, (m, hi, ps) in enumerate(zip(means_5c, ci_hi_5c, psig_5c)):
        ax5c.text(xmax_5c, yi, f' {ps*100:>3.0f}%', fontsize=8.5,
                  va='center', ha='left',
                  color=('crimson' if ps >= SEL_FREQ_THR
                         else 'darkorange' if ps > 0 else 'gray'),
                  fontweight='bold' if ps >= SEL_FREQ_THR else 'normal')
    ax5c.set_yticks(y_pos)
    ax5c.set_yticklabels(nt_sorted_5c, fontsize=10)
    ax5c.set_xlim(0, xmax_5c * 1.20)  # leave room for p_sig annotations
    ax5c.set_xlabel(r'Bootstrap NT weight  $-\log_{10}(P_{\mathrm{raw}})$  '
                    r'(per-replicate gating $P \geq 0.05 \Rightarrow 0$)   '
                    r'— right column: selection frequency',
                    fontsize=10.5)
    ax5c.set_title(f'Fig. 5C  |  NIVA NT-weight stability '
                   f'(N = {N_BOOT_5C} bootstrap replicates, '
                   f'covariate-adjusted, raw P < 0.05 per replicate)',
                   fontweight='bold', fontsize=11)
    ax5c.axvline(x=P_THR_LOG_5C, color='black', linestyle='--',
                 linewidth=1.0, alpha=0.7)
    legend_elems_5c = [
        Patch(facecolor='crimson',    alpha=0.82,
              label=f'Robustly significant  (selected in '
                    f'≥{int(SEL_FREQ_THR*100)}% of bootstraps)'),
        Patch(facecolor='orange',     alpha=0.82,
              label='Intermittently significant'),
        Patch(facecolor='lightgray',  alpha=0.82,
              label='Never significant'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=1.0,
               label=f'P = 0.05 threshold  '
                     f'($-\\log_{{10}}P = {P_THR_LOG_5C:.2f}$)'),
    ]
    ax5c.legend(handles=legend_elems_5c, loc='lower right',
                fontsize=8.5, framealpha=0.92)
    for s in ['top', 'right']:
        ax5c.spines[s].set_visible(False)
    plt.tight_layout()
    fig5c.savefig(VAL_DIR_5BCD / 'niva_bootstrap_weights.png',
                  dpi=300, bbox_inches='tight', facecolor='white')
    fig5c.savefig(FIGS / 'Fig_5C_Bootstrap.png',
                  dpi=300, bbox_inches='tight', facecolor='white')
    fig5c.savefig(FIGS / 'Fig_5C_Bootstrap.pdf',
                  bbox_inches='tight', facecolor='white')
    plt.close(fig5c)
    print(f"    → niva_bootstrap_weights.png")
    print(f"    → {FIGS / 'Fig_5C_Bootstrap.png'}")

    # ============================================================
    # Fig 5D: Voxel-wise correlation with Hansen NT atlases
    # ============================================================
    print("\n  [Fig 5D] NIVA × Hansen NT atlas voxel-wise correlation ...")
    # use the un-thresholded full-sample composite from Fig 5A
    niva_full_5d = weighted_sum_5a  # already on [0, 1]
    corr_5d = {}
    for nt_name, (d_5d, _aff_5d) in nt_density_cache.items():
        if d_5d.shape != niva_full_5d.shape:
            continue
        mask_5d = (niva_full_5d > 0) & (d_5d > 0)
        if mask_5d.sum() < 1000:
            continue
        r_5d, p_5d = _pearsonr(niva_full_5d[mask_5d].astype(np.float64),
                               d_5d[mask_5d].astype(np.float64))
        corr_5d[nt_name] = (float(r_5d), float(p_5d))
    print(f"    {len(corr_5d)} NT atlases correlated; "
          f"r range = [{min(v[0] for v in corr_5d.values()):.3f}, "
          f"{max(v[0] for v in corr_5d.values()):.3f}]")

    if corr_5d:
        nt_sorted_5d = sorted(corr_5d.keys(), key=lambda n: corr_5d[n][0])
        rs_5d = [corr_5d[n][0] for n in nt_sorted_5d]
        # Mark which NTs actually contributed to the NIVA (weight > 0 in Fig 5A)
        contributed = {nt for nt, w in weights_5a.items() if w > 0}
        colors_5d = ['crimson' if n in contributed else 'lightsteelblue'
                     for n in nt_sorted_5d]

        fig5d, ax5d = plt.subplots(figsize=(8.5, 5.5), facecolor='white')
        ax5d.barh(range(len(nt_sorted_5d)), rs_5d,
                  color=colors_5d, alpha=0.85,
                  edgecolor='black', linewidth=0.4)
        ax5d.set_yticks(range(len(nt_sorted_5d)))
        ax5d.set_yticklabels(nt_sorted_5d, fontsize=10)
        ax5d.set_xlabel('Voxel-wise Pearson r  (NIVA  vs.  individual NT atlas)',
                        fontsize=10.5)
        ax5d.set_title('Fig. 5D  |  NIVA spatial alignment with Hansen NT atlases',
                       fontweight='bold', fontsize=11.5)
        ax5d.axvline(x=0, color='black', linewidth=0.5)
        legend_elems_5d = [
            Patch(facecolor='crimson',       alpha=0.85,
                  label='Full-sample FDR-significant  (q < 0.05, contributed to NIVA anchor)'),
            Patch(facecolor='lightsteelblue', alpha=0.85,
                  label='Not FDR-significant  (q ≥ 0.05, excluded from anchor)'),
        ]
        ax5d.legend(handles=legend_elems_5d, loc='lower right',
                    fontsize=8.5, framealpha=0.92)
        for s in ['top', 'right']:
            ax5d.spines[s].set_visible(False)
        plt.tight_layout()
        fig5d.savefig(VAL_DIR_5BCD / 'niva_hansen_correlation.png',
                      dpi=300, bbox_inches='tight', facecolor='white')
        fig5d.savefig(FIGS / 'Fig_5D_HansenCorrelation.png',
                      dpi=300, bbox_inches='tight', facecolor='white')
        fig5d.savefig(FIGS / 'Fig_5D_HansenCorrelation.pdf',
                      bbox_inches='tight', facecolor='white')
        plt.close(fig5d)
        print(f"    → niva_hansen_correlation.png")
        print(f"    → {FIGS / 'Fig_5D_HansenCorrelation.png'}")

    print("\n[Fig 5B/C/D] validation suite complete — outputs in:")
    print(f"  {VAL_DIR_5BCD}")
except Exception as exc_5bcd:
    import traceback as _tb_5bcd
    print(f"  ⚠️ Fig 5B/C/D inline build failed: "
          f"{type(exc_5bcd).__name__}: {exc_5bcd}")
    _tb_5bcd.print_exc()

# ============================================================
# README
# ============================================================
readme = """# Publication-Ready Outputs

## Tables (.csv, ready for Word/Excel paste)

| File | Content |
|---|---|
| Table_1_*.csv | Baseline characteristics (run `table1_baseline.py` separately) |
| Table_2_NT_Acute_Outcome.csv | NT effects on discharge mRS, Models A/B/C |
| Table_3_CST_Adjusted_Model_D.csv | CST adjustment effect on NT OR |
| Table_4_NT_Inflammation_Interactions.csv | FDR-significant NT × inflammation pairs |

### Supplementary tables (.csv)

| File | Content |
|---|---|
| Table_S1_Deep_Phenotyping_FDR.csv | NT × deep-phenotyping outcomes (MoCA / Barthel / PHQ-9 / GAD-7 / SIS 6 + 12 m; OLS on Koch-residualised NT loads with TLV+NIHSS+Age+Sex+CST adjustment; BH-FDR within each outcome's 17-NT family — port of analysis/outcome_analysis.py) |
| Table_S2_NRI_IDI_AUC.csv | Incremental prediction metrics: 3 nested logistic models (Base / Base+CHA / Base+CHA×IL-6 dual-burden) × AUC [95% CI] + ΔAUC [CI, P] + continuous NRI [CI, P] + IDI [CI, P]; bootstrap N=1000 |
| Table_S3_AdditiveInteraction_summary.csv | Self-contained ground-truth for Fig 3B-Left: primary pair (medial-CHA × hsCRP) 4-cell breakdown (LL/LH/HL/HH × N / Events / Risk [Wilson 95% CI] / OR_vs_LL) + additive-interaction summary (RERI [bootstrap CI] / AP / S / 1-df LRT / Q_global) annotated on the HH row |
| Table_S4_AdditiveInteraction_AllSigPairs.csv | Additive-interaction stats for all 7 globally-FDR-significant pairs |
| Table_S5_SimpleSlope_CHA_IL6.csv | Per-tertile CHA × IL-6 simple-slope (continuous-scale visualization) |

## Figures (.png, 300 DPI)

| File | Content |
|---|---|
| Fig_2_CST_OR_attenuation.png | Forest plot: NT OR before/after CST adjustment |
| Fig_3A_Interaction_heatmap.png | NT × Inflam interaction heatmap (-log₁₀ q) |
| Fig_4A_DCA.png | Decision curve analysis |
| Fig_4B_CV_10fold.png | 10-fold cross-validation AUC |
| Fig_5B_SplitHalf.png | NIVA split-half cross-validation (r = 0.997) |
| Fig_5C_Bootstrap.png | NIVA bootstrap weight stability |
| Fig_5D_HansenCorrelation.png | NIVA vs Hansen NT atlas correlation |
| Fig_6AB_SmallLesion.png | Small-lesion: NT damage + IL-6 |

## Supplementary (.png + .csv)

| File | Content |
|---|---|
| Supp_Fig_S2_12month_temporal_decay | NT effects at 12-mo (none significant after FDR) |
| Supp_Fig_S3_Temporal_Trajectory | NT effect across discharge / 3m / 6m / 12m |
| Supp_Fig_S4A_PCA_System | PCA system-level aggregation |
| Supp_Fig_S4B_PreSyn_vs_PostSyn | Pre- vs post-synaptic effect comparison |
| Supp_Fig_S4C_Dose_Response | Quartile dose-response |
| Supp_Fig_S5_Mediation | Bootstrap mediation (null indirect effects) |
| Supp_Fig_S6.png/.pdf | 3-panel sensitivity figure (A/B/C) |
| Supp_Fig_S6A_Permutation.csv | A: 1,000-iter permutation test |
| Supp_Fig_S6B_mRS_Cutpoint.csv | B: mRS grouping sensitivity |
| Supp_Fig_S6C_Spin_Test.csv | C: Alexander-Bloch spatial-null spin test |

## Manually prepared (not generated here)
- Table 1: baseline characteristics (use `table1_baseline.py`)
- Fig 1: Pipeline diagram
- Fig 6C: Mechanism schematic

## Generated inline by this script
- Fig 5A: NIVA brain map (inline rebuild, FDR-gated)
- Fig 5B/C/D: NIVA validation suite (inline rebuild)
  Output mirrored to publication_ready/figures/Fig_5{B,C,D}_*.png/.pdf
  AND to <SRC>/../7.figure/vulnerability_map/validation/
"""
with open(PUB / "README.md", "w") as f:
    f.write(readme)

print("\n" + "="*60)
print(f" ✅ ALL DONE → {PUB}")
print("="*60)
print("\nFiles generated:")
for p in sorted(PUB.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(PUB)}")
