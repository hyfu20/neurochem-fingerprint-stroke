#!/usr/bin/env python3
"""
Figure 4: High-Dimensional Symptom Mapping (Double Dissociation Heatmap)
=========================================================================
输出: /data/usersdir/liuzhengxin/Stepbystep/7.figure/figure4/
"""
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_ROOT = Path("/data/usersdir/liuzhengxin/Stepbystep/7.figure/figure4")

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
OUTCOME_FALLBACKS = {
    "D_MRS":["D_MRS","d_mrs"], "m3_mRS":["m3_mRS","M03_mRS"],
    "m12_mRS":["m12_mRS","M12_mRS"], "MoCA_12m":["MoCA_12m","M12_MoCA","MoCA"],
    "SIS_Emotion_6m":["SIS_Emotion_6m","M06_SIS_Emotion","VA6_SIS_emotion","SIS_emotion"],
    "Barthel_12m":["Barthel_12m","M12_Barthel","Barthel"],
    "PHQ9":["PHQ9","BSL_PHQ9","PHQ_9"], "GAD7":["GAD7","BSL_GAD7","GAD_7"],
}
OUTCOME_META = [
    ("D_MRS","D-mRS","pos"), ("m3_mRS","3m-mRS","pos"), ("m12_mRS","12m-mRS","pos"),
    ("MoCA_12m","MoCA-12m","neg"), ("SIS_Emotion_6m","SIS-Emo-6m","neg"),
    ("Barthel_12m","Barthel-12m","neg"), ("PHQ9","PHQ-9","pos"), ("GAD7","GAD-7","pos"),
]

def find_col(df, cands):
    for c in cands:
        if c in df.columns: return c
    return None

def main():
    data_path = ("/data/usersdir/liuzhengxin/Stepbystep/"
                 "6.NeurotransmitterMapping/3.variable_outcom_merge_data/merged_neuro_data.csv")
    if not os.path.exists(data_path):
        print(f"⚠️ 未找到: {data_path}"); return

    import pandas as pd, statsmodels.api as sm

    df = pd.read_csv(data_path, low_memory=False)
    print(f"数据: {df.shape}")

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

    out_cols, out_labels, out_dirs = [], [], []
    for key, label, d in OUTCOME_META:
        c = find_col(df, OUTCOME_FALLBACKS.get(key, [key]))
        if c: out_cols.append(c); out_labels.append(label); out_dirs.append(d)
    print(f"结局: {out_labels}")
    if len(out_cols) < 2: print("⚠️ 结局不足"); return

    # ── Koch 残差化: NT 对 TLV 回归取残差，消除共线性 ──
    tlv_col = find_col(df, ["TLV"])
    if tlv_col:
        df[tlv_col] = pd.to_numeric(df[tlv_col], errors="coerce")
        log_tlv = np.log1p(df[tlv_col].clip(lower=0))
        print(f"Koch 残差化: NT 对 log1p(TLV) 回归取残差...")
        resid_map = {}
        for nc in nt_ordered:
            sub_r = df[[nc]].copy()
            sub_r["log_tlv"] = log_tlv
            sub_r = sub_r.dropna()
            if len(sub_r) < 50:
                resid_map[nc] = nc
                continue
            X_r = sm.add_constant(sub_r[["log_tlv"]])
            try:
                res_r = sm.OLS(sub_r[nc].astype(float), X_r).fit()
                resid_col = f"Resid_{nc.replace(pfx, '')}"
                df[resid_col] = np.nan
                df.loc[sub_r.index, resid_col] = res_r.resid
                resid_map[nc] = resid_col
            except Exception:
                resid_map[nc] = nc
        # 替换 nt_ordered 为残差列
        nt_ordered_resid = [resid_map[nc] for nc in nt_ordered]
        print(f"  已残差化 {sum(1 for v in resid_map.values() if v.startswith('Resid_'))}/{len(nt_ordered)} 个 NT")
    else:
        nt_ordered_resid = nt_ordered
        print("  ⚠️ 无 TLV 列，跳过残差化")

    # ── 协变量: 残差化后不再包含 TLV（已消除）──
    covars = []
    for cands in [["A_NIHSS","BSL_NIHSS","NIHSS"],["AGE","Age"],["SEX","Sex","GENDER"],["CST_Load"]]:
        c = find_col(df, cands)
        if c: covars.append(c)
    print(f"协变量(不含TLV): {covars}")

    n_nt, n_out = len(nt_ordered_resid), len(out_cols)
    beta_mat = np.full((n_nt, n_out), np.nan)
    pval_mat = np.full((n_nt, n_out), np.nan)
    n_mat = np.zeros((n_nt, n_out), dtype=int)

    print(f"计算 {n_nt}×{n_out} 回归 (Koch 残差 + NIHSS+Age+Sex)...")
    for i, nc in enumerate(nt_ordered_resid):
        for j, oc in enumerate(out_cols):
            needed = [oc, nc] + [c for c in covars if c in df.columns]
            sub = df[needed].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < 30: continue
            n_mat[i,j] = len(sub)
            for col in needed:
                if col.upper() in ("SEX","GENDER"): continue
                s = sub[col].std()
                if s > 1e-10: sub[col] = (sub[col]-sub[col].mean())/s
            X = sm.add_constant(sub[[nc]+[c for c in covars if c in sub.columns]])
            try:
                res = sm.OLS(sub[oc].astype(float), X).fit()
                beta_mat[i,j] = res.params[nc]; pval_mat[i,j] = res.pvalues[nc]
            except: pass

    flat = pval_mat.flatten(); valid = np.isfinite(flat)
    qflat = np.full_like(flat, np.nan)
    if valid.sum() > 0:
        try:
            from statsmodels.stats.multitest import multipletests
            _, q, _, _ = multipletests(flat[valid], method="fdr_bh"); qflat[valid] = q
        except:
            p_v = flat[valid]; n = len(p_v); ranks = np.empty(n)
            ranks[np.argsort(p_v)] = np.arange(1,n+1); qflat[valid] = np.minimum(1, p_v*n/ranks)
    qval_mat = qflat.reshape(pval_mat.shape)

    nt_labels = [NT_RENAME.get(c.replace(pfx,""), c.replace(pfx,"")) for c in nt_ordered]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8,n_out*1.2+3), max(6,n_nt*0.5+2)), facecolor='white')
    vmax = max(np.nanmax(np.abs(beta_mat))*0.9, 0.1)
    im = ax.imshow(beta_mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')

    for i in range(n_nt):
        for j in range(n_out):
            b, q, p = beta_mat[i,j], qval_mat[i,j], pval_mat[i,j]
            if np.isnan(b): ax.text(j,i,'–',ha='center',va='center',fontsize=7,color='#CCC'); continue
            stars = '***' if q<0.001 else ('**' if q<0.01 else ('*' if q<0.05 else ('†' if p<0.05 else '')))
            fc = 'white' if abs(b)>vmax*0.5 else 'black'
            ax.text(j,i,f'{b:.2f}{stars}',ha='center',va='center',fontsize=6.5,color=fc,
                    fontweight='bold' if q<0.05 else 'normal')

    ax.set_xticks(range(n_out)); ax.set_xticklabels(out_labels, rotation=45, ha='right', fontsize=10, fontweight='bold')
    ax.set_yticks(range(n_nt)); ax.set_yticklabels(nt_labels, fontsize=10)
    for i, c in enumerate(nt_ordered):
        bare = c.replace(pfx,"")
        for sn, nts in NT_SYSTEMS.items():
            if bare in nts: ax.get_yticklabels()[i].set_color(SYS_COLORS[sn]); ax.get_yticklabels()[i].set_fontweight('bold'); break
    for j, d in enumerate(out_dirs):
        ax.text(j, n_nt+0.3, '↑worse' if d=='pos' else '↓worse', ha='center', va='top', fontsize=7, color='#888', style='italic')
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02); cbar.set_label('Standardized β', fontsize=11, fontweight='bold')
    ax.set_title('Supplementary Figure S2. NT Independent Effects (Koch Residuals)\n*** q<0.001  ** q<0.01  * q<0.05  † P<0.05\nβ: TLV-orthogonalized NT → Outcome, adjusted for NIHSS+Age+Sex', fontsize=10, fontweight='bold', pad=15)
    plt.tight_layout(rect=[0.08,0.05,1,0.93])
    fig.savefig(OUTPUT_ROOT/'figureS2_symptom_heatmap_null.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_ROOT/'figureS2_symptom_heatmap_null.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ figureS2_symptom_heatmap_null.png/.pdf → {OUTPUT_ROOT}")

    pd.DataFrame(beta_mat, index=nt_labels, columns=out_labels).to_csv(OUTPUT_ROOT/'figureS2_beta_matrix.csv')
    pd.DataFrame(qval_mat, index=nt_labels, columns=out_labels).to_csv(OUTPUT_ROOT/'figureS2_qval_matrix.csv')
    print("✅ figureS2_beta_matrix.csv + figureS2_qval_matrix.csv")

if __name__ == "__main__":
    main()
