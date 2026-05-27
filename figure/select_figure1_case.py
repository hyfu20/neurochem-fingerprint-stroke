#!/usr/bin/env python3
"""
==========================================================
Figure 1 最佳病例筛选器
==========================================================
在云电脑上运行，自动从 merged_neuro_data.csv 中找出
最适合放 Figure 1 的代表性病例。

筛选标准（优先级从高到低）：
  1. 小病灶 + 差预后 (TLV < Q1, mRS ≥ 3)  → "小病灶大影响"
  2. CST_Load 高（病灶切到了皮质脊髓束）
  3. NT load 高（特别是胆碱能通路: Medial_Path, Lateral_Path）
  4. 影像数据完整（DWI + T1 + lesion_MNI + 图谱叠加都能生成）

运行:  python3 select_figure1_case.py
==========================================================
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# 数据路径
# ============================================================
DATA_CANDIDATES = [
    "/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/3.variable_outcom_merge_data/merged_neuro_data.csv",
    "/data/usersdir/liuzhengxin/Stepbystep/merged_neuro_data.csv",
    "/data/usersdir/liuzhengxin/merged_neuro_data.csv",
]
# 影像完整性检查路径
DWI_ROOT = "/data/usersdir/liuzhengxin/Stepbystep/4.deepisles_script/deepisles_ORG"
MNI_ROOT = "/data/usersdir/liuzhengxin/Stepbystep/5.MNI/output"

# ============================================================
# 加载数据
# ============================================================
df = None
for dp in DATA_CANDIDATES:
    if os.path.exists(dp):
        df = pd.read_csv(dp)
        print(f"✅ 加载数据: {dp}  ({len(df)} 例)")
        break
if df is None:
    print("⚠️ 未找到 merged_neuro_data.csv")
    print("  请检查路径或用 find 命令定位:")
    print("  find /data/usersdir/liuzhengxin -name 'merged_neuro_data.csv'")
    sys.exit(1)

# ============================================================
# 识别关键列
# ============================================================
id_col = 'ID' if 'ID' in df.columns else df.columns[0]

tlv_candidates = ['TLV', 'TLV_mm3', 'LesionVolume', 'lesion_vol']
tlv_col = None
for c in tlv_candidates:
    if c in df.columns:
        tlv_col = c
        break

mrs_candidates = ['m3_mRS', 'm6_mRS', 'D_MRS', 'mRS']
mrs_col = None
for c in mrs_candidates:
    if c in df.columns:
        mrs_col = c
        break

cst_candidates = ['CST_Load', 'CST_load', 'Load_CST']
cst_col = None
for c in cst_candidates:
    if c in df.columns:
        cst_col = c
        break

# NT load 列
nt_names = ['NAT', 'DAT', 'D1', 'D2', '5HT1a', '5HT1b', '5HT2a', '5HT4',
            '5HT6', '5HTT', 'A4B2', 'M1', 'VAChT', 'human_CHA',
            'JHU_EC', 'Lateral_Path', 'Medial_Path']
nt_cols = [c for c in df.columns if c in nt_names or
           c.replace('Load_', '') in nt_names]

print(f"\n关键列:")
print(f"  ID:     {id_col}")
print(f"  TLV:    {tlv_col}")
print(f"  mRS:    {mrs_col}")
print(f"  CST:    {cst_col}")
print(f"  NT列:   {len(nt_cols)} 个")

# ============================================================
# 筛选
# ============================================================
if tlv_col is None or mrs_col is None:
    print("\n⚠️ 缺少 TLV 或 mRS 列，无法自动筛选")
    print(f"  可用列: {list(df.columns[:20])}...")
    sys.exit(1)

df[tlv_col] = pd.to_numeric(df[tlv_col], errors='coerce')
df[mrs_col] = pd.to_numeric(df[mrs_col], errors='coerce')
df = df.dropna(subset=[tlv_col, mrs_col])

q1 = df[tlv_col].quantile(0.25)
q2 = df[tlv_col].quantile(0.50)
print(f"\n病灶体积分布:")
print(f"  Q1 = {q1:.0f}, Median = {q2:.0f}, Max = {df[tlv_col].max():.0f}")

# 筛选: 小病灶(< Q1) + 差预后(mRS ≥ 3)
small_severe = df[(df[tlv_col] < q1) & (df[mrs_col] >= 3)].copy()
print(f"\n小病灶 + 差预后 (TLV < {q1:.0f}, mRS ≥ 3): {len(small_severe)} 例")

# 如果太少，放宽到中等体积
if len(small_severe) < 5:
    small_severe = df[(df[tlv_col] < q2) & (df[mrs_col] >= 3)].copy()
    print(f"  放宽到 TLV < {q2:.0f}: {len(small_severe)} 例")

if len(small_severe) == 0:
    print("  未找到符合条件的病例，展示所有 mRS ≥ 3")
    small_severe = df[df[mrs_col] >= 3].copy()

# ============================================================
# 影像完整性检查
# ============================================================
def check_imaging(sid):
    """检查影像数据是否完整"""
    checks = {
        'DWI': os.path.exists(os.path.join(DWI_ROOT, sid, 'dwi_stripped.nii.gz')) or
               os.path.exists(os.path.join(DWI_ROOT, sid, 'dwi_stripped_bet.nii.gz')),
        'Lesion': os.path.exists(os.path.join(DWI_ROOT, sid, 'results', 'lesion_msk.nii.gz')),
        'MNI': os.path.exists(os.path.join(MNI_ROOT, sid, 'lesion_MNI.nii.gz')),
    }
    return checks

print("\n" + "=" * 80)
print("  排名  |  ID                |  TLV      |  mRS  |  CST_Load  |  影像完整性")
print("=" * 80)

# 排序: 优先 CST_Load 高 → TLV 小
if cst_col and cst_col in small_severe.columns:
    small_severe[cst_col] = pd.to_numeric(small_severe[cst_col], errors='coerce')
    small_severe = small_severe.sort_values(cst_col, ascending=False)
else:
    small_severe = small_severe.sort_values(tlv_col, ascending=True)

top_n = min(20, len(small_severe))
best_candidates = []

for rank, (idx, row) in enumerate(small_severe.head(top_n).iterrows(), 1):
    sid = str(row[id_col])
    tlv = row[tlv_col]
    mrs = row[mrs_col]
    cst = row[cst_col] if cst_col and cst_col in row.index else float('nan')
    
    checks = check_imaging(sid)
    complete = all(checks.values())
    status = "✅ 完整" if complete else "❌ " + ",".join(k for k, v in checks.items() if not v)
    
    print(f"  {rank:>4}  |  {sid:<18} |  {tlv:>8.0f} |  {mrs:.0f}    |  {cst:>9.4f} |  {status}")
    
    if complete:
        best_candidates.append({
            'rank': rank, 'id': sid, 'tlv': tlv, 'mrs': mrs, 'cst': cst
        })

print("=" * 80)

if best_candidates:
    best = best_candidates[0]
    print(f"\n🏆 推荐病例: {best['id']}")
    print(f"   TLV = {best['tlv']:.0f} mm³ (小病灶)")
    print(f"   mRS = {best['mrs']:.0f} (差预后)")
    if not np.isnan(best['cst']):
        print(f"   CST_Load = {best['cst']:.4f} (皮质脊髓束受累)")
    print(f"\n   运行命令:")
    print(f"   python3 generate_figure1_panels.py {best['id']}")
    print(f"   python3 figure1_combine.py")
    
    # 展示 NT load 情况
    case_match = df[df[id_col].astype(str) == str(best['id'])]
    if len(case_match) > 0:
        case_row = case_match.iloc[0]
        print(f"\n   NT load 分布:")
        for nt in ['NAT', 'DAT', '5HT1a', 'A4B2', 'Medial_Path', 'Lateral_Path']:
            for col in [nt, f'Load_{nt}']:
                if col in case_row.index:
                    val = case_row[col]
                    if pd.notna(val):
                        pct = (df[col].dropna() < val).mean() * 100
                        print(f"     {nt:>15}: {val:.4f}  (第 {pct:.0f} 百分位)")
                    break
else:
    print("\n⚠️ 没有找到影像数据完整的候选病例")
    print("  可能的原因: DWI_ROOT 或 MNI_ROOT 路径不对")
    print(f"  DWI_ROOT: {DWI_ROOT}")
    print(f"  MNI_ROOT: {MNI_ROOT}")
    print("\n  手动查看:")
    print(f"  ls {DWI_ROOT} | head -10")

# ============================================================
# 对比当前病例
# ============================================================
print("\n" + "=" * 80)
print("💡 选病例的逻辑（给审稿人讲故事）:")
print("=" * 80)
print("""
最佳 Figure 1 病例应该是:

  1. 小病灶 + 差预后 → 说明"体积不是一切"
  2. 病灶在基底节/内囊区 → 经典 MCA 卒中，读者一看就懂
  3. CST_Load 高 → "虽然病灶小，但切到了关键运动通路"
  4. NT load 高（特别是胆碱能通路）→ 直接支持你的核心发现

这样的病例让审稿人一看 Figure 1 就理解:
  "啊，这就是为什么需要做 NT mapping，
   光看体积根本解释不了预后差异"
""")
