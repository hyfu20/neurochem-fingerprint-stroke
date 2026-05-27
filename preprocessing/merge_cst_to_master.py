#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 CST_Load 合并到主数据表 merged_neuro_data.csv
基于 ID 进行 left join

用法:
    python3 merge_cst_to_master.py

输入:
    - all_cst_load.csv        (来自 extract_cst_load.sh)
    - merged_neuro_data.csv   (现有主数据表)

输出:
    - merged_neuro_data.csv   (原地覆盖, 新增 CST_Load 列)
    - merged_neuro_data_backup_YYYYMMDD.csv  (备份)
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = Path("/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping")
CST_CSV = BASE_DIR / "cst_load_output" / "all_cst_load.csv"
MASTER_CSV = BASE_DIR / "3.variable_outcom_merge_data" / "merged_neuro_data.csv"

print("=" * 60)
print("  合并 CST_Load 到主数据表")
print("=" * 60)

# ============================================================
# 1. 读取数据
# ============================================================
if not CST_CSV.exists():
    print(f"❌ CST_Load 文件不存在: {CST_CSV}")
    print(f"   请先运行: bash extract_cst_load.sh")
    sys.exit(1)

if not MASTER_CSV.exists():
    print(f"❌ 主数据表不存在: {MASTER_CSV}")
    sys.exit(1)

df_cst = pd.read_csv(CST_CSV)
df_master = pd.read_csv(MASTER_CSV)

print(f"\n📊 CST_Load 表:   {df_cst.shape[0]} 行 × {df_cst.shape[1]} 列")
print(f"📊 主数据表:       {df_master.shape[0]} 行 × {df_master.shape[1]} 列")

# ============================================================
# 2. 清洗 ID 列 (与 merge_csv.py 保持一致)
# ============================================================
df_cst["ID"] = df_cst["ID"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

# 主表的 ID 列可能叫 "ID" 或 "code_n"
id_col_master = None
for cand in ["ID", "code_n", "id"]:
    if cand in df_master.columns:
        id_col_master = cand
        break

if id_col_master is None:
    print("❌ 主数据表中未找到 ID 列 (尝试了 ID, code_n, id)")
    sys.exit(1)

df_master[id_col_master] = (df_master[id_col_master]
                            .astype(str).str.strip()
                            .str.replace(r'\.0$', '', regex=True))

print(f"  主表 ID 列:     {id_col_master}")

# ============================================================
# 3. 检查是否已有 CST_Load 列
# ============================================================
if "CST_Load" in df_master.columns:
    n_existing = df_master["CST_Load"].notna().sum()
    print(f"\n⚠️  主表已有 CST_Load 列 ({n_existing} 个有效值)")
    print(f"   将覆盖更新")
    df_master = df_master.drop(columns=["CST_Load"])

# ============================================================
# 4. 合并
# ============================================================
# CST_Load 转为数值
df_cst["CST_Load"] = pd.to_numeric(df_cst["CST_Load"], errors="coerce")
df_cst_clean = df_cst.dropna(subset=["CST_Load"])

print(f"\n  CST_Load 有效值: {len(df_cst_clean)} / {len(df_cst)}")

# Left join: 保留主表所有行
df_merged = df_master.merge(
    df_cst_clean[["ID", "CST_Load"]],
    left_on=id_col_master,
    right_on="ID",
    how="left"
)

# 如果 merge 产生了额外的 ID 列, 去掉
if id_col_master != "ID" and "ID_y" in df_merged.columns:
    df_merged = df_merged.drop(columns=["ID_y"])
    if "ID_x" in df_merged.columns:
        df_merged = df_merged.rename(columns={"ID_x": "ID"})
elif "ID" in df_merged.columns and id_col_master == "ID":
    pass  # ID 列保持不变

n_matched = df_merged["CST_Load"].notna().sum()
pct = n_matched / len(df_merged) * 100
print(f"\n✅ 合并完成:")
print(f"   匹配到 CST_Load: {n_matched} / {len(df_merged)} ({pct:.1f}%)")

# CST_Load 基本统计
cst_vals = df_merged["CST_Load"].dropna()
print(f"\n📈 CST_Load 统计:")
print(f"   Mean ± SD:  {cst_vals.mean():.2f} ± {cst_vals.std():.2f}")
print(f"   Median:     {cst_vals.median():.2f}")
print(f"   Range:      [{cst_vals.min():.2f}, {cst_vals.max():.2f}]")
print(f"   零值 (无重叠): {(cst_vals == 0).sum()} ({(cst_vals == 0).sum()/len(cst_vals)*100:.1f}%)")

# ============================================================
# 5. 备份 & 保存
# ============================================================
timestamp = datetime.now().strftime("%Y%m%d")
backup_path = MASTER_CSV.parent / f"merged_neuro_data_backup_{timestamp}.csv"
df_master_orig = pd.read_csv(MASTER_CSV)
df_master_orig.to_csv(backup_path, index=False)
print(f"\n💾 备份已保存: {backup_path}")

df_merged.to_csv(MASTER_CSV, index=False)
print(f"💾 主表已更新: {MASTER_CSV}")
print(f"   新维度: {df_merged.shape[0]} × {df_merged.shape[1]}")

print(f"\n{'=' * 60}")
print(f"  ✅ CST_Load 已成功合并到主数据表")
print(f"  下一步: 运行 Master_NT_Analysis_v4.py (CST_Load 自动纳入 Model C)")
print(f"{'=' * 60}")
