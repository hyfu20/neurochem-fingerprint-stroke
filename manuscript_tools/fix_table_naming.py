#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_table_naming.py
====================
把 publication_ready/ 目录里残留的旧编号 CSV 重命名为稿件正式引用的编号。

旧编号 → 新编号映射（与稿件 Methods/Results/Figure Legends 中的引用一致）：
  Table_S5_AdditiveInteraction_AllSigPairs.csv → Table_S4_AdditiveInteraction_AllSigPairs.csv
  Table_S6_SimpleSlope_CHA_IL6.csv             → Table_S5_SimpleSlope_CHA_IL6.csv
  Table_S4_AdditiveInteraction_4cell.csv       → Table_S3_AdditiveInteraction_4cell.csv
      （4cell 数据被稿件并入 S3 引用范畴）

不动正确编号：
  Table_S1_Deep_Phenotyping_FDR.csv ✓
  Table_S2_NRI_IDI_AUC.csv ✓
  Table_S3_AdditiveInteraction_summary.csv ✓
  Table_2_NT_Acute_Outcome.csv ✓
  Table_3_CST_Adjusted_Model_D.csv ✓
  Table_4_NT_Inflammation_Interactions.csv ✓

用法：
  python fix_table_naming.py /path/to/publication_ready
  python fix_table_naming.py            # 默认当前目录
  python fix_table_naming.py --dry-run  # 仅显示，不真改
"""
import sys
import argparse
from pathlib import Path

RENAME_MAP = {
    "Table_S5_AdditiveInteraction_AllSigPairs.csv": "Table_S4_AdditiveInteraction_AllSigPairs.csv",
    "Table_S6_SimpleSlope_CHA_IL6.csv":             "Table_S5_SimpleSlope_CHA_IL6.csv",
    # 4cell 不在稿件直接引用，但保留并归入 S3 命名空间
    "Table_S4_AdditiveInteraction_4cell.csv":       "Table_S3_AdditiveInteraction_4cell.csv",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", nargs="?", default=".",
                        help="包含 Table_S*.csv 的目录（默认当前目录）")
    parser.add_argument("--dry-run", action="store_true", help="仅打印计划，不真改名")
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.exists():
        sys.exit(f"❌ 目录不存在: {folder}")

    print(f"扫描目录: {folder}")
    print("-" * 70)

    n_renamed = 0
    n_skipped = 0
    n_conflict = 0

    for old, new in RENAME_MAP.items():
        old_p = folder / old
        new_p = folder / new

        if not old_p.exists():
            print(f"  [skip]   {old}  （不存在）")
            n_skipped += 1
            continue

        if new_p.exists():
            # 已经有目标文件，检查是否相同
            if old_p.read_bytes() == new_p.read_bytes():
                print(f"  [dup]    {old}  已存在同名 {new}，删除旧文件")
                if not args.dry_run:
                    old_p.unlink()
                n_renamed += 1
            else:
                print(f"  ⚠️ [conflict] {old} → {new}  目标已存在且内容不同！跳过")
                n_conflict += 1
            continue

        print(f"  [rename] {old}  →  {new}")
        if not args.dry_run:
            old_p.rename(new_p)
        n_renamed += 1

    print("-" * 70)
    print(f"重命名/合并: {n_renamed}    跳过: {n_skipped}    冲突: {n_conflict}")
    if args.dry_run:
        print("⚠️  dry-run 模式，未实际修改文件。去掉 --dry-run 真正执行。")
    else:
        print("✅ 完成。")

    # 最终校验：列出目录里所有 Table_*.csv
    print()
    print("当前目录里的 Table_*.csv 文件:")
    for f in sorted(folder.glob("Table_*.csv")):
        print(f"  ✓ {f.name}  ({f.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
