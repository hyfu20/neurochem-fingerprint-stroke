#!/usr/bin/env python3
"""
图谱 1mm 标准化转换脚本
- 递质图谱 (连续密度): trilinear 插值
- 解剖 Mask/Labels:    nearestneighbour 插值
"""

import subprocess
import sys
from pathlib import Path

# ==============================================================================
# 1. 路径配置
# ==============================================================================
DEST_DIR = Path("/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/human_CHA")
REF_1MM = Path("/home/liuzhengxin/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz")

NT_SRC_DIR = Path("/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/Neurotransmitters")
CHOL_SRC_DIR = Path("/data/usersdir/liuzhengxin/5.cholinergic_project")

# 胆碱能/解剖掩码文件列表
CHOL_FILES = [
    "human_CHA_2mm.nii.gz",
    "mask_Lateral_Path.nii.gz",
    "mask_Medial_Path.nii.gz",
    "mask_JHU_EC_2mm.nii.gz",
]


def diagnose_paths():
    """诊断：检查源目录是否存在，并列出其中的 .nii.gz 文件"""
    print("=" * 60)
    print("诊断: 检查源目录和文件")
    print("=" * 60)

    for label, d in [("NT_SRC_DIR", NT_SRC_DIR), ("CHOL_SRC_DIR", CHOL_SRC_DIR)]:
        print(f"\n[{label}]")
        print(f"  路径: {d}")
        if not d.exists():
            print("  ❌ 目录不存在!")
            # 尝试列出父目录内容帮助定位
            parent = d.parent
            if parent.exists():
                print(f"  父目录 {parent} 中的内容:")
                for item in sorted(parent.iterdir()):
                    print(f"    {'📁' if item.is_dir() else '📄'} {item.name}")
        else:
            print("  ✅ 目录存在")
            nii_files = sorted(d.glob("*.nii.gz"))
            if nii_files:
                print(f"  找到 {len(nii_files)} 个 .nii.gz 文件:")
                for f in nii_files:
                    size_bytes = f.stat().st_size
                    if size_bytes < 1024:
                        size_str = f"{size_bytes} B"
                    elif size_bytes < 1024 * 1024:
                        size_str = f"{size_bytes / 1024:.1f} KB"
                    else:
                        size_str = f"{size_bytes / 1024 / 1024:.1f} MB"
                    print(f"    {f.name}  ({size_str})")
            else:
                print("  ⚠️  目录下没有 .nii.gz 文件")
                print("  目录中的内容:")
                for item in sorted(d.iterdir())[:30]:
                    print(f"    {'📁' if item.is_dir() else '📄'} {item.name}")

    print("\n" + "=" * 60)
    print()


def run_flirt(src: Path, ref: Path, out: Path, interp: str):
    """调用 FSL flirt 进行重采样"""
    cmd = [
        "flirt",
        "-in", str(src),
        "-ref", str(ref),
        "-applyxfm", "-init", str(Path(subprocess.os.environ.get("FSLDIR", "/usr/local/fsl")) / "etc/flirtsch/ident.mat"),
        "-interp", interp,
        "-out", str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ERROR: flirt 失败\n    {result.stderr.strip()}")
        return False
    return True


def main():
    # 检查参考图
    if not REF_1MM.exists():
        sys.exit(f"ERROR: 参考图不存在: {REF_1MM}")

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # 先运行诊断，确认路径正确
    diagnose_paths()

    print(">>>> 开始 1mm 重采样任务 <<<<")

    # ==================================================================
    # 步骤 1: 递质图谱 — trilinear
    # ==================================================================
    print("\n正在转换递质图谱 (trilinear)...")
    nt_files = sorted(NT_SRC_DIR.glob("functionnectome_anat_*.nii.gz"))
    if not nt_files:
        print("  WARNING: 未找到任何 functionnectome_anat_*.nii.gz 文件")

    for f in nt_files:
        name = f.name.replace(".nii.gz", "")
        out_file = DEST_DIR / f"{name}_1mm.nii.gz"
        if out_file.exists():
            print(f"  SKIP (已存在): {name}")
            continue
        print(f"  Processing: {name}")
        run_flirt(f, REF_1MM, out_file, "trilinear")

    # ==================================================================
    # 步骤 2: 胆碱能/解剖掩码 — nearestneighbour
    # ==================================================================
    print("\n正在转换胆碱能及掩码图谱 (nearestneighbour)...")
    for fname in CHOL_FILES:
        src = CHOL_SRC_DIR / fname
        if not src.exists():
            print(f"  WARNING: 未找到文件 {fname}")
            continue
        name = fname.replace(".nii.gz", "")
        out_file = DEST_DIR / f"{name}_1mm.nii.gz"
        if out_file.exists():
            print(f"  SKIP (已存在): {name}")
            continue
        print(f"  Processing Mask: {name}")
        run_flirt(src, REF_1MM, out_file, "nearestneighbour")

    # ==================================================================
    # 汇总
    # ==================================================================
    print(f"\n>>>> 全部转换完成！输出目录: {DEST_DIR} <<<<")
    results = sorted(DEST_DIR.glob("*.nii.gz"))
    for r in results:
        size_bytes = r.stat().st_size
        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{size_bytes / 1024 / 1024:.1f} MB"
        print(f"  {r.name}  ({size_str})")
    print(f"共 {len(results)} 个文件")


if __name__ == "__main__":
    main()
