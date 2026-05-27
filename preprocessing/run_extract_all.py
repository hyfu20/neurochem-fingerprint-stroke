#!/usr/bin/env python3
"""
一键提取 5000 人 × 17 图谱 加权受损负荷 (Weighted Lesion Load)
Koch et al. (2025, Brain) 方法学

用法:
    python3 run_extract_all.py          # 默认 50 并发
    python3 run_extract_all.py 80       # 指定并发数

输出:
    6.NeurotransmitterMapping/output_loads/{ID}_nt_load.csv  (单人)
    6.NeurotransmitterMapping/NT_Imaging_Load_Master.csv     (汇总大表)

预计耗时: 112核服务器 50并发 约 10-20 分钟
"""

import subprocess
import sys
import os
import time
import logging
from pathlib import Path
from multiprocessing import Pool

# ============================================================
# 路径配置
# ============================================================
FSLDIR = "/home/liuzhengxin/fsl"
FSLSTATS = f"{FSLDIR}/bin/fslstats"
LESION_ROOT = Path("/data/usersdir/liuzhengxin/Stepbystep/5.MNI/output")
ATLAS_DIR = Path("/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/atlas1mm")
OUT_ROOT = Path("/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/output_loads")
MASTER_CSV = Path("/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/NT_Imaging_Load_Master.csv")
LOG_FILE = Path("/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/extract_load.log")
FAIL_LOG = Path("/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/extract_load_failures.log")

os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"

HEADER = "ID,TLV,5HT1a,5HT1b,5HT2a,5HT4,5HT6,5HTT,A4B2,D1,D2,DAT,M1,NAT,VAChT,human_CHA,JHU_EC,Lateral_Path,Medial_Path"

# 硬编码 17 张图谱, 顺序锁死
ATLAS_LIST = [
    "functionnectome_anat_5HT1a_1mm.nii.gz",
    "functionnectome_anat_5HT1b_1mm.nii.gz",
    "functionnectome_anat_5HT2a_1mm.nii.gz",
    "functionnectome_anat_5HT4_1mm.nii.gz",
    "functionnectome_anat_5HT6_1mm.nii.gz",
    "functionnectome_anat_5HTT_1mm.nii.gz",
    "functionnectome_anat_A4B2_1mm.nii.gz",
    "functionnectome_anat_D1_1mm.nii.gz",
    "functionnectome_anat_D2_1mm.nii.gz",
    "functionnectome_anat_DAT_1mm.nii.gz",
    "functionnectome_anat_M1_1mm.nii.gz",
    "functionnectome_anat_NAT_1mm.nii.gz",
    "functionnectome_anat_VAChT_1mm.nii.gz",
    "human_CHA_2mm_1mm.nii.gz",
    "mask_JHU_EC_2mm_1mm.nii.gz",
    "mask_Lateral_Path_1mm.nii.gz",
    "mask_Medial_Path_1mm.nii.gz",
]


# ============================================================
# 核心提取函数 (直接在进程池中运行, 不再额外 spawn Python)
# ============================================================
def extract_one(sid):
    """提取单个被试, 返回 (sid, 状态, 原因)"""
    mni_lesion = LESION_ROOT / sid / "lesion_MNI.nii.gz"
    out_file = OUT_ROOT / f"{sid}_nt_load.csv"

    if out_file.exists():
        return (sid, "skip", "已完成")

    if not mni_lesion.exists():
        return (sid, "skip", "无lesion_MNI")

    try:
        # TLV
        r = subprocess.run(
            [FSLSTATS, str(mni_lesion), "-V"],
            capture_output=True, text=True, timeout=60
        )
        parts = r.stdout.strip().split()
        if len(parts) < 2:
            return (sid, "fail", "fslstats -V 输出异常")
        tlv_mm3 = parts[1]

        if float(tlv_mm3) == 0:
            return (sid, "fail", "TLV=0")

        # 逐图谱
        loads = []
        for atlas_file in ATLAS_LIST:
            atlas_path = ATLAS_DIR / atlas_file
            if not atlas_path.exists():
                loads.append("NA")
                continue
            r = subprocess.run(
                [FSLSTATS, str(atlas_path), "-k", str(mni_lesion), "-M"],
                capture_output=True, text=True, timeout=60
            )
            mean_val = r.stdout.strip()
            if not mean_val or mean_val in ("nan", "NaN"):
                mean_val = "0"
            try:
                load_val = float(mean_val) * float(tlv_mm3)
                loads.append(f"{load_val:.6f}")
            except ValueError:
                loads.append("0.000000")

        line = f"{sid},{tlv_mm3},{','.join(loads)}\n"
        out_file.write_text(line)
        return (sid, "ok", "")

    except subprocess.TimeoutExpired:
        return (sid, "fail", "超时")
    except Exception as e:
        return (sid, "fail", str(e))


def setup_logging():
    """设置日志: 同时输出到屏幕和文件"""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("extract")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    # 屏幕
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    # 文件
    fh = logging.FileHandler(str(LOG_FILE), mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def main():
    njobs = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    log = setup_logging()

    log.info("=" * 60)
    log.info("  Koch (2025) 加权受损负荷 — 一键提取")
    log.info(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  日志文件: {LOG_FILE}")
    log.info("=" * 60)

    # --- 预检查 ---
    if not Path(FSLSTATS).exists():
        log.error(f"ERROR: fslstats 不存在: {FSLSTATS}")
        sys.exit(1)
    if not LESION_ROOT.exists():
        log.error(f"ERROR: 病灶目录不存在: {LESION_ROOT}")
        sys.exit(1)
    if not ATLAS_DIR.exists():
        log.error(f"ERROR: 图谱目录不存在: {ATLAS_DIR}")
        sys.exit(1)

    missing_atlas = [a for a in ATLAS_LIST if not (ATLAS_DIR / a).exists()]
    if missing_atlas:
        log.warning(f"  ⚠️  缺少 {len(missing_atlas)} 张图谱:")
        for a in missing_atlas:
            log.warning(f"      {a}")
    else:
        log.info(f"  ✓ 17 张图谱全部就位")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # --- Step 1: 扫描被试 ---
    log.info(f"[Step 1] 扫描被试...")
    all_ids = sorted([d.name for d in LESION_ROOT.iterdir() if d.is_dir()])
    has_lesion = [s for s in all_ids
                  if (LESION_ROOT / s / "lesion_MNI.nii.gz").exists()]
    already_done = [s for s in has_lesion
                    if (OUT_ROOT / f"{s}_nt_load.csv").exists()]
    todo = [s for s in has_lesion if s not in set(already_done)]

    log.info(f"  总文件夹:      {len(all_ids)}")
    log.info(f"  有 lesion_MNI: {len(has_lesion)}")
    log.info(f"  已完成:        {len(already_done)}")
    log.info(f"  本次待处理:    {len(todo)}")

    # --- Step 2: 并行提取 ---
    if todo:
        est_min = len(todo) / njobs * 9 / 60
        log.info(f"[Step 2] 并行提取 ({njobs} 并发, {len(todo)} 人, 预计 ~{est_min:.0f} 分钟)...")

        t0 = time.time()
        ok_count, fail_count, skip_count = 0, 0, 0
        failures = []

        with Pool(njobs) as pool:
            for i, (sid, status, reason) in enumerate(
                pool.imap_unordered(extract_one, todo), 1
            ):
                if status == "ok":
                    ok_count += 1
                elif status == "fail":
                    fail_count += 1
                    failures.append((sid, reason))
                else:
                    skip_count += 1

                if i % 100 == 0 or i == len(todo):
                    elapsed = time.time() - t0
                    speed = i / elapsed if elapsed > 0 else 0
                    eta = (len(todo) - i) / speed if speed > 0 else 0
                    log.info(f"  [{i:>5}/{len(todo)}] "
                             f"✓{ok_count} ✗{fail_count} ⏭{skip_count} | "
                             f"{speed:.1f} 人/s | ETA {eta:.0f}s")

        elapsed = time.time() - t0
        log.info(f"  提取完成! 耗时 {elapsed:.0f}s ({elapsed/60:.1f}min)")
        log.info(f"  成功: {ok_count}, 失败: {fail_count}, 跳过: {skip_count}")

        # 失败列表写入单独日志
        if failures:
            log.warning(f"  失败列表 ({len(failures)} 个):")
            with open(FAIL_LOG, "w", encoding="utf-8") as ff:
                ff.write(f"# 失败被试列表 {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                ff.write(f"# 总失败: {len(failures)}\n")
                ff.write("ID,原因\n")
                for sid, reason in failures:
                    ff.write(f"{sid},{reason}\n")
                    log.warning(f"    {sid}: {reason}")
            log.info(f"  失败详情: {FAIL_LOG}")
    else:
        log.info(f"[Step 2] 全部已完成, 跳过")

    # --- Step 3: 合并大表 ---
    log.info(f"[Step 3] 合并 NT_Imaging_Load_Master.csv...")

    result_files = sorted(OUT_ROOT.glob("*_nt_load.csv"))
    lines = []
    bad_lines = 0
    for f in result_files:
        content = f.read_text().strip()
        if not content:
            continue
        if len(content.split(",")) == 19:
            lines.append(content)
        else:
            bad_lines += 1

    with open(MASTER_CSV, "w") as fout:
        fout.write(HEADER + "\n")
        for line in lines:
            fout.write(line + "\n")

    log.info(f"{'=' * 60}")
    log.info(f"  ✅ 完成!")
    log.info(f"{'=' * 60}")
    log.info(f"  输出文件:   {MASTER_CSV}")
    log.info(f"  数据行数:   {len(lines)} (不含表头)")
    log.info(f"  列数:       19 (ID + TLV + 17 图谱)")
    if bad_lines:
        log.warning(f"  ⚠️  列数异常跳过: {bad_lines}")

    log.info(f"  预览:")
    log.info(f"  {HEADER}")
    for line in lines[:3]:
        parts = line.split(",")
        short = ",".join(parts[:4]) + ",...," + ",".join(parts[-2:])
        log.info(f"  {short}")

    log.info(f"  日志文件: {LOG_FILE}")
    log.info(f"  验证命令:")
    log.info(f"    wc -l {MASTER_CSV}")
    log.info(f"    head -3 {MASTER_CSV} | column -t -s,")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
