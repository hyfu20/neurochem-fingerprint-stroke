#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量提取 CST 重叠体积 (CST_Load)
用于在 Model C 中控制皮质脊髓束物理损伤

原理:
    CST_Load = 病灶 ∩ CST 双侧模板 的重叠体积 (mm³)
    fslmaths lesion -mul CST → overlap
    fslstats overlap -V → 体积

输入:
    - CST 双侧模板: atlas1mm/CST_Bilateral_1mm.nii.gz
    - 3528 个病人的病灶: 5.MNI/output/{ID}/lesion_MNI.nii.gz

输出:
    - 单人: cst_load_output/{ID}_cst_load.csv
    - 汇总: cst_load_output/all_cst_load.csv  (列: ID, CST_Load)

用法:
    python3 extract_cst_load.py          # 默认 50 并发
    python3 extract_cst_load.py 80       # 指定并发数

预计耗时: 112 核服务器 50 并发 约 5-15 分钟
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
FSLMATHS = f"{FSLDIR}/bin/fslmaths"
FSLSTATS = f"{FSLDIR}/bin/fslstats"

LESION_ROOT = Path("/data/usersdir/liuzhengxin/Stepbystep/5.MNI/output")
CST_TEMPLATE = Path("/data/usersdir/liuzhengxin/Stepbystep/"
                     "6.NeurotransmitterMapping/1.atlas/atlas1mm/"
                     "CST_Bilateral_1mm.nii.gz")

OUT_DIR = Path("/data/usersdir/liuzhengxin/Stepbystep/"
               "6.NeurotransmitterMapping/cst_load_output")
FINAL_CSV = OUT_DIR / "all_cst_load.csv"
LOG_FILE = OUT_DIR / "extract_cst_load.log"
FAIL_LOG = OUT_DIR / "extract_cst_load_failures.log"
TMP_DIR = OUT_DIR / "tmp"

os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"


# ============================================================
# 单被试提取函数 (在进程池中运行)
# ============================================================
def extract_one(sid):
    """
    计算单个被试的 CST_Load (mm³).
    返回 (sid, status, cst_load_mm3, reason)
    """
    mni_lesion = LESION_ROOT / sid / "lesion_MNI.nii.gz"
    out_file = OUT_DIR / f"{sid}_cst_load.csv"
    tmp_overlap = TMP_DIR / f"{sid}_overlap.nii.gz"

    # 跳过已完成
    if out_file.exists():
        content = out_file.read_text().strip()
        if content and "MISSING" not in content and "ERROR" not in content:
            return (sid, "skip", None, "已完成")

    # 检查病灶文件
    if not mni_lesion.exists():
        out_file.write_text(f"{sid},MISSING_LESION\n")
        return (sid, "skip", None, "无 lesion_MNI")

    try:
        # 1. 计算重叠: lesion × CST → 临时文件
        r = subprocess.run(
            [FSLMATHS, str(mni_lesion), "-mul", str(CST_TEMPLATE),
             str(tmp_overlap)],
            capture_output=True, text=True, timeout=120
        )
        if r.returncode != 0:
            out_file.write_text(f"{sid},FSLMATHS_ERROR\n")
            tmp_overlap.unlink(missing_ok=True)
            return (sid, "fail", None, f"fslmaths 错误: {r.stderr.strip()}")

        # 2. 提取重叠体积 (mm³)
        r = subprocess.run(
            [FSLSTATS, str(tmp_overlap), "-V"],
            capture_output=True, text=True, timeout=60
        )
        parts = r.stdout.strip().split()
        if len(parts) < 2:
            out_file.write_text(f"{sid},FSLSTATS_ERROR\n")
            tmp_overlap.unlink(missing_ok=True)
            return (sid, "fail", None, "fslstats -V 输出异常")

        overlap_voxels = int(parts[0])
        overlap_mm3 = float(parts[1])

        # 零重叠也是有效数据（病灶没碰到 CST）
        if overlap_voxels == 0:
            overlap_mm3 = 0.0

        # 3. 写出结果
        out_file.write_text(f"{sid},{overlap_mm3:.6f}\n")

        # 清理临时文件
        tmp_overlap.unlink(missing_ok=True)

        return (sid, "ok", overlap_mm3, "")

    except subprocess.TimeoutExpired:
        tmp_overlap.unlink(missing_ok=True)
        return (sid, "fail", None, "超时")
    except Exception as e:
        tmp_overlap.unlink(missing_ok=True)
        return (sid, "fail", None, str(e))


# ============================================================
# 日志
# ============================================================
def setup_logging():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cst_load")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(str(LOG_FILE), mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ============================================================
# 主函数
# ============================================================
def main():
    njobs = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    log = setup_logging()

    log.info("=" * 60)
    log.info("  CST 重叠体积提取 (CST_Load)")
    log.info(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    # ── 预检查 ──
    if not Path(FSLMATHS).exists():
        log.error(f"❌ fslmaths 不存在: {FSLMATHS}")
        sys.exit(1)
    if not Path(FSLSTATS).exists():
        log.error(f"❌ fslstats 不存在: {FSLSTATS}")
        sys.exit(1)
    if not LESION_ROOT.exists():
        log.error(f"❌ 病灶目录不存在: {LESION_ROOT}")
        sys.exit(1)
    if not CST_TEMPLATE.exists():
        log.error(f"❌ CST 模板不存在: {CST_TEMPLATE}")
        log.error("   请先确认 CST_Bilateral_1mm.nii.gz 已生成")
        sys.exit(1)

    log.info(f"  CST 模板:  {CST_TEMPLATE}")
    log.info(f"  病灶输入:  {LESION_ROOT}")
    log.info(f"  输出目录:  {OUT_DIR}")
    log.info(f"  并行数:    {njobs}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: 扫描被试 ──
    log.info(f"\n[Step 1/3] 扫描被试...")
    all_ids = sorted([d.name for d in LESION_ROOT.iterdir() if d.is_dir()])
    has_lesion = [s for s in all_ids
                  if (LESION_ROOT / s / "lesion_MNI.nii.gz").exists()]
    already_done = [s for s in has_lesion
                    if (OUT_DIR / f"{s}_cst_load.csv").exists()
                    and "MISSING" not in
                    (OUT_DIR / f"{s}_cst_load.csv").read_text()
                    and "ERROR" not in
                    (OUT_DIR / f"{s}_cst_load.csv").read_text()]
    todo = [s for s in has_lesion if s not in set(already_done)]

    log.info(f"  总文件夹:      {len(all_ids)}")
    log.info(f"  有 lesion_MNI: {len(has_lesion)}")
    log.info(f"  已完成:        {len(already_done)}")
    log.info(f"  本次待处理:    {len(todo)}")

    # ── Step 2: 并行提取 ──
    if todo:
        est_min = len(todo) / njobs * 3 / 60  # 每人约 3 秒
        log.info(f"\n[Step 2/3] 并行提取 CST_Load "
                 f"({njobs} 并发, {len(todo)} 人, "
                 f"预计 ~{est_min:.0f} 分钟)...")

        t0 = time.time()
        ok_count, fail_count, skip_count = 0, 0, 0
        failures = []
        cst_values = []

        with Pool(njobs) as pool:
            for i, (sid, status, cst_val, reason) in enumerate(
                pool.imap_unordered(extract_one, todo), 1
            ):
                if status == "ok":
                    ok_count += 1
                    if cst_val is not None:
                        cst_values.append(cst_val)
                elif status == "fail":
                    fail_count += 1
                    failures.append((sid, reason))
                else:
                    skip_count += 1

                if i % 200 == 0 or i == len(todo):
                    elapsed = time.time() - t0
                    speed = i / elapsed if elapsed > 0 else 0
                    eta = (len(todo) - i) / speed if speed > 0 else 0
                    log.info(f"  [{i:>5}/{len(todo)}] "
                             f"✓{ok_count} ✗{fail_count} ⏭{skip_count} | "
                             f"{speed:.1f} 人/s | ETA {eta:.0f}s")

        elapsed = time.time() - t0
        log.info(f"  提取完成! 耗时 {elapsed:.0f}s ({elapsed/60:.1f}min)")
        log.info(f"  成功: {ok_count}, 失败: {fail_count}, 跳过: {skip_count}")

        # 简要统计
        if cst_values:
            import statistics
            mean_v = statistics.mean(cst_values)
            median_v = statistics.median(cst_values)
            n_zero = sum(1 for v in cst_values if v == 0)
            n_pos = sum(1 for v in cst_values if v > 0)
            log.info(f"  本轮 CST_Load 统计:")
            log.info(f"    Mean:   {mean_v:.2f} mm³")
            log.info(f"    Median: {median_v:.2f} mm³")
            log.info(f"    有重叠: {n_pos} ({n_pos/len(cst_values)*100:.1f}%)")
            log.info(f"    无重叠: {n_zero} ({n_zero/len(cst_values)*100:.1f}%)")

        # 失败日志
        if failures:
            log.warning(f"  失败列表 ({len(failures)} 个):")
            with open(FAIL_LOG, "w", encoding="utf-8") as ff:
                ff.write(f"# CST_Load 提取失败列表 "
                         f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                ff.write("ID,原因\n")
                for sid, reason in failures:
                    ff.write(f"{sid},{reason}\n")
                    log.warning(f"    {sid}: {reason}")
            log.info(f"  失败详情: {FAIL_LOG}")
    else:
        log.info(f"\n[Step 2/3] 全部已完成, 跳过")

    # ── Step 3: 汇总 CSV ──
    log.info(f"\n[Step 3/3] 汇总 all_cst_load.csv...")

    result_files = sorted(OUT_DIR.glob("*_cst_load.csv"))
    lines = []
    bad_count = 0
    for f in result_files:
        content = f.read_text().strip()
        if not content:
            continue
        parts = content.split(",")
        # 有效行: ID,数值 (2 列, 且第 2 列不是错误标签)
        if (len(parts) == 2
                and parts[1] not in ("MISSING_LESION", "FSLMATHS_ERROR",
                                     "FSLSTATS_ERROR")):
            try:
                float(parts[1])  # 确认是数值
                lines.append(content)
            except ValueError:
                bad_count += 1
        else:
            bad_count += 1

    with open(FINAL_CSV, "w", encoding="utf-8") as fout:
        fout.write("ID,CST_Load\n")
        for line in lines:
            fout.write(line + "\n")

    log.info("=" * 60)
    log.info("  ✅ CST_Load 提取完成!")
    log.info("=" * 60)
    log.info(f"  输出文件:   {FINAL_CSV}")
    log.info(f"  有效数据:   {len(lines)} 行 (不含表头)")
    if bad_count:
        log.warning(f"  ⚠️  跳过无效: {bad_count}")
    log.info(f"  预览 (前 5 行):")
    log.info(f"  ID,CST_Load")
    for line in lines[:5]:
        log.info(f"  {line}")

    log.info("")
    log.info("  下一步:")
    log.info("    python3 merge_cst_to_master.py")
    log.info("=" * 60)

    # 清理 tmp
    import shutil
    shutil.rmtree(TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
