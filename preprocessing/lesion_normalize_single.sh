#!/bin/bash
# 不用 set -e，改为手动检查，避免一个被试出错终止全部批处理
set -uo pipefail

# ============================================================
#  单被试: 病灶 → MNI 标准化 (仅配准，不叠加图谱)
#  用法: bash lesion_normalize_single.sh <ID> <OUT_ROOT>
#
#  输入:
#    DWI_ROOT/{ID}/results/lesion_msk*.nii.gz   — 原始病灶 mask
#    DWI_ROOT/{ID}/dwi_stripped_bet.nii.gz       — DWI 去颅骨
#    T1_ROOT/{ID}/*2D-T1w-TRA.nii.gz            — T1 结构像
#
#  输出:
#    OUT_ROOT/{ID}/lesion_MNI.nii.gz  — MNI 空间病灶 mask
#    OUT_ROOT/{ID}/result_line.txt    — 状态/体积记录
# ============================================================

# --- 0. 参数检查 ---
if [ $# -lt 2 ]; then
    echo "用法: $0 <ID> <OUT_ROOT>"
    exit 1
fi

ID=$1
OUT_ROOT=$2

# --- 1. 环境与绝对路径 ---
export FSLDIR=/home/liuzhengxin/fsl
. "${FSLDIR}/etc/fslconf/fsl.sh"
export PATH="${FSLDIR}/bin:${PATH}"

DWI_ROOT="/data/usersdir/liuzhengxin/Stepbystep/4.deepisles_script/deepisles_ORG"
T1_ROOT="/data/shares/CNSR3/NIFITI/Release/CNSR3_all_CodeN_nifti/CNSR3-13012-T1w-nii_v20190625"
STANDARD_REF="${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz"

work_dir="${OUT_ROOT}/${ID}"
mkdir -p "${work_dir}"

# --- 全局统一日志 (所有被试写同一个文件) ---
GLOBAL_LOG="${OUT_ROOT}/pipeline_all.log"
STEP_LOG="${work_dir}/_step_tmp.log"

# 写一行日志到全局（短消息用）
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [${ID}] $*"
    echo "$msg"
    ( flock -w 5 200 && echo "$msg" >> "${GLOBAL_LOG}"; ) 200>"${GLOBAL_LOG}.lock"
}

# 把临时日志整块追加到全局日志（FSL 命令输出用，只锁一次）
flush_step_log() {
    if [ -s "${STEP_LOG}" ]; then
        ( flock -w 5 200 && sed "s/^/[$(date '+%Y-%m-%d %H:%M:%S')] [${ID}]   /" "${STEP_LOG}" >> "${GLOBAL_LOG}"; ) 200>"${GLOBAL_LOG}.lock"
        rm -f "${STEP_LOG}"
    fi
}

log "====== 开始处理 ${ID} ======"

# --- 2. 稳健寻找文件 ---
# 优先精确匹配 lesion_msk.nii.gz，避免抓到 _corrected 等中间文件
LESION_NATIVE="${DWI_ROOT}/${ID}/results/lesion_msk.nii.gz"
if [ ! -f "${LESION_NATIVE}" ]; then
    # fallback: 模糊匹配（取字典序最后一个，通常是最终版本）
    LESION_NATIVE=$(find "${DWI_ROOT}/${ID}/results" -maxdepth 1 -name 'lesion_msk*.nii.gz' -print 2>/dev/null | sort | tail -1)
fi
DWI_BET="${DWI_ROOT}/${ID}/dwi_stripped_bet.nii.gz"
T1_FILE=$(find "${T1_ROOT}/${ID}" -maxdepth 1 -name '*2D-T1w-TRA.nii.gz' -print 2>/dev/null | sort | head -n 1)

# 如果缺文件，记录并跳过
missing=""
[ -z "${LESION_NATIVE}" ] && missing="${missing} lesion_mask"
[ -z "${T1_FILE}" ]       && missing="${missing} T1"
[ ! -f "${DWI_BET}" ]     && missing="${missing} dwi_bet"

if [ -n "${missing}" ]; then
    log "INPUT_MISSING:${missing}"
    echo "${ID},INPUT_MISSING" > "${work_dir}/result_line.txt"
    exit 0
fi

log "病灶文件: ${LESION_NATIVE}"
log "T1  文件: ${T1_FILE}"
log "DWI 文件: ${DWI_BET}"

# --- 3. 配准: DWI → T1 → MNI (已有结果则跳过) ---
if [ ! -f "${work_dir}/lesion_MNI.nii.gz" ]; then

    # 3a. T1 去颅骨
    log "BET 去颅骨..."
    if ! bet "${T1_FILE}" "${work_dir}/t1_brain" -f 0.3 -Z > "${STEP_LOG}" 2>&1; then
        flush_step_log
        log "ERROR: bet 命令执行失败"
        echo "${ID},BET_FAILED" > "${work_dir}/result_line.txt"
        exit 0
    fi
    if [ ! -f "${work_dir}/t1_brain.nii.gz" ]; then
        log "ERROR: bet 未生成输出文件"
        echo "${ID},BET_FAILED" > "${work_dir}/result_line.txt"
        exit 0
    fi
    flush_step_log

    # 3b. DWI → T1 (刚性 6-DOF, 全角度搜索)
    log "FLIRT: DWI → T1 ..."
    if ! flirt -in "${DWI_BET}" \
          -ref "${work_dir}/t1_brain" \
          -omat "${work_dir}/dwi_to_t1.mat" \
          -dof 6 \
          -searchrx -180 180 -searchry -180 180 -searchrz -180 180 \
          > "${STEP_LOG}" 2>&1; then
        flush_step_log
        log "ERROR: flirt DWI→T1 失败"
        echo "${ID},FLIRT_DWI2T1_FAILED" > "${work_dir}/result_line.txt"
        exit 0
    fi
    flush_step_log

    # 3c-prep. 将病灶 mask 变换到 T1 空间，生成代价函数遮盖 (Cost-Function Masking)
    log "生成 T1 空间病灶遮盖 (CFM)..."
    flirt -in "${LESION_NATIVE}" \
          -ref "${work_dir}/t1_brain" \
          -applyxfm -init "${work_dir}/dwi_to_t1.mat" \
          -interp nearestneighbour \
          -out "${work_dir}/lesion_in_t1"
    # 反转：病灶=0 健康=1，作为配准权重
    fslmaths "${work_dir}/lesion_in_t1" -binv "${work_dir}/lesion_inv_mask_t1"

    # 3c. T1 → MNI (仿射 12-DOF, 代价函数遮盖 + 全角度搜索)
    log "FLIRT: T1 → MNI (with CFM)..."
    if ! flirt -in "${work_dir}/t1_brain" \
          -ref "${STANDARD_REF}" \
          -omat "${work_dir}/t1_to_mni.mat" \
          -inweight "${work_dir}/lesion_inv_mask_t1" \
          -dof 12 \
          -searchrx -180 180 -searchry -180 180 -searchrz -180 180 \
          > "${STEP_LOG}" 2>&1; then
        flush_step_log
        log "ERROR: flirt T1→MNI 失败"
        echo "${ID},FLIRT_T12MNI_FAILED" > "${work_dir}/result_line.txt"
        exit 0
    fi
    flush_step_log

    # 3d. 合并变换矩阵  DWI → MNI
    log "合并变换矩阵..."
    if ! convert_xfm -omat "${work_dir}/dwi_to_mni.mat" \
                -concat "${work_dir}/t1_to_mni.mat" \
                "${work_dir}/dwi_to_t1.mat"; then
        log "ERROR: convert_xfm 失败"
        echo "${ID},XFM_FAILED" > "${work_dir}/result_line.txt"
        exit 0
    fi

    # 3e. 将病灶应用到 MNI 空间 (最近邻插值保持二值)
    log "applywarp: 病灶 → MNI ..."
    if ! applywarp --in="${LESION_NATIVE}" \
              --ref="${STANDARD_REF}" \
              --premat="${work_dir}/dwi_to_mni.mat" \
              --interp=nn \
              --out="${work_dir}/lesion_MNI"; then
        log "ERROR: applywarp 失败"
        echo "${ID},APPLYWARP_FAILED" > "${work_dir}/result_line.txt"
        exit 0
    fi

    # 检查输出
    if [ ! -f "${work_dir}/lesion_MNI.nii.gz" ]; then
        log "ERROR: applywarp 未生成 lesion_MNI.nii.gz"
        echo "${ID},APPLYWARP_FAILED" > "${work_dir}/result_line.txt"
        exit 0
    fi

    # 3f. 生成 QC 截图: 标准脑 + 病灶轮廓叠加 (轴位中层)
    log "生成 QC 截图..."
    slicer "${STANDARD_REF}" "${work_dir}/lesion_MNI" -s 2 -z 0.5 "${work_dir}/qc_axial.png" 2>/dev/null || true
    slicer "${STANDARD_REF}" "${work_dir}/lesion_MNI" -s 2 -x 0.5 "${work_dir}/qc_sagittal.png" 2>/dev/null || true
    slicer "${STANDARD_REF}" "${work_dir}/lesion_MNI" -s 2 -y 0.5 "${work_dir}/qc_coronal.png" 2>/dev/null || true
    # 拼合三视图为一张总图
    if command -v pngappend &>/dev/null; then
        pngappend "${work_dir}/qc_axial.png" + "${work_dir}/qc_sagittal.png" + "${work_dir}/qc_coronal.png" "${work_dir}/qc_registration.png" 2>/dev/null || true
        rm -f "${work_dir}/qc_axial.png" "${work_dir}/qc_sagittal.png" "${work_dir}/qc_coronal.png"
    fi

    log "配准完成 ✓"
else
    log "lesion_MNI.nii.gz 已存在，跳过配准"
fi

# --- 4. 计算病灶 MNI 体积并记录结果 ---
total_v=$(fslstats "${work_dir}/lesion_MNI" -V | awk '{print $2}')
echo "${ID},${total_v}" > "${work_dir}/result_line.txt"
log "病灶 MNI 体积: ${total_v} mm³"
log "结果已写入 result_line.txt"

# --- 5. 清理中间文件 (保留 lesion_MNI.nii.gz) ---
rm -f "${work_dir}/t1_brain.nii.gz" \
      "${work_dir}/t1_brain_mask.nii.gz" \
      "${work_dir}/lesion_in_t1.nii.gz" \
      "${work_dir}/lesion_inv_mask_t1.nii.gz" \
      "${work_dir}"/*.mat \
      "${STEP_LOG}"

log "====== ${ID} 处理完成 ======"
