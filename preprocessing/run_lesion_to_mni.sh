#!/bin/bash
# 不用 set -e，改为手动检查，避免一个被试出错终止全部批处理
set -uo pipefail

# ============================================================
#  病灶 → MNI 标准化 + 胆碱能图谱叠加提取
#  用法: bash run_lesion_to_mni.sh <ID> <OUT_ROOT>
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

PROJ_DIR="/data/usersdir/liuzhengxin/cholinergic_project"
DWI_ROOT="/data/usersdir/liuzhengxin/deepisles_ORG"
T1_ROOT="/data/shares/CNSR3/NIFITI/Release/CNSR3_all_CodeN_nifti/CNSR3-13012-T1w-nii_v20190625"
STANDARD_REF="${FSLDIR}/data/standard/MNI152_T1_2mm_brain.nii.gz"

work_dir="${OUT_ROOT}/${ID}"
mkdir -p "${work_dir}"

# --- 全局统一日志 (所有被试写同一个文件) ---
GLOBAL_LOG="${OUT_ROOT}/pipeline_all.log"
STEP_LOG="${work_dir}/_step_tmp.log"   # 单步临时日志，跑完一次性写入全局

# 写一行日志到全局（短消息用）
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [${ID}] $*"
    echo "$msg"
    ( flock -w 5 200 && echo "$msg" >> "${GLOBAL_LOG}"; ) 200>"${GLOBAL_LOG}.lock"
}

# 把临时日志整块追加到全局日志（FSL命令输出用，只锁一次）
flush_step_log() {
    if [ -s "${STEP_LOG}" ]; then
        ( flock -w 5 200 && sed "s/^/[$(date '+%Y-%m-%d %H:%M:%S')] [${ID}]   /" "${STEP_LOG}" >> "${GLOBAL_LOG}"; ) 200>"${GLOBAL_LOG}.lock"
        rm -f "${STEP_LOG}"
    fi
}

log "====== 开始处理 ${ID} ======"

# --- 2. 稳健寻找文件 ---
LESION_NATIVE=$(find "${DWI_ROOT}/${ID}/results" -maxdepth 1 -name 'lesion_msk*' -print -quit 2>/dev/null || true)
DWI_BET="${DWI_ROOT}/${ID}/dwi_stripped_bet.nii.gz"
T1_FILE=$(find "${T1_ROOT}/${ID}" -maxdepth 1 -name '*2D-T1w-TRA.nii.gz' -print -quit 2>/dev/null || true)

# 如果缺文件，记录并跳过
missing=""
[ -z "${LESION_NATIVE}" ] && missing="${missing} lesion_mask"
[ -z "${T1_FILE}" ]       && missing="${missing} T1"
[ ! -f "${DWI_BET}" ]     && missing="${missing} dwi_bet"

if [ -n "${missing}" ]; then
    log "INPUT_MISSING:${missing}"
    echo "${ID},INPUT_MISSING" > "${work_dir}/neurochem_line.txt"
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
        echo "${ID},BET_FAILED" > "${work_dir}/neurochem_line.txt"
        exit 0
    fi
    if [ ! -f "${work_dir}/t1_brain.nii.gz" ]; then
        log "ERROR: bet 未生成输出文件"
        echo "${ID},BET_FAILED" > "${work_dir}/neurochem_line.txt"
        exit 0
    fi
    flush_step_log

    # 3b. DWI → T1 (刚性 6-DOF)
    log "FLIRT: DWI → T1 ..."
    if ! flirt -in "${DWI_BET}" \
          -ref "${work_dir}/t1_brain" \
          -omat "${work_dir}/dwi_to_t1.mat" \
          -dof 6 > "${STEP_LOG}" 2>&1; then
        flush_step_log
        log "ERROR: flirt DWI→T1 失败"
        echo "${ID},FLIRT_DWI2T1_FAILED" > "${work_dir}/neurochem_line.txt"
        exit 0
    fi
    flush_step_log

    # 3c. T1 → MNI (仿射 12-DOF)
    log "FLIRT: T1 → MNI ..."
    if ! flirt -in "${work_dir}/t1_brain" \
          -ref "${STANDARD_REF}" \
          -omat "${work_dir}/t1_to_mni.mat" \
          -dof 12 > "${STEP_LOG}" 2>&1; then
        flush_step_log
        log "ERROR: flirt T1→MNI 失败"
        echo "${ID},FLIRT_T12MNI_FAILED" > "${work_dir}/neurochem_line.txt"
        exit 0
    fi
    flush_step_log

    # 3d. 合并变换矩阵  DWI → MNI
    log "合并变换矩阵..."
    if ! convert_xfm -omat "${work_dir}/dwi_to_mni.mat" \
                -concat "${work_dir}/t1_to_mni.mat" \
                "${work_dir}/dwi_to_t1.mat"; then
        log "ERROR: convert_xfm 失败"
        echo "${ID},XFM_FAILED" > "${work_dir}/neurochem_line.txt"
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
        echo "${ID},APPLYWARP_FAILED" > "${work_dir}/neurochem_line.txt"
        exit 0
    fi

    # 检查输出
    if [ ! -f "${work_dir}/lesion_MNI.nii.gz" ]; then
        log "ERROR: applywarp 未生成 lesion_MNI.nii.gz"
        echo "${ID},APPLYWARP_FAILED" > "${work_dir}/neurochem_line.txt"
        exit 0
    fi
    log "配准完成 ✓"
else
    log "lesion_MNI.nii.gz 已存在，跳过配准"
fi

# --- 4. 提取胆碱能受损数据 ---
log "提取 26 个标签的体积..."
total_v=$(fslstats "${work_dir}/lesion_MNI" -V | awk '{print $2}')

cha_results=""
for i in $(seq 1 26); do
    label_file="${PROJ_DIR}/label_library/label_${i}.nii.gz"
    if [ ! -f "${label_file}" ]; then
        log "WARNING: 标签文件缺失 ${label_file}"
        val="NA"
    else
        # 图谱标签作为输入，病灶作为遮罩
        val=$(fslstats "${label_file}" -k "${work_dir}/lesion_MNI" -V | awk '{print $2}')
    fi
    cha_results="${cha_results}${val},"
done

# 写出结果 (去掉末尾逗号)
echo "${ID},${total_v},${cha_results%,}" > "${work_dir}/neurochem_line.txt"
log "结果已写入 neurochem_line.txt"

# --- 5. 清理中间文件 (保留 lesion_MNI) ---
rm -f "${work_dir}/t1_brain.nii.gz" \
      "${work_dir}/t1_brain_mask.nii.gz" \
      "${work_dir}"/*.mat \
      "${STEP_LOG}"

log "====== ${ID} 处理完成 ======"
