#!/bin/bash
# 不用 set -e，手动检查
set -uo pipefail

# ============================================================
#  单被试: 提取 13 种递质图谱的 加权受损负荷 (Weighted Lesion Load)
#  基于 Koch et al. (2025, Brain) 方法学
#
#  用法: bash extract_weighted_load_single.sh <ID> <LESION_ROOT> <OUT_CSV_DIR>
#
#  输入:
#    LESION_ROOT/{ID}/lesion_MNI.nii.gz     — 已配准到 MNI 1mm 的病灶 mask
#    ATLAS_DIR/functionnectome_anat_*_1mm.nii.gz — 13 张递质密度图谱
#
#  输出:
#    OUT_CSV_DIR/{ID}_neurochem_load.csv     — 一行: ID, TLV, Load_xxx, ...
#
#  核心公式 (Koch 2025, p.3937):
#    SL_NT-LES = Σ(atlas_value × voxel_volume)  ≈  Mean × Volume(非零体素)
#    SL_LES    = Total Lesion Volume (TLV)
# ============================================================

# --- 0. 参数检查 ---
if [ $# -lt 3 ]; then
    echo "用法: $0 <ID> <LESION_ROOT> <OUT_CSV_DIR>"
    exit 1
fi

ID=$1
LESION_ROOT=$2
OUT_CSV_DIR=$3

# --- 1. 环境与路径 ---
export FSLDIR=${FSLDIR:-/home/liuzhengxin/fsl}
. "${FSLDIR}/etc/fslconf/fsl.sh"
export PATH="${FSLDIR}/bin:${PATH}"

# 1mm 重采样后的递质图谱目录
ATLAS_DIR="/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/human_CHA"

# 胆碱能通路图谱 (也可以包含)
CHOL_ATLAS_DIR="${ATLAS_DIR}"

MNI_LESION="${LESION_ROOT}/${ID}/lesion_MNI.nii.gz"

mkdir -p "${OUT_CSV_DIR}"

# --- 日志函数 ---
GLOBAL_LOG="${OUT_CSV_DIR}/extract_load.log"
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [${ID}] $*"
    echo "$msg"
    ( flock -w 5 200 && echo "$msg" >> "${GLOBAL_LOG}"; ) 200>"${GLOBAL_LOG}.lock"
}

# --- 2. 检查病灶文件 ---
if [ ! -f "${MNI_LESION}" ]; then
    log "SKIP: lesion_MNI.nii.gz 不存在"
    echo "${ID},MISSING_LESION" > "${OUT_CSV_DIR}/${ID}_neurochem_load.csv"
    exit 0
fi

# --- 3. 提取 TLV (Total Lesion Volume) ---
# fslstats -V 输出: <非零体素数> <非零体素总体积mm³>
tlv_output=$(fslstats "${MNI_LESION}" -V)
tlv_voxels=$(echo "${tlv_output}" | awk '{print $1}')
tlv_mm3=$(echo "${tlv_output}" | awk '{print $2}')

if [ "${tlv_voxels}" = "0" ] || [ -z "${tlv_mm3}" ]; then
    log "WARNING: 病灶体积为零"
    echo "${ID},ZERO_LESION" > "${OUT_CSV_DIR}/${ID}_neurochem_load.csv"
    exit 0
fi

log "TLV = ${tlv_mm3} mm³ (${tlv_voxels} voxels)"

# --- 4. 对每个递质图谱提取加权受损负荷 ---
# 核心公式: Weighted_Load = Mean_within_lesion × Volume_within_lesion
# 即: fslstats ${atlas} -k ${lesion} -M  ×  fslstats ${atlas} -k ${lesion} -V 的第2列
#
# 注意: 我们用 atlas 作为主输入, lesion 作为 mask (-k)
#        -M  = lesion 内 atlas 值的均值
#        -V  = lesion 内非零 atlas 体素的 <个数> <体积mm³>
#
# 但更精确做法: 直接用 atlas 在 lesion 内的均值 × lesion的体积
# 因为 lesion 是二值 mask, 所以:
#   Load = mean(atlas within lesion) × N_lesion_voxels × voxel_volume
# 等价于: fslstats atlas -k lesion -M 乘以 lesion 的总体积

results="${ID},${tlv_mm3}"

# 收集递质图谱 (functionnectome_anat_*_1mm.nii.gz)
atlas_files=( $(find "${ATLAS_DIR}" -maxdepth 1 -name "functionnectome_anat_*_1mm.nii.gz" 2>/dev/null | sort) )

if [ ${#atlas_files[@]} -eq 0 ]; then
    log "ERROR: 未找到任何递质图谱文件在 ${ATLAS_DIR}"
    echo "${ID},NO_ATLAS" > "${OUT_CSV_DIR}/${ID}_neurochem_load.csv"
    exit 0
fi

log "找到 ${#atlas_files[@]} 个递质图谱"

for atlas in "${atlas_files[@]}"; do
    atlas_name=$(basename "${atlas}" .nii.gz | sed 's/functionnectome_anat_//' | sed 's/_1mm//')

    # 方法1 (Koch 2025): 加权受损负荷 = 病灶内图谱均值 × 病灶体积
    # 这等价于病灶内图谱值的总和 × 单体素体积
    mean_val=$(fslstats "${atlas}" -k "${MNI_LESION}" -M 2>/dev/null || echo "0")

    # 处理 NaN 或空值
    if [ -z "${mean_val}" ] || [ "${mean_val}" = "nan" ] || [ "${mean_val}" = "NaN" ]; then
        mean_val="0"
    fi

    # Weighted Load = Mean × TLV(mm³)
    # 注意: 这里 Mean 是 atlas 在 lesion mask 区域内所有体素的均值
    # 乘以 TLV 即得到受损负荷 (= 所有体素的 atlas 值之和 × 体素体积)
    weighted_load=$(echo "${mean_val} ${tlv_mm3}" | awk '{printf "%.6f", $1 * $2}')

    results="${results},${weighted_load}"
    log "  ${atlas_name}: mean=${mean_val}, load=${weighted_load}"
done

# --- 5. 可选: 提取胆碱能通路图谱的负荷 ---
chol_maps=( "human_CHA_2mm_1mm" "mask_Lateral_Path_1mm" "mask_Medial_Path_1mm" )
for cmap_name in "${chol_maps[@]}"; do
    cmap="${CHOL_ATLAS_DIR}/${cmap_name}.nii.gz"
    if [ -f "${cmap}" ]; then
        mean_val=$(fslstats "${cmap}" -k "${MNI_LESION}" -M 2>/dev/null || echo "0")
        if [ -z "${mean_val}" ] || [ "${mean_val}" = "nan" ]; then
            mean_val="0"
        fi
        weighted_load=$(echo "${mean_val} ${tlv_mm3}" | awk '{printf "%.6f", $1 * $2}')
        results="${results},${weighted_load}"
        log "  ${cmap_name}: mean=${mean_val}, load=${weighted_load}"
    else
        results="${results},NA"
        log "  ${cmap_name}: 文件不存在, 跳过"
    fi
done

# --- 6. 写出结果 ---
echo "${results}" > "${OUT_CSV_DIR}/${ID}_neurochem_load.csv"
log "结果已保存: ${OUT_CSV_DIR}/${ID}_neurochem_load.csv"
log "====== ${ID} 提取完成 ======"
