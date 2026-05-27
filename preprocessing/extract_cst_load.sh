#!/bin/bash
set -uo pipefail

# ============================================================
#  批量提取 CST 重叠体积 (CST_Load)
#  用于在 Model C 中控制皮质脊髓束物理损伤
#
#  输入:
#    - CST 双侧模板: 1.atlas/atlas1mm/CST_Bilateral_1mm.nii.gz
#    - 3528 个病人的病灶: 5.MNI/output/{ID}/lesion_MNI.nii.gz
#
#  输出:
#    - 单人 CSV: OUT_DIR/{ID}_cst_load.csv
#    - 汇总 CSV: OUT_DIR/all_cst_load.csv
#      列: ID, CST_Load_mm3
#
#  原理:
#    CST_Load = 病灶 ∩ CST 模板 的重叠体积 (mm³)
#    即 fslmaths 做乘法 → fslstats -V 取体积
#
#  用法:
#    bash extract_cst_load.sh [并行数, 默认 50]
# ============================================================

NJOBS=${1:-50}

# ── 路径配置 ──
export FSLDIR=${FSLDIR:-/home/liuzhengxin/fsl}
. "${FSLDIR}/etc/fslconf/fsl.sh"
export PATH="${FSLDIR}/bin:${PATH}"

LESION_ROOT="/data/usersdir/liuzhengxin/Stepbystep/5.MNI/output"
CST_TEMPLATE="/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/atlas1mm/CST_Bilateral_1mm.nii.gz"
OUT_DIR="/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/cst_load_output"
FINAL_CSV="${OUT_DIR}/all_cst_load.csv"
LOG_FILE="${OUT_DIR}/extract_cst_load.log"
TMPDIR_BASE="${OUT_DIR}/tmp"

mkdir -p "${OUT_DIR}" "${TMPDIR_BASE}"

echo "============================================================"
echo "  CST 重叠体积提取 (CST_Load)"
echo "============================================================"
echo "  CST 模板:    ${CST_TEMPLATE}"
echo "  病灶输入:    ${LESION_ROOT}"
echo "  输出目录:    ${OUT_DIR}"
echo "  并行数:      ${NJOBS}"
echo "============================================================"

# ── 检查 CST 模板 ──
if [ ! -f "${CST_TEMPLATE}" ]; then
    echo "❌ CST 模板不存在: ${CST_TEMPLATE}"
    echo "   请先确认 CST_Bilateral_1mm.nii.gz 已生成"
    exit 1
fi

# ── 单被试提取函数 (export 给 xargs 调用) ──
extract_one() {
    local ID=$1
    local MNI_LESION="${LESION_ROOT}/${ID}/lesion_MNI.nii.gz"
    local OUT_CSV="${OUT_DIR}/${ID}_cst_load.csv"
    local TMP_OVERLAP="${TMPDIR_BASE}/${ID}_overlap.nii.gz"
    local LOGMSG

    # 跳过已完成
    if [ -f "${OUT_CSV}" ] && ! grep -q "MISSING\|ZERO" "${OUT_CSV}" 2>/dev/null; then
        return 0
    fi

    # 检查病灶文件
    if [ ! -f "${MNI_LESION}" ]; then
        echo "${ID},MISSING_LESION" > "${OUT_CSV}"
        return 0
    fi

    # 1. 计算重叠: lesion × CST → 临时文件
    fslmaths "${MNI_LESION}" -mul "${CST_TEMPLATE}" "${TMP_OVERLAP}" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "${ID},FSLMATHS_ERROR" > "${OUT_CSV}"
        rm -f "${TMP_OVERLAP}"
        return 0
    fi

    # 2. 提取重叠体积 (mm³)
    local overlap_output
    overlap_output=$(fslstats "${TMP_OVERLAP}" -V 2>/dev/null)
    local overlap_voxels overlap_mm3

    if [ -z "${overlap_output}" ]; then
        echo "${ID},FSLSTATS_ERROR" > "${OUT_CSV}"
        rm -f "${TMP_OVERLAP}"
        return 0
    fi

    overlap_voxels=$(echo "${overlap_output}" | awk '{print $1}')
    overlap_mm3=$(echo "${overlap_output}" | awk '{print $2}')

    # 处理零体积
    if [ "${overlap_voxels}" = "0" ] || [ -z "${overlap_mm3}" ]; then
        overlap_mm3="0.000000"
    fi

    # 3. 写出结果
    echo "${ID},${overlap_mm3}" > "${OUT_CSV}"

    LOGMSG="[$(date '+%H:%M:%S')] ${ID}: CST_Load=${overlap_mm3} mm³"
    echo "${LOGMSG}"
    echo "${LOGMSG}" >> "${LOG_FILE}"

    # 清理临时文件
    rm -f "${TMP_OVERLAP}"
}
export -f extract_one
export LESION_ROOT CST_TEMPLATE OUT_DIR TMPDIR_BASE LOG_FILE

# ══════════════════════════════════════════════════════════════
# Step 1: 收集被试 ID
# ══════════════════════════════════════════════════════════════
echo ""
echo "[Step 1/3] 扫描被试..."

all_ids=()
for d in ${LESION_ROOT}/*/; do
    [ -d "$d" ] || continue
    id=$(basename "$d")
    if [ -f "${d}/lesion_MNI.nii.gz" ]; then
        all_ids+=("$id")
    fi
done

echo "  找到 ${#all_ids[@]} 个有 lesion_MNI.nii.gz 的被试"

# 过滤已完成
todo_ids=()
done_count=0
for id in "${all_ids[@]}"; do
    result="${OUT_DIR}/${id}_cst_load.csv"
    if [ -f "$result" ] && ! grep -qE "MISSING|ZERO|ERROR" "$result" 2>/dev/null; then
        done_count=$((done_count + 1))
    else
        todo_ids+=("$id")
    fi
done

echo "  已完成: ${done_count}"
echo "  待处理: ${#todo_ids[@]}"

# ══════════════════════════════════════════════════════════════
# Step 2: 并行提取
# ══════════════════════════════════════════════════════════════
if [ ${#todo_ids[@]} -eq 0 ]; then
    echo "  全部已完成，跳到汇总阶段"
else
    echo ""
    echo "[Step 2/3] 并行提取 CST_Load... (${NJOBS} 并发)"
    echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    printf '%s\n' "${todo_ids[@]}" | xargs -P ${NJOBS} -I {} bash -c 'extract_one "$@"' _ {}

    echo ""
    echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
fi

# ══════════════════════════════════════════════════════════════
# Step 3: 汇总 CSV
# ══════════════════════════════════════════════════════════════
echo ""
echo "[Step 3/3] 汇总结果..."

echo "ID,CST_Load" > "${FINAL_CSV}"

success=0
failed=0
for id in "${all_ids[@]}"; do
    result="${OUT_DIR}/${id}_cst_load.csv"
    if [ -f "$result" ] && ! grep -qE "MISSING|ZERO|ERROR" "$result" 2>/dev/null; then
        cat "$result" >> "${FINAL_CSV}"
        success=$((success + 1))
    else
        failed=$((failed + 1))
    fi
done

echo ""
echo "============================================================"
echo "  ✅ CST_Load 提取完成"
echo "============================================================"
echo "  成功: ${success} 个被试"
echo "  失败/跳过: ${failed} 个被试"
echo ""
echo "  汇总文件: ${FINAL_CSV}"
echo "  总行数: $(wc -l < "${FINAL_CSV}") (含表头)"
echo ""
echo "  数据预览 (前 5 行):"
head -6 "${FINAL_CSV}" | column -t -s,
echo ""
echo "============================================================"
echo "  下一步: 合并到主数据表"
echo "  python3 merge_cst_to_master.py"
echo "============================================================"

# 清理 tmp 目录
rm -rf "${TMPDIR_BASE}"
