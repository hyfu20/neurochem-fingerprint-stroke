#!/bin/bash
set -uo pipefail

# ============================================================
#  批量提取递质加权受损负荷 (Weighted Lesion Load)
#  基于 Koch et al. (2025, Brain) 方法学
#
#  用法: bash extract_weighted_load_batch.sh <LESION_ROOT> <OUT_DIR> [并行数]
#
#  说明:
#    LESION_ROOT: 包含 {ID}/lesion_MNI.nii.gz 的根目录
#                 (来自 lesion_normalize_batch.sh 的输出)
#    OUT_DIR:     输出目录，最终生成 all_neurochem_loads.csv
#    并行数:      默认 8
#
#  输出:
#    OUT_DIR/all_neurochem_loads.csv — 完整数据表 (用于后续 Python 分析)
#    OUT_DIR/{ID}_neurochem_load.csv — 每个被试的单独结果
#    OUT_DIR/extract_load.log       — 运行日志
# ============================================================

LESION_ROOT=${1:?  "用法: $0 <LESION_ROOT> <OUT_DIR> [并行数]"}
OUT_DIR=${2:?  "用法: $0 <LESION_ROOT> <OUT_DIR> [并行数]"}
NJOBS=${3:-8}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- 环境 ---
export FSLDIR=${FSLDIR:-/home/liuzhengxin/fsl}
. "${FSLDIR}/etc/fslconf/fsl.sh"
export PATH="${FSLDIR}/bin:${PATH}"

ATLAS_DIR="/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/human_CHA"

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "  递质加权受损负荷提取 (Koch et al., 2025)"
echo "============================================================"
echo "  病灶输入:  ${LESION_ROOT}"
echo "  输出目录:  ${OUT_DIR}"
echo "  图谱目录:  ${ATLAS_DIR}"
echo "  并行数:    ${NJOBS}"
echo "============================================================"

# --- 1. 收集被试 ID ---
all_ids=()
for d in ${LESION_ROOT}/*/; do
    [ -d "$d" ] || continue
    id=$(basename "$d")
    if [ -f "${d}/lesion_MNI.nii.gz" ]; then
        all_ids+=("$id")
    fi
done

echo "找到 ${#all_ids[@]} 个已有 lesion_MNI.nii.gz 的被试"

# --- 2. 过滤已完成的 ---
todo_ids=()
done_count=0
for id in "${all_ids[@]}"; do
    result="${OUT_DIR}/${id}_neurochem_load.csv"
    if [ -f "$result" ]; then
        # 检查是否是有效结果 (不包含 MISSING/ZERO/NO_ATLAS)
        if grep -qE "MISSING|ZERO|NO_ATLAS" "$result" 2>/dev/null; then
            rm -f "$result"
            todo_ids+=("$id")
        else
            done_count=$((done_count + 1))
        fi
    else
        todo_ids+=("$id")
    fi
done

echo "已完成: ${done_count}"
echo "待处理: ${#todo_ids[@]}"
echo "-----------------------------------"

if [ ${#todo_ids[@]} -eq 0 ]; then
    echo "全部提取完成，跳过到汇总阶段"
else
    echo "开始并行提取... (日志: ${OUT_DIR}/extract_load.log)"
    echo ""

    # --- 3. 并行执行 ---
    printf '%s\n' "${todo_ids[@]}" | \
        xargs -P ${NJOBS} -I {} bash "${SCRIPT_DIR}/extract_weighted_load_single.sh" {} "${LESION_ROOT}" "${OUT_DIR}"
fi

# --- 4. 汇总 CSV ---
echo ""
echo "============ 汇总结果 ============"

# 动态生成表头: 从图谱文件名提取递质名称
header="ID,TLV_mm3"

atlas_files=( $(find "${ATLAS_DIR}" -maxdepth 1 -name "functionnectome_anat_*_1mm.nii.gz" 2>/dev/null | sort) )
for atlas in "${atlas_files[@]}"; do
    nt_name=$(basename "${atlas}" .nii.gz | sed 's/functionnectome_anat_/Load_/' | sed 's/_1mm//')
    header="${header},${nt_name}"
done

# 胆碱能通路列
header="${header},Load_CHA,Load_Lateral_Path,Load_Medial_Path"

# 写表头
FINAL_CSV="${OUT_DIR}/all_neurochem_loads.csv"
echo "${header}" > "${FINAL_CSV}"

# 合并所有有效结果
success=0
failed=0
for id in "${all_ids[@]}"; do
    result="${OUT_DIR}/${id}_neurochem_load.csv"
    if [ -f "$result" ] && ! grep -qE "MISSING|ZERO|NO_ATLAS" "$result" 2>/dev/null; then
        cat "$result" >> "${FINAL_CSV}"
        success=$((success + 1))
    else
        failed=$((failed + 1))
    fi
done

echo ""
echo "成功提取: ${success} 个被试"
echo "失败/跳过: ${failed} 个被试"
echo ""
echo "输出文件: ${FINAL_CSV}"
echo "总行数: $(wc -l < "${FINAL_CSV}") (含表头)"
echo ""
echo "表头预览:"
head -1 "${FINAL_CSV}"
echo ""
echo "数据预览 (前5行):"
head -6 "${FINAL_CSV}" | column -t -s,
echo ""
echo "============================================================"
echo "  下一步: 运行残差分析"
echo "  python3 residual_analysis.py ${FINAL_CSV} clinical_data.csv"
echo "============================================================"
