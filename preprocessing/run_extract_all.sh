#!/bin/bash
# ============================================================
#  三步走: 生成 NT_Imaging_MasterMatrix.csv
#
#  用法: bash run_extract_all.sh [并发数, 默认50]
#
#  Step 1: 生成被试列表
#  Step 2: GNU Parallel 并行提取
#  Step 3: 合并成最终大表
# ============================================================
set -uo pipefail

NJOBS=${1:-50}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LESION_ROOT="/data/usersdir/liuzhengxin/Stepbystep/5.MNI/output"
OUT_ROOT="/data/usersdir/liuzhengxin/Stepbystep/7.Results_NT_Load"
SUBJ_LIST="/data/usersdir/liuzhengxin/Stepbystep/subj_list.txt"
MASTER_CSV="/data/usersdir/liuzhengxin/Stepbystep/NT_Imaging_MasterMatrix.csv"

mkdir -p "${OUT_ROOT}"

echo "============================================================"
echo "  Koch (2025) 加权受损负荷批量提取"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# =====================
# Step 1: 生成被试列表
# =====================
echo ""
echo "[Step 1/3] 生成被试列表..."
ls -d ${LESION_ROOT}/*/ 2>/dev/null | xargs -I{} basename {} > "${SUBJ_LIST}"
total=$(wc -l < "${SUBJ_LIST}")
echo "  找到 ${total} 个被试文件夹"

# 统计有 lesion_MNI.nii.gz 的数量
has_lesion=0
while read id; do
    [ -f "${LESION_ROOT}/${id}/lesion_MNI.nii.gz" ] && has_lesion=$((has_lesion + 1))
done < "${SUBJ_LIST}"
echo "  其中 ${has_lesion} 个有 lesion_MNI.nii.gz"

# 统计已完成的
done_count=$(ls ${OUT_ROOT}/*_nt_load.csv 2>/dev/null | wc -l)
echo "  已完成: ${done_count}"
echo "  待处理: $((has_lesion - done_count)) (估计)"
echo ""

# =====================
# Step 2: 并行提取
# =====================
echo "[Step 2/3] 开始并行提取 (${NJOBS} 并发)..."
echo "  脚本: ${SCRIPT_DIR}/extract_load_v1.sh"
start_time=$(date +%s)

cat "${SUBJ_LIST}" | parallel -j ${NJOBS} bash "${SCRIPT_DIR}/extract_load_v1.sh" {}

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo ""
echo "  提取完成! 耗时: ${elapsed} 秒"

# 统计结果
result_count=$(ls ${OUT_ROOT}/*_nt_load.csv 2>/dev/null | wc -l)
echo "  生成结果文件: ${result_count} 个"

# =====================
# Step 3: 合并大表
# =====================
echo ""
echo "[Step 3/3] 合并成 NT_Imaging_MasterMatrix.csv..."

# 写表头 (与 extract_load_v1.sh 中 atlas_list 顺序严格一致)
echo "ID,TLV,5HT1a,5HT1b,5HT2a,5HT4,5HT6,5HTT,A4B2,D1,D2,DAT,M1,NAT,VAChT,human_CHA,JHU_EC,Lateral_Path,Medial_Path" > "${MASTER_CSV}"

# 追加所有数据 (按 ID 排序)
cat ${OUT_ROOT}/*_nt_load.csv | sort >> "${MASTER_CSV}"

data_rows=$(($(wc -l < "${MASTER_CSV}") - 1))

echo ""
echo "============================================================"
echo "  完成!"
echo "============================================================"
echo "  最终数据表: ${MASTER_CSV}"
echo "  数据行数:   ${data_rows} (不含表头)"
echo "  列数:       19 (ID + TLV + 17 图谱)"
echo ""
echo "  预览 (前5行):"
head -6 "${MASTER_CSV}" | column -t -s,
echo ""
echo "  检查列数一致性:"
awk -F, '{print NF}' "${MASTER_CSV}" | sort | uniq -c | sort -rn | head -5
echo ""
echo "  下一步: 把 NT_Imaging_MasterMatrix.csv 与临床变量表用 ID 合并"
echo "============================================================"
