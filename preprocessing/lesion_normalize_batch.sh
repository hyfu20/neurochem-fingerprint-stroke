#!/bin/bash
# ============================================================
#  批量病灶 → MNI 标准化：自动跳过已完成的被试，支持断点续跑
#  用法: bash lesion_normalize_batch.sh <OUT_ROOT> [并行数, 默认4]
#
#  输出文件 (全部在 OUT_ROOT 下)：
#    pipeline_all.log   — 所有被试的详细运行日志 (实时)
#    master_log.tsv     — 每个被试一行的汇总表
#    all_results.csv    — 成功被试的 ID + MNI 病灶体积
# ============================================================

OUT_ROOT=${1:?  "用法: $0 <OUT_ROOT> [并行数]"}
NJOBS=${2:-4}

DWI_ROOT="/data/usersdir/liuzhengxin/Stepbystep/4.deepisles_script/deepisles_ORG"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GLOBAL_LOG="${OUT_ROOT}/pipeline_all.log"

mkdir -p "${OUT_ROOT}"

# --- 初始化全局日志 ---
echo "================================================================" >> "${GLOBAL_LOG}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 批量任务启动, 并行数=${NJOBS}"      >> "${GLOBAL_LOG}"
echo "================================================================" >> "${GLOBAL_LOG}"

# --- 收集所有被试 ID ---
all_ids=( $(ls -d ${DWI_ROOT}/*/ 2>/dev/null | xargs -I{} basename {}) )
echo "总被试数: ${#all_ids[@]}"

# --- 过滤：跳过已有正常结果的 ---
todo_ids=()
done_count=0
fail_count=0

for id in "${all_ids[@]}"; do
    result="${OUT_ROOT}/${id}/result_line.txt"
    if [ -f "$result" ]; then
        if grep -qE "MISSING|FAILED" "$result" 2>/dev/null; then
            rm -f "$result" "${OUT_ROOT}/${id}/lesion_MNI.nii.gz"
            todo_ids+=("$id")
            fail_count=$((fail_count + 1))
        else
            done_count=$((done_count + 1))
        fi
    else
        todo_ids+=("$id")
    fi
done

echo "已完成: ${done_count}"
echo "之前失败需重跑: ${fail_count}"
echo "待处理: ${#todo_ids[@]}"
echo "并行数: ${NJOBS}"
echo "-----------------------------------"

if [ ${#todo_ids[@]} -eq 0 ]; then
    echo "全部完成，无需处理！"
    exit 0
fi

echo "实时日志: tail -f ${GLOBAL_LOG}"
echo "-----------------------------------"

# --- 并行执行 ---
printf '%s\n' "${todo_ids[@]}" | \
    xargs -P ${NJOBS} -I {} bash "${SCRIPT_DIR}/lesion_normalize_single.sh" {} "${OUT_ROOT}"

# --- 汇总报告 ---
echo ""
echo "============ 汇总报告 ============"
total=$(ls -d ${OUT_ROOT}/*/ 2>/dev/null | wc -l)
success=$(grep -rL "MISSING\|FAILED" ${OUT_ROOT}/*/result_line.txt 2>/dev/null | wc -l)
input_miss=$(grep -rl "INPUT_MISSING" ${OUT_ROOT}/*/result_line.txt 2>/dev/null | wc -l)
failed=$(grep -rl "FAILED" ${OUT_ROOT}/*/result_line.txt 2>/dev/null | wc -l)
no_result=$(for d in ${OUT_ROOT}/*/; do [ ! -f "$d/result_line.txt" ] && echo "$d"; done | wc -l)

echo "总目录数:    ${total}"
echo "成功:        ${success}"
echo "输入缺失:    ${input_miss}"
echo "处理失败:    ${failed}"
echo "无结果文件:  ${no_result}"
echo ""

if [ "$failed" -gt 0 ]; then
    echo "--- 失败的被试 ---"
    grep -rl "FAILED" ${OUT_ROOT}/*/result_line.txt | while read f; do
        id=$(basename $(dirname "$f"))
        reason=$(cat "$f" | cut -d, -f2)
        echo "  ${id}: ${reason}"
    done
fi

# --- 生成 master_log.tsv ---
MASTER_LOG="${OUT_ROOT}/master_log.tsv"
echo -e "ID\t状态\t病灶体积mm3\t错误原因" > "${MASTER_LOG}"

for d in ${OUT_ROOT}/*/; do
    [ ! -d "$d" ] && continue
    id=$(basename "$d")
    result="$d/result_line.txt"

    if [ ! -f "$result" ]; then
        echo -e "${id}\t无结果\t-\t未生成result_line.txt" >> "${MASTER_LOG}"
    elif grep -q "INPUT_MISSING" "$result" 2>/dev/null; then
        echo -e "${id}\t输入缺失\t-\tINPUT_MISSING" >> "${MASTER_LOG}"
    elif grep -q "FAILED" "$result" 2>/dev/null; then
        reason=$(cut -d, -f2 "$result")
        echo -e "${id}\t失败\t-\t${reason}" >> "${MASTER_LOG}"
    else
        vol=$(cut -d, -f2 "$result")
        echo -e "${id}\t成功\t${vol}\t-" >> "${MASTER_LOG}"
    fi
done

# 排序：失败 > 输入缺失 > 无结果 > 成功
head -1 "${MASTER_LOG}" > "${MASTER_LOG}.tmp"
tail -n +2 "${MASTER_LOG}" | sort -t$'\t' -k2,2 -k1,1 >> "${MASTER_LOG}.tmp"
mv "${MASTER_LOG}.tmp" "${MASTER_LOG}"

echo ""
echo "========= 输出文件 ========="
echo "  详细日志: ${GLOBAL_LOG}"
echo "  汇总表:   ${MASTER_LOG}"
echo ""
echo "  成功: $(grep -c '成功' "${MASTER_LOG}") 个"
echo "  失败: $(grep -c '失败' "${MASTER_LOG}") 个"
echo "  输入缺失: $(grep -c '输入缺失' "${MASTER_LOG}") 个"
echo "  无结果: $(grep -c '无结果' "${MASTER_LOG}") 个"

# 合并成功结果 (带表头)
echo ""
header="ID,lesion_volume_mm3"
echo "${header}" > "${OUT_ROOT}/all_results.csv"
grep -rL "MISSING\|FAILED" ${OUT_ROOT}/*/result_line.txt 2>/dev/null | \
    sort | xargs cat >> "${OUT_ROOT}/all_results.csv"
echo "数值结果: ${OUT_ROOT}/all_results.csv ($(wc -l < "${OUT_ROOT}/all_results.csv") 行)"
echo ""
echo "快速查看:"
echo "  tail -f ${GLOBAL_LOG}                          # 实时看日志"
echo "  grep ERROR ${GLOBAL_LOG}                       # 只看报错"
echo "  grep '某个ID' ${GLOBAL_LOG}                    # 看某个被试"
echo "  column -t -s \$'\\t' ${MASTER_LOG} | head -20   # 看汇总表"
