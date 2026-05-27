#!/bin/bash
# ==============================================================================
# 图谱 1mm 标准化转换脚本
# 针对 NT 连续密度图谱使用 trilinear，针对解剖 Mask 使用 nearestneighbour
# ==============================================================================
set -euo pipefail

# 路径配置
DEST_DIR="/data/usersdir/liuzhengxin/Stepbystep/6.Neurotransmitter Mapping/human_CHA"
REF_1MM="/home/liuzhengxin/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"
NT_SRC_DIR="/data/usersdir/liuzhengxin/Stepbystep/6.Neurotransmitter Mapping/Neurotransmitters' white matter mapping unveils the neurochemical fingerprints of stroke"
CHOL_SRC_DIR="/data/usersdir/liuzhengxin/cholinergic_project"

mkdir -p "$DEST_DIR"

# 检查参考图
[ -f "$REF_1MM" ] || { echo "ERROR: 参考图不存在"; exit 1; }

echo ">>>> 开始 1mm 重采样任务 <<<<"

# ==============================================================================
# 步骤 1: 处理递质图谱 (根据 Hansen/Koch 论文，这是连续密度图，用 trilinear)
# ==============================================================================
echo "正在转换递质图谱 (使用线性插值)..."
# 使用 find 处理可能存在的空格路径
find "$NT_SRC_DIR" -maxdepth 1 -name "functionnectome_anat_*.nii.gz" | while read -r file; do
    name=$(basename "$file" .nii.gz)
    out_file="$DEST_DIR/${name}_1mm.nii.gz"

    if [ ! -f "$out_file" ]; then
        echo "  Processing: $name"
        flirt -in "$file" \
              -ref "$REF_1MM" \
              -applyxfm -init "$FSLDIR/etc/flirtsch/ident.mat" \
              -interp trilinear \
              -out "$out_file"
    fi
done

# ==============================================================================
# 步骤 2: 处理胆碱能/解剖图谱 (这些是 Mask/Labels，用 nearestneighbour)
# ==============================================================================
echo ""
echo "正在转换胆碱能及掩码图谱 (使用最近邻插值)..."
CHOL_FILES=(
    "human_CHA_2mm.nii.gz"
    "mask_Lateral_Path.nii.gz"
    "mask_Medial_Path.nii.gz"
    "mask_JHU_EC_2mm.nii.gz"
)

for file_name in "${CHOL_FILES[@]}"; do
    src_path="$CHOL_SRC_DIR/$file_name"
    if [ -f "$src_path" ]; then
        name=$(basename "$file_name" .nii.gz)
        out_file="$DEST_DIR/${name}_1mm.nii.gz"

        if [ ! -f "$out_file" ]; then
            echo "  Processing Mask: $name"
            flirt -in "$src_path" \
                  -ref "$REF_1MM" \
                  -applyxfm -init "$FSLDIR/etc/flirtsch/ident.mat" \
                  -interp nearestneighbour \
                  -out "$out_file"
        fi
    else
        echo "  Warning: 未找到文件 $file_name"
    fi
done

echo ""
echo ">>>> 全部转换完成！输出目录: $DEST_DIR <<<<"
ls -lh "$DEST_DIR"
