#!/usr/bin/env python3
"""
==========================================================
Figure 1 Pipeline 截图生成器
==========================================================
用法: 在云电脑上运行
    python3 generate_figure1_panels.py <SUBJECT_ID>

会生成 figure1_panels/ 文件夹，包含：
  A1_DWI_b0.png          — 原始 DWI b=0
  A2_DWI_b1000.png       — 原始 DWI b=1000
  A3_T1w.png             — 原始 T1w
  B1_lesion_native.png   — DeepISLES 分割结果叠加
  C1_lesion_MNI.png      — MNI 空间病灶 (配准后)
  D1_NT_NAT_overlay.png  — NAT 密度图 + 病灶轮廓
  D2_NT_A4B2_overlay.png — A4B2 密度图 + 病灶轮廓
  D3_NT_5HT1a_overlay.png— 5HT1a 密度图 + 病灶轮廓

然后下载到本地，用 PowerPoint / BioRender 拼成 Figure 1

依赖: nibabel, matplotlib, numpy
    pip install nibabel matplotlib numpy
==========================================================
"""

import sys
import os
import numpy as np
from pathlib import Path

try:
    import nibabel as nib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    print("请先安装: pip install nibabel matplotlib numpy")
    sys.exit(1)

# ============================================================
# 路径配置 — 修改这里匹配你的云电脑目录结构
# ============================================================
DWI_ROOT = "/data/usersdir/liuzhengxin/Stepbystep/4.deepisles_script/deepisles_ORG"
# 原始 CNSR3 DWI 数据（可能包含 b=1000）
DWI_RAW_ROOT = "/data/shares/CNSR3/NIFITI/Release"
MNI_ROOT = "/data/usersdir/liuzhengxin/Stepbystep/5.MNI/output"
# 图谱目录 — 自动搜索多个可能路径
_ATLAS_CANDIDATES = [
    "/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/1.atlas/atlas1mm",
    "/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping/atlas1mm",
    "/data/usersdir/liuzhengxin/Stepbystep/6.Neurotransmitter Mapping/1.atlas/atlas1mm",
]
ATLAS_DIR = None
for _p in _ATLAS_CANDIDATES:
    if Path(_p).exists():
        ATLAS_DIR = _p
        break
if ATLAS_DIR is None:
    ATLAS_DIR = _ATLAS_CANDIDATES[0]  # fallback
FSLDIR = os.environ.get("FSLDIR", "/home/liuzhengxin/fsl")
MNI_TEMPLATE = f"{FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz"
T1_ROOT = "/data/shares/CNSR3/NIFITI/Release/CNSR3_all_CodeN_nifti/CNSR3-13012-T1w-nii_v20190625"

# 代表性 NT 图谱 (选 3 张展示)
NT_DISPLAY = {
    "NAT":   "functionnectome_anat_NAT_1mm.nii.gz",
    "A4B2":  "functionnectome_anat_A4B2_1mm.nii.gz",
    "5HT1a": "functionnectome_anat_5HT1a_1mm.nii.gz",
}

# 输出目录 — 保存在脚本所在文件夹的子目录下
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "figure1_panels"


def get_middle_slice(data, axis=2):
    """获取中间层面索引"""
    # 找到有非零值的范围的中间
    nonzero = np.any(data > 0, axis=tuple(i for i in range(3) if i != axis))
    indices = np.where(nonzero)[0]
    if len(indices) == 0:
        return data.shape[axis] // 2
    return indices[len(indices) // 2]


def plot_single_slice(bg_data, slice_idx, ax, cmap='gray', vmin=None, vmax=None):
    """绘制单层轴位切片 — 拉强对比度"""
    slc = bg_data[:, :, slice_idx].T
    valid = slc[slc > 0]
    if vmin is None:
        vmin = np.percentile(valid, 2) if len(valid) > 0 else 0
    if vmax is None:
        vmax = np.percentile(valid, 98) if len(valid) > 0 else 1
    ax.imshow(slc, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax,
              aspect='equal', interpolation='bilinear')
    ax.axis('off')


def plot_overlay(bg_data, overlay_data, slice_idx, ax, 
                 bg_cmap='gray', overlay_cmap='hot', alpha=0.5,
                 contour=False, contour_color='red'):
    """绘制叠加图 — 红色轮廓 + 内部微光"""
    bg_slc = bg_data[:, :, slice_idx].T
    ov_slc = overlay_data[:, :, slice_idx].T
    
    valid = bg_slc[bg_slc > 0]
    vmin = np.percentile(valid, 2) if len(valid) > 0 else 0
    vmax = np.percentile(valid, 98) if len(valid) > 0 else 1
    
    ax.imshow(bg_slc, cmap=bg_cmap, origin='lower', vmin=vmin, vmax=vmax,
              aspect='equal', interpolation='bilinear')
    
    if ov_slc.max() > 0:
        # 内部极轻微的透明红
        masked = np.ma.masked_where(ov_slc <= 0, ov_slc)
        ax.imshow(masked, cmap='Reds', origin='lower', alpha=0.35,
                  vmin=0, vmax=1, aspect='equal', interpolation='nearest')
        # 醒目的正红色轮廓
        ax.contour(ov_slc, levels=[0.5], colors='#E64B35', linewidths=2.0)
    ax.axis('off')


def save_fig(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='black',
                pad_inches=0, edgecolor='none')
    plt.close(fig)
    print(f"  ✅ {name}")


def add_scale_bar(ax, data_shape, voxel_size_mm=1.0, bar_length_mm=20):
    """在右下角添加比例尺（黑底白字）"""
    bar_length_vox = bar_length_mm / voxel_size_mm
    x_start = data_shape[0] * 0.62
    y_start = data_shape[1] * 0.06
    # 半透明黑底背景条
    from matplotlib.patches import FancyBboxPatch
    bg_rect = FancyBboxPatch((x_start - 5, y_start - 3), bar_length_vox + 10, data_shape[1] * 0.08,
                              boxstyle='round,pad=2', facecolor='black', alpha=0.6, edgecolor='none')
    ax.add_patch(bg_rect)
    ax.plot([x_start, x_start + bar_length_vox], [y_start, y_start],
            color='white', linewidth=3, solid_capstyle='butt')
    ax.text(x_start + bar_length_vox / 2, y_start + data_shape[1] * 0.04,
            f'{bar_length_mm} mm', color='white', fontsize=9, ha='center',
            fontweight='bold')


def add_panel_label(ax, label, data_shape):
    """在左上角添加 A/B/C/D 标签（带黑色描边）"""
    import matplotlib.patheffects as pe
    ax.text(data_shape[0] * 0.04, data_shape[1] * 0.94, label,
            color='white', fontsize=22, fontweight='bold',
            va='top', ha='left', fontfamily='sans-serif',
            path_effects=[pe.withStroke(linewidth=3, foreground='black')])


def generate_panels(subject_id):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*50}")
    print(f"  生成 Figure 1 面板: {subject_id}")
    print(f"  输出目录: {OUT_DIR}")
    print(f"  图谱目录: {ATLAS_DIR}")
    if ATLAS_DIR and Path(ATLAS_DIR).exists():
        atlas_files = list(Path(ATLAS_DIR).glob('*.nii.gz'))
        print(f"  图谱数量: {len(atlas_files)} 个 .nii.gz 文件")
        if atlas_files:
            print(f"  示例: {atlas_files[0].name}")
    else:
        print(f"  ⚠️ 图谱目录不存在! 请检查路径")
        print(f"  尝试查找...")
        import subprocess
        result = subprocess.run(['find', '/data/usersdir/liuzhengxin/Stepbystep/6.NeurotransmitterMapping', 
                                 '-name', 'functionnectome_anat_NAT*', '-type', 'f'],
                                capture_output=True, text=True, timeout=10)
        if result.stdout.strip():
            print(f"  找到: {result.stdout.strip().split(chr(10))[0]}")
            print(f"  请修改脚本中的 ATLAS_DIR 路径!")
    print(f"{'='*50}\n")

    # ── A. Input Images ──
    # 先找病灶最大层面，统一用于所有原始空间图片 (A1, A2, B1)
    lesion_native = Path(DWI_ROOT) / subject_id / "results" / "lesion_msk.nii.gz"
    native_slice = None
    if lesion_native.exists():
        les_tmp = nib.load(str(lesion_native)).get_fdata()
        les_sum_tmp = les_tmp.sum(axis=(0, 1))
        native_slice = int(np.argmax(les_sum_tmp)) if les_sum_tmp.max() > 0 else None
        print(f"  原始空间统一层面: Z={native_slice} (病灶最大层)")

    # DWI — 优先用原始 DWI（未去颅骨），显示效果更好
    dwi_bet = Path(DWI_ROOT) / subject_id / "dwi_stripped_bet.nii.gz"
    dwi_raw = Path(DWI_ROOT) / subject_id / "dwi_stripped.nii.gz"
    dwi_display = dwi_raw if dwi_raw.exists() else dwi_bet
    if dwi_display.exists():
        print("[A] 输入图像...")
        img = nib.load(str(dwi_display))
        data = img.get_fdata()
        if data.ndim == 4:
            data = data[..., 0]
        mid = native_slice if native_slice is not None and native_slice < data.shape[2] else get_middle_slice(data)
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), facecolor='black')
        plot_single_slice(data, mid, ax)
        add_panel_label(ax, 'a', data[:,:,mid].T.shape)
        save_fig(fig, "A1_DWI_b0.png")
        
        # b=1000 — 直接从患者文件夹找
        b1000_data = None
        b1000_candidates = [
            Path(DWI_ROOT) / subject_id / "dwi.nii.gz",           # 原始 b=1000 (509KB)
            Path(DWI_ROOT) / subject_id / "DWI.nii.gz",
            Path(DWI_ROOT) / subject_id / "dwi_stripped.nii.gz",   # 可能是4D
        ]
        
        for cand in b1000_candidates:
            if cand.exists():
                tmp = nib.load(str(cand)).get_fdata()
                if tmp.ndim == 4 and tmp.shape[-1] > 1:
                    b1000_data = tmp[..., -1]  # 最后一个volume通常是最高b值
                    print(f"    b=1000 来源: {cand.name} (4D, vol[-1])")
                    break
                elif tmp.ndim == 3 and cand != dwi_display:
                    b1000_data = tmp
                    print(f"    b=1000 来源: {cand.name} (3D)")
                    break
        
        if b1000_data is not None:
            mid_b = get_middle_slice(b1000_data)
            fig, ax = plt.subplots(1, 1, figsize=(5, 5), facecolor='black')
            add_panel_label(ax, 'a\'', b1000_data[:,:,mid_b].T.shape)
            save_fig(fig, "A2_DWI_b1000.png")
        else:
            print("    ⚠️ 未找到 b=1000 DWI，跳过 A2")
            print(f"    搜索过: {[str(c) for c in b1000_candidates[:3]]}")
    else:
        print(f"  ⚠️ 未找到 DWI: {dwi_bet}")

    # T1w
    t1_dir = Path(T1_ROOT) / subject_id
    t1_files = list(t1_dir.glob("*2D-T1w-TRA.nii.gz")) if t1_dir.exists() else []
    if t1_files:
        t1_img = nib.load(str(t1_files[0]))
        t1_data = t1_img.get_fdata()
        mid_t1 = get_middle_slice(t1_data)
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), facecolor='black')
        plot_single_slice(t1_data, mid_t1, ax)
        add_panel_label(ax, 'a\u2033', t1_data[:,:,mid_t1].T.shape)
        save_fig(fig, "A3_T1w.png")
    else:
        print(f"  ⚠️ 未找到 T1w: {t1_dir}")

    # ── A4. BET skull stripping 前后对比 ──
    # T1 原始 vs BET 去颅骨（展示 -f 0.3 -Z 的效果）
    t1_bet_file = Path(MNI_ROOT) / subject_id / "t1_brain.nii.gz"
    if t1_files and t1_bet_file.exists():
        print("[A] BET 去颅骨对比...")
        t1_raw = nib.load(str(t1_files[0])).get_fdata()
        t1_bet_data = nib.load(str(t1_bet_file)).get_fdata()
        mid_bet = get_middle_slice(t1_bet_data)
        # 调整层面以匹配（如果形状不同就用各自中间层）
        mid_raw = min(mid_bet, t1_raw.shape[2] - 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor='black')
        plot_single_slice(t1_raw, mid_raw, axes[0])
        add_panel_label(axes[0], 'T1 raw', t1_raw[:,:,mid_raw].T.shape)
        plot_single_slice(t1_bet_data, mid_bet, axes[1])
        add_panel_label(axes[1], 'BET -f 0.3 -Z', t1_bet_data[:,:,mid_bet].T.shape)
        # 在两图之间加箭头
        fig.text(0.50, 0.50, '→', fontsize=40, ha='center', va='center',
                 color='#E74C3C', fontweight='bold')
        plt.subplots_adjust(wspace=0.12)
        save_fig(fig, "A4_BET_comparison.png")
    elif t1_bet_file.exists():
        # 即使没有原始T1，也单独展示BET结果
        print("[A] BET 结果...")
        t1_bet_data = nib.load(str(t1_bet_file)).get_fdata()
        mid_bet = get_middle_slice(t1_bet_data)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4), facecolor='black')
        plot_single_slice(t1_bet_data, mid_bet, ax)
        add_panel_label(ax, 'BET', t1_bet_data[:,:,mid_bet].T.shape)
        save_fig(fig, "A4_BET_result.png")
    else:
        print("  ⚠️ 未找到 BET 结果（t1_brain.nii.gz），可能已被清理")
        print("  提示：配准脚本默认会删除中间文件，需在 lesion_normalize_single.sh 中保留 t1_brain.nii.gz")

    # ── B. Lesion Segmentation ──
    # 使用与 A 面板相同的层面
    dwi_raw_candidates = [
        Path(DWI_ROOT) / subject_id / "dwi.nii.gz",       # b=1000 作为背景最好
        Path(DWI_ROOT) / subject_id / "dwi_stripped.nii.gz",
        Path(DWI_ROOT) / subject_id / "adc.nii.gz",
        dwi_bet,
    ]
    dwi_bg_file = None
    for c in dwi_raw_candidates:
        if c.exists():
            dwi_bg_file = c
            break
    if lesion_native.exists() and dwi_bg_file is not None:
        print("[B] 病灶分割...")
        les_data = nib.load(str(lesion_native)).get_fdata()
        bg_data = nib.load(str(dwi_bg_file)).get_fdata()
        if bg_data.ndim == 4:
            bg_data = bg_data[..., -1]
        
        # 统一使用 native_slice
        mid_les = native_slice if native_slice is not None else (int(np.argmax(les_data.sum(axis=(0,1)))) if les_data.max() > 0 else bg_data.shape[2] // 2)
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), facecolor='black')
        plot_overlay(bg_data, les_data, mid_les, ax)
        add_panel_label(ax, 'b', bg_data[:,:,mid_les].T.shape)
        save_fig(fig, "B1_lesion_native.png")
    else:
        print(f"  ⚠️ 未找到病灶 mask")

    # ── C. MNI Registration ──
    lesion_mni = Path(MNI_ROOT) / subject_id / "lesion_MNI.nii.gz"
    if lesion_mni.exists() and Path(MNI_TEMPLATE).exists():
        print("[C] MNI 配准...")
        mni_bg = nib.load(MNI_TEMPLATE).get_fdata()
        mni_les = nib.load(str(lesion_mni)).get_fdata()
        
        les_sum = mni_les.sum(axis=(0, 1))
        mid_mni = np.argmax(les_sum) if les_sum.max() > 0 else mni_bg.shape[2] // 2
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), facecolor='black')
        plot_overlay(mni_bg, mni_les, mid_mni, ax)
        slc_shape = mni_bg[:,:,mid_mni].T.shape
        add_panel_label(ax, 'c', slc_shape)
        add_scale_bar(ax, slc_shape, voxel_size_mm=1.0, bar_length_mm=20)
        save_fig(fig, "C1_lesion_MNI.png")
    else:
        print(f"  ⚠️ 未找到 MNI 病灶")

    # ── D. NT Atlas Overlay ──
    if lesion_mni.exists() and Path(MNI_TEMPLATE).exists():
        print("[D] 递质图谱叠加...")
        mni_bg = nib.load(MNI_TEMPLATE).get_fdata()
        mni_les = nib.load(str(lesion_mni)).get_fdata()
        
        les_sum = mni_les.sum(axis=(0, 1))
        mid_mni = np.argmax(les_sum) if les_sum.max() > 0 else mni_bg.shape[2] // 2
        
        for nt_name, nt_file in NT_DISPLAY.items():
            atlas_path = Path(ATLAS_DIR) / nt_file
            if atlas_path.exists():
                atlas_data = nib.load(str(atlas_path)).get_fdata()
                
                # 确保形状匹配
                if atlas_data.shape != mni_bg.shape:
                    print(f"    ⚠️ {nt_name} 形状不匹配，跳过")
                    continue
                
                fig, ax = plt.subplots(1, 1, figsize=(5, 5), facecolor='black')
                
                # 底图：MNI
                bg_slc = mni_bg[:, :, mid_mni].T
                ax.imshow(bg_slc, cmap='gray', origin='lower',
                          vmin=np.percentile(bg_slc[bg_slc>0], 2),
                          vmax=np.percentile(bg_slc[bg_slc>0], 98),
                          aspect='equal', interpolation='bilinear')
                
                # 中层：NT 密度 — 只显示前60%高密度区
                at_slc = atlas_data[:, :, mid_mni].T
                threshold = np.percentile(at_slc[at_slc > 0], 40) if np.any(at_slc > 0) else 0
                masked_at = np.ma.masked_where(at_slc < threshold, at_slc)
                ax.imshow(masked_at, cmap='inferno', origin='lower', alpha=0.75,
                          aspect='equal', interpolation='bicubic')
                
                # 顶层：病灶轮廓 — 亮青色加粗
                les_slc = mni_les[:, :, mid_mni].T
                if les_slc.max() > 0:
                    ax.contour(les_slc, levels=[0.5], colors='#00FFFF', linewidths=2.5)
                
                # panel 标签 + scale bar，不加 colorbar
                d_idx = list(NT_DISPLAY.keys()).index(nt_name)
                d_labels = ['d', 'd\'', 'd\u2033']
                add_panel_label(ax, d_labels[d_idx], bg_slc.shape)
                add_scale_bar(ax, bg_slc.shape, voxel_size_mm=1.0, bar_length_mm=20)
                
                ax.axis('off')
                save_fig(fig, f"D{d_idx+1}_NT_{nt_name}_overlay.png")
            else:
                print(f"    ⚠️ 未找到图谱: {nt_file}")

    print(f"\n{'='*50}")
    print(f"  所有面板已保存到: {OUT_DIR.absolute()}")
    print(f"  下载到本地后用 PowerPoint/BioRender 拼图")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 尝试自动找一个有完整数据的被试
        print("用法: python3 generate_figure1_panels.py <SUBJECT_ID>")
        print("")
        print("建议选一个病灶在外囊/基底节区域的典型 MCA 患者")
        print("让图看起来最有说服力")
        sys.exit(1)
    
    sid = sys.argv[1]
    generate_panels(sid)
