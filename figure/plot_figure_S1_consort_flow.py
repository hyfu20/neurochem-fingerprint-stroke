#!/usr/bin/env python3
"""
==========================================================
Supplementary Figure 1 — CONSORT/STROBE 患者筛选流程图
==========================================================
白底、专业排版的患者筛选流程图，适合顶刊投稿。
需要你根据实际数据填入准确数字。

运行:  python3 figure_consort_flow.py
输出:  FigureS1_CONSORT_flow.png / .tiff  (300 dpi)
==========================================================
"""

import os
import sys
import math
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# 字体
# ============================================================
_script_dir = os.path.dirname(os.path.abspath(__file__))

_bold_candidates = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
    '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
]
_regular_candidates = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
    '/System/Library/Fonts/Supplemental/Arial.ttf',
]

def _find(candidates):
    for fp in candidates:
        if os.path.exists(fp):
            return fp
    return None

_bold_path = _find(_bold_candidates)
_regular_path = _find(_regular_candidates)

def get_font(size, bold=True):
    path = _bold_path if bold else (_regular_path or _bold_path)
    try:
        return ImageFont.truetype(path, size) if path else ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()

# ============================================================
# 流程图数据 — 请根据实际数据修改数字！
# ============================================================
# 每个 box: {"text": 显示文字, "n": 样本量, "type": "main"/"exclude"}
# "main" = 主流程框(蓝色边框)
# "exclude" = 右侧排除框(红色边框)

FLOW = [
    # Step 0: CNSR-III 总注册
    {"text": "Patients enrolled in the CNSR-III registry\n201 hospitals across China, Aug 2015 – Mar 2018",
     "n": "N = 15,166", "type": "main"},

    # Step 1: 临床排除（仅保留与本研究相关的）
    {"exclude_text": "Clinical exclusion:\n"
     "\u2022 Transient ischemic attack  (n = 1,020)\n"
     "\u2022 Hemorrhagic transformation  (n = 193)\n"
     "\u2022 Stroke recurrence within 3 months  (n = 944)",
     "type": "exclude"},

    # Step 1 result
    {"text": "Patients with first-ever acute ischemic stroke",
     "n": "N = 13,009", "type": "main"},

    # Step 2: NIHSS 筛选
    {"exclude_text": "Stroke severity selection (n = 8,959):\n"
     "\u2022 NIHSS < 5 or > 15\n"
     "\u2022 Missing available DWI",
     "type": "exclude"},

    # Step 2 result
    {"text": "Moderate stroke severity\n(admission NIHSS 5\u201315) with available DWI",
     "n": "N = 4,050", "type": "main"},

    # Step 3: 影像排除
    {"exclude_text": "Neuroimaging exclusion (n = 274):\n"
     "\u2022 Failed automated lesion\n"
     "  segmentation (DeepISLES)  (n = 2)\n"
     "\u2022 Failed spatial normalization\n"
     "  to MNI152 (FLIRT QC)  (n = 272)",
     "type": "exclude"},

    # Step 3 result
    {"text": "Patients with successful lesion\nmapping in MNI152 standard space",
     "n": "N = 3,776", "type": "main"},

    # Step 4: 随访排除
    {"exclude_text": "Data completeness exclusion (n = 194):\n"
     "\u2022 Missing 3-month mRS or\n"
     "  baseline clinical data  (n = 194)",
     "type": "exclude"},

    # Final
    {"text": "Final analytical cohort\nComplete imaging + clinical data + 3-month follow-up",
     "n": "N = 3,582", "type": "final"},
]

# ============================================================
# 布局参数
# ============================================================
BOX_W = 480
BOX_H = 80
EXCL_W = 380
EXCL_H = 120
V_GAP = 50          # 主框间距
H_OFFSET = 280      # 排除框右偏
MARGIN = 80

# 计算画布
main_boxes = [f for f in FLOW if f["type"] in ("main", "final")]
excl_boxes = [f for f in FLOW if f["type"] == "exclude"]
n_main = len(main_boxes)

CANVAS_W = MARGIN + BOX_W + H_OFFSET + EXCL_W + MARGIN
CANVAS_H = MARGIN + n_main * (BOX_H + V_GAP) + MARGIN + 40

# 颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_GRAY = (50, 50, 50)
MID_GRAY = (120, 120, 120)
BLUE_BORDER = (41, 98, 166)
BLUE_FILL = (235, 243, 252)
RED_BORDER = (200, 50, 50)
RED_FILL = (255, 240, 240)
GREEN_BORDER = (39, 139, 71)
GREEN_FILL = (232, 248, 237)

# ============================================================
# 绘图
# ============================================================
def draw_box(drw, x, y, w, h, text, n_text, border_color, fill_color,
             f_text, f_bold):
    """绘制圆角矩形框 + 文字"""
    drw.rounded_rectangle([(x, y), (x + w, y + h)], radius=12,
                          outline=border_color, width=2, fill=fill_color)
    # 文字
    lines = text.split('\n')
    total_lines = len(lines) + (1 if n_text else 0)
    line_h = f_text.size + 4
    start_y = y + (h - total_lines * line_h) // 2

    for i, line in enumerate(lines):
        bbox = drw.textbbox((0, 0), line, font=f_text)
        tw = bbox[2] - bbox[0]
        drw.text((x + (w - tw) // 2, start_y + i * line_h),
                 line, fill=DARK_GRAY, font=f_text)

    if n_text:
        bbox = drw.textbbox((0, 0), n_text, font=f_bold)
        tw = bbox[2] - bbox[0]
        drw.text((x + (w - tw) // 2, start_y + len(lines) * line_h),
                 n_text, fill=BLACK, font=f_bold)


def draw_arrow_v(drw, x, y1, y2, color=DARK_GRAY, width=2, head=10):
    """垂直箭头"""
    drw.line([(x, y1), (x, y2)], fill=color, width=width)
    angle = math.pi / 2  # 向下
    for s in [-1, 1]:
        a = angle + s * 0.4
        drw.line([(x, y2), (int(x - head * math.cos(a)),
                              int(y2 - head * math.sin(a)))], fill=color, width=width)


def draw_arrow_h(drw, x1, y, x2, color=DARK_GRAY, width=2, head=10):
    """水平箭头"""
    drw.line([(x1, y), (x2, y)], fill=color, width=width)
    for s in [-1, 1]:
        a = s * 0.4
        drw.line([(x2, y), (int(x2 - head * math.cos(a)),
                              int(y - head * math.sin(a)))], fill=color, width=width)


def main():
    canvas = Image.new('RGB', (CANVAS_W, CANVAS_H), WHITE)
    drw = ImageDraw.Draw(canvas)

    f_text = get_font(17, bold=False)
    f_bold = get_font(19, bold=True)
    f_excl = get_font(15, bold=False)
    f_excl_title = get_font(15, bold=True)
    f_caption = get_font(16, bold=True)

    main_x = MARGIN + (BOX_W // 2) - (BOX_W // 2)
    main_cx = main_x + BOX_W // 2

    main_idx = 0
    excl_idx = 0
    positions = []  # (cx, bottom_y) for each main box

    y = MARGIN
    for item in FLOW:
        if item["type"] in ("main", "final"):
            border = GREEN_BORDER if item["type"] == "final" else BLUE_BORDER
            fill = GREEN_FILL if item["type"] == "final" else BLUE_FILL
            draw_box(drw, main_x, y, BOX_W, BOX_H,
                     item["text"], item["n"], border, fill, f_text, f_bold)
            positions.append((main_cx, y, y + BOX_H))

            # 向下箭头到下一个 main box
            if item["type"] != "final":
                draw_arrow_v(drw, main_cx, y + BOX_H, y + BOX_H + V_GAP,
                             color=DARK_GRAY, width=2)

            y += BOX_H + V_GAP
            main_idx += 1

        elif item["type"] == "exclude":
            # 排除框在右侧，与前一个 main box 的底部对齐
            prev_cx, prev_top, prev_bottom = positions[-1]
            excl_x = main_x + BOX_W + 60
            excl_y = prev_bottom - 10

            # 排除框可能更高
            excl_lines = item["exclude_text"].split('\n')
            excl_h = max(EXCL_H, len(excl_lines) * 20 + 20)

            drw.rounded_rectangle(
                [(excl_x, excl_y), (excl_x + EXCL_W, excl_y + excl_h)],
                radius=10, outline=RED_BORDER, width=2, fill=RED_FILL)

            # 排除框文字
            for i, line in enumerate(excl_lines):
                font = f_excl_title if i == 0 else f_excl
                drw.text((excl_x + 15, excl_y + 12 + i * 20),
                         line, fill=RED_BORDER, font=font)

            # 水平箭头: main → exclude
            arrow_y = prev_bottom + V_GAP // 2
            # 从主线拉一条水平线到排除框
            mid_y = (prev_bottom + excl_y + excl_h // 2) // 2
            drw.line([(main_cx, prev_bottom + V_GAP // 2 - 5),
                      (excl_x - 5, prev_bottom + V_GAP // 2 - 5)],
                     fill=RED_BORDER, width=1)
            # 箭头
            ax = excl_x - 5
            ay = prev_bottom + V_GAP // 2 - 5
            for s in [-1, 1]:
                a = s * 0.4
                drw.line([(ax, ay), (int(ax - 8 * math.cos(a)),
                                      int(ay - 8 * math.sin(a)))],
                         fill=RED_BORDER, width=1)

    # 底部 caption
    cap = ("Supplementary Figure 1  |  Patient screening and enrollment flowchart. "
           "From the CNSR-III registry (N = 15,166), patients were sequentially excluded for TIA, "
           "hemorrhagic transformation, and early stroke recurrence. Patients with moderate stroke severity "
           "(admission NIHSS 5-15) and available DWI underwent automated lesion segmentation (DeepISLES) "
           "and spatial normalization to MNI152 (FLIRT). The final analytical cohort comprised 3,582 patients "
           "with complete neuroimaging, clinical, and 3-month follow-up data.")
    drw.text((MARGIN, CANVAS_H - 35), cap, fill=MID_GRAY, font=f_caption)

    # 保存
    out_path = os.path.join(_script_dir, 'FigureS1_CONSORT_flow.png')
    canvas.save(out_path, dpi=(300, 300), quality=95)
    print(f"\n✅ CONSORT flow diagram: {out_path}")

    try:
        tiff_path = os.path.join(_script_dir, 'FigureS1_CONSORT_flow.tiff')
        canvas.save(tiff_path, dpi=(300, 300), compression='tiff_lzw')
        print(f"   TIFF: {tiff_path}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
