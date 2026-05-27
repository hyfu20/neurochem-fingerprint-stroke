#!/usr/bin/env python3
"""
==========================================================
Figure 1 — Publication-quality Pipeline Figure  (v3)
==========================================================
白底、流程箭头、步骤标注、大量留白，顶刊投稿级。

布局:
  Row 1 (4 panels, pipeline):
    A. DWI (b=1000)  →  B. T1-weighted  →  C. Lesion segmentation  →  D. MNI152 space
                                                   ↓ dashed (Lesion in MNI)
  Row 2 (3 panels, centered, with breathing room):
                      E. NAT       E'. α4β2       E''. 5-HT₁ₐ
                                                              } Neurochemical
                                                                fingerprinting

- 删掉 BET skull-stripping（Methods 细节，不上 Fig 1）
- 面板内原始白色小字母用黑色遮罩覆盖
- 右侧括号文字完整不截断
- 300 dpi, ~180 mm 宽

运行:  python3 figure1_combine.py
==========================================================
"""

import os
import sys
import math
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# 1. 路径自动搜索
# ============================================================
_script_dir = os.path.dirname(os.path.abspath(__file__))
_panel_candidates = [
    os.path.join(_script_dir, 'figure1_panels'),
    os.path.join(_script_dir, '..', 'figure1_panels'),
    '/data/usersdir/liuzhengxin/Stepbystep/7.writefigure1/figure1_panels',
    '/data/usersdir/liuzhengxin/Stepbystep/figure1_panels',
]
panel_dir = None
for _pd in _panel_candidates:
    if os.path.isdir(_pd):
        panel_dir = _pd
        break
if panel_dir is None:
    print("\u26a0\ufe0f 未找到 figure1_panels 文件夹")
    print(f"  搜索过: {_panel_candidates}")
    sys.exit(1)
print(f"\u2705 面板目录: {panel_dir}")

# ============================================================
# 2. 字体搜索
# ============================================================
_bold_candidates = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
    '/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf',
    '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
]
_regular_candidates = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
    '/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf',
    '/System/Library/Fonts/Supplemental/Arial.ttf',
]
_italic_candidates = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf',
    '/System/Library/Fonts/Supplemental/Arial Italic.ttf',
]

def _find(candidates):
    for fp in candidates:
        if os.path.exists(fp):
            return fp
    return None

_bold_path = _find(_bold_candidates)
_regular_path = _find(_regular_candidates)
_italic_path = _find(_italic_candidates)

def get_font(size, bold=True):
    path = _bold_path if bold else (_regular_path or _bold_path)
    try:
        return ImageFont.truetype(path, size) if path else ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()

def get_italic_font(size):
    path = _italic_path or _regular_path or _bold_path
    try:
        return ImageFont.truetype(path, size) if path else ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()

# ============================================================
# 3. 面板配置 — 科学准确版 (3×2 对称布局)
# ============================================================
#  Row 1: A → B → C  (删掉T1，追踪病灶视觉流)
ROW1 = [
    {"file": "A1_DWI_b0.png",          "label": "A",  "title": "DWI (b = 1000)",
     "desc": "Diffusion-weighted imaging\naxial view, native space"},
    {"file": "B1_lesion_native.png",   "label": "B",  "title": "Lesion segmentation",
     "desc": "Automated infarct\nsegmentation (DeepISLES v2)"},
    {"file": "C1_lesion_MNI.png",      "label": "C",  "title": "MNI152 space",
     "desc": "Lesion mask registered to\nstandard space (FLIRT)"},
]

#  Row 2: D, D', D''  (3 个NT图谱，与Row1完美对齐)
ROW2 = [
    {"file": "D1_NT_NAT_overlay.png",  "label": "D",          "title": "NAT",
     "desc": "Noradrenaline transporter\ndensity map"},
    {"file": "D2_NT_A4B2_overlay.png", "label": "D\u2032",    "title": "\u03b14\u03b22",
     "desc": "Nicotinic acetylcholine\nreceptor density map"},
    {"file": "D3_NT_5HT1a_overlay.png","label": "D\u2033",    "title": "5-HT\u2081\u2090",
     "desc": "Serotonin receptor\ndensity map"},
]

# ============================================================
# 4. 布局参数
# ============================================================
CELL_W, CELL_H = 500, 500         # 3列图更大更清晰
H_GAP = 120                        # 水平箭头区（更宽敞）
V_GAP = 190                        # 行间距
MARGIN_L = 70
MARGIN_R = 280                     # 右侧留空给大括号+文字
MARGIN_T = 55
MARGIN_B = 65
LABEL_H = 58                       # 标签+标题高度
DESC_H = 60                        # 描述高度

# 两行都是 3 列 — 完美 3×2 对称
R1_COLS = 3
R1_W = R1_COLS * CELL_W + (R1_COLS - 1) * H_GAP
R2_COLS = 3
R2_W = R2_COLS * CELL_W + (R2_COLS - 1) * H_GAP

CANVAS_W = MARGIN_L + R1_W + MARGIN_R
ROW_H = LABEL_H + CELL_H + DESC_H
CANVAS_H = MARGIN_T + ROW_H + V_GAP + ROW_H + MARGIN_B

# 颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_GRAY = (50, 50, 50)
MID_GRAY = (120, 120, 120)
LIGHT_GRAY = (210, 210, 210)
BLUE_ACCENT = (41, 98, 166)       # 虚线箭头
BRACKET_CLR = (70, 70, 70)

# 原始面板图内白色标签的遮罩尺寸 (px) — 加大以完全覆盖旧字母
MASK_W, MASK_H = 160, 110

# ============================================================
# 5. 工具函数
# ============================================================
def r1_cell_xy(col):
    """Row 1 面板左上角"""
    x = MARGIN_L + col * (CELL_W + H_GAP)
    y = MARGIN_T + LABEL_H
    return x, y

def r2_cell_xy(col):
    """Row 2 面板左上角（与 Row 1 完美对齐）"""
    x = MARGIN_L + col * (CELL_W + H_GAP)
    y = MARGIN_T + ROW_H + V_GAP + LABEL_H
    return x, y

def draw_arrow(drw, x1, y1, x2, y2, color=DARK_GRAY, width=3, head=14):
    drw.line([(x1, y1), (x2, y2)], fill=color, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    for s in [-1, 1]:
        a = angle + s * 0.4
        drw.line([(x2, y2), (int(x2 - head * math.cos(a)),
                              int(y2 - head * math.sin(a)))], fill=color, width=width)

def draw_dashed_arrow(drw, x1, y1, x2, y2, color=DARK_GRAY, width=2,
                      dash=10, gap=7, head=13):
    length = math.hypot(x2 - x1, y2 - y1)
    angle = math.atan2(y2 - y1, x2 - x1)
    ca, sa = math.cos(angle), math.sin(angle)
    pos = 0
    while pos < length - head:
        end = min(pos + dash, length - head)
        drw.line([(int(x1 + pos * ca), int(y1 + pos * sa)),
                  (int(x1 + end * ca), int(y1 + end * sa))],
                 fill=color, width=width)
        pos = end + gap
    for s in [-1, 1]:
        a = angle + s * 0.4
        drw.line([(x2, y2), (int(x2 - head * math.cos(a)),
                              int(y2 - head * math.sin(a)))], fill=color, width=width)

def text_cx(drw, cx, y, text, font, fill):
    bbox = drw.textbbox((0, 0), text, font=font)
    drw.text((cx - (bbox[2] - bbox[0]) // 2, y), text, fill=fill, font=font)

def multiline_cx(drw, cx, y, text, font, fill, spacing=4):
    for i, line in enumerate(text.split('\n')):
        text_cx(drw, cx, y + i * (font.size + spacing), line, font, fill)

def place_panel(canvas, drw, p, cx, cy, f_label, f_title, f_desc):
    """放置一个面板：标签、图片（带遮罩）、描述"""
    # —— 标签 ——
    drw.text((cx, cy - LABEL_H + 2), p["label"], fill=BLACK, font=f_label)
    lbl_bbox = drw.textbbox((0, 0), p["label"], font=f_label)
    lbl_w = lbl_bbox[2] - lbl_bbox[0]
    drw.text((cx + lbl_w + 14, cy - LABEL_H + 12), p["title"],
             fill=DARK_GRAY, font=f_title)

    # —— 图片 ——
    fpath = os.path.join(panel_dir, p["file"])
    if os.path.exists(fpath):
        img = Image.open(fpath).convert('RGB')
        img.thumbnail((CELL_W - 4, CELL_H - 4), Image.LANCZOS)
        iw, ih = img.size
        px = cx + (CELL_W - iw) // 2
        py = cy + (CELL_H - ih) // 2
        # 用黑色矩形覆盖原始面板图左上角的白色小字母
        img_draw = ImageDraw.Draw(img)
        img_draw.rectangle([(0, 0), (MASK_W, MASK_H)], fill=(0, 0, 0))
        # 细边框 + 贴图
        drw.rectangle([(px - 2, py - 2), (px + iw + 1, py + ih + 1)],
                      outline=LIGHT_GRAY, width=1)
        canvas.paste(img, (px, py))
    else:
        drw.rectangle([(cx, cy), (cx + CELL_W, cy + CELL_H)],
                      outline=LIGHT_GRAY, width=2, fill=(245, 245, 245))
        text_cx(drw, cx + CELL_W // 2, cy + CELL_H // 2,
                f"[{p['file']}]", f_desc, MID_GRAY)
        print(f"  \u26a0\ufe0f 缺少: {p['file']}")

    # —— 描述文字 ——
    multiline_cx(drw, cx + CELL_W // 2, cy + CELL_H + 8,
                 p["desc"], f_desc, MID_GRAY)

# ============================================================
# 6. 主绘图
# ============================================================
def main():
    canvas = Image.new('RGB', (CANVAS_W, CANVAS_H), WHITE)
    drw = ImageDraw.Draw(canvas)

    f_label = get_font(36, bold=True)
    f_title = get_font(22, bold=True)
    f_desc  = get_font(17, bold=False)
    f_arrow = get_font(15, bold=False)
    f_arrow_b = get_font(15, bold=True)
    f_bracket = get_font(19, bold=True)
    f_bracket_sm = get_font(16, bold=False)
    f_caption = get_font(17, bold=True)

    # ── Row 1: A → B → C  (3 panels) ──
    for col, p in enumerate(ROW1):
        cx, cy = r1_cell_xy(col)
        place_panel(canvas, drw, p, cx, cy, f_label, f_title, f_desc)

    # Row 1 水平箭头 + 步骤标注 (只有 2 个箭头)
    arrow_labels = ["", "Deep learning\nsegmentation\n(DeepISLES)",
                    "Spatial normalization\n(FLIRT)"]
    for col in range(R1_COLS - 1):
        x1, y1 = r1_cell_xy(col)
        x2, _  = r1_cell_xy(col + 1)
        ax1, ax2 = x1 + CELL_W + 8, x2 - 8
        ay = y1 + CELL_H // 2
        draw_arrow(drw, ax1, ay, ax2, ay, color=DARK_GRAY, width=3)
        mid_x = (ax1 + ax2) // 2
        lbl = arrow_labels[col + 1]
        if lbl:
            multiline_cx(drw, mid_x, ay - 40, lbl, f_arrow, MID_GRAY)

    # ── Row 2: D, D', D'' (3 panels, 与 Row 1 完美对齐) ──
    for col, p in enumerate(ROW2):
        cx, cy = r2_cell_xy(col)
        place_panel(canvas, drw, p, cx, cy, f_label, f_title, f_desc)

    # Row 2: 不画水平箭头！三个NT系统是平行的，不是序列

    # ── 垂直虚线箭头: C (MNI) → 分叉 → 每个 D 面板 ──
    cx_c, cy_c = r1_cell_xy(2)   # Panel C (MNI space)
    c_bottom = cy_c + CELL_H + DESC_H + 8
    c_cx = cx_c + CELL_W // 2

    # 分叉点 y 坐标
    e0x, e0y = r2_cell_xy(0)
    e2x, _ = r2_cell_xy(2)
    fork_y = c_bottom + 18

    # C 底部中心 → 分叉线
    draw_dashed_arrow(drw, c_cx, c_bottom + 2, c_cx, fork_y,
                      color=BLUE_ACCENT, width=2, head=0)

    # 水平分叉线（横跨三个 D 面板）
    drw.line([(e0x + CELL_W // 2, fork_y), (e2x + CELL_W // 2, fork_y)],
             fill=BLUE_ACCENT, width=2)

    # 从分叉线垂直向下到每个 D 面板
    for col in range(R2_COLS):
        ex, ey = r2_cell_xy(col)
        e_top = ey - LABEL_H + 2
        arr_x = ex + CELL_W // 2
        draw_dashed_arrow(drw, arr_x, fork_y, arr_x, e_top,
                          color=BLUE_ACCENT, width=2)

    # 分叉线旁标注
    drw.text((e2x + CELL_W // 2 + 15, fork_y - 20),
             "Lesion mask in MNI space", fill=BLUE_ACCENT, font=f_arrow_b)

    # ── 右侧大括号: D D' D'' → 汇总标注 ──
    e2x, e2y = r2_cell_xy(2)
    bk_x = e2x + CELL_W + 25
    bk_top = e2y + 40
    bk_bot = e2y + CELL_H - 40
    bk_mid = (bk_top + bk_bot) // 2
    bk_w = 20
    # 上钩
    drw.line([(bk_x, bk_top), (bk_x + bk_w // 2, bk_top)], fill=BRACKET_CLR, width=2)
    drw.line([(bk_x + bk_w // 2, bk_top), (bk_x + bk_w // 2, bk_mid - 14)],
             fill=BRACKET_CLR, width=2)
    # 尖
    drw.line([(bk_x + bk_w // 2, bk_mid - 14), (bk_x + bk_w, bk_mid)],
             fill=BRACKET_CLR, width=2)
    drw.line([(bk_x + bk_w, bk_mid), (bk_x + bk_w // 2, bk_mid + 14)],
             fill=BRACKET_CLR, width=2)
    # 下钩
    drw.line([(bk_x + bk_w // 2, bk_mid + 14), (bk_x + bk_w // 2, bk_bot)],
             fill=BRACKET_CLR, width=2)
    drw.line([(bk_x, bk_bot), (bk_x + bk_w // 2, bk_bot)], fill=BRACKET_CLR, width=2)

    # 括号右侧文字（完整不截断）
    txt_x = bk_x + bk_w + 14
    drw.text((txt_x, bk_mid - 38), "Neurochemical", fill=BRACKET_CLR, font=f_bracket)
    drw.text((txt_x, bk_mid - 14), "fingerprinting", fill=BRACKET_CLR, font=f_bracket)
    drw.text((txt_x, bk_mid + 14), "across 17 receptor/", fill=BRACKET_CLR, font=f_bracket_sm)
    drw.text((txt_x, bk_mid + 34), "transporter systems", fill=BRACKET_CLR, font=f_bracket_sm)

    # ── 底部 Figure caption ──
    cap = ("Figure 1  |  High-throughput neurochemical lesion mapping pipeline and case illustration. "
           "(A) Raw DWI (b = 1000) of a representative patient with a strategic small infarct (TLV = 816 mm\u00b3). "
           "(B) Automated ischemic core segmentation via the DeepISLES ensemble model (red contour). "
           "(C) Lesion mask registered to MNI152 standard space using cost-function masking (FLIRT). "
           "(D\u2013D\u2033) Individual neurochemical fingerprinting: the standardized lesion mask (cyan contour) is "
           "spatially intersected with normative PET-derived neurotransmitter density maps (inferno colormap) "
           "for the noradrenergic (NAT), cholinergic (\u03b14\u03b22), and serotonergic (5-HT\u2081\u2090) systems. "
           "Despite the minimal lesion volume, the infarct strategically transects high-density hubs of the "
           "noradrenergic and cholinergic networks, while the serotonergic system is relatively spared.")
    # 自动折行
    words = cap.split()
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        bbox = drw.textbbox((0, 0), test, font=f_desc)
        if bbox[2] - bbox[0] > CANVAS_W - MARGIN_L - 80:
            lines.append(cur)
            cur = w
        else:
            cur = test
    if cur:
        lines.append(cur)
    cap_y = CANVAS_H - MARGIN_B + 5
    for i, line in enumerate(lines):
        text_cx(drw, CANVAS_W // 2, cap_y + i * 22, line, f_desc, MID_GRAY)

    # ── 保存 ──
    out_path = os.path.join(_script_dir, 'Figure1_combined.png')
    canvas.save(out_path, dpi=(300, 300), quality=95)
    print(f"\n\u2705 Figure 1 已保存: {out_path}")
    print(f"   尺寸: {CANVAS_W}\u00d7{CANVAS_H} px @ 300 dpi")
    print(f"   \u2248 {CANVAS_W / 300 * 25.4:.0f} \u00d7 {CANVAS_H / 300 * 25.4:.0f} mm")

    try:
        tiff_path = os.path.join(_script_dir, 'Figure1_combined.tiff')
        canvas.save(tiff_path, dpi=(300, 300), compression='tiff_lzw')
        print(f"   TIFF: {tiff_path}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
