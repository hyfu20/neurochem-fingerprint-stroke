#!/usr/bin/env bash
# ============================================================================
# render_all_tables_on_server.sh
# ----------------------------------------------------------------------------
# 在远端服务器上一键完成：
#   1) Table 1 目录       → 渲染 table1_baseline.csv
#   2) publication_ready/ → 修正命名 + 渲染所有 Table_*.csv
#   3) 把两边的图汇总到 /data/.../8.tables/all_table_images/
#   4) 生成一份 All_Tables_For_Supervisor.pdf
#
# 用法：直接 bash render_all_tables_on_server.sh
# ============================================================================

set -euo pipefail

# ---- 路径（按你的服务器结构来）-----------------------------------------------
TABLE1_DIR="/data/usersdir/liuzhengxin/Stepbystep/8.tables/table1_results"
PUB_DIR="/data/usersdir/liuzhengxin/Stepbystep/8.tables/othertable/publication_ready"
SUMMARY_DIR="/data/usersdir/liuzhengxin/Stepbystep/8.tables/all_table_images"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_FIX="${SCRIPT_DIR}/fix_table_naming.py"
PY_RENDER="${SCRIPT_DIR}/tables_to_images.py"

mkdir -p "${SUMMARY_DIR}"

echo "================================================================"
echo " STEP 1  修正 publication_ready 里的表格编号"
echo "================================================================"
if [[ -d "${PUB_DIR}" ]]; then
    python "${PY_FIX}" "${PUB_DIR}"
else
    echo "  ⚠️ ${PUB_DIR} 不存在，跳过"
fi

echo
echo "================================================================"
echo " STEP 2  渲染 Table 1（基线表）"
echo "================================================================"
python "${PY_RENDER}" "${TABLE1_DIR}" --out "${TABLE1_DIR}/figures_tables"

echo
echo "================================================================"
echo " STEP 3  渲染 publication_ready 下所有补充表与主表 2/3/4"
echo "================================================================"
if [[ -d "${PUB_DIR}" ]]; then
    python "${PY_RENDER}" "${PUB_DIR}" --out "${PUB_DIR}/figures_tables"
fi

echo
echo "================================================================"
echo " STEP 4  汇总到 ${SUMMARY_DIR}"
echo "================================================================"
# 拷贝 PNG/PDF，不要 All_Tables.pdf（一会儿重新合）
find "${TABLE1_DIR}/figures_tables" "${PUB_DIR}/figures_tables" \
    -maxdepth 1 -type f \( -name "Table*.png" -o -name "table*.png" \
                          -o -name "Table*.pdf" -o -name "table*.pdf" \) \
    -not -name "All_Tables*.pdf" 2>/dev/null \
    -exec cp -v {} "${SUMMARY_DIR}/" \;

# 用 pypdf 合并成一份给导师看
python - <<PYEOF
from pathlib import Path
summary = Path("${SUMMARY_DIR}")
try:
    from pypdf import PdfWriter
except ImportError:
    try:
        from PyPDF2 import PdfWriter
    except ImportError:
        print("⚠️ pypdf 未安装，无法合并 PDF。可执行: pip install pypdf")
        raise SystemExit(0)

# 按主表（Table_1..4）→ 补充表（Table_S1..S5）的顺序合并
def sort_key(p):
    n = p.stem.lower()
    if n.startswith("table1_baseline"):
        return (0, 1, n)
    if n.startswith("table_") and len(n) > 6 and n[6].isdigit():
        return (0, int(n[6]) + 1, n)
    if n.startswith("table_s"):
        # 取 S 后面的数字
        try:
            num = int(n.split("_s", 1)[1].split("_")[0])
            return (1, num, n)
        except Exception:
            return (1, 99, n)
    return (2, 0, n)

pdfs = sorted(summary.glob("*.pdf"), key=sort_key)
print(f"待合并 PDF: {len(pdfs)} 个")
for p in pdfs:
    print(f"  + {p.name}")

if pdfs:
    w = PdfWriter()
    for p in pdfs:
        w.append(str(p))
    out = summary / "All_Tables_For_Supervisor.pdf"
    with open(out, "wb") as f:
        w.write(f)
    print(f"\n📚 合并完成: {out}")
PYEOF

echo
echo "================================================================"
echo " ✅ 全部完成"
echo "   单张图位置:   ${SUMMARY_DIR}/"
echo "   汇总 PDF:    ${SUMMARY_DIR}/All_Tables_For_Supervisor.pdf"
echo "================================================================"
