#!/usr/bin/env python3
"""
Merge 6 manuscript section .md files into a single
`manuscript/Full_Manuscript.md` ready for md_to_word.py conversion
and Zotero processing.

Order:
  1. Manuscript_FrontMatter.md
  2. Introduction_Section.md
  3. Methods_Section.md
  4. Results_Section.md
  5. Discussion_Section.md
  6. figure/Figure_Legends.md
"""
from __future__ import annotations

import datetime as _dt
import shutil
from pathlib import Path

BASE = Path("/Users/liuzhengxin/VS_code/病灶标准化nmi图谱")
OUT = BASE / "manuscript" / "Full_Manuscript.md"
BACKUP_DIR = BASE / "manuscript" / "_superseded"

SECTIONS = [
    BASE / "Manuscript_FrontMatter.md",
    BASE / "Introduction_Section.md",
    BASE / "Methods_Section.md",
    BASE / "Results_Section.md",
    BASE / "Discussion_Section.md",
    BASE / "figure" / "Figure_Legends.md",
]

SEP = "\n\n---\n\n"


def main() -> None:
    # backup existing
    if OUT.exists():
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M")
        backup = BACKUP_DIR / f"Full_Manuscript_{ts}.md"
        shutil.copy2(OUT, backup)
        print(f"[backup] {OUT.name} -> {backup.relative_to(BASE)}")

    # merge
    chunks: list[str] = []
    for src in SECTIONS:
        if not src.exists():
            print(f"[skip] missing: {src.relative_to(BASE)}")
            continue
        text = src.read_text(encoding="utf-8").rstrip()
        chunks.append(text)
        print(f"[ok]   {src.relative_to(BASE)}  ({len(text.splitlines())} lines)")

    merged = SEP.join(chunks) + "\n"
    OUT.write_text(merged, encoding="utf-8")

    n_lines = len(merged.splitlines())
    n_words = len(merged.split())
    n_chars = len(merged)
    print()
    print(f"[done] {OUT.relative_to(BASE)}")
    print(f"       lines = {n_lines:>6}")
    print(f"       words = {n_words:>6}")
    print(f"       chars = {n_chars:>6}")


if __name__ == "__main__":
    main()
