#!/usr/bin/env python3
"""修掉已生成知识库里标题带的音频后缀（.mp3 之类）。

只改标题行，不动正文；store 和 .kb.md 一起改，保持一致。

用法：
    .venv/bin/python scripts/_kb_fix_kb_title.py "文档名"
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.pipeline import _strip_media_suffix  # noqa: E402
from app.store import DocumentStore  # noqa: E402


def main() -> int:
    doc_id = sys.argv[1]
    store = DocumentStore()
    doc = store.load(doc_id)
    if doc is None:
        print(f"✗ 找不到文档: {doc_id}")
        return 1
    kb = doc.get("knowledge_base") or ""
    if not kb.strip():
        print("✗ 知识库为空")
        return 1
    lines = kb.splitlines()
    m = re.match(r"^(#\s+)(.+?)\s*$", lines[0])
    if not m:
        print(f"✗ 首行不是标题: {lines[0]!r}")
        return 1
    old = m.group(2)
    new = _strip_media_suffix(old)
    if old == new:
        print(f"· 标题本来就干净: {old}")
        return 0
    lines[0] = m.group(1) + new
    fixed = "\n".join(lines)
    doc["knowledge_base"] = fixed
    store.save(doc)
    kbmd = config.OUTPUT_DIR / doc_id / f"{doc_id}.kb.md"
    if kbmd.exists():
        kbmd.write_text(fixed if fixed.endswith("\n") else fixed + "\n",
                        encoding="utf-8")
        print(f"  ✓ 已改 {kbmd.name}")
    print(f"  标题: {old!r} → {new!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
