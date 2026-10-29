#!/usr/bin/env python3
"""改写知识库「待确认」小节里的某一条（按子串定位）。

用法：
    .venv/bin/python scripts/_kb_replace_pending.py "文档名" "旧条目片段" "新条目全文"
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.store import DocumentStore  # noqa: E402


def main() -> int:
    doc_id, needle, new_item = sys.argv[1], sys.argv[2], sys.argv[3]
    store = DocumentStore()
    doc = store.load(doc_id)
    if doc is None:
        print(f"✗ 找不到文档: {doc_id}")
        return 1
    kb = doc.get("knowledge_base") or ""
    lines = kb.splitlines()
    hit = [i for i, l in enumerate(lines)
           if l.strip().startswith("-") and needle in l]
    if not hit:
        print(f"✗ 没找到含「{needle}」的待确认条目")
        return 1
    if len(hit) > 1:
        print(f"✗ 命中 {len(hit)} 条，无法确定改哪条")
        return 1
    old = lines[hit[0]]
    lines[hit[0]] = new_item if new_item.startswith("-") else "- " + new_item
    fixed = "\n".join(lines)
    doc["knowledge_base"] = fixed
    store.save(doc)
    kbmd = config.OUTPUT_DIR / doc_id / f"{doc_id}.kb.md"
    if kbmd.exists():
        kbmd.write_text(fixed if fixed.endswith("\n") else fixed + "\n",
                        encoding="utf-8")
        print(f"  ✓ 已改 {kbmd.name}")
    print(f"  - {old}")
    print(f"  + {lines[hit[0]]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
