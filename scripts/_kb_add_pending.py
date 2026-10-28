#!/usr/bin/env python3
"""往知识库的「待确认」小节补一条指定条目（已存在就不重复加）。

用法：
    .venv/bin/python scripts/_kb_add_pending.py "文档名" "待确认条目内容"
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.store import DocumentStore  # noqa: E402


def main() -> int:
    doc_id, item = sys.argv[1], sys.argv[2]
    store = DocumentStore()
    doc = store.load(doc_id)
    if doc is None:
        print(f"✗ 找不到文档: {doc_id}")
        return 1
    kb = doc.get("knowledge_base") or ""
    if not kb.strip():
        print("✗ 知识库为空")
        return 1
    if item in kb:
        print("· 已存在，未重复添加")
        return 0

    m = re.search(r"(?m)^##\s*待确认\s*$", kb)
    if not m:
        print("✗ 没有「## 待确认」小节")
        return 1
    head, tail = kb[:m.end()], kb[m.end():]
    tail = tail.rstrip("\n")
    # 插在小节最前面（待确认按重要性排，最要紧的在前）
    new_kb = f"{head}\n- {item}\n{tail}\n"

    doc["knowledge_base"] = new_kb
    store.save(doc)
    kbmd = config.OUTPUT_DIR / doc_id / f"{doc_id}.kb.md"
    if kbmd.exists():
        kbmd.write_text(new_kb, encoding="utf-8")
        print(f"  ✓ 已改 {kbmd.name}")
    print(f"  已添加: {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
