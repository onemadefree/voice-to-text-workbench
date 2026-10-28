#!/usr/bin/env python3
"""只生成知识库、不写 store 的干跑：验证 no_think 链路端到端能出合格知识库。

用法：
    .venv/bin/python scripts/_kb_dryrun.py "文档名"
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.llm import build_knowledge_base, _kb_complete, _kb_looks_like_kb  # noqa: E402
from app.store import DocumentStore, segment_final_text  # noqa: E402

SECTIONS = ("## 摘要", "## 要点", "## 术语表", "## 待确认")


def main() -> int:
    doc_id = sys.argv[1]
    store = DocumentStore()
    doc = store.load(doc_id)
    if doc is None:
        print(f"✗ 找不到文档: {doc_id}")
        return 1
    segs = doc.get("segments") or []
    text = "\n".join(segment_final_text(s) for s in segs)
    print(f"正文 {len(text)} 字｜{len(segs)} 段", flush=True)

    from app.llm import LocalLLM
    llm = LocalLLM(backend="custom")
    t0 = time.time()
    kb = build_knowledge_base(llm, text, title=doc_id)
    dt = time.time() - t0

    missing = [s for s in SECTIONS if not re.search(r"(?m)^" + re.escape(s) + r"\s*$", kb)]
    has_title = bool(re.search(r"(?m)^#\s+\S", kb))
    print(f"\n耗时 {dt:.0f}s｜{len(kb)} 字符")
    print(f"  缺小节: {missing or '无'}")
    print(f"  有标题: {has_title}")
    print(f"  _kb_complete: {_kb_complete(kb)}")
    print(f"  _kb_looks_like_kb: {_kb_looks_like_kb(kb)}")
    print(f"  「- 无」出现: {kb.count('- 无')} 处")

    out = ROOT / "output" / "_dryrun_kb.md"
    out.write_text(kb, encoding="utf-8")
    print(f"  已写出 {out}")
    return 0 if _kb_complete(kb) else 2


if __name__ == "__main__":
    raise SystemExit(main())
