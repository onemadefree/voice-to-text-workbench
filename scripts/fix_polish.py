#!/usr/bin/env python3
"""检查并修复润色结果：把与原文不对应（模型漏行导致错位）的段落重新润色。

用法：
    python scripts/fix_polish.py                 # 只检查
    python scripts/fix_polish.py --fix           # 重新润色错位段落
    python scripts/fix_polish.py --fix --kb      # 修复后重建知识库
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.llm import _aligned  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.store import DocumentStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("doc_id", nargs="?", default=None, help="文档 id（默认取 source 下第一个）")
    ap.add_argument("--fix", action="store_true", help="重新润色错位段落")
    ap.add_argument("--kb", action="store_true", help="修复后重建知识库")
    ap.add_argument("--threshold", type=float, default=0.30)
    args = ap.parse_args()

    store = DocumentStore()
    doc_id = args.doc_id
    if not doc_id:
        ids = store.list_ids()
        if not ids:
            print("没有可用的转写文档")
            return 1
        doc_id = ids[0]
    doc = store.load(doc_id)
    if doc is None:
        print(f"找不到文档: {doc_id}")
        return 1

    segs = doc.get("segments") or []
    bad = [
        i for i, s in enumerate(segs)
        if s.get("polished") and not _aligned(s.get("raw") or "", s.get("polished") or "", args.threshold)
    ]
    print(f"文档 {doc_id}：共 {len(segs)} 段，错位 {len(bad)} 段（阈值 {args.threshold}）")
    for i in bad[:5]:
        print(f"  #{i} 原文：{segs[i]['raw'][:60]}")
        print(f"       当前：{segs[i]['polished'][:60]}")

    if not args.fix:
        return 0

    pipeline = Pipeline(store)
    print(f"\n重新润色 {len(bad)} 段…")
    pipeline.run_polish(doc_id, indices=bad, force_polish=True)

    doc = store.load(doc_id) or {}
    still = [
        i for i, s in enumerate(doc.get("segments") or [])
        if s.get("polished") and not _aligned(s.get("raw") or "", s.get("polished") or "", args.threshold)
    ]
    print(f"修复后仍错位：{len(still)} 段")

    if args.kb:
        print("重建知识库…")
        pipeline.run_knowledge_base(doc_id, title=doc.get("source", ""))
        print("知识库完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
