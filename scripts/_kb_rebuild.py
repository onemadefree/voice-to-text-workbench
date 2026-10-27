#!/usr/bin/env python3
"""重建指定文档的知识库并导出 .kb.md。

用途：知识库被内容审核拦截时会退化成 46 字符的空壳
（「## 摘要 / - 无 / ## 要点 / - 无 / ## 术语表 / - 无 / ## 待确认 / - 无」），
四节齐全但内容全空。这种空壳要重跑才能恢复。

默认走 minimax 后端：历史上它的知识库产出稳定（2700~3800 字符），
内网模型和审核拦截都容易出空壳。

用法：
    .venv/bin/python scripts/_kb_rebuild.py "文档名" ["文档名" ...]
    .venv/bin/python scripts/_kb_rebuild.py --backend custom "文档名"
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.store import DocumentStore  # noqa: E402

SECTIONS = ("## 摘要", "## 要点", "## 术语表", "## 待确认")


def main() -> int:
    args = sys.argv[1:]
    backend = "minimax"
    if args and args[0] == "--backend":
        backend = args[1]
        args = args[2:]
    if not args:
        print(__doc__)
        return 1

    store = DocumentStore()
    pipeline = Pipeline(store)
    rc = 0
    for doc_id in args:
        doc = store.load(doc_id)
        if doc is None:
            print(f"✗ {doc_id}: 找不到文档")
            rc = 1
            continue
        print(f"=== 重建知识库：{doc_id}（后端 {backend}）===", flush=True)
        pipeline.run_knowledge_base(doc_id, title=doc.get("source", ""), backend=backend)
        doc = store.load(doc_id) or {}
        kb = doc.get("knowledge_base") or ""
        missing = [s for s in SECTIONS if s not in kb]
        empty = kb.count("- 无")
        print(f"  {len(kb)} 字符｜缺小节 {missing or '无'}｜「- 无」{empty} 处", flush=True)
        if not kb.strip() or missing or empty >= 3:
            print("  ✗ 仍然退化，需要换后端再试")
            rc = 1
            continue
        kbmd = config.OUTPUT_DIR / doc_id / f"{doc_id}.kb.md"
        kbmd.write_text(kb if kb.endswith("\n") else kb + "\n", encoding="utf-8")
        print(f"  ✓ 导出 {kbmd.name}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
