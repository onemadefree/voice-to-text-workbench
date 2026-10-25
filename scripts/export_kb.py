#!/usr/bin/env python3
"""把每篇的知识库导出成独立 Markdown 文件（output/<文档名>/<文档名>.kb.md）。

用法：
    python scripts/export_kb.py            # 全部文档
    python scripts/export_kb.py 某个文档名  # 只导出某篇
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402


def export(doc_id: str) -> bool:
    d = config.OUTPUT_DIR / doc_id
    path = d / "transcript.json"
    if not path.exists():
        print(f"  ✗ {doc_id}: 没有 transcript.json")
        return False
    doc = json.loads(path.read_text(encoding="utf-8"))
    kb = (doc.get("knowledge_base") or "").strip()
    if not kb:
        print(f"  ✗ {doc_id}: 知识库为空")
        return False
    out = d / f"{doc_id}.kb.md"
    out.write_text(kb if kb.endswith("\n") else kb + "\n", encoding="utf-8")
    print(f"  ✓ {doc_id}.kb.md（{len(kb)} 字符）")
    return True


def main() -> int:
    ids = sys.argv[1:]
    if not ids:
        ids = sorted(p.name for p in config.OUTPUT_DIR.iterdir()
                     if p.is_dir() and (p / "transcript.json").exists())
    ok = sum(1 for i in ids if export(i))
    print(f"\n共 {len(ids)} 篇，成功 {ok} 篇")
    return 0 if ok == len(ids) else 1


if __name__ == "__main__":
    raise SystemExit(main())
