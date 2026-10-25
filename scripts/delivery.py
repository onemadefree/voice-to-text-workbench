#!/usr/bin/env python3
"""交付清单：列出每篇稿件的转写/润色/知识库/导出文件与体量。

用法：python scripts/delivery.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402

KB_SECTIONS = ("## 摘要", "## 要点", "## 术语表", "## 待确认")


def kb_of(doc_id: str) -> str:
    p = config.OUTPUT_DIR / doc_id / "transcript.json"
    if not p.exists():
        return ""
    return (json.loads(p.read_text(encoding="utf-8")).get("knowledge_base") or "").strip()


def main() -> int:
    ids = sorted(d.name for d in config.OUTPUT_DIR.iterdir()
                 if d.is_dir() and (d / "transcript.json").exists())
    rows = []
    for doc_id in ids:
        doc = json.loads((config.OUTPUT_DIR / doc_id / "transcript.json").read_text(encoding="utf-8"))
        segs = doc.get("segments") or []
        pol = sum(1 for s in segs if (s.get("polished") or "").strip())
        kb = kb_of(doc_id)
        missing = [s for s in KB_SECTIONS if s not in kb]
        txt = config.OUTPUT_DIR / doc_id / f"{doc_id}.txt"
        kbm = config.OUTPUT_DIR / doc_id / f"{doc_id}.kb.md"
        rows.append({
            "文档": doc_id,
            "段数": len(segs),
            "已润色": pol,
            "状态": doc.get("status"),
            "知识库字数": len(kb),
            "缺小节": "、".join(missing) if missing else "无",
            "逐字稿": f"{txt.stat().st_size // 1024}KB" if txt.exists() else "缺",
            "知识库文件": "有" if kbm.exists() else "缺",
        })
    if not rows:
        print("没有文档")
        return 1
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
    total_segs = sum(r["段数"] for r in rows)
    total_pol = sum(r["已润色"] for r in rows)
    print(f"\n共 {len(rows)} 篇，{total_pol}/{total_segs} 段已润色")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
