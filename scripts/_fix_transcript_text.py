#!/usr/bin/env python3
"""按听原音核对的结论，定点修正逐字稿里的 ASR 错字。

背景（2026-10-29 听原音核实）：
    逐字稿第 20 行「刚才郭老师介绍过。其实我蒋一元也做了。」
    原音是「刚才…老师介绍过我，其实我讲义也做了，到时候会发给大家」。
    「讲义」（课件）被 ASR 听成了「蒋一元」——它不是人名，
    知识库模型却把它当成了讲者姓名，写进了摘要和要点。

用法：
    .venv/bin/python scripts/_fix_transcript_text.py "文档名" 段号 原文片段 正确文本
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.store import DocumentStore  # noqa: E402


def main() -> int:
    doc_id, idx_s, needle, correct = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    store = DocumentStore()
    doc = store.load(doc_id)
    if doc is None:
        print(f"✗ 找不到文档: {doc_id}")
        return 1
    segs = doc.get("segments") or []
    if not (0 <= idx_s < len(segs)):
        print(f"✗ 段号越界: {idx_s}（共 {len(segs)} 段）")
        return 1
    seg = segs[idx_s]
    changed: list[str] = []
    for field in ("raw", "clean", "polished"):
        old = seg.get(field)
        if isinstance(old, str) and needle in old:
            seg[field] = old.replace(needle, correct)
            changed.append(f"  {field}: {old[:60]} → {seg[field][:60]}")
    if not changed:
        print(f"✗ 段 {idx_s} 里没有「{needle}」")
        for field in ("raw", "clean", "polished"):
            print(f"    {field} = {seg.get(field)!r}")
        return 1
    store.save(doc)

    # 同步导出 .txt（逐字稿导出物：只有正文，不带知识库）
    from app.store import segment_final_text
    body = "\n".join(segment_final_text(s) for s in segs)
    txt = config.OUTPUT_DIR / doc_id / f"{doc_id}.txt"
    txt.write_text(body, encoding="utf-8")
    print(f"  ✓ 已改 {len(changed)} 个字段：")
    for c in changed:
        print(c)
    print(f"  ✓ 已同步导出 {txt.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
