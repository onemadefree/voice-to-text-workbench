#!/usr/bin/env python3
"""6 篇稿件的交付前核对：润色覆盖率、错位检测、知识库结构、导出文本。

用法：
    python scripts/verify_all.py            # 全部文档
    python scripts/verify_all.py 某个文档名   # 只看某篇
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.llm import _aligned, _numbers_preserved  # noqa: E402
from app.store import segment_final_text  # noqa: E402

KB_SECTIONS = ("## 摘要", "## 要点", "## 术语表", "## 待确认")
THRESHOLD = 0.30


def check(doc_id: str) -> bool:
    path = config.OUTPUT_DIR / doc_id / "transcript.json"
    if not path.exists():
        print(f"{doc_id}: 没有 transcript.json")
        return False
    doc = json.loads(path.read_text(encoding="utf-8"))
    segs = doc.get("segments") or []
    kb = doc.get("knowledge_base") or ""

    polished = [s for s in segs if (s.get("polished") or "").strip()]
    bad = [i for i, s in enumerate(segs)
           if s.get("polished") and not _aligned(s.get("raw") or "", s.get("polished") or "", THRESHOLD)]
    # 数字保真：润色结果不能把原文的年份读丢（如 二零二零年 → 二零年）
    lost_num = [i for i, s in enumerate(segs)
                if s.get("polished") and not _numbers_preserved(s.get("raw") or "",
                                                               s.get("polished") or "")]
    # 参考信息：模型原样返回的段落数（原文本来就干净时是正常的，不作为错误判据）
    echo = sum(1 for s in segs
               if s.get("polished") and (s.get("polished") or "").strip() == (s.get("raw") or "").strip())
    missing_kb = [sec for sec in KB_SECTIONS if sec not in kb]

    txt = config.OUTPUT_DIR / doc_id / f"{doc_id}.txt"
    txt_ok = txt.exists() and txt.stat().st_size > 0
    txt_lines = 0
    if txt_ok:
        txt_lines = len([ln for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()])

    ok = (len(polished) == len(segs) and len(segs) > 0
          and not bad and not lost_num and not missing_kb and kb.strip() and txt_ok)

    print(f"{'✅' if ok else '⚠️ '} {doc_id}")
    print(f"    段数 {len(segs)}｜已润色 {len(polished)}｜错位 {len(bad)}｜丢数字 {len(lost_num)}"
          f"｜原样返回 {echo}"
          f"｜知识库 {len(kb)} 字符｜缺小节 {missing_kb or '无'}"
          f"｜导出 {'有' if txt_ok else '无'}({txt_lines} 行)")
    if bad:
        for i in bad[:3]:
            print(f"      错位 #{i} 原文：{(segs[i].get('raw') or '')[:50]}")
            print(f"             当前：{(segs[i].get('polished') or '')[:50]}")
    if lost_num:
        for i in lost_num[:3]:
            print(f"      丢数字 #{i} 原文：{(segs[i].get('raw') or '')[:50]}")
            print(f"              当前：{(segs[i].get('polished') or '')[:50]}")
    return ok


def main() -> int:
    ids = sys.argv[1:]
    if not ids:
        ids = sorted(d.name for d in config.OUTPUT_DIR.iterdir()
                     if d.is_dir() and (d / "transcript.json").exists())
    results = [check(i) for i in ids]
    bad = sum(1 for r in results if not r)
    print(f"\n共 {len(results)} 篇，全部通过 {len(results) - bad} 篇，待处理 {bad} 篇")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
