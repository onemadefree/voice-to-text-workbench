#!/usr/bin/env python3
"""临时脚本：不经服务，直接在本地重刷某篇的润色 + 知识库 + 导出。

用于服务端任务已结束、但需要用新代码（思考块剥离 / token 预算 / 拆半重试）重跑的文档。
走网络 API 后端，不需要 GPU。

用法：.venv/bin/python scripts/_local_repolish.py "<文档名>" ["<文档名>" ...]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.llm import LocalLLM, load_user_config  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.status import job_running  # noqa: E402
from app.store import DocumentStore, segment_final_text  # noqa: E402


def main() -> int:
    doc_ids = sys.argv[1:]
    if not doc_ids:
        print("用法: _local_repolish.py <文档名> [...]")
        return 1
    backend = load_user_config().get("backend") or "minimax"
    model = load_user_config().get("model")
    print(f"后端 {backend} / 模型 {model}\n", flush=True)

    store = DocumentStore()
    pipeline = Pipeline(store)
    for doc_id in doc_ids:
        doc = store.load(doc_id)
        if doc is None:
            print(f"✗ {doc_id}: 找不到文档")
            continue

        # 别的进程正在处理这篇（比如用户自己在网页上点了润色）→ 别碰，
        # 两个进程同时改同一个 transcript.json 会互相覆盖。
        if job_running(doc_id):
            print(f"⏭ {doc_id}: 另一个进程正在处理，跳过")
            continue

        segs = doc.get("segments") or []
        t0 = time.time()
        polished_all = bool(segs) and all((s.get("polished") or "").strip() for s in segs)
        kb_empty = not (doc.get("knowledge_base") or "").strip()
        if polished_all and kb_empty:
            # 已经有人润色完了，只差知识库——别把几小时的润色再跑一遍。
            print(f"=== {doc_id}（{len(segs)} 段）已润色过，只补知识库 ===", flush=True)
        else:
            print(f"=== {doc_id}（{len(segs)} 段）开始重刷 ===", flush=True)
            pipeline.run_polish(doc_id, level="standard", force_polish=True, backend=backend)
        doc = store.load(doc_id) or {}
        segs = doc.get("segments") or []
        echo = sum(1 for s in segs
                   if (s.get("polished") or "").strip() == (s.get("raw") or "").strip())
        print(f"  润色完成 {len(segs)} 段，耗时 {(time.time() - t0) / 60:.1f} 分钟，"
              f"原样未改 {echo} 段（{echo * 100 // max(1, len(segs))}%）", flush=True)

        pipeline.run_knowledge_base(doc_id, title=doc.get("source", ""), backend=backend)
        doc = store.load(doc_id) or {}
        kb = doc.get("knowledge_base") or ""
        print(f"  知识库 {len(kb)} 字符", flush=True)

        out = config.OUTPUT_DIR / doc_id / f"{doc_id}.txt"
        out.write_text("\n".join(segment_final_text(s) for s in doc.get("segments") or []),
                       encoding="utf-8")
        kbmd = config.OUTPUT_DIR / doc_id / f"{doc_id}.kb.md"
        if kb:
            kbmd.write_text(kb if kb.endswith("\n") else kb + "\n", encoding="utf-8")
        print(f"  导出 {out.name} / {kbmd.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
