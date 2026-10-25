#!/usr/bin/env python3
"""定向重刷：修复后重新规整 + 重润色受影响的段落，并重建知识库与导出。

只动「clean 因数字保护而变化」的段落（默认 54 段），不重跑全量。
流程：更新 clean → run_polish(force_polish) → run_knowledge_base → 导出 txt/kb.md

用法：.venv/bin/python scripts/_repolish_targets.py [--no-kb] [文档名 ...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config, textproc  # noqa: E402
from app.llm import load_user_config  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.store import DocumentStore, segment_final_text  # noqa: E402

PLAN = ROOT / "output" / "_repolish_plan.json"


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    no_kb = "--no-kb" in sys.argv
    skip_polish = "--skip-polish" in sys.argv
    if not PLAN.exists():
        print("✗ 缺少重刷计划 output/_repolish_plan.json，先跑 _plan_repolish.py")
        return 1
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if args:
        plan = {k: v for k, v in plan.items() if k in args}

    backend = load_user_config().get("backend") or "minimax"
    model = load_user_config().get("model")
    print(f"后端 {backend} / 模型 {model}\n", flush=True)

    store = DocumentStore()
    pipeline = Pipeline(store)
    for doc_id, idxs in plan.items():
        idxs = [int(i) for i in idxs]
        if not idxs:
            continue
        doc = store.load(doc_id)
        if doc is None:
            print(f"✗ {doc_id}: 找不到文档")
            continue
        segs = doc.get("segments") or []

        # 1) 用新规整逻辑更新受影响的 clean
        changed = 0
        for i in idxs:
            raw = segs[i].get("raw") or ""
            new_clean = textproc.clean_text(raw, level="light")
            if new_clean != (segs[i].get("clean") or ""):
                segs[i]["clean"] = new_clean
                changed += 1
        doc = store.save(doc)
        print(f"=== {doc_id}: 更新 clean {changed}/{len(idxs)} 段，开始重润色 ===", flush=True)

        # 2) 定向重润色
        t0 = time.time()
        if skip_polish:
            print("    跳过重润色（--skip-polish）", flush=True)
        else:
            pipeline.run_polish(doc_id, indices=idxs, level="standard",
                                force_polish=True, backend=backend)
            doc = store.load(doc_id) or {}
            segs = doc.get("segments") or []
            print(f"    重润色 {len(idxs)} 段，耗时 {(time.time() - t0) / 60:.1f} 分钟", flush=True)

        # 3) 重建知识库
        kb = ""
        if not no_kb:
            t1 = time.time()
            pipeline.run_knowledge_base(doc_id, title=doc.get("source", ""), backend=backend)
            doc = store.load(doc_id) or {}
            kb = doc.get("knowledge_base") or ""
            print(f"    知识库 {len(kb)} 字符，耗时 {(time.time() - t1) / 60:.1f} 分钟", flush=True)

        # 4) 导出
        out = config.OUTPUT_DIR / doc_id / f"{doc_id}.txt"
        out.write_text("\n".join(segment_final_text(s) for s in segs), encoding="utf-8")
        if kb:
            kbmd = config.OUTPUT_DIR / doc_id / f"{doc_id}.kb.md"
            kbmd.write_text(kb if kb.endswith("\n") else kb + "\n", encoding="utf-8")
        print(f"    导出 {out.name} / {doc_id}.kb.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
