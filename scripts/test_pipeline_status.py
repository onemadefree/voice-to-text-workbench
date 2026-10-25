#!/usr/bin/env python3
"""自测：流水线跑润色/知识库时会正确上报状态，且不影响原有行为。

后台重刷任务会用到新代码，这里守住两条：
1. 润色照常生效（polished 有内容、进度写回 doc）；
2. 状态文件同步更新（阶段 / 进度 / 结束标记），且结束后是 done。
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config, status as st  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.store import DocumentStore, new_document  # noqa: E402

DOC = "__流水线状态自测__"
FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}　{detail}")
        FAILED.append(name)


class StubLLM:
    """假模型：把每条都改写一遍，格式与真实模型一致。"""

    @property
    def budget_extra(self) -> int:
        return 0

    def complete(self, prompt, system, max_tokens=2048, temperature=0.3) -> str:
        import json
        import re
        pairs = re.findall(r"^(\d+)\. (.+)$", prompt, re.M)
        return "\n".join(
            json.dumps({"i": int(n), "t": f"{t}（已顺句）"}, ensure_ascii=False)
            for n, t in pairs
        )


def main() -> int:
    store = DocumentStore()
    pipe = Pipeline(store)
    out_dir = config.OUTPUT_DIR / DOC
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = new_document(DOC, Path("/tmp/none.wav"), {"duration": 1.0})
    doc["segments"] = [
        {"id": i, "raw": f"这是第{i}段原始转写内容", "clean": f"这是第{i}段原始转写内容"}
        for i in range(5)
    ]
    doc["status"] = "ready"
    store.save(doc)

    # 把 LLM 换掉，不打真实 API
    pipe.get_llm = lambda backend=None: StubLLM()  # type: ignore

    try:
        print("· 润色")
        pipe.run_polish(DOC, level="standard", force_polish=True, backend="stub")
        doc2 = store.load(DOC) or {}
        segs = doc2.get("segments") or []
        polished = [s.get("polished") or "" for s in segs]
        check("全部段落都润色了", all(p for p in polished), str(polished[:2]))
        check("润色内容正确", all("（已顺句）" in p for p in polished), str(polished[:1]))
        check("polish_progress 已清除", "polish_progress" not in doc2, str(doc2.get("polish_progress")))

        print("· 状态文件")
        jobs = [j for j in st.read_all() if j["id"] == DOC]
        check("有状态记录", len(jobs) == 1, str(jobs))
        if jobs:
            j = jobs[0]
            check("kind 是 polish", j["kind"] == "polish", j["kind"])
            check("结束标记已写", j["state"] == "done", j["state"])
            check("进度是满的", j["done"] == 5 and j["total"] == 5, str(j))
            check("有 finished_at", j["finished_at"] is not None, str(j["finished_at"]))

        print("· 知识库")
        # 知识库要调 API，这里只验证状态上报的前置动作不炸
        def fake_kb(llm, text, title=""):
            return "## 摘要\n- 测\n## 要点\n- 测\n## 术语表\n- 测\n## 待确认\n- 无"
        import app.llm as llm_mod
        old = llm_mod.build_knowledge_base
        llm_mod.build_knowledge_base = fake_kb
        try:
            pipe.run_knowledge_base(DOC, title="测试")
        finally:
            llm_mod.build_knowledge_base = old
        jobs = [j for j in st.read_all() if j["id"] == DOC]
        check("知识库阶段已上报", jobs and jobs[0]["kind"] == "kb", str(jobs[:1]))
        check("知识库完成后是 done", jobs and jobs[0]["state"] == "done", str(jobs[:1]))
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
        st._path(DOC).unlink(missing_ok=True)

    if FAILED:
        print(f"\n失败 {len(FAILED)} 项：{FAILED}")
        return 1
    print("\n全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
