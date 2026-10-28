#!/usr/bin/env python3
"""端到端：跑着的润色任务被仪表盘暂停 / 恢复 / 取消。

用假的润色实现（慢 2 秒返回）代替真模型，这样测试不用花 token，
但走的还是 pipeline.run_polish 的真实路径：批次边界检查控制信号、
状态落盘、暂停时挂起、恢复后继续、取消后停下。
"""
from __future__ import annotations

import json
import shutil
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app import llm as llm_mod  # noqa: E402
from app import status as st  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.store import DocumentStore  # noqa: E402

DOC = "__测试_控制E2E__"
FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}　{detail}")
        FAILED.append(name)


def _fake_polish(llm, segments, context_head="", batch_size=16, **kw):
    """假的润色：睡 3 秒，按原文返回（够慢，能观察到批次边界）。"""
    time.sleep(3)
    return [(s.get("text") or s.get("raw") or "") for s in segments]


def _stage_of(doc_id: str) -> str:
    for j in st.read_all():
        if j.get("id") == doc_id:
            return j.get("stage") or ""
    return ""


def _wait_stage(doc_id: str, want: str, limit: float = 20.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < limit:
        if _stage_of(doc_id) == want:
            return True
        time.sleep(0.3)
    return False


def run() -> int:
    llm_mod.polish_segments = _fake_polish

    print("· 造一篇测试稿")
    out_dir = config.OUTPUT_DIR / DOC
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "id": DOC,
        "source": "test.mp3",
        "status": "ready",
        "segments": [
            {"id": i, "start": i * 1.0, "end": i * 1.0 + 0.5,
             "raw": f"这是第{i}段测试文本。", "clean": f"这是第{i}段测试文本。"}
            for i in range(600)
        ],
    }
    (out_dir / "transcript.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    store = DocumentStore()
    pipeline = Pipeline(store)
    done = {}

    def worker():
        pipeline.run_polish(DOC, level="standard", force_polish=True, backend="custom")
        done["finished"] = True

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    try:
        print("· 任务跑起来了")
        check("状态是 polishing", _wait_stage(DOC, "polishing"), _stage_of(DOC))

        print("· 暂停")
        st.set_control(DOC, "pause")
        check("任务挂起（状态 paused）", _wait_stage(DOC, "paused", 30), _stage_of(DOC))
        check("挂起时线程还活着（进度不丢）", th.is_alive())

        print("· 恢复")
        st.clear_control(DOC)
        check("恢复后回到 polishing", _wait_stage(DOC, "polishing", 30), _stage_of(DOC))

        print("· 取消")
        st.set_control(DOC, "cancel")
        check("任务停下（状态 cancelled）", _wait_stage(DOC, "cancelled", 30), _stage_of(DOC))
        th.join(timeout=10)
        check("线程结束了", not th.is_alive())
        # 取消掉的那部分不该有润色结果（取消是正常提前返回，不是异常）
        cur = store.load(DOC) or {}
        segs = cur.get("segments") or []
        filled = sum(1 for s in segs if (s.get("polished") or "").strip())
        check("只润色了取消前的部分", 0 < filled < len(segs),
              f"{filled}/{len(segs)}")
    finally:
        st.clear_control(DOC)
        shutil.rmtree(out_dir, ignore_errors=True)
        try:
            st._path(DOC).unlink(missing_ok=True)
        except Exception:
            pass

    print()
    if FAILED:
        print(f"❌ 失败 {len(FAILED)} 项：" + "、".join(FAILED))
        return 1
    print("✅ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
