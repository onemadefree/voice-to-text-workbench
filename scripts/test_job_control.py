#!/usr/bin/env python3
"""自测：作业的模型标注 + 暂停 / 恢复 / 取消。

要点：
1. 状态文件里要带上这项作业用的模型（仪表盘要显示）；
2. 控制信号走文件（output/_ctl/<文档名>.cmd），任务在批次边界读它；
3. pause = 挂起等待（进程不死、进度不丢），cancel = 停下，resume = 接着跑；
4. 控制信号读写绝不能影响主流程。
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import status as st  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.store import DocumentStore  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}　{detail}")
        FAILED.append(name)


def run() -> int:
    doc = "__控制测试文档__"

    print("· 模型标注")
    st.publish(doc, "polish", "polishing", "润色中", 5, 50,
               model="qwen3.8-27b", backend="custom")
    jobs = [j for j in st.read_all() if j["id"] == doc]
    check("状态文件带模型", bool(jobs) and jobs[0]["model"] == "qwen3.8-27b",
          str(jobs[:1]))
    check("状态文件带后端", bool(jobs) and jobs[0]["backend"] == "custom",
          str(jobs[:1]))
    # 后续 publish 不带 model 时要沿用上一个值，不能被抹掉
    st.publish(doc, "polish", "polishing", "润色中", 6, 50)
    jobs = [j for j in st.read_all() if j["id"] == doc]
    check("后续更新不丢模型", jobs[0]["model"] == "qwen3.8-27b", jobs[0]["model"])

    print("· 控制信号读写")
    check("初始没有信号", st.get_control(doc) == "", repr(st.get_control(doc)))
    st.set_control(doc, "pause")
    check("pause 信号写进去了", st.get_control(doc) == "pause", st.get_control(doc))
    check("有信号时 pending 为真", st.control_pending(doc) is True)
    st.clear_control(doc)
    check("清掉后没有信号", st.get_control(doc) == "", repr(st.get_control(doc)))
    try:
        st.set_control(doc, "乱写")
        check("非法动作要报错", False, "居然没报错")
    except ValueError:
        check("非法动作要报错", True)

    print("· paused 状态能被识别")
    st.publish(doc, "polish", "paused", "已暂停，等待恢复", 6, 50)
    jobs = [j for j in st.read_all() if j["id"] == doc]
    check("state = paused", jobs[0]["state"] == "paused", jobs[0]["state"])

    print("· 管道层的控制闸")
    pipeline = Pipeline(DocumentStore())
    # 没有信号 → 直接放行
    check("无信号时放行", pipeline._control_gate(doc, "polish") == "",
          repr(pipeline._control_gate(doc, "polish")))

    # pause：任务线程挂起，等控制信号被清掉才继续
    st.set_control(doc, "pause")
    result = {}

    def worker():
        result["action"] = pipeline._control_gate(doc, "polish")

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    time.sleep(1.0)
    check("pause 后任务挂起（线程还活着）", th.is_alive())
    jobs = [j for j in st.read_all() if j["id"] == doc]
    check("挂起时状态是 paused", jobs[0]["state"] == "paused", jobs[0]["state"])

    # 恢复
    st.clear_control(doc)
    th.join(timeout=5)
    check("恢复后任务继续", (not th.is_alive()) and result.get("action") == "",
          f"alive={th.is_alive()} result={result}")

    print("· cancel：任务停下")
    st.set_control(doc, "cancel")
    check("cancel 被识别", pipeline._control_gate(doc, "polish") == "cancel",
          repr(pipeline._control_gate(doc, "polish")))
    jobs = [j for j in st.read_all() if j["id"] == doc]
    check("取消后状态是 cancelled", jobs[0]["stage"] == "cancelled", jobs[0]["stage"])
    check("取消后标记已结束", jobs[0]["finished_at"] is not None)

    # 清理
    st.clear_control(doc)
    try:
        st._path(doc).unlink(missing_ok=True)
    except Exception:
        pass
    try:
        st._ctl_path(doc).unlink(missing_ok=True)
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
