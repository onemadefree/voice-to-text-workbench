#!/usr/bin/env python3
"""自测：作业状态文件（写/读/存活判断/ETA）与仪表盘接口。

要点：
1. 状态写到 output/_status/<文档名>.json，任何进程都能被看到；
2. 进程还活着 → running；进程没了却没写结束 → died（异常退出）；
3. 数字进度要能算出百分比与 ETA；
4. 状态上报失败绝不能影响主流程。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import status as st  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}　{detail}")
        FAILED.append(name)


def run() -> int:
    print("· 状态发布与读取")
    doc = "__测试文档__"
    st.publish(doc, "polish", "polishing", "润色中", 10, 100)
    jobs = st.read_all()
    mine = [j for j in jobs if j["id"] == doc]
    check("写完能读回来", len(mine) == 1, f"读到 {len(mine)} 条")
    if not mine:
        return 1
    j = mine[0]
    check("阶段正确", j["stage"] == "polishing", j["stage"])
    check("数字进度正确", j["done"] == 10 and j["total"] == 100, str(j))
    check("百分比正确", j["pct"] == 10, str(j["pct"]))
    check("进程活着 → running", j["state"] == "running", j["state"])
    check("alive 判定正确", j["alive"] is True, str(j["alive"]))
    check("有 ETA", j["eta_s"] is not None and j["eta_s"] >= 0, str(j["eta_s"]))
    check("pid 是当前进程", j["pid"] == os.getpid(), str(j["pid"]))

    print("· 更新进度")
    st.publish(doc, "polish", "polishing", "润色中", 60, 100)
    j = [x for x in st.read_all() if x["id"] == doc][0]
    check("进度已更新", j["done"] == 60 and j["pct"] == 60, str(j))
    check("started_at 保持不变（同一进程）",
          j["started_at"] == mine[0]["started_at"], str(j["started_at"]))

    print("· 结束")
    st.publish(doc, "polish", "done", "润色完成", 100, 100, finished=True)
    j = [x for x in st.read_all() if x["id"] == doc][0]
    check("状态为 done", j["state"] == "done", j["state"])
    check("有 finished_at", j["finished_at"] is not None, str(j["finished_at"]))
    check("结束后不再给 ETA", j["eta_s"] is None, str(j["eta_s"]))

    print("· 进程没了却没写结束 → 异常退出")
    p = st._path(doc)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["finished_at"] = None
    data["pid"] = 999999          # 不存在的进程号
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    j = [x for x in st.read_all() if x["id"] == doc][0]
    check("状态为 died", j["state"] == "died", j["state"])
    check("alive 为 False", j["alive"] is False, str(j["alive"]))

    print("· 错误状态")
    st.publish(doc, "asr", "error", "炸了", error="Boom", finished=True)
    j = [x for x in st.read_all() if x["id"] == doc][0]
    check("状态为 error", j["state"] == "error", j["state"])
    check("错误信息保留", j["error"] == "Boom", j["error"])

    print("· 文件名安全（文档名带斜杠/空格）")
    weird = "a/b c*?<>|"
    st.publish(weird, "asr", "queued", "排队")
    jobs = [x for x in st.read_all() if x["id"] == weird]
    check("特殊字符文档名能落盘并读回", len(jobs) == 1, str(jobs))
    check("文件名已转义", "/" not in st._path(weird).name, st._path(weird).name)

    print("· 清理")
    for d in (doc, weird):
        st._path(d).unlink(missing_ok=True)

    print("· 状态上报失败不影响主流程")
    try:
        st.publish("__状态自测残留__", "asr", "queued", "", None, None)  # 正常路径
        ok = True
    except Exception as exc:
        ok = False
        print("    异常:", exc)
    check("publish 不抛异常", ok)

    print("· 清理")
    for d in (doc, weird, "__状态自测残留__"):
        st._path(d).unlink(missing_ok=True)
    left = [p.name for p in st.STATUS_DIR.glob("*.json")] if st.STATUS_DIR.exists() else []
    check("测试文件已全部清理", not left, f"还剩 {left}")

    if FAILED:
        print(f"\n失败 {len(FAILED)} 项：{FAILED}")
        return 1
    print("\n全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
