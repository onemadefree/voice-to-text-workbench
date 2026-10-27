"""作业状态：文件化，让任何进程都能被仪表盘看到。

原来作业状态只在 Pipeline 的内存字典里（self.jobs），有两个问题：
1. 服务重启即丢；
2. **命令行脚本跑的长任务，服务端完全看不见**（今晚踩的就是这个）。

现在改成写文件：output/_status/<文档名>.json。任何进程（网页服务、
命令行脚本）跑流水线时都会写同一份状态，仪表盘统一读它。
output/ 已经在 .gitignore 里，不会进 Git。
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config

STATUS_DIR = config.OUTPUT_DIR / "_status"

# 超过这个时间没更新，就认为状态可能卡住（秒）
STALE_SECONDS = 900
# 结束多久之后就不再展示了（秒）
KEEP_FINISHED_SECONDS = 24 * 3600

_SLUG_RE = re.compile(r"[^\w一-鿿.-]+")


def _slug(doc_id: str) -> str:
    s = _SLUG_RE.sub("_", str(doc_id)).strip("_")
    return s or "job"


def _path(doc_id: str) -> Path:
    return STATUS_DIR / f"{_slug(doc_id)}.json"


def _pid_alive(pid: Any) -> bool:
    """进程还活着吗？用纯 Python 判断，不需要 ps（沙箱里 ps 是 EPERM）。"""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # 进程存在，只是没权限发信号
        return True
    except OSError:
        return False


def publish(
    doc_id: str,
    kind: str,
    stage: str,
    message: str = "",
    done: Optional[int] = None,
    total: Optional[int] = None,
    *,
    error: str = "",
    finished: bool = False,
) -> Optional[Dict[str, Any]]:
    """发布一次作业状态。

    kind: asr / polish / kb / rules
    stage: queued / probing / converting / segmenting / loading_model /
           cleaning / preview / polishing / knowledge_base / done / error
    """
    try:
        STATUS_DIR.mkdir(parents=True, exist_ok=True)
        path = _path(doc_id)
        prev: Dict[str, Any] = {}
        if path.exists():
            try:
                prev = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                prev = {}
        now = time.time()
        pid = os.getpid()
        # 换了进程就重新计开始时间
        started = prev.get("started_at") if prev.get("pid") == pid else now
        data = {
            "id": str(doc_id),
            "doc_id": str(doc_id),
            "kind": kind,
            "stage": stage,
            "message": message,
            "done": done,
            "total": total,
            "pid": pid,
            "error": error,
            "started_at": started,
            "updated_at": now,
            "finished_at": now if finished else None,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        return data
    except Exception:
        # 状态上报绝不能影响主流程
        return None


def _eta(started_at: float, done: Optional[int], total: Optional[int],
         now: float) -> Optional[int]:
    if not done or not total or done >= total or done <= 0:
        return None
    elapsed = max(1.0, now - float(started_at or now))
    return int(elapsed / done * (total - done))


def read_all() -> List[Dict[str, Any]]:
    """读出全部作业状态，带存活判断、百分比、ETA。"""
    jobs: List[Dict[str, Any]] = []
    if not STATUS_DIR.exists():
        return jobs
    now = time.time()
    for path in sorted(STATUS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        finished_at = data.get("finished_at")
        alive = _pid_alive(data.get("pid"))
        updated_at = float(data.get("updated_at") or 0)
        stale = bool(alive and not finished_at and (now - updated_at) > STALE_SECONDS)
        if finished_at and (now - float(finished_at)) > KEEP_FINISHED_SECONDS:
            continue
        done = data.get("done")
        total = data.get("total")
        pct = None
        if done is not None and total:
            pct = max(0, min(100, round(float(done) / float(total) * 100)))
        if finished_at:
            state = "error" if data.get("error") else "done"
        elif alive:
            state = "stale" if stale else "running"
        else:
            # 进程没了却没写结束 → 异常退出
            state = "died"
        jobs.append({
            "id": data.get("id") or data.get("doc_id") or path.stem,
            "kind": data.get("kind") or "unknown",
            "stage": data.get("stage") or "unknown",
            "message": data.get("message") or "",
            "done": done,
            "total": total,
            "pct": pct,
            "eta_s": _eta(float(data.get("started_at") or now), done, total, now)
            if state == "running" else None,
            "pid": data.get("pid"),
            "alive": alive,
            "stale": stale,
            "state": state,
            "error": data.get("error") or "",
            "started_at": data.get("started_at"),
            "updated_at": updated_at,
            "finished_at": finished_at,
            "elapsed_s": int(now - float(data.get("started_at") or now)),
        })
    # 正在跑的排前面，其次按开始时间倒序
    order = {"running": 0, "stale": 1, "died": 2, "error": 3, "done": 4}
    jobs.sort(key=lambda j: (order.get(j["state"], 9), -(j.get("started_at") or 0)))
    return jobs


def job_running(doc_id: str) -> bool:
    """这篇当前有没有别的进程在处理（没写结束标记，且进程还活着）。

    用来避免两个进程同时润色同一篇——比如用户自己在网页上点了润色，
    而后台脚本也想跑一遍。
    """
    path = _path(doc_id)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if data.get("finished_at"):
        return False
    return _pid_alive(data.get("pid"))


def clear_finished() -> int:
    """清掉过期与已结束的状态文件。"""
    n = 0
    if not STATUS_DIR.exists():
        return 0
    now = time.time()
    for path in STATUS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            path.unlink(missing_ok=True)
            n += 1
            continue
        finished_at = data.get("finished_at")
        if finished_at and (now - float(finished_at)) > KEEP_FINISHED_SECONDS:
            path.unlink(missing_ok=True)
            n += 1
    return n
