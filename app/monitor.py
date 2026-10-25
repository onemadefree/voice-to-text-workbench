"""系统指标采集：只用 macOS 内建命令，不引入新依赖。

采集项与方式（都是本机实测可用的）：
  GPU 占用/显存  ioreg -r -d 1 -w 0 -c AGXAccelerator
  内存          top -l 1 -n 0 的 PhysMem 行（取不到则退 vm_stat）
  总体 CPU      top -l 1 -n 0 的 CPU usage 行
  进程 CPU/内存 ps -o pid,pcpu,rss,etime,comm -p <pid>
  磁盘          df -k <目录>

设计原则：
1. **任何一项采集失败都返回 None**，界面显示「—」，看板绝不能把服务搞挂；
2. 结果缓存 2 秒——top -l 1 本身要 1 秒，不能每个请求都采一遍；
3. GPU 数字是**系统级**的（含浏览器等其他程序），不是本任务独占。
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

INTERVAL = 2.0
_CACHE: Dict[str, Any] = {}
_CACHE_TS = 0.0
_LOCK = threading.Lock()
_THREAD: Optional[threading.Thread] = None


def _run(cmd: List[str], timeout: float = 3.0) -> str:
    """跑一条命令拿 stdout；失败/超时返回空串。"""
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout,
                              check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.decode("utf-8", "ignore")


# ---------------------------------------------------------------- 各指标

def _gpu() -> Dict[str, Any]:
    out = _run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"])
    if not out:
        return {}
    pct = None
    m = re.search(r'"Device Utilization %"\s*=\s*(\d+)', out)
    if m:
        pct = int(m.group(1))
    mem_bytes = None
    m = re.search(r'"In use system memory"\s*=\s*(\d+)', out)
    if m:
        mem_bytes = int(m.group(1))
    res: Dict[str, Any] = {}
    if pct is not None:
        res["gpu_pct"] = pct
    if mem_bytes is not None:
        res["gpu_mem_gb"] = round(mem_bytes / (1024 ** 3), 2)
    return res


def _top() -> Dict[str, Any]:
    """top 一次，解析 CPU 与内存（拿不到就都返回 None）。"""
    out = _run(["top", "-l", "1", "-n", "0"], timeout=4.0)
    if not out:
        return {}
    res: Dict[str, Any] = {}

    m = re.search(r"CPU usage:\s*([\d.]+)%\s*user,\s*([\d.]+)%\s*sys,\s*([\d.]+)%\s*idle",
                  out)
    if m:
        res["cpu_pct"] = round(float(m.group(1)) + float(m.group(2)), 1)
        res["cpu_user_pct"] = float(m.group(1))
        res["cpu_sys_pct"] = float(m.group(2))

    m = re.search(r"PhysMem:\s*([\d.]+)([MG]) used.*?([\d.]+)([MG]) unused", out)
    if m:
        used = float(m.group(1)) * (1024 ** 3 if m.group(2) == "G" else 1024 ** 2)
        unused = float(m.group(3)) * (1024 ** 3 if m.group(4) == "G" else 1024 ** 2)
        total = used + unused
        res["mem_total_gb"] = round(total / (1024 ** 3), 2)
        res["mem_used_gb"] = round(used / (1024 ** 3), 2)
        res["mem_pct"] = round(used / total * 100) if total else None
    return res


def _mem_fallback() -> Dict[str, Any]:
    """top 拿不到时的退路。

    首选 memory_pressure：它给的是系统自己的口径（空闲百分比），
    比「总量 - 空闲页」靠谱——macOS 的 inactive 是可回收的文件缓存，
    按页数减法会把 16GB 机器显示成「已用 99%」。

    再退一步是 vm_stat。注意 vm_stat 里除了页数还有一堆
    「Pages zero filled: 1173027309」这种累计计数器，不能一股脑加进去，
    那是事件次数不是页数，会把内存算成 19TB。
    """
    out = _run(["memory_pressure", "-Q"], timeout=4.0)
    if out:
        m = re.search(r"has\s+(\d+)\s+\(", out)
        m2 = re.search(r"free percentage:\s*(\d+)%", out)
        if m and m2:
            total = int(m.group(1))
            free_pct = int(m2.group(1))
            used = total * (100 - free_pct) / 100
            return {
                "mem_total_gb": round(total / (1024 ** 3), 2),
                "mem_used_gb": round(used / (1024 ** 3), 2),
                "mem_pct": 100 - free_pct,
                "mem_free_pct": free_pct,
            }

    out = _run(["vm_stat"])
    if not out:
        return {}
    m = re.search(r"page size of (\d+) bytes", out)
    page = int(m.group(1)) if m else 4096
    cats: Dict[str, int] = {}
    for name, value in re.findall(r"^Pages\s+([a-zA-Z \-]+):\s*(\d+)\.", out, re.M):
        cats[name.strip()] = int(value)
    total_cats = ("active", "inactive", "speculative", "throttled",
                  "wired down", "occupied by compressor")
    total_pages = sum(cats.get(c, 0) for c in total_cats)
    if not total_pages:
        return {}
    free_pages = cats.get("free", 0)
    total = total_pages * page
    used = max(0, total_pages - free_pages) * page
    return {
        "mem_total_gb": round(total / (1024 ** 3), 2),
        "mem_used_gb": round(used / (1024 ** 3), 2),
        "mem_pct": round(used / total * 100) if total else None,
    }


def _disk(path: str) -> Dict[str, Any]:
    out = _run(["df", "-k", path])
    if not out:
        return {}
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if len(lines) < 2:
        return {}
    parts = lines[-1].split()
    if len(parts) < 5:
        return {}
    try:
        total_k = int(parts[1])
        used_k = int(parts[2])
        free_k = int(parts[3])
    except (ValueError, IndexError):
        return {}
    return {
        "disk_total_gb": round(total_k / 1024 / 1024, 1),
        "disk_used_gb": round(used_k / 1024 / 1024, 1),
        "disk_free_gb": round(free_k / 1024 / 1024, 1),
        "disk_pct": round(used_k / total_k * 100) if total_k else None,
    }


def _proc(pid: Any) -> Dict[str, Any]:
    """单个进程的 CPU 占用与常驻内存。"""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return {}
    if pid <= 0:
        return {}
    out = _run(["ps", "-o", "pcpu=,rss=,etime=,comm=", "-p", str(pid)])
    if not out.strip():
        return {}
    parts = out.split()
    if len(parts) < 2:
        return {}
    res: Dict[str, Any] = {}
    try:
        res["cpu_pct"] = float(parts[0])
        res["rss_mb"] = round(int(parts[1]) / 1024, 1)
    except ValueError:
        return {}
    return res


# ---------------------------------------------------------------- 采集与缓存

def collect(pids: Optional[List[int]] = None) -> Dict[str, Any]:
    """采一次全部指标。"""
    data: Dict[str, Any] = {
        "cpu_pct": None,
        "cpu_cores": os.cpu_count(),
        "mem_total_gb": None,
        "mem_used_gb": None,
        "mem_pct": None,
        "gpu_pct": None,
        "gpu_mem_gb": None,
        "disk_free_gb": None,
        "disk_pct": None,
        "sampled_at": time.time(),
    }
    data.update(_top())
    if data.get("mem_total_gb") is None:
        data.update(_mem_fallback())
    data.update(_gpu())
    data.update(_disk(str(os.path.expanduser("~"))))
    if pids:
        procs = {}
        for pid in pids:
            info = _proc(pid)
            if info:
                procs[str(pid)] = info
        if procs:
            data["processes"] = procs
    return data


def _refresh_loop() -> None:
    global _CACHE, _CACHE_TS
    while True:
        try:
            from .status import read_all
            pids = [j["pid"] for j in read_all() if j.get("pid")]
            pids.append(os.getpid())
        except Exception:
            pids = [os.getpid()]
        try:
            data = collect(pids)
        except Exception:
            data = {}
        with _LOCK:
            _CACHE = data
            _CACHE_TS = time.time()
        time.sleep(INTERVAL)


def _ensure_thread() -> None:
    global _THREAD
    if _THREAD is not None and _THREAD.is_alive():
        return
    _THREAD = threading.Thread(target=_refresh_loop, name="sys-monitor", daemon=True)
    _THREAD.start()


def snapshot() -> Dict[str, Any]:
    """拿一份系统指标快照（带缓存）。"""
    global _CACHE, _CACHE_TS
    _ensure_thread()
    with _LOCK:
        age = time.time() - _CACHE_TS
        if _CACHE and age < INTERVAL * 3:
            return dict(_CACHE)
    # 缓存太旧（后台线程可能没起来）→ 同步采一次
    try:
        data = collect()
    except Exception:
        data = {}
    with _LOCK:
        _CACHE = data
        _CACHE_TS = time.time()
    return dict(data)
