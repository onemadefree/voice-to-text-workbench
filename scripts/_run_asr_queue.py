#!/usr/bin/env python3
"""把 5 篇未转写的音频排进 8787 服务的转写队列（GPU 串行）。

润色走在线 API 不吃 GPU，可以和这里并行。
每篇转写完就接着下一篇；失败的会重试一次。

用法：.venv/bin/python scripts/_run_asr_queue.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE = "http://127.0.0.1:8787"
SOURCE = ROOT / "source"

# 短的那个先跑，快速验证链路
QUEUE = [
    "人力资源组织部-结业讲话",
    "AI 赋能企业项目管理的应用与实践-张涛",
    "以场景为导向的 AI 项目端到端应用落地方法论-王峰",
    "数字经济时代的产业创新：机遇与挑战-高太山",
    "组织变革管理与变革领导力-程建刚",
]


def req(path: str, payload=None):
    url = f"{BASE}{urllib.parse.quote(path, safe='/')}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    r = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET")
    with urllib.request.urlopen(r, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def source_name(doc_id: str) -> str:
    for ext in (".mp3", ".wav", ".m4a", ".aac", ".flac"):
        p = SOURCE / f"{doc_id}{ext}"
        if p.exists():
            return p.name
    return f"{doc_id}.mp3"


def wait_done(doc_id: str, limit_s: int = 7200) -> bool:
    t0 = time.time()
    last = ""
    while time.time() - t0 < limit_s:
        try:
            d = req(f"/api/doc/{doc_id}")
        except Exception as exc:
            time.sleep(5)
            continue
        job = d.get("job") or {}
        prog = d.get("progress") or {}
        segs = len(d.get("segments") or [])
        line = (f"{job.get('stage')} {job.get('message')} "
                f"{prog.get('done')}/{prog.get('total')} 段={segs}")
        if line != last:
            print(f"    [{time.strftime('%H:%M:%S')}] {line}", flush=True)
            last = line
        if job.get("error"):
            print(f"    ✗ 失败：{job['error']}", flush=True)
            return False
        if not job.get("running") and d.get("status") in ("ready", "error"):
            print(f"    ✓ 完成 {segs} 段，用时 {(time.time() - t0) / 60:.1f} 分钟", flush=True)
            return True
        time.sleep(10)
    print("    ⏳ 超时", flush=True)
    return False


def main() -> int:
    print(f"=== ASR 队列开始 {time.strftime('%m-%d %H:%M:%S')} ===", flush=True)
    results = []
    for doc_id in QUEUE:
        name = source_name(doc_id)
        print(f"\n--- {doc_id}（{name}）---", flush=True)
        try:
            req("/api/transcribe", {"id": doc_id, "name": name,
                                    "engine": "qwen3", "align": True, "rules": True})
        except Exception as exc:
            print(f"    ✗ 提交失败：{exc}", flush=True)
            results.append((doc_id, False))
            continue
        ok = wait_done(doc_id)
        if not ok:
            # 重试一次（可能只是这次撞上了审核/网络）
            print("    ↻ 重试一次", flush=True)
            try:
                req("/api/transcribe", {"id": doc_id, "name": name,
                                        "engine": "qwen3", "align": True,
                                        "rules": True, "force": True})
                ok = wait_done(doc_id)
            except Exception as exc:
                print(f"    ✗ 重试失败：{exc}", flush=True)
        results.append((doc_id, ok))

    print(f"\n=== ASR 队列结束 {time.strftime('%m-%d %H:%M:%S')} ===", flush=True)
    for doc_id, ok in results:
        print(f"  {'✓' if ok else '✗'} {doc_id}")
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
