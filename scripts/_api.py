"""驱动 8787 服务的转写任务（中文文档名要正确编码）。

用法：
  .venv/bin/python scripts/_api.py transcribe "<文档名>"      # 提交转写
  .venv/bin/python scripts/_api.py status "<文档名>"          # 查状态
  .venv/bin/python scripts/_api.py wait "<文档名>" [超时秒]   # 等转写完成
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8787"


def _req(path: str, payload=None):
    url = f"{BASE}{urllib.parse.quote(path, safe='/')}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def doc_state(doc_id: str):
    return _req(f"/api/doc/{doc_id}")


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    cmd, doc_id = sys.argv[1], sys.argv[2]
    if cmd == "transcribe":
        print(json.dumps(_req("/api/transcribe", {
            "id": doc_id, "engine": "qwen3", "align": True, "rules": True,
        }), ensure_ascii=False))
        return 0
    if cmd == "status":
        d = doc_state(doc_id)
        job = d.get("job") or {}
        prog = d.get("progress") or {}
        print(f"status={d.get('status')} stage={job.get('stage')} "
              f"msg={job.get('message')!r} running={job.get('running')} "
              f"done={prog.get('done')}/{prog.get('total')} err={job.get('error')!r}")
        return 0
    if cmd == "wait":
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else 3600
        t0 = time.time()
        last = ""
        while time.time() - t0 < limit:
            d = doc_state(doc_id)
            job = d.get("job") or {}
            prog = d.get("progress") or {}
            line = (f"{job.get('stage')} {job.get('message')} "
                    f"{prog.get('done')}/{prog.get('total')} "
                    f"段={len(d.get('segments') or [])}")
            if line != last:
                print(f"[{time.strftime('%H:%M:%S')}] {line}", flush=True)
                last = line
            if job.get("error"):
                print(f"✗ 失败：{job['error']}")
                return 1
            if not job.get("running") and d.get("status") in ("ready", "error"):
                print(f"✓ 完成，{len(d.get('segments') or [])} 段，"
                      f"用时 {(time.time() - t0) / 60:.1f} 分钟")
                return 0
            time.sleep(10)
        print("⏳ 超时")
        return 2
    print("未知命令")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
