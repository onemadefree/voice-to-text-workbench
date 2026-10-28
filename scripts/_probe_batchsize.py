#!/usr/bin/env python3
"""探针：内网模型在不同批大小下的成功率与耗时。

拆半重试的代价 = 1 + 2 + 4 … 次请求；批越小越不容易被 max_tokens 截断，
但请求次数变多。这里量一下到底哪个批大小最划算。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.llm import LocalLLM, polish_segments  # noqa: E402


def main() -> int:
    d = json.loads((ROOT / "output/数字经济时代的产业创新：机遇与挑战-高太山"
                    "/transcript.json").read_text(encoding="utf-8"))
    segs = d["segments"]
    base = 1153  # 高太山卡住的那一段附近

    for size in (16, 8, 4):
        work = [{"raw": segs[base + i].get("clean") or segs[base + i].get("raw") or "",
                 "text": segs[base + i].get("raw") or ""}
                for i in range(size)]
        llm = LocalLLM(backend="custom")
        t = time.time()
        res = polish_segments(llm, work, batch_size=size)
        dt = time.time() - t
        changed = sum(1 for i, r in enumerate(res)
                      if r.strip() and r.strip() != work[i]["raw"].strip())
        print(f"批大小 {size:2d}：{dt:6.1f}s | 拿到结果 {sum(1 for r in res if r.strip()):2d}/{size}"
              f" | 改写 {changed:2d} 段", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
