#!/usr/bin/env python3
"""修掉「文档明明有内容、状态却是 error」的脏状态。

起因：早先网页「开始转写」有个 bug——按不带扩展名的 id 找不到源文件时，
会回退去探测 source 目录本身，ffprobe 报 "Is a directory"，
pipeline 随即把整篇文档的 status 标成 error。那篇文档的段落、润色、
知识库其实都还在，只是状态被写坏了，界面上就显示红色「出错」。

本脚本把这类文档的状态纠正回 ready，并把失败原因挪到 last_error。
幂等：没有脏状态就什么都不做。

用法：.venv/bin/python scripts/_fix_doc_status.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    fixed = []
    for path in sorted(config.OUTPUT_DIR.glob("*/transcript.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"跳过（读不了）{path.parent.name}: {exc}")
            continue
        segs = doc.get("segments") or []
        if doc.get("status") != "error" or not segs:
            continue
        err = doc.get("error") or doc.get("last_error") or ""
        doc["status"] = "ready"
        doc["last_error"] = err
        doc.pop("error", None)
        atomic_write(path, json.dumps(doc, ensure_ascii=False, indent=2))
        fixed.append((path.parent.name, len(segs), err[:60]))
    for name, n, err in fixed:
        print(f"已修正 {name}：{n} 段，状态 error → ready（原因：{err}）")
    if not fixed:
        print("没有需要修正的文档")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
