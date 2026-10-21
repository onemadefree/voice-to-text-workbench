#!/usr/bin/env python3
"""修正文档元数据：清掉陈旧的错误状态，把 source_path 指到当前真实源文件。

用途：
1. 早前任务被中断/失败留下的 status=error 会让界面误判文档不可用；
2. 源文件改名后，transcript.json 里记的还是旧路径（例如目录改名、音频重命名）。

用法：
    python scripts/fix_meta.py            # 处理 output 下所有文档
    python scripts/fix_meta.py 某个文档名
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402


def fix(doc_id: str) -> None:
    path = config.OUTPUT_DIR / doc_id / "transcript.json"
    if not path.exists():
        return
    doc = json.loads(path.read_text(encoding="utf-8"))
    segs = doc.get("segments") or []
    changed = []

    # 1. 陈旧错误：有段落且没在跑 → 状态归位
    if doc.get("status") == "error" and segs:
        doc["status"] = "ready"
        doc["error"] = ""
        changed.append("清理陈旧 error")

    # 2. source_path 指向当前真实源文件
    stem = doc.get("source") or ""
    cand = None
    for ext in sorted(config.AUDIO_EXTS):
        p = config.SOURCE_DIR / f"{doc_id}{ext}"
        if p.exists():
            cand = p
            break
    if cand is None and stem:
        for ext in sorted(config.AUDIO_EXTS):
            p = config.SOURCE_DIR / f"{Path(stem).stem}{ext}"
            if p.exists():
                cand = p
                break
    if cand is not None:
        if doc.get("source_path") != str(cand):
            doc["source_path"] = str(cand)
            changed.append("更新 source_path")
        if doc.get("source") != cand.name:
            doc["source"] = cand.name
            changed.append("更新 source")

    if changed:
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  ✓ {doc_id}: {'；'.join(changed)}")
    else:
        print(f"  · {doc_id}: 无需修改")


def main() -> int:
    ids = sys.argv[1:] or sorted(d.name for d in config.OUTPUT_DIR.iterdir()
                                if d.is_dir() and (d / "transcript.json").exists())
    for doc_id in ids:
        fix(doc_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
