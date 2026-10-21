#!/usr/bin/env python3
"""对已转写的文档做退化段落清理（幻觉串 / 零时长噪声 / 连续刷屏区）。

识别完成后单独跑一遍，把模型刷屏产生的垃圾段清掉，并给零时长的真实口语
补上时间轴。清完再送去润色，避免 LLM 在垃圾上浪费算力。

用法：
    python scripts/postprocess.py "20261021 下午尚水利"
    python scripts/postprocess.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.pipeline import _drop_degenerate  # noqa: E402


def clean_doc(doc_id: str, dry: bool = False) -> tuple[int, int]:
    path = config.OUTPUT_DIR / doc_id / "transcript.json"
    if not path.exists():
        print(f"  ✗ 找不到 {path}")
        return 0, 0
    doc = json.loads(path.read_text(encoding="utf-8"))
    segs = doc.get("segments") or []
    before = len(segs)
    kept = _drop_degenerate(segs)
    doc["segments"] = kept
    doc["segments_cleaned"] = {"before": before, "after": len(kept)}
    if not dry:
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  段落 {before} → {len(kept)}（丢弃 {before - len(kept)}）"
          f"{'  [dry-run]' if dry else ''}")
    return before, len(kept)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("doc_ids", nargs="*")
    ap.add_argument("--all", action="store_true", help="处理 output 下所有文档")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    ids = list(args.doc_ids)
    if args.all or not ids:
        ids = sorted(d.name for d in config.OUTPUT_DIR.iterdir()
                     if d.is_dir() and (d / "transcript.json").exists())
    for doc_id in ids:
        print(f"\n=== {doc_id} ===")
        clean_doc(doc_id, args.dry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
