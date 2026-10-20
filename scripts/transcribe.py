#!/usr/bin/env python3
"""命令行批处理：把 source 目录里的音频转成逐字稿。

用法：
    python scripts/transcribe.py                      # 处理 source 下所有未转写文件
    python scripts/transcribe.py 周三上午闫树          # 处理指定文件（不带扩展名）
    python scripts/transcribe.py 周三上午闫树.mp3     # 也可以带扩展名
    python scripts/transcribe.py --engine qwen3-hq    # 换高精度模型
    python scripts/transcribe.py --rules standard     # 更彻底的去口语
    python scripts/transcribe.py --polish             # 转写后再跑 LLM 润色
    python scripts/transcribe.py --kb                 # 再生成知识库资料
    python scripts/transcribe.py --force              # 忽略已有结果，重新转写
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.store import DocumentStore, segment_final_text  # noqa: E402


def find_source(name: str) -> Path:
    p = config.SOURCE_DIR / name
    if p.exists():
        return p
    stem = Path(name).stem
    for q in config.SOURCE_DIR.glob(f"{stem}.*"):
        return q
    raise SystemExit(f"找不到源文件: {name}（目录 {config.SOURCE_DIR}）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="文件名（可省略，默认处理全部）")
    ap.add_argument("--engine", default="qwen3", choices=["qwen3", "qwen3-hq"])
    ap.add_argument("--lang", default=None, help="语言，默认 Chinese")
    ap.add_argument("--no-align", action="store_true", help="不做词级强制对齐")
    ap.add_argument("--rules", default="light", choices=["none", "light", "standard"])
    ap.add_argument("--polish", action="store_true", help="转写后跑本地 LLM 润色")
    ap.add_argument("--kb", action="store_true", help="生成知识库资料")
    ap.add_argument("--force", action="store_true", help="重新转写")
    ap.add_argument("--hotwords", default="", help="热词，逗号分隔")
    args = ap.parse_args()

    config.ensure_dirs()
    store = DocumentStore()
    pipeline = Pipeline(store)

    if args.names:
        sources = [find_source(n) for n in args.names]
    else:
        sources = sorted(
            p for p in config.SOURCE_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in config.AUDIO_EXTS
        )
    if not sources:
        print(f"目录 {config.SOURCE_DIR} 里没有音频文件")
        return 1

    hotwords = [w.strip() for w in args.hotwords.split(",") if w.strip()]

    for src in sources:
        doc_id = src.stem
        print(f"\n=== {src.name} ===", flush=True)
        pipeline.start_transcribe(
            doc_id=doc_id,
            source_path=src,
            engine=args.engine,
            language=args.lang,
            align=not args.no_align,
            use_rules=args.rules != "none",
            force=args.force,
            hotwords=hotwords,
        )
        job = pipeline.get_job(doc_id)
        last = -1
        while job and job.running:
            time.sleep(1.0)
            doc = store.load(doc_id) or {}
            p = doc.get("progress") or {}
            if p.get("done") != last:
                last = p.get("done")
                print(f"  {p.get('stage')} {p.get('done')}/{p.get('total')}  {p.get('message','')}", flush=True)
        if job and job.error:
            print(f"  ✗ 失败: {job.error}")
            continue
        print(f"  ✓ 转写完成，段落数 {len((store.load(doc_id) or {}).get('segments') or [])}")

        if args.rules != "none":
            pipeline.run_rules(doc_id, level=args.rules)
            print(f"  ✓ 规则清理（{args.rules}）")

        if args.polish:
            print("  … LLM 润色中")
            pipeline.run_polish(doc_id, level=args.rules if args.rules != "none" else "standard")
            print("  ✓ 润色完成")

        if args.kb:
            print("  … 知识库提炼中")
            pipeline.run_knowledge_base(doc_id, title=src.stem)
            print("  ✓ 知识库完成")

        doc = store.load(doc_id) or {}
        out_txt = config.OUTPUT_DIR / doc_id / f"{doc_id}.txt"
        out_txt.write_text(
            "\n".join(segment_final_text(s) for s in doc.get("segments") or []),
            encoding="utf-8",
        )
        print(f"  → {out_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
