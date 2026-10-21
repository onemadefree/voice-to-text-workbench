#!/usr/bin/env python3
"""通过本地服务批量处理 source 下的音频（转写 → 清理 → 润色 → 知识库 → 导出）。

为什么走服务：沙箱里的新进程拿不到 Metal/GPU，而服务进程有。服务负责跑
模型，本脚本负责编排与后处理（退化段落清理在本地用新代码执行）。

用法：
    python scripts/batch_service.py                       # 处理所有未完成的
    python scripts/batch_service.py "20261021 上午闫树"     # 只处理指定文件
    python scripts/batch_service.py --engine qwen3-hq     # 用高精度模型
    python scripts/batch_service.py --llm minimax          # 润色走网络 API
    python scripts/batch_service.py --no-polish            # 只转写
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.store import segment_final_text  # noqa: E402

BASE = f"http://{config.HOST}:{config.PORT}"


def api(method: str, path: str, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"{method} {path} → HTTP {exc.code}: {exc.read().decode('utf-8','ignore')[:200]}")


def wait_job(doc_id: str, label: str, timeout: float = 14400) -> dict:
    """等任务跑完。"""
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout:
        try:
            doc = api("GET", f"/api/doc/{urllib.parse.quote(doc_id)}")
        except SystemExit:
            time.sleep(3)
            continue
        job = doc.get("job") or {}
        prog = doc.get("progress") or {}
        stage = prog.get("stage") or job.get("stage") or ""
        done, total = prog.get("done"), prog.get("total")
        msg = prog.get("message") or job.get("message") or ""
        line = f"{stage} {done}/{total} {msg}".strip()
        if line != last:
            print(f"    {line}", flush=True)
            last = line
        if job.get("error"):
            raise SystemExit(f"{label} 失败: {job['error']}")
        if not job.get("running") and job:
            return doc
        time.sleep(5)
    raise SystemExit(f"{label} 超时")


def export_txt(doc_id: str) -> Path:
    doc = api("GET", f"/api/doc/{urllib.parse.quote(doc_id)}")
    out = config.OUTPUT_DIR / doc_id / f"{doc_id}.txt"
    out.write_text("\n".join(segment_final_text(s) for s in doc.get("segments") or []),
                   encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--engine", default="qwen3")
    ap.add_argument("--llm", default="")
    ap.add_argument("--no-polish", action="store_true")
    ap.add_argument("--no-kb", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", help="已有结果就跳过")
    args = ap.parse_args()

    if args.names:
        files = [config.SOURCE_DIR / n for n in args.names]
        files = [f if f.exists() else next(
            config.SOURCE_DIR.glob(Path(n).stem + ".*"), f) for n, f in zip(args.names, files)]
    else:
        files = sorted(p for p in config.SOURCE_DIR.iterdir()
                       if p.is_file() and p.suffix.lower() in config.AUDIO_EXTS)
    if not files:
        print("没有可处理的音频")
        return 1

    for src in files:
        doc_id = src.stem
        print(f"\n=== {src.name} ===", flush=True)
        if args.skip_existing and (config.OUTPUT_DIR / doc_id / "transcript.json").exists():
            print("  · 已有结果，跳过")
            continue

        api("POST", "/api/transcribe", {
            "id": doc_id, "name": src.name, "engine": args.engine,
            "align": True, "rules": True, "force": args.force,
        })
        wait_job(doc_id, "转写")

        # 用新代码做退化段落清理（服务里跑的是旧管线，这里补上）
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "scripts" / "postprocess.py"), doc_id],
                       check=False, cwd=str(ROOT))

        if not args.no_polish:
            api("POST", f"/api/doc/{urllib.parse.quote(doc_id)}/polish",
                {"level": "standard", "backend": args.llm})
            wait_job(doc_id, "润色")
        if not args.no_kb:
            api("POST", f"/api/doc/{urllib.parse.quote(doc_id)}/knowledge_base",
                {"backend": args.llm})
            wait_job(doc_id, "知识库")
        print(f"  → {export_txt(doc_id)}", flush=True)

    print("\n全部完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
