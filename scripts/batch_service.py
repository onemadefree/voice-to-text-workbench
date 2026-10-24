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


def wait_job(doc_id: str, label: str, timeout: float = 14400,
             not_before: float = 0.0, done_when=None) -> dict:
    """等任务跑完。

    not_before 是发起请求的时刻：任务线程是异步启动的，刚 POST 完立刻查询
    可能拿到上一次的旧 job（已 finished）而被误判为「已完成」，所以必须
    等到 job.started_at 晚于请求时刻才算数。

    done_when(doc) 是内容层面的完成判定，比时间戳可靠：润色看「所有段落都有
    润色结果」，知识库看「knowledge_base 非空」。旧服务不返回 started_at 时
    靠它兜底。
    """
    t0 = time.time()
    last = ""
    seen_running = not_before <= 0
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
        if job.get("running"):
            seen_running = True
        started = job.get("started_at")
        if started is not None:
            if started >= not_before and not job.get("running"):
                return doc
        else:
            # 旧服务没有 started_at：等看到运行中再等它结束
            if job and seen_running and not job.get("running"):
                return doc
            # 任务可能瞬间完成（无事可做），用内容判定兜底
            if done_when and done_when(doc):
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
    ap.add_argument("--polish-only", action="store_true",
                    help="跳过转写，只做清理+润色+知识库+导出")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--force-polish", action="store_true",
                    help="重刷已润色段落（换模型后想统一质量时用）")
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

        if args.polish_only:
            print("  · 仅润色模式：跳过转写", flush=True)
        else:
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
            t = time.time()
            api("POST", f"/api/doc/{urllib.parse.quote(doc_id)}/polish",
                {"level": "standard", "backend": args.llm, "force": bool(args.force_polish)})
            wait_job(doc_id, "润色", not_before=t,
                     done_when=lambda d: bool(d.get("segments")) and all(
                         s.get("polished") for s in d.get("segments") or []))
        if not args.no_kb:
            t = time.time()
            api("POST", f"/api/doc/{urllib.parse.quote(doc_id)}/knowledge_base",
                {"backend": args.llm})
            wait_job(doc_id, "知识库", not_before=t,
                     done_when=lambda d: bool(d.get("knowledge_base")))
        print(f"  → {export_txt(doc_id)}", flush=True)

    print("\n全部完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
