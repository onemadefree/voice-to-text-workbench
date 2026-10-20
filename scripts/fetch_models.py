#!/usr/bin/env python3
"""稳扎稳打的模型下载器：对每个文件用 curl 断点续传 + 重试。

用法：
    python scripts/fetch_models.py            # 下载默认全部
    python scripts/fetch_models.py asr aligner llm
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
GROUPS = {
    "asr": [config.ASR_MODEL],
    "hq": [config.ASR_MODEL_HQ],
    "aligner": [config.ALIGNER_MODEL],
    "llm": [config.LLM_MODEL],
}
SKIP = {".gitattributes"}


def list_files(repo: str) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi(endpoint=ENDPOINT)
    return api.list_repo_files(repo)


def fetch(repo: str, fname: str, target: Path) -> bool:
    url = f"{ENDPOINT}/{repo}/resolve/main/{fname}"
    part = target.with_suffix(target.suffix + ".part")
    cmd = [
        "curl", "-L", "--fail", "--silent", "--show-error",
        "-C", "-", "--retry", "8", "--retry-delay", "3",
        "--retry-all-errors", "--connect-timeout", "20",
        "--speed-time", "60", "--speed-limit", "2048",
        "-o", str(part), url,
    ]
    for attempt in range(1, 6):
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode == 0 and part.exists() and part.stat().st_size > 0:
            size_mb = part.stat().st_size / 1e6
            part.replace(target)
            print(f"  ✓ {fname} ({size_mb:.1f} MB)", flush=True)
            return True
        print(f"  … {fname} 第 {attempt} 次重试（{r.stderr.decode('utf-8','ignore').strip()[:80]}）", flush=True)
    return False


def download(repo: str) -> bool:
    target_dir = config.MODELS_DIR / repo.replace("/", "__")
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"→ {repo}", flush=True)
    try:
        files = list_files(repo)
    except Exception as exc:
        print(f"  ✗ 拿不到文件清单: {exc}", flush=True)
        return False
    ok = True
    for fname in files:
        if fname in SKIP or fname.endswith("/"):
            continue
        dest = target_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  · 已存在 {fname}", flush=True)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not fetch(repo, fname, dest):
            ok = False
    return ok


def main() -> int:
    which = sys.argv[1:] or ["asr", "aligner", "llm"]
    repos: list[str] = []
    for key in which:
        if key not in GROUPS:
            print(f"未知分组: {key}，可选: {', '.join(GROUPS)}")
            return 2
        repos.extend(GROUPS[key])
    config.ensure_dirs()
    failed = [r for r in repos if not download(r)]
    if failed:
        print("失败: " + ", ".join(failed))
        return 1
    print("全部完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
