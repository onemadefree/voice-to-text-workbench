#!/usr/bin/env python3
"""环境自检：依赖、模型、音频工具、目录。"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'✓' if ok else '✗'}] {name}{('  ' + detail) if detail else ''}")
    return ok


def main() -> int:
    ok = True
    print("▸ 运行环境")
    ok &= check("ffmpeg", shutil.which("ffmpeg") is not None)
    ok &= check("ffprobe", shutil.which("ffprobe") is not None)

    print("▸ Python 依赖")
    try:
        import mlx.core as mx  # noqa: F401
        ok &= check("mlx + Metal", True, f"metal={mx.metal.is_available()}")
    except Exception as exc:
        ok &= check("mlx", False, str(exc))
    for mod, label in [("mlx_audio", "mlx-audio"), ("mlx_lm", "mlx-lm"),
                       ("fastapi", "fastapi"), ("uvicorn", "uvicorn")]:
        try:
            __import__(mod)
            ok &= check(label, True)
        except Exception as exc:
            ok &= check(label, False, str(exc))

    print("▸ 模型")
    for label, repo in [("ASR", config.ASR_MODEL), ("对齐器", config.ALIGNER_MODEL),
                        ("润色 LLM", config.LLM_MODEL)]:
        local = config.MODELS_DIR / repo.replace("/", "__")
        has_weights = local.exists() and any(local.glob("*.safetensors"))
        ok &= check(f"{label} {repo}", has_weights,
                    "" if has_weights else "未下载（运行 scripts/fetch_models.py）")

    print("▸ 目录")
    config.ensure_dirs()
    ok &= check(f"source  {config.SOURCE_DIR}", config.SOURCE_DIR.exists())
    ok &= check(f"output  {config.OUTPUT_DIR}", config.OUTPUT_DIR.exists())

    print()
    if ok:
        print("✓ 环境就绪")
        return 0
    print("✗ 存在未满足项，请按提示处理")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
