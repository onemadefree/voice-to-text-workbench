#!/usr/bin/env python3
"""跨平台环境探针：报告这台机器能跑哪些 ASR 引擎。

在任何一台候选部署机上跑一次，把输出贴回来，就能直接对照《跨平台 ASR 方案》
判断该走哪条路线。不需要联网，不加载模型，只看环境。

用法：
    python scripts/probe_env.py            # 人看的报告
    python scripts/probe_env.py --json     # 机器读的 JSON
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 引擎 → 需要的 Python 包（按 extras 分组）
ENGINE_DEPS: Dict[str, List[str]] = {
    "qwen3 / whisper（MLX）": ["mlx_audio", "mlx_lm"],
    "faster-whisper": ["faster_whisper"],
    "funasr": ["funasr"],
    "sherpa-onnx": ["sherpa_onnx"],
    "whisper.cpp（CLI）": [],          # 外部二进制，不看 Python 包
    "onnxruntime": ["onnxruntime"],
    "onnxruntime-directml": ["onnxruntime"],
}


def _run(cmd: List[str], timeout: float = 5.0) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (p.stdout or "") + (p.stderr or "")
    except Exception:
        return ""


def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def probe_os() -> Dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "is_apple_silicon": (platform.system() == "Darwin"
                              and platform.machine() in ("arm64", "aarch64")),
    }


def probe_tools() -> Dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    return {
        "ffmpeg": ffmpeg or "",
        "ffmpeg_ok": bool(ffmpeg),
        "ffprobe_ok": bool(ffprobe),
        "whisper_cpp": shutil.which("whisper-cli") or shutil.which("main") or "",
        "nvidia_smi": shutil.which("nvidia-smi") or "",
        "rocm_smi": shutil.which("rocm-smi") or "",
    }


def probe_gpu() -> Dict[str, Any]:
    """尽量把 GPU 型号/显存抓出来。"""
    out: Dict[str, Any] = {}
    if shutil.which("nvidia-smi"):
        txt = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader"])
        if txt.strip():
            out["nvidia"] = [ln.strip() for ln in txt.strip().splitlines() if ln.strip()]
    if shutil.which("rocm-smi"):
        txt = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram"])
        if txt.strip():
            out["rocm"] = txt.strip().splitlines()[:12]
    if platform.system() == "Darwin":
        txt = _run(["system_profiler", "SPDisplaysDataType"])
        for ln in txt.splitlines():
            if "Chipset Model" in ln or "VRAM" in ln:
                out.setdefault("apple_gpu", []).append(ln.strip())
    # onnxruntime 的 provider 列表最直接
    if _has("onnxruntime"):
        try:
            import onnxruntime as ort  # type: ignore
            out["onnxruntime_providers"] = list(ort.get_available_providers())
            out["onnxruntime_version"] = ort.__version__
        except Exception as exc:
            out["onnxruntime_error"] = str(exc)
    if _has("torch"):
        try:
            import torch  # type: ignore
            out["torch_version"] = torch.__version__
            out["torch_cuda"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                out["torch_cuda_version"] = torch.version.cuda
            out["torch_hip"] = bool(getattr(torch.version, "hip", None))
        except Exception as exc:
            out["torch_error"] = str(exc)
    if _has("mlx.core"):
        out["mlx"] = True
    return out


def probe_packages() -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    for engine, deps in ENGINE_DEPS.items():
        res[engine] = {d: _has(d) for d in deps}
    return res


def _dir_size(path: Path) -> int:
    """目录递归大小（stat() 对目录只给 inode 大小，必须自己走一遍）。"""
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def probe_models() -> Dict[str, Any]:
    """models/ 目录里已经有哪些模型（本地路径约定：<repo> 里的 / 换成 __）。"""
    models = ROOT / "models"
    if not models.exists():
        return {}
    return {p.name: _dir_size(p) for p in models.iterdir() if p.is_dir()}


def available_engines(os_info: Dict[str, Any], pkgs: Dict[str, Any],
                       tools: Dict[str, Any]) -> List[str]:
    """按当前环境推断「装好依赖就能跑」的引擎。"""
    out: List[str] = []
    apple = os_info["is_apple_silicon"]
    if apple and all(pkgs["qwen3 / whisper（MLX）"].values()):
        out.append("qwen3 / whisper（MLX）")
    if pkgs["faster-whisper"].get("faster_whisper"):
        out.append("faster-whisper（CUDA / CPU）")
    if pkgs["funasr"].get("funasr"):
        out.append("funasr（CUDA / ROCm / CPU）")
    if pkgs["sherpa-onnx"].get("sherpa_onnx"):
        out.append("sherpa-onnx（CUDA / DirectML / CPU）")
    if tools.get("whisper_cpp"):
        out.append("whisper.cpp（CUDA / Vulkan / CPU）")
    if pkgs["onnxruntime"].get("onnxruntime"):
        out.append("onnxruntime（CPU / CUDA / DirectML）")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    os_info = probe_os()
    tools = probe_tools()
    gpu = probe_gpu()
    pkgs = probe_packages()
    models = probe_models()
    avail = available_engines(os_info, pkgs, tools)

    report = {
        "os": os_info,
        "tools": tools,
        "gpu": gpu,
        "packages": pkgs,
        "models": models,
        "available_engines": avail,
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    print("=" * 62)
    print("跨平台 ASR 环境探针")
    print("=" * 62)
    print(f"\n【系统】{os_info['system']} {os_info['release']} ({os_info['machine']})")
    print(f"【Python】{os_info['python']}"
          f"{'  ·  Apple Silicon' if os_info['is_apple_silicon'] else ''}")

    print("\n【外部工具】")
    for k, label in (("ffmpeg_ok", "ffmpeg（切块/静音检测必需）"),
                      ("ffprobe_ok", "ffprobe"),
                      ("whisper_cpp", "whisper.cpp 可执行文件"),
                      ("nvidia_smi", "nvidia-smi"),
                      ("rocm_smi", "rocm-smi")):
        v = tools.get(k)
        ok = bool(v)
        print(f"  {'✅' if ok else '❌'} {label:28s} {v if isinstance(v, str) else ''}")

    print("\n【GPU】")
    if not gpu:
        print("  （没探测到）")
    for k, v in gpu.items():
        print(f"  · {k}: {v}")

    print("\n【Python 依赖】")
    for engine, deps in pkgs.items():
        if not deps:
            continue
        marks = "  ".join(f"{d}{'✅' if ok else '❌'}" for d, ok in deps.items())
        print(f"  {engine:26s} {marks}")

    print("\n【已下载模型】")
    if not models:
        print("  （models/ 是空的）")
    for name, size in models.items():
        print(f"  · {name}  ({size / 1024 ** 3:.2f} GB)")

    print("\n【按当前环境可用的引擎】")
    if avail:
        for e in avail:
            print(f"  ✅ {e}")
    else:
        print("  ❌ 还没有任何引擎的依赖 —— 按《跨平台 ASR 方案》装对应 extras")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
