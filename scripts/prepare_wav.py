#!/usr/bin/env python3
"""预生成带增益的 16k 单声道 wav，让服务端沿用「按文件自适应」的静音阈值。

背景：服务进程里跑的是固定 -32 dB 的静音阈值，而实测各录音的噪声底相差
11 dB（噪声底 -28 dB 的文件用 -32 dB 阈值等于完全没切静音，底噪整段送进
ASR）。这里按每个文件自己的分布算出应有阈值 T，再给音频加一个增益
G = -32 - T，这样服务端用 -32 dB 检测时，等价于在原始音频上用 T。

增益幅度很小（±4 dB），不会改变识别难度，但能让切分真正按文件自适应。

用法：
    python scripts/prepare_wav.py                 # 处理所有还没 wav 的文件
    python scripts/prepare_wav.py --force         # 重建
    python scripts/prepare_wav.py "某个文件.mp3"
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402

SR = 16000
BLOCK = SR  # 1 秒


def profile(src: Path) -> dict:
    """1 秒块 RMS 分布（p25 噪声底 / p75 语音电平）。"""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(src),
           "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    x = np.frombuffer(out, dtype="<f4").astype(np.float32)
    n = len(x) // BLOCK
    if not n:
        return {}
    frames = x[: n * BLOCK].reshape(n, BLOCK)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    p25, p75 = float(np.percentile(db, 25)), float(np.percentile(db, 75))
    thr = max(-48.0, min(-28.0, p25 + 0.55 * (p75 - p25)))
    return {"p25": p25, "p75": p75, "adaptive_db": round(thr, 1),
            "gain_db": round(config.SILENCE_NOISE_DB - thr, 1)}


def prepare(src: Path, force: bool = False) -> Path | None:
    doc_id = src.stem
    out_dir = config.OUTPUT_DIR / doc_id
    wav = out_dir / "audio_16k.wav"
    if wav.exists() and wav.stat().st_size > 44 and not force:
        print(f"  · {doc_id}: 已有 wav，跳过")
        return wav
    p = profile(src)
    if not p:
        print(f"  ✗ {src.name}: 音频太短")
        return None
    gain = p["gain_db"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-nostdin", "-v", "error", "-i", str(src),
           "-vn", "-ac", "1", "-ar", str(SR)]
    if abs(gain) > 0.05:
        cmd += ["-af", f"volume={gain:.2f}dB"]
    cmd += ["-c:a", "pcm_s16le", str(wav)]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"  ✓ {doc_id}: 噪声底 {p['p25']:.1f} / 语音 {p['p75']:.1f} dB → "
          f"阈值 {p['adaptive_db']:.1f} dB，增益 {gain:+.1f} dB")
    return wav


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.files:
        files = []
        for n in args.files:
            f = config.SOURCE_DIR / n
            if not f.exists():
                f = next(config.SOURCE_DIR.glob(Path(n).stem + ".*"), f)
            files.append(f)
    else:
        files = sorted(p for p in config.SOURCE_DIR.iterdir()
                       if p.is_file() and p.suffix.lower() in config.AUDIO_EXTS)
    for src in files:
        prepare(src, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
