#!/usr/bin/env python3
"""音频特征分析：找出「非语音段」并给出可切片的统计。

动机：不同录音的底噪/动态范围差别很大，纯静音阈值（silencedetect）会把
「很响但不是语音」的内容（音乐、视频、削波失真）误当成语音送进 ASR，
模型就会在原地刷屏（幻觉）。本脚本用一整遍解码 + 分窗特征，
把这类段落标出来，供管线按文件自适应地跳过。

用法：
    python scripts/analyze_audio.py 音频.mp3                 # 整段报告
    python scripts/analyze_audio.py 音频.mp3 --top 10        # 列出最可疑的时间段
    python scripts/analyze_audio.py 音频.mp3 --json out.json # 机器可读
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
FRAME = int(SR * 0.5)  # 每 0.5 秒一帧


def decode(src: Path, start: float | None = None, duration: float | None = None) -> np.ndarray:
    """解码为 16k 单声道 float32（-1..1）。"""
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(src), "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype="<f4").astype(np.float32)


def features(x: np.ndarray) -> dict[str, np.ndarray]:
    """逐帧特征：响度、削波、过零率、高频占比、谱平坦度、谱质心。"""
    n = len(x) // FRAME
    if n == 0:
        return {}
    frames = x[: n * FRAME].reshape(n, FRAME)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1) + 1e-12)
    peak = np.abs(frames).max(axis=1)
    zcr = (np.diff(np.signbit(frames).astype(np.int8), axis=1) != 0).sum(axis=1) / FRAME

    spec = np.abs(np.fft.rfft(frames * np.hanning(FRAME), axis=1)) + 1e-12
    freqs = np.fft.rfftfreq(FRAME, 1.0 / SR)
    total = (spec ** 2).sum(axis=1)
    hf = (spec[:, freqs > 4000] ** 2).sum(axis=1)
    mid = (spec[:, (freqs > 200) & (freqs <= 4000)] ** 2).sum(axis=1)
    centroid = (spec * freqs).sum(axis=1) / spec.sum(axis=1)
    # 谱平坦度：几何均值/算术均值，接近 1 表示噪声/宽带内容
    log_s = np.log(spec)
    flat = np.exp(log_s.mean(axis=1)) / spec.mean(axis=1)
    # 谐波性：自相关峰值
    ac = np.fft.irfft(np.abs(np.fft.rfft(frames, axis=1)) ** 2, axis=1)[:, :FRAME]
    ac0 = np.maximum(ac[:, 0], 1e-12)
    harm = ac[:, 20:400].max(axis=1) / ac0

    return {
        "t": (np.arange(n) + 0.5) * (FRAME / SR),
        "rms_db": 20 * np.log10(rms),
        "peak_db": 20 * np.log10(peak),
        "clip": (np.abs(frames) >= 0.985).mean(axis=1),
        "zcr": zcr,
        "hf_ratio": 10 * np.log10(hf / total + 1e-12),
        "mid_ratio": 10 * np.log10(mid / total + 1e-12),
        "centroid": centroid,
        "flatness": flat,
        "harmony": harm,
    }


def speech_score(f: dict[str, np.ndarray]) -> np.ndarray:
    """语音似然评分（0~1，越大越像连续语音）。

    依据：连续语音的能量集中在 0~4 kHz、过零率适中、频谱不平坦（谐波结构）、
    幅度起伏大；音乐/宽带噪声/削波失真则相反。
    """
    s = np.ones_like(f["rms_db"])
    # 谐波结构明显 → 加分（语音）
    s += np.clip((f["harmony"] - 0.30) / 0.25, -1.0, 1.0)
    # 高频占比过高 → 减分（宽带/音乐）
    s -= np.clip((f["hf_ratio"] + 22.0) / 12.0, 0.0, 1.5)
    # 谱太平坦 → 减分（噪声）
    s -= np.clip((f["flatness"] - 0.10) / 0.25, 0.0, 1.5)
    # 削波严重 → 减分
    s -= np.clip(f["clip"] / 0.02, 0.0, 1.5)
    # 过零率异常（太低=单频嗡鸣，太高=白噪）
    s -= np.clip((f["zcr"] - 0.12) / 0.10, 0.0, 1.0)
    return np.clip(s / 2.0, 0.0, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--top", type=int, default=0, help="列出最可疑的 N 段")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    report = []
    for name in args.files:
        src = Path(name)
        if not src.exists():
            src = config.SOURCE_DIR / name
        x = decode(src, args.start, args.duration)
        f = features(x)
        if not f:
            print(f"{src.name}: 音频太短")
            continue
        score = speech_score(f)
        good = score >= 0.5
        print(f"\n=== {src.name} ===")
        print(f"  时长 {len(x)/SR:.1f} 秒　帧数 {len(score)}（每帧 0.5 秒）")
        print(f"  响度  中位 {np.median(f['rms_db']):.1f} dB　最响 {f['rms_db'].max():.1f} dB")
        print(f"  削波  中位 {np.median(f['clip'])*100:.3f}%　最大 {f['clip'].max()*100:.2f}%")
        print(f"  谱平坦 中位 {np.median(f['flatness']):.3f}　谐波性 中位 {np.median(f['harmony']):.3f}")
        print(f"  语音似然  中位 {np.median(score):.2f}　判为非语音占比 {100*(~good).mean():.1f}%")
        # 连续非语音段
        runs = []
        i = 0
        while i < len(good):
            if not good[i]:
                j = i
                while j < len(good) and not good[j]:
                    j += 1
                if (j - i) * 0.5 >= 5:
                    runs.append((f["t"][i], f["t"][j - 1]))
                i = j
            else:
                i += 1
        print(f"  连续非语音段(≥5秒) {len(runs)} 段")
        for s0, e0 in runs[:12]:
            print(f"     {s0:8.1f}s → {e0:8.1f}s  ({e0-s0:.0f} 秒)")
        if args.top:
            idx = np.argsort(score)[: args.top]
            print(f"  最可疑的 {args.top} 个帧：")
            for i in sorted(idx):
                print(f"     t={f['t'][i]:8.1f}s  score={score[i]:.2f}  rms={f['rms_db'][i]:6.1f}dB"
                      f"  clip={f['clip'][i]*100:6.2f}%  flat={f['flatness'][i]:.3f}"
                      f"  harm={f['harmony'][i]:.3f}  hf={f['hf_ratio'][i]:6.1f}dB")
        report.append({
            "file": src.name,
            "duration": len(x) / SR,
            "rms_median": float(np.median(f["rms_db"])),
            "clip_median": float(np.median(f["clip"])),
            "clip_max": float(f["clip"].max()),
            "flat_median": float(np.median(f["flatness"])),
            "harm_median": float(np.median(f["harmony"])),
            "score_median": float(np.median(score)),
            "nonspeech_ratio": float((~good).mean()),
            "nonspeech_runs": [[float(a), float(b)] for a, b in runs],
        })
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n→ {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
