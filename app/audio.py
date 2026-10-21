"""音频预处理：探测信息、转码、静音检测、切片、波形峰值、预览音频。

统一走 ffmpeg（系统已装），不依赖 soundfile / librosa，内存占用恒定。
"""

from __future__ import annotations

import json
import math
import struct
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from . import config


class AudioError(RuntimeError):
    pass


def _run(cmd: List[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "ignore").strip().splitlines()[-6:]
        raise AudioError(f"{cmd[0]} 失败: {' | '.join(tail)}")
    return proc


@dataclass
class MediaInfo:
    duration: float
    sample_rate: int
    channels: int
    codec: str
    size: int
    bit_rate: int

    def to_dict(self) -> dict:
        return asdict(self)


def probe(path: Path) -> MediaInfo:
    proc = _run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration,bit_rate,size",
            "-show_entries", "stream=codec_type,codec_name,sample_rate,channels",
            "-of", "json", str(path),
        ],
        timeout=120,
    )
    data = json.loads(proc.stdout.decode() or "{}")
    fmt = data.get("format", {}) or {}
    streams = data.get("streams", []) or []
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    try:
        duration = float(fmt.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    return MediaInfo(
        duration=duration,
        sample_rate=int(audio.get("sample_rate") or 0),
        channels=int(audio.get("channels") or 0),
        codec=str(audio.get("codec_name") or ""),
        size=int(fmt.get("size") or 0),
        bit_rate=int(fmt.get("bit_rate") or 0),
    )


def to_wav16k(src: Path, dst: Path) -> Path:
    """转成 16kHz 单声道 PCM WAV（ASR 友好的标准格式）。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 44:
        return dst
    _run([
        "ffmpeg", "-y", "-nostdin", "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(config.SAMPLE_RATE),
        "-c:a", "pcm_s16le", str(dst),
    ])
    return dst


def level_profile(wav: Path, block_sec: float = 1.0) -> Dict[str, float]:
    """统计 16k 单声道 wav 的逐块 RMS（dB）分布。

    不同录音的响度差别很大（实测 11 dB），固定静音阈值对某些文件会误切
    轻声说话、对另一些文件又切不掉噪声。用本文件的分布自适应定阈值。
    块长取 1 秒：silencedetect 是在细粒度上判静音，用 1 秒块才能代表
    「有人说话」时的实际电平。
    """
    import math
    import wave

    import numpy as np

    block = int(config.SAMPLE_RATE * block_sec)
    levels: List[float] = []
    with wave.open(str(wav), "rb") as w:
        while True:
            data = w.readframes(block)
            if not data:
                break
            x = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
            if x.size == 0:
                continue
            rms = float(np.sqrt((x.astype(np.float64) ** 2).mean() + 1e-12))
            levels.append(20 * math.log10(rms))
    if not levels:
        return {}
    arr = np.array(levels)
    return {
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def adaptive_silence_db(wav: Path) -> float:
    """按本文件的响度分布推静音阈值（dB）。

    p25 近似噪声底，p75 近似语音电平，阈值取两者之间偏噪声底一侧
    （SILENCE_OFFSET 是「从噪声底往语音方向走」的比例，0.4 ≈ 噪声底 +40% 行程）。
    再夹到 [SILENCE_MIN_DB, SILENCE_MAX_DB] 防止离谱。
    """
    profile = level_profile(wav)
    if not profile:
        return config.SILENCE_NOISE_DB
    floor = profile["p25"]
    speech = profile["p75"]
    ratio = min(max(config.SILENCE_OFFSET, 0.0), 1.0)
    thr = floor + ratio * (speech - floor)
    thr = max(config.SILENCE_MIN_DB, min(config.SILENCE_MAX_DB, thr))
    return round(thr, 1)


def detect_silence(src: Path, noise_db: float | None = None, min_sec: float | None = None) -> List[Tuple[float, float]]:
    """用 ffmpeg silencedetect 找出静音区间，返回 [(start, end), ...]。"""
    noise_db = config.SILENCE_NOISE_DB if noise_db is None else noise_db
    min_sec = config.SILENCE_MIN_SEC if min_sec is None else min_sec
    proc = _run(
        [
            "ffmpeg", "-nostdin", "-i", str(src),
            "-af", f"silencedetect=noise={noise_db}dB:d={min_sec}",
            "-f", "null", "-",
        ],
        timeout=3600,
    )
    stderr = proc.stderr.decode("utf-8", "ignore")
    silences: List[Tuple[float, float]] = []
    pending: float | None = None
    for line in stderr.splitlines():
        if "silence_start:" in line:
            try:
                pending = float(line.split("silence_start:")[1].split()[0])
            except (IndexError, ValueError):
                pending = None
        elif "silence_end:" in line and pending is not None:
            try:
                end = float(line.split("silence_end:")[1].split()[0])
                silences.append((pending, end))
            except (IndexError, ValueError):
                pass
            pending = None
    return silences


def speech_regions(silences: List[Tuple[float, float]], duration: float) -> List[Tuple[float, float]]:
    """由静音区间推导语音区间，并合并过短的停顿。"""
    regions: List[Tuple[float, float]] = []
    cursor = 0.0
    for s_start, s_end in silences:
        if s_start - cursor > 0.01:
            regions.append((cursor, s_start))
        cursor = max(cursor, s_end)
    if duration - cursor > 0.01:
        regions.append((cursor, duration))

    merged: List[Tuple[float, float]] = []
    for start, end in regions:
        if merged and start - merged[-1][1] < config.MIN_GAP_MERGE:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    out: List[Tuple[float, float]] = []
    for start, end in merged:
        if end - start < config.SEG_MIN_SECONDS:
            # 过短的碎块并入上一段
            if out:
                out[-1] = (out[-1][0], end)
            continue
        out.append((start, end))
    return out


def chunk_regions(
    regions: List[Tuple[float, float]],
    silences: List[Tuple[float, float]],
    max_sec: float | None = None,
    min_sec: float | None = None,
) -> List[Tuple[float, float]]:
    """把语音区间整理成适合识别的块。

    - 短于 min_sec 的块与相邻块合并（提供上下文，提升识别准确率）
    - 长于 max_sec 的块在附近的静音处切开（不切断句子）
    """
    max_sec = max_sec or config.SEG_MAX_SECONDS
    min_sec = min_sec or config.SEG_MIN_MERGE
    if not regions:
        return []

    # ---- 1. 合并过短的块
    merged: List[Tuple[float, float]] = []
    for start, end in regions:
        if merged and (end - start) < min_sec and (merged[-1][1] - merged[-1][0]) < max_sec:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    # ---- 2. 切开过长的块（切点吸附到静音中点）
    silence_mids = sorted((s + e) / 2 for s, e in silences)
    out: List[Tuple[float, float]] = []
    for start, end in merged:
        if end - start <= max_sec:
            out.append((start, end))
            continue
        pieces = _split_at_silence(start, end, max_sec, silence_mids)
        out.extend(pieces)
    return out


def _split_at_silence(
    start: float,
    end: float,
    max_sec: float,
    silence_mids: List[float],
) -> List[Tuple[float, float]]:
    """在长区间内找静音点做切分，切点尽量靠近等分点。"""
    n = max(1, math.ceil((end - start) / max_sec))
    step = (end - start) / n
    cuts: List[float] = []
    for i in range(1, n):
        ideal = start + step * i
        window = step * 0.4
        best = None
        best_dist = None
        for mid in silence_mids:
            if start < mid < end and abs(mid - ideal) <= window:
                d = abs(mid - ideal)
                if best_dist is None or d < best_dist:
                    best, best_dist = mid, d
        cuts.append(best if best is not None else ideal)

    bounds = [start] + cuts + [end]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def cut(src: Path, start: float, end: float, dst: Path) -> Path:
    """切出 [start, end) 区间的 16k 单声道 wav（精确到毫秒）。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y", "-nostdin",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(config.SAMPLE_RATE),
        "-c:a", "pcm_s16le", str(dst),
    ], timeout=300)
    return dst


def make_preview(src: Path, dst: Path) -> Path:
    """生成小体积预览音频，浏览器播放/seek 更顺滑。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 1024:
        return dst
    _run([
        "ffmpeg", "-y", "-nostdin", "-i", str(src),
        "-vn", "-ac", "1", "-ar", "22050", "-b:a", config.PREVIEW_BITRATE,
        "-movflags", "+faststart", str(dst),
    ], timeout=3600)
    return dst


def compute_peaks(src: Path, buckets_per_sec: int | None = None) -> bytes:
    """计算波形峰值包络，返回 int16 小端二进制（每桶 min/max 两个采样）。

    通过 ffmpeg 抽成 8kHz 单声道 s16le 流后按桶聚合，内存占用与音频时长成正比，
    2 小时音频约 57MB，可接受。
    """
    buckets_per_sec = buckets_per_sec or config.PEAK_BUCKETS_PER_SEC
    info = probe(src)
    rate = 8000
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", str(src), "-vn", "-ac", "1", "-ar", str(rate),
         "-f", "s16le", "-acodec", "pcm_s16le", "-"],
        capture_output=True, timeout=3600,
    )
    if proc.returncode != 0:
        raise AudioError("波形峰值计算失败")
    data = proc.stdout
    samples_per_bucket = max(1, rate // buckets_per_sec)
    n_buckets = len(data) // 2 // samples_per_bucket
    out = bytearray()
    for b in range(n_buckets):
        lo = 32767
        hi = -32768
        base = b * samples_per_bucket * 2
        for i in range(samples_per_bucket):
            off = base + i * 2
            if off + 2 > len(data):
                break
            v = struct.unpack_from("<h", data, off)[0]
            if v < lo:
                lo = v
            if v > hi:
                hi = v
        out += struct.pack("<hh", lo, hi)
    return bytes(out)


def fmt_ts(seconds: float, with_ms: bool = True) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if with_ms:
        return f"{h:02d}:{m:02d}:{s:06.3f}"
    return f"{h:02d}:{m:02d}:{int(s):02d}"
