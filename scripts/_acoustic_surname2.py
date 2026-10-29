#!/usr/bin/env python3
"""声学判别 v2：把姓氏单独切成一个音节再比对（陈 chén vs 程 chéng）。

v1 的问题（已确认）：
  1. 模板是「陈毅/程度」这种**双音节词**，目标却是「陈老师」**三音节**，
     DTW 会把「老师」硬塞进第二个模板音节里，比的主要是「老/师 vs 毅/伟/度/程」，
     姓氏本身的差异被淹没 → 结论不可用。
  2. 模板全是**讲者**的声音，目标里只有一处是讲者本人，另两处是主持人/学员。
  3. 滑动窗可能根本没落在姓氏上（最小代价窗口可能在别处）。

v2 的做法：
  · 按能量谷做音节切分，**只取姓氏那一个音节**（都是词首音节）。
  · 模板与目标都来自讲者本人（其中「企业家引述」那处是讲者复述企业家的话），
    说话人一致；跨说话人的两处仅作参考。
  · 加 CMN（逐系数去均值）压掉信道/音色差异，让鼻音尾的差异凸显。
  · 打印切分点，方便人耳复核；同时把切好的单音节存成 wav 直接听。

用法：
    .venv/bin/python scripts/_acoustic_surname2.py
"""
from __future__ import annotations

import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SR = 16000
N_FFT = 512
HOP = 160          # 10ms
N_MELS = 24
FRAME_LEN = 400    # 25ms

OUT = ROOT / "output" / "_listen"


def load(src: Path, start: float, dur: float) -> np.ndarray:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}",
           "-to", f"{start + dur:.3f}", "-i", str(src), "-ac", "1", "-ar", str(SR),
           "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def save_wav(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(x, -1.0, 1.0)
    pcm = (pcm * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def frames(x: np.ndarray) -> np.ndarray:
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    n = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx] * np.hanning(N_FFT)[None, :]


def logmel(x: np.ndarray) -> np.ndarray:
    spec = np.abs(np.fft.rfft(frames(x), axis=1)) ** 2

    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    lo, hi = hz2mel(100.0), hz2mel(SR / 2)
    pts = mel2hz(np.linspace(lo, hi, N_MELS + 2))
    bins = np.floor((N_FFT + 1) * pts / SR).astype(int)
    fb = np.zeros((N_MELS, spec.shape[1]))
    for i in range(N_MELS):
        a, b, c = bins[i], bins[i + 1], bins[i + 2]
        b = max(b, a + 1)
        c = max(c, b + 1)
        b = min(b, spec.shape[1])
        c = min(c, spec.shape[1])
        fb[i, a:b] = np.linspace(0, 1, b - a, endpoint=False)
        fb[i, b:c] = np.linspace(1, 0, c - b, endpoint=False)
    mel = spec @ fb.T
    return np.log(mel + 1e-8)


def cmn(m: np.ndarray) -> np.ndarray:
    """逐系数去均值：压掉信道/音色差异，突出鼻音尾的形状。"""
    if len(m) == 0:
        return m
    return m - m.mean(axis=0, keepdims=True)


def energy_env(x: np.ndarray) -> np.ndarray:
    """25ms 帧短时能量（dB），10ms 步进。"""
    n = 1 + max(0, (len(x) - FRAME_LEN) // HOP)
    idx = np.arange(FRAME_LEN)[None, :] + HOP * np.arange(n)[:, None]
    fr = x[idx]
    e = np.sqrt((fr ** 2).mean(axis=1) + 1e-12)
    return 20 * np.log10(e)


def syllable_bounds(x: np.ndarray, max_syl: int = 4) -> list[int]:
    """用能量谷找音节边界，返回样本点索引（升序，含 0 和 len(x)）。

    做法：帧能量 → 中值平滑 → 找局部极小；在极小值之间按「谷深于全局峰值 12dB」
    挑出真正的音节交界。
    """
    e = energy_env(x)
    if len(e) < 5:
        return [0, len(x)]
    # 中值平滑去毛刺
    k = 5
    pad = np.pad(e, (k // 2, k // 2), mode="edge")
    sm = np.array([np.median(pad[i:i + k]) for i in range(len(e))])
    peak = sm.max()
    # 候选谷：局部极小且低于 峰值-12dB
    cand = []
    for i in range(1, len(sm) - 1):
        if sm[i] <= sm[i - 1] and sm[i] <= sm[i + 1] and sm[i] < peak - 12:
            cand.append(i)
    # 谷之间的「峰」高度 ≥ 谷底 + 10dB 才算真交界（否则是颤音）
    real = []
    for i in cand:
        lo = max(0, i - 12)
        hi = min(len(sm), i + 13)
        if sm[lo:hi].max() - sm[i] >= 10:
            real.append(i)
    # 取最强的前 (max_syl-1) 个谷
    real.sort(key=lambda i: sm[i])
    real = real[: max_syl - 1]
    real.sort()
    bounds = [0] + [i * HOP for i in real] + [len(x)]
    # 去重 + 去掉过近的
    out = []
    for b in bounds:
        if not out or b - out[-1] > int(0.06 * SR):
            out.append(b)
    return out


def dtw(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 1e9
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        drow = d[i - 1]
        prev = acc[i - 1]
        cur = acc[i]
        for j in range(1, m + 1):
            best = prev[j]
            if cur[j - 1] < best:
                best = cur[j - 1]
            if prev[j - 1] < best:
                best = prev[j - 1]
            cur[j] = drow[j - 1] + best
    return float(acc[n, m] / (n + m))


def main() -> int:
    src = ROOT / "source" / "组织变革管理与变革领导力-程建刚.mp3"

    # (标签, 起点, 时长, 取第几个音节)  —— 姓氏都在词首
    chén_items = [
        ("陈伟", 2252.10, 0.9, 0),
        ("陈毅", 4434.82, 1.0, 0),
    ]
    chéng_items = [
        ("程度", 4450.32, 1.1, 0),   # chéng dù  → 程度
        ("过程", 1855.40, 1.0, 1),   # guò chéng → 第 2 个音节是 chéng
    ]
    targets = [
        ("开场白(主持人)", 1.92, 6.40, 0),
        ("企业家引述(讲者)", 2749.60, 11.5, 0),
        ("学员提问(学员)", 3638.00, 9.5, 0),
    ]

    def syll(x: np.ndarray, k: int) -> np.ndarray:
        b = syllable_bounds(x)
        if len(b) <= k + 1:
            return x
        return x[b[k]:b[k + 1]]

    print("=== 切分并导出单音节 ===")
    tpl: dict[str, list[np.ndarray]] = {"chen": [], "cheng": []}
    for group, items in (("chen", chén_items), ("cheng", chéng_items)):
        for label, st, du, k in items:
            x = load(src, st, du)
            s = syll(x, k)
            b = syllable_bounds(x)
            times = [round(v / SR, 3) for v in b]
            print(f"  {group:6s} {label:6s} @{st:8.2f}  边界(s)={times}  音节时长={len(s)/SR:.3f}s")
            save_wav(OUT / f"syl_{group}_{label}.wav", s)
            tpl[group].append(cmn(logmel(s)))

    tgt: dict[str, np.ndarray] = {}
    for label, st, du, k in targets:
        x = load(src, st, du)
        b = syllable_bounds(x)
        # 目标所在词「陈老师」在片段里位置靠后：先按能量谷找所有音节，取第 1 个
        s = syll(x, k)
        times = [round(v / SR, 3) for v in b]
        print(f"  目标   {label:18s} @{st:8.2f}  边界(s)={times}  音节时长={len(s)/SR:.3f}s")
        save_wav(OUT / f"syl_target_{label.split('(')[0]}.wav", s)
        tgt[label] = cmn(logmel(s))

    print("\n=== 单音节 DTW 比对（越小越像） ===")
    for label, t in tgt.items():
        c = min(dtw(t, x) for x in tpl["chen"])
        g = min(dtw(t, x) for x in tpl["cheng"])
        win = "chén(陈)" if c < g else "chéng(程)"
        print(f"  {label:18s} → chén {c:7.3f}｜chéng {g:7.3f}｜更像 {win}"
              f"（相差 {abs(c-g)/max(c,g)*100:.1f}%）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
