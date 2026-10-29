#!/usr/bin/env python3
"""声学判别：讲者姓氏到底是「陈」(chén) 还是「程」(chéng)。

思路：用**同一个文件里已经确定无疑的参照音**做模板——
  · 「陈毅」「陈伟」 → 一定是 chén（chén yì / chén wěi，历史上确有其人）
  · 「程度」「过程」 → 一定是 chéng（chéng dù / guò chéng）
  · 待判目标：主持人说的「陈老师」

方法：13 维 log-mel 频谱 + DTW 动态时间规整，滑动窗在目标段里找最佳匹配，
比较它跟 chén 模板和 chéng 模板谁更接近。同一录音、同一麦克风、同一说话人
（参照音来自讲者本人），所以是公平比较。

用法：
    .venv/bin/python scripts/_acoustic_surname.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SR = 16000
N_FFT = 512
HOP = 160          # 10ms
N_MELS = 13        # 够用且快
WIN_FRAMES = 60    # 600ms 滑窗
STEP_FRAMES = 15   # 150ms 步进


def load(src: Path, start: float, dur: float) -> np.ndarray:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}",
           "-to", f"{start + dur:.3f}", "-i", str(src), "-ac", "1", "-ar", str(SR),
           "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def logmel(x: np.ndarray) -> np.ndarray:
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    n = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    frames = x[idx] * np.hanning(N_FFT)[None, :]
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2

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


def dtw(a: np.ndarray, b: np.ndarray) -> float:
    """标准 DTW，返回平均代价（越小越像）。

    注意：不能用「整行向量化」的写法——D[i][j] 依赖同一行左边的 D[i][j-1]，
    numpy 右值先整体求值会把这条依赖算成 inf。距离矩阵先算好，
    再按标量递推（57×77 ≈ 4.4k 次迭代，够快）。
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 1e9
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        drow = d[i - 1]
        for j in range(1, m + 1):
            best = acc[i - 1, j]
            if acc[i, j - 1] < best:
                best = acc[i, j - 1]
            if acc[i - 1, j - 1] < best:
                best = acc[i - 1, j - 1]
            acc[i, j] = drow[j - 1] + best
    return float(acc[n, m] / (n + m))


def best_matches(x: np.ndarray, templates: dict[str, list[np.ndarray]]) -> dict[str, float]:
    best = {k: 1e9 for k in templates}
    span = WIN_FRAMES * HOP
    if len(x) < span:
        return best
    n_windows = (len(x) - span) // (STEP_FRAMES * HOP) + 1
    for wi in range(n_windows):
        start = wi * STEP_FRAMES * HOP
        seg = logmel(x[start:start + span])
        for k, tpls in templates.items():
            for tpl in tpls:
                c = dtw(seg, tpl)
                if c < best[k]:
                    best[k] = c
    return best


def main() -> int:
    src = ROOT / "source" / "组织变革管理与变革领导力-程建刚.mp3"

    # 参照音（讲者本人 / 画面内人名，确定无疑）
    chen_refs = [
        ("陈毅", 4434.90, 0.8),   # chén yì
        ("陈伟", 2252.10, 0.8),   # chén wěi
    ]
    cheng_refs = [
        ("程度", 4450.85, 0.9),   # chéng dù
        ("过程", 1855.90, 0.9),   # guò chéng
    ]
    # 待判目标：主持人称「陈老师」的每一处
    targets = [
        ("开场白", 1.92, 6.30),
        ("企业家引述", 2750.53, 2760.13),
        ("学员提问", 3638.00, 3647.00),
    ]

    templates: dict[str, list[np.ndarray]] = {"chen": [], "cheng": []}
    print("载入参照音（同一录音、同一说话人）…")
    for label, st, du in chen_refs:
        templates["chen"].append(logmel(load(src, st, du)))
        print(f"  chén  参照: {label} @{st:.2f}s")
    for label, st, du in cheng_refs:
        templates["cheng"].append(logmel(load(src, st, du)))
        print(f"  chéng 参照: {label} @{st:.2f}s")

    print("\n目标逐一比对：")
    votes = {"chen": 0, "cheng": 0}
    for label, st, du in targets:
        x = load(src, st, du)
        b = best_matches(x, templates)
        c, g = b["chen"], b["cheng"]
        win = "chén (陈)" if c < g else "chéng (程)"
        votes["chen" if c < g else "cheng"] += 1
        margin = abs(c - g) / max(c, g) * 100
        print(f"  {label:10s} → chén {c:7.3f}｜chéng {g:7.3f}"
              f"｜更像 {win}（相差 {margin:.1f}%）")
    print(f"\n合计：chén {votes['chen']} 票，chéng {votes['cheng']} 票")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
