#!/usr/bin/env python3
"""为每个音频算出自适应静音阈值与建议增益。

背景：不同录音响度相差 11 dB，固定 -32 dB 的静音阈值对安静文件会误切。
本脚本从每个文件的 1 秒块 RMS 分布推出 (噪声底 p25, 语音电平 p75)，
给出阈值；并给出把「语音电平」拉到 -22 dB 的增益。
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SR = 16000
BLOCK = SR  # 1 秒


def profile(src: Path) -> dict:
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
    p25, p50, p75, p90 = (float(np.percentile(db, p)) for p in (25, 50, 75, 90))
    thr = p25 + 0.55 * (p75 - p25)
    thr = max(-48.0, min(-28.0, thr))
    return {"p25": p25, "p50": p50, "p75": p75, "p90": p90,
            "adaptive_db": round(thr, 1),
            "gain_db": round(-22.0 - p75, 1)}


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    src_dir = root / "source"
    out = {}
    for f in sorted(src_dir.glob("*.mp3")):
        p = profile(f)
        if not p:
            continue
        out[f.stem] = p
        print(f"{f.stem[:34]:36s} p25={p['p25']:6.1f} p50={p['p50']:6.1f} "
              f"p75={p['p75']:6.1f} → 自适应阈值 {p['adaptive_db']:6.1f} dB  "
              f"建议增益 {p['gain_db']:+.1f} dB", flush=True)
    Path("/tmp/thresholds.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print("\n→ /tmp/thresholds.json")
