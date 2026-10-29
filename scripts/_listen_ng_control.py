#!/usr/bin/env python3
"""正对照：这套「只输出拼音」的听音方法，能不能分辨 chén 和 chéng？

用同一录音里**确定无疑**的词：
  · 「流程」liú chéng（讲者，30+ 次）、「程度」chéng dù（讲者）
  · 「陈伟」chén wěi（讲者念的人名）、「陈毅」chén yì（讲者念的人名）
如果模型对 流程/程度 输出带 g 的 chéng、对 陈伟/陈毅 输出不带 g 的 chén，
说明这套方法确实能分辨 -ng 尾音 → 那它对本篇「X老师」给 chén 就是有效信号。
反之若它对 流程 也只给 chén，说明方法不敏感，两种答案都不可用。
"""
from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "output" / "_listen"

PROMPT = (
    "请只输出这段音频中被提到的那个讲师的**姓氏**的汉语拼音（带声调，例如 zhāng）。"
    "只输出拼音本身，不要写汉字，不要写解释，不要写其他内容。"
)

# 词义不同，提示词跟着换，但保持「只输出拼音」的形状
WORD_PROMPT = (
    "请只输出这段音频中那个词的汉语拼音（带声调）。"
    "只输出拼音本身，不要写汉字，不要写解释，不要写其他内容。"
)

REPEAT = 3

CASES = [
    # (标签, 起点, 时长, 提示词, 期望含 chéng)
    ("pos_liucheng", 1122.9, 2.2, WORD_PROMPT, True),   # 流程
    ("pos_chengdu", 4449.9, 2.0, WORD_PROMPT, True),    # 程度
    ("pos_liucheng2", 2391.0, 2.2, WORD_PROMPT, True),  # 流程是啥意思
    ("neg_chenwei", 2251.7, 1.9, WORD_PROMPT, False),   # 陈伟
    ("neg_chenyi", 4434.4, 1.9, WORD_PROMPT, False),    # 陈毅
]


def upload(path: Path) -> str:
    out = subprocess.run(["mcode-tools", "upload_temp_url", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)["temp_url"]


def listen(url: str, tag: str, k: int, prompt: str) -> str:
    args = {"audio_info": {"url": url, "mime_type": "audio/mpeg", "prompt": prompt}}
    ap = D / f"args_{tag}_{k}.json"
    ap.write_text(json.dumps(args, ensure_ascii=False), encoding="utf-8")
    out = subprocess.run(
        ["mcode-tools", "connector", "call", "connector__matrix__listen_audio",
         "--args-file", str(ap)], capture_output=True, text=True)
    raw = (out.stdout or out.stderr).strip()
    try:
        return str(json.loads(raw).get("transcription", raw)).strip()
    except Exception:
        return raw


def main() -> int:
    src = ROOT / "source" / "组织变革管理与变革领导力-程建刚.mp3"
    for tag, st, dur, prompt, want_ng in CASES:
        clip = D / f"pc_{tag}.mp3"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-ss", f"{st}", "-to", f"{st + dur}", "-i", str(src),
                        "-ac", "1", "-ar", "16000", str(clip)], check=True)
        url = upload(clip)
        raw = [listen(url, tag, k, prompt) for k in range(REPEAT)]
        joined = " | ".join(raw)
        got_ng = "g" in joined.lower().replace(" ", "")
        hit = "✅" if got_ng == want_ng else "❌"
        print(f"{hit} {tag:16s} 期望{'带g(chéng)' if want_ng else '不带g(chén)'} → {joined}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
