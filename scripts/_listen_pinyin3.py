#!/usr/bin/env python3
"""拼音听音对照实验 v3。

v2 的坏处：提示词里写了「某老师」，模型有时把它原样复读成 "mǒu lǎoshī"，
            5 个对照组里错了 3 个 → 方法不可信。
v3 改法：
  · 提示词去掉「某老师」这类会被复读的字样，只让模型「输出被提到的那位讲师的姓氏拼音」；
  · 每处重复 N 次取多数，抵消单次抖动；
  · 对照组保持不变，方法在对照组上全对才算可用。
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "output" / "_listen"

PROMPT = (
    "请只输出这段音频中被提到的那个讲师的**姓氏**的汉语拼音（带声调，例如 zhāng）。"
    "只输出拼音本身，不要写汉字，不要写解释，不要写其他内容。"
)

REPEAT = 3

# (标签, 源文件, 起点, 时长, 说明, 期望)
SPECS = [
    ("cheng_intro", "组织变革管理与变革领导力-程建刚.mp3", 2.5, 4.0, "本篇·主持人开场", "chéng"),
    ("cheng_quote_w", "组织变革管理与变革领导力-程建刚.mp3", 2752.0, 6.0, "本篇·讲者复述(宽)", "chéng"),
    ("cheng_quote_t", "组织变革管理与变革领导力-程建刚.mp3", 2754.4, 2.4, "本篇·讲者复述(紧)", "chéng"),
    ("cheng_qa", "组织变革管理与变革领导力-程建刚.mp3", 3641.0, 5.5, "本篇·学员提问", "chéng"),
    ("ctrl_zhangtao", "AI 赋能企业项目管理的应用与实践-张涛.mp3", 5.0, 5.0, "对照·张涛", "zhāng"),
    ("ctrl_zhaorong", "20261020PM企业 AI 转型前沿与组织能力体系化建设-赵荣.mp3", 7.5, 3.5, "对照·赵荣", "zhào"),
    ("ctrl_yanshu", "20261021 上午闫树.mp3", 0.0, 3.5, "对照·闫树", "yán"),
    ("ctrl_gaotaishan", "数字经济时代的产业创新：机遇与挑战-高太山.mp3", 23.0, 5.0, "对照·高太山", "gāo"),
    ("ctrl_zhangrongchen", "20261020AM加强党性修养做好党性分析-张荣臣.mp3", 7.5, 3.5, "对照·张荣臣", "zhāng"),
    ("ctrl_wangfeng", "以场景为导向的 AI 项目端到端应用落地方法论-王峰.mp3", 6.5, 3.5, "对照·王峰", "wáng"),
]


def upload(path: Path) -> str:
    out = subprocess.run(["mcode-tools", "upload_temp_url", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)["temp_url"]


def listen(url: str, tag: str, k: int) -> str:
    args = {"audio_info": {"url": url, "mime_type": "audio/mpeg", "prompt": PROMPT}}
    ap = D / f"args_{tag}_{k}.json"
    ap.write_text(json.dumps(args, ensure_ascii=False), encoding="utf-8")
    out = subprocess.run(
        ["mcode-tools", "connector", "call", "connector__matrix__listen_audio",
         "--args-file", str(ap)],
        capture_output=True, text=True)
    raw = (out.stdout or out.stderr).strip()
    try:
        return str(json.loads(raw).get("transcription", raw)).strip()
    except Exception:
        return raw


def norm(s: str) -> str:
    """归一化：小写、去空白/标点，只留字母和声调符号。"""
    out = []
    for ch in s.lower():
        if ch.isalpha() or ch in "āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜüńň":
            out.append(ch)
    return "".join(out)


def main() -> int:
    only = sys.argv[1:]
    ok = bad = 0
    for tag, src, st, dur, note, expect in SPECS:
        if only and tag not in only:
            continue
        src_path = ROOT / "source" / src
        if not src_path.exists():
            print(f"{tag:18s} {note:20s} → 源文件缺失")
            continue
        clip = D / f"py_{tag}.mp3"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-ss", f"{st}", "-to", f"{st + dur}", "-i", str(src_path),
                        "-ac", "1", "-ar", "16000", str(clip)], check=True)
        url = upload(clip)
        raw = []
        for k in range(REPEAT):
            raw.append(listen(url, tag, k))
        votes = Counter(norm(r) for r in raw if norm(r))
        top, cnt = (votes.most_common(1)[0] if votes else ("", 0))
        hit = "✅" if top == norm(expect) else "❌"
        if top == norm(expect):
            ok += 1
        else:
            bad += 1
        print(f"{hit} {tag:18s} {note:20s} 期望={expect:6s} 多数={top:10s}"
              f" ({cnt}/{REPEAT})  原始={raw}")
    print(f"\n对照+目标合计：命中 {ok}，未命中 {bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
