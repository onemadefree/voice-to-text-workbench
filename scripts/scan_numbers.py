#!/usr/bin/env python3
"""体检全量语料：检查中文数字（年份/金额）有没有被规整阶段压坏。

背景：叠字压缩会把「二零二零年」压成「二零年」、「三三五元」压成「三五元」，
年份/金额少一位就是事实性错误。修复后清洗阶段不再碰数字，这里做全量核对。

用法：.venv/bin/python scripts/scan_numbers.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import textproc  # noqa: E402
from app.llm import _digit_weight, _numbers_preserved  # noqa: E402

CN_DIGIT = "零〇一二三四五六七八九"
RE_CN_RUN = re.compile(rf"[{CN_DIGIT}幺]{{2,}}")
# 跟这些单位相邻的数字串一定是真数字（三元 / 二号 / 三届）
NUM_UNIT = "年月日元万亿号届季％%倍秒厘"


def _is_number_run(raw: str, m) -> bool:
    """判断这个数字串是真数字，还是「统一一下」这种词里带的「一」。

    真数字的三种特征：三位以上、后面跟数量单位、后面紧跟另一串数字。
    """
    size = m.end() - m.start()
    if size >= 3:
        return True
    after = raw[m.end():m.end() + 1]
    if after and (after in NUM_UNIT or after in CN_DIGIT + "幺"):
        return True
    return False


def scan_corpus() -> int:
    """核对存储的 clean 是否丢了原文数字。"""
    hard = 0
    soft = 0
    for tj in sorted((ROOT / "output").glob("*/transcript.json")):
        if tj.parent.name.startswith("_"):
            continue
        doc = json.loads(tj.read_text(encoding="utf-8"))
        segs = doc.get("segments") or []
        bad = []
        for i, s in enumerate(segs):
            raw = s.get("raw") or ""
            clean = s.get("clean") or ""
            if not raw or not clean:
                continue
            if not RE_CN_RUN.search(raw):
                continue
            lost_runs = {m.group(0) for m in RE_CN_RUN.finditer(raw)
                         if m.group(0) not in clean}
            if not lost_runs:
                continue
            high = [r for r in lost_runs
                    if any(_is_number_run(raw, m) and m.group(0) == r
                           for m in RE_CN_RUN.finditer(raw))]
            if high:
                hard += 1
                bad.append((i, sorted(high), raw[:60], clean[:60]))
            else:
                soft += 1
        flag = "⚠️" if bad else "✅"
        note = "" if not bad else f"｜另有 {soft} 处为词内数字（统一一下）"
        print(f"{flag} {tj.parent.name}: 段数 {len(segs)}｜数字被压坏 {len(bad)}{note}")
        for i, runs, raw, clean in bad[:8]:
            print(f"    #{i} 数字串 {runs}")
            print(f"        raw  : {raw}")
            print(f"        clean: {clean}")
    print(f"\n合计真数字被压坏：{hard} 段（词内数字不计）")
    return hard


def scan_kb() -> None:
    """检查知识库里有没有三位年份这种被压坏的写法。"""
    print("\n===== 知识库里的可疑年份 =====")
    for kb in sorted((ROOT / "output").glob("*/*.kb.md")):
        if kb.parent.name.startswith("_"):
            continue
        txt = kb.read_text(encoding="utf-8")
        bad = re.findall(r"[零〇一二三四五六七八九]{2,3}年", txt)
        if bad:
            print(f"  ⚠️ {kb.parent.name}: {sorted(set(bad))}")
        else:
            print(f"  ✅ {kb.parent.name}: 无")


def scan_new_clean() -> None:
    """用当前代码重算 clean，确认规整阶段不再丢数字。"""
    print("\n===== 用当前代码重算 clean =====")
    for tj in sorted((ROOT / "output").glob("*/transcript.json")):
        if tj.parent.name.startswith("_"):
            continue
        doc = json.loads(tj.read_text(encoding="utf-8"))
        segs = doc.get("segments") or []
        lost = 0
        for s in segs:
            raw = s.get("raw") or ""
            new = textproc.clean_text(raw, level="light")
            if raw and new and (not _numbers_preserved(raw, new)
                                or _digit_weight(new) < _digit_weight(raw)):
                lost += 1
        flag = "⚠️" if lost else "✅"
        print(f"{flag} {tj.parent.name}: 重算后仍丢数字 {lost}")


if __name__ == "__main__":
    bad = scan_corpus()
    scan_kb()
    scan_new_clean()
    raise SystemExit(1 if bad else 0)
