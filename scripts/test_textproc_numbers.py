#!/usr/bin/env python3
"""回归测试：规整阶段不得压坏中文数字（年份、金额、口述数字）。

背景：叠字压缩把「二零二零年」压成「二零年」、「三三五元」压成「三五元」，
年份/金额少一位就是事实性错误。修复后中文数字串（三位及以上）必须原样保留。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.llm import _digits_intact, _numbers_preserved, _safe_base  # noqa: E402
from app.textproc import clean_text, strip_stutter  # noqa: E402

# (输入, 期望输出, 说明)
CASES = [
    # —— 年份：修复前的实际受害样本 ——
    ("二零二零年八月份，欧元发布了意见。", "二零二零年八月份，欧元发布了意见。", "2020 年"),
    ("直到二零二二年十二月的一个很重磅的制度", "直到二零二二年十二月的一个很重磅的制度", "2022 年"),
    ("总总书记啊，在二零二二年中职班是这样讲的。", "总书记啊，在二零二二年中职班是这样讲的。", "2022 年（总总叠字照常压）"),
    ("二〇二五年", "二〇二五年", "2025 年（〇 写法）"),
    ("二零二零年、二零二九年，甚至二零二八年", "二零二零年、二零二九年，甚至二零二八年", "多个年份连排"),
    ("二零零年的事", "二零零年的事", "2000 年"),
    ("我们从二二年开始，然后到现在", "我们从二二年开始，然后到现在", "2022 简写"),
    ("九零后、零零后的同志", "九零后、零零后的同志", "00 后"),
    # —— 金额 / 口述数字 ——
    ("北京的三三五元呢", "北京的三三五元呢", "335 元"),
    ("三三五元，大家知道是哪三块钱吗？", "三三五元，大家知道是哪三块钱吗？", "335 元"),
    ("三三五零，给大家最为神秘的地方", "三三五零，给大家最为神秘的地方", "3350"),
    ("五五二十五", "五五二十五", "5×5=25"),
    ("五八五八。", "五八五八。", "5858"),
    ("八八八，你把那三条说出来。", "八八八，你把那三条说出来。", "888"),
    ("四五四五的阶段", "四五四五的阶段", "4545"),
    ("到二二六年的Q四了", "到二二六年的Q四了", "226 年"),
    ("第三一二二点零的一个基础", "第三一二二点零的一个基础", "3122.0"),
    ("八八幺幺的一个分配", "八八幺幺的一个分配", "88:11"),
    # —— 真正的口吃仍然要压掉（非数字）——
    ("我我我想说", "我想说", "单字叠字"),
    ("这个这个很重要", "这个很重要", "词组叠字"),
    ("看看这个这个", "看这个", "叠字词"),
    # —— 含数字的词组：数字要留住，口吃交给 LLM ——
    ("统一一下核心价值观", "统一一下核心价值观", "一一 属词不压"),
    ("大杀四四方", "大杀四四方", "四四 属词不压"),
    ("一一一个小时", "一一一个小时", "一一一 属词不压"),
]


def run() -> int:
    failed = []
    for src, want, why in CASES:
        got = strip_stutter(src)
        if got != want:
            failed.append((src, want, got, why))
    # 整条清洗流水线也要守住（normalize/strip_fillers 不得二次破坏）
    for src, want, why in CASES:
        got = clean_text(src)
        if want not in got:
            failed.append((src, want, got, why + "（clean_text）"))

    # 纵深防御：规整文本丢数字时，回退基准必须退回原始转写稿
    safe = _safe_base("二零年八月份", "二零二零年八月份")
    if safe != "二零二零年八月份":
        failed.append(("_safe_base 回退", "二零二零年八月份", safe, "数字丢了要退回原文"))
    if not _numbers_preserved("二零二零年八月份", "二零二零年八月份"):
        failed.append(("_numbers_preserved", "True", "False", "同值应通过"))

    # 数字守卫：中文数字改成阿拉伯数字不算丢，丢字符才算
    guard_cases = [
        ("二零二零年八月份", "2020年8月份", True, "中文改阿拉伯不算丢"),
        ("二零二零年八月份", "二零年八月份", False, "丢一位要拦"),
        ("五五二十五", "五二十五", False, "丢一位要拦"),
        ("五五二十五", "5×5=25", True, "换写法不算丢"),
        ("二二年", "2022年", True, "年份换写法"),
        ("二二年", "二年", False, "年份少一位要拦"),
        ("零零后的同志", "00后的同志", True, "00 换写法"),
        ("零零后的同志", "零后的同志", False, "00 丢一位要拦"),
        ("八八幺幺的分配", "八八幺的分配", False, "88:11 丢一位要拦"),
        ("二二六年的Q四了", "2026年的Q四了", True, "226 年换写法"),
        # —— 词内叠字不算数字，模型正常顺句不能被误判 ——
        ("遇到了一一定时间的困境", "遇到了一定时间的困境", True, "口吃顺句不算丢"),
        ("认识是一一致的", "认识是一致的", True, "口吃顺句不算丢"),
        ("统一一下核心价值观", "统一核心价值观", True, "口吃顺句不算丢"),
        ("大杀四四方", "大杀四方", True, "成语顺句不算丢"),
        ("三个人的团队", "仨人的团队", True, "普通词不算丢数字"),
        ("五五个人", "五个人", True, "五五（人数）不算丢"),
    ]
    for src, cand, want_ok, why in guard_cases:
        got_ok = _digits_intact(src, cand)
        if got_ok != want_ok:
            failed.append((f"_digits_intact({src!r},{cand!r})",
                           str(want_ok), str(got_ok), why))

    print(f"用例 {len(CASES) * 2 + 2 + len(guard_cases)} 项")
    if failed:
        for src, want, got, why in failed:
            print(f"  ✗ {why}\n      输入 {src}\n      期望 {want}\n      实际 {got}")
        print(f"\n失败 {len(failed)} 项")
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
