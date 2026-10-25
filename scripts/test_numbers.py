#!/usr/bin/env python3
"""数字保真校验的单元测试。

场景：模型把四位中文数字年份读成三位（「二零二零年」→「二零年」），
年份属于事实性信息，丢了就是硬伤，必须拦下来退回原文。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.llm import _numbers_preserved  # noqa: E402

CASES = [
    # (原文, 润色结果, 期望通过?)
    ("二零二零年八月份，发布了意见。", "二零年八月份，发布了意见。", False),
    ("直到二零二二年十二月的一个制度。", "二零二年底，一个制度。", False),
    ("它第一次出现是在二零二零年。", "二零年。", False),
    ("二零二零年八月份，发布了意见。", "2020年八月份，发布了意见。", True),
    ("二零二零年八月份，发布了意见。", "2020年8月份，发布了意见。", True),
    ("二零二零年八月份，发布了意见。", "二零二零年八月份，发布了意见。", True),
    ("二零二二年年底，疫情还没有结束。", "2022年年底，疫情还没有完全结束。", True),
    ("原文没有年份的普通句子。", "改写后依然没有年份的句子。", True),
    ("二零二〇年三月。", "2020年3月。", True),
    ("两个年份：二零一九年和二零二一年。", "两个年份：2019年和2021年。", True),
    ("两个年份：二零一九年和二零二一年。", "两个年份：二零一九年和二零年。", False),
]


def main() -> int:
    bad = 0
    for src, cand, want in CASES:
        got = _numbers_preserved(src, cand)
        ok = (got == want)
        if not ok:
            bad += 1
        print(f"{'✅' if ok else '❌'} 通过={got} 期望={want} | {src} → {cand}")
    print(f"\n共 {len(CASES)} 项，失败 {bad} 项")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
