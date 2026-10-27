#!/usr/bin/env python3
"""自测：写坏的年份必须能还原，真年份必须不能被改。

背景：语音识别把「二零二五年」写成「二五二五年」（逐位读是 2525）。
原来的年份校验会把 2525 当成事实性数字硬保护，于是模型改对成 2025年
反被判成「丢年份」，整段退回原文——高太山/王峰/闫树那几处年份就这么修不掉。

判据：
- 合理年份（1900~2100）：改掉就是丢，必须拦下
- 不合理年份（2525/1415 这类写坏的）：还原成合理年份要放行
- 真实历史年份（1679/1847/1848）：转成阿拉伯数字要放行
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.llm import _digit_weight, _numbers_preserved, _years_of  # noqa: E402

FAILED: list[str] = []


def check(label: str, got, want) -> None:
    ok = got == want
    print(f"  {'✓' if ok else '✗'} {label}: {got}")
    if not ok:
        print(f"      期望: {want}")
        FAILED.append(label)


def main() -> int:
    print("· 写坏的年份要能还原")
    check("二五二五年 → 2025年 放行",
          _numbers_preserved("这个是二五二五年春节", "这个是 2025年春节"), True)
    check("二六二六年 → 2026年 放行",
          _numbers_preserved("所以二六二六年", "所以 2026年"), True)
    check("七六二五年 → 2025年 放行",
          _numbers_preserved("七六二五年第一次", "2025年第一次"), True)
    check("一八一九年 → 2019年 放行",
          _numbers_preserved("一八一九年的时候", "2019年的时候"), True)
    check("一四一五年 → 2015年 放行",
          _numbers_preserved("一四一五年那会儿", "2015年那会儿"), True)

    print("· 真年份不能被改坏")
    check("二零二零年 → 2020年 放行",
          _numbers_preserved("二零二零年的事", "2020年的事"), True)
    check("一八四七年 → 1847年 放行",
          _numbers_preserved("到了一八四七年", "到了 1847年"), True)
    check("一六七九年 → 1679年 放行",
          _numbers_preserved("一六七九年，在英国", "1679年，在英国"), True)
    check("二零二五年 → 2035年 拦下",
          _numbers_preserved("二零二五年", "2035年"), False)
    check("二零二零年 → 二零年 拦下（丢位）",
          _numbers_preserved("二零二零年", "二零年"), False)
    check("二零二五年 → 2525年 拦下（没还原成合理年份）",
          _numbers_preserved("二零二五年", "2525年"), False)

    print("· 数字重量：年份串按 4 位算")
    check("二二零二三年 与 2023年 等重",
          _digit_weight("二二零二三年") == _digit_weight("2023年"), True)
    check("二零二零年 与 2020年 等重",
          _digit_weight("二零二零年") == _digit_weight("2020年"), True)
    check("丢掉年份要变轻",
          _digit_weight("二零二零年") > _digit_weight("二零年"), True)

    print("· 年份识别")
    check("2020年 认得", 2020 in _years_of("到了 2020年"), True)
    check("1679年 认得", 1679 in _years_of("一六七九年"), True)
    check("五位年份取后四位", 2023 in _years_of("二二零二三年"), True)

    print()
    if FAILED:
        print(f"✗ {len(FAILED)} 项未通过: {', '.join(FAILED)}")
        return 1
    print("✓ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
