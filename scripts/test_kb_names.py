#!/usr/bin/env python3
"""知识库「不加戏」校验的单元测试。

场景：原文只说「总书记」，知识库却写成「习近平指出」——
这是模型自行补全了原文没有的信息，违反"只做提炼"的要求。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.llm import _kb_adds_names  # noqa: E402

CASES = [
    ("总书记指出，干部要终身努力。",
     "总书记指出，干部的党性修养不会随年龄自然提高。", False),
    ("总书记指出，干部要终身努力。",
     "习近平指出，干部的党性修养不会随年龄自然提高。", True),
    ("我们学习了总书记关于党性修养的讲话。",
     "要点：学贯总书记关于党性修养的重要论述。", False),
    ("马克思在《共产党宣言》里提出…",
     "马克思和恩格斯创立了科学社会主义理论。", True),
    ("刘少奇在《论共产党员的修养》中指出…",
     "刘少奇在《论共产党员的修养》中指出了党性标准。", False),
    ("这是一段完全没提到任何人物的原文。",
     "知识库里提到了马克思的基本原理。", True),
]


def main() -> int:
    bad = 0
    for src, kb, want in CASES:
        got = _kb_adds_names(kb, src)
        ok = (got == want)
        if not ok:
            bad += 1
        print(f"{'✅' if ok else '❌'} 加戏={got} 期望={want} | {src[:24]} → {kb[:34]}")
    print(f"\n共 {len(CASES)} 项，失败 {bad} 项")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
