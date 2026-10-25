#!/usr/bin/env python3
"""自测：剥离推理模型塞在正文里的 <think>…</think> 思考块。

MiniMax M 系列会把思考过程和正文一起放进 content，
不剥掉会破坏 JSON 行解析，甚至把思维链写进稿子。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.llm import _parse_indexed, _strip_think  # noqa: E402

FAILED: list[str] = []


def check(label: str, got, want) -> None:
    ok = got == want
    print(f"  {'✓' if ok else '✗'} {label}: {got!r}")
    if not ok:
        print(f"      期望: {want!r}")
        FAILED.append(label)


def main() -> int:
    print("· 思考块剥离")
    check("完整 think 块", _strip_think('<think>先想一下</think>{"i":1,"t":"改好"}'),
          '{"i":1,"t":"改好"}')
    check("多行 think 块", _strip_think('<think>第一行\n第二行\n第三行</think>正文'),
          "正文")
    check("未闭合 think 块（后面全丢）", _strip_think('正文\n<think>后面是思考'),
          "正文")
    check("没有 think 块原样返回", _strip_think('{"i":1,"t":"x"}'),
          '{"i":1,"t":"x"}')
    check("空串", _strip_think(""), "")
    check("大小写混合", _strip_think('<THINK>x</THINK>正文'), "正文")

    print("· 剥离后 JSON 行能正常解析")
    raw = '<think>We need output JSON lines 16. Item 1 maybe "较好的数据"</think>\n' \
          '{"i":1,"t":"改写一"}\n{"i":2,"t":"改写二"}'
    clean = _strip_think(raw)
    parsed = _parse_indexed(clean, 2)
    check("解析出 2 条", len(parsed), 2)
    check("第 1 条内容", parsed.get(0), "改写一")
    check("第 2 条内容", parsed.get(1), "改写二")

    print("· 思考块里带编号也不能把解析带偏")
    raw2 = '<think>1. 这是思考\n2. 也是思考</think>\n{"i":1,"t":"真一"}\n{"i":2,"t":"真二"}'
    p2 = _parse_indexed(_strip_think(raw2), 2)
    check("仍解析出 2 条", len(p2), 2)
    check("内容正确", [p2.get(0), p2.get(1)], ["真一", "真二"])

    print("· JSON 后面接思考文字（推理模型常见）")
    raw3 = '{"i":1,"t":"改写一"}\nSome thinking about the answer\n{"i":2,"t":"改写二"}'
    p3 = _parse_indexed(raw3, 2)
    check("解析出 2 条", len(p3), 2)
    check("内容正确", [p3.get(0), p3.get(1)], ["改写一", "改写二"])

    print("· 整批都是思考文字时不能把乱码当改写结果")
    raw4 = "We need answer strictly structure Chinese Markdown.\nMaybe the title is X.\nSo the bullets are..."
    p4 = _parse_indexed(raw4, 3)
    check("解析结果为空", p4, {})

    print("· JSON 被截断时不能把半截内容塞进去")
    raw5 = '{"i":1,"t":"算力是人工智能的基础'
    p5 = _parse_indexed(raw5, 1)
    check("截断的 JSON 不被采纳", p5, {})

    print("· 段数不足时兜底切分不乱填")
    raw6 = "只有一行文字"
    p6 = _parse_indexed(raw6, 3)
    check("不足 3 段不硬切", p6, {})

    print()
    if FAILED:
        print(f"✗ {len(FAILED)} 项未通过: {', '.join(FAILED)}")
        return 1
    print("✓ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
