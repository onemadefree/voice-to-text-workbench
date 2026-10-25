#!/usr/bin/env python3
"""回归测试：润色批次拿不全条目时，缺的那些要拆小重试，而不是悄悄退回规整稿。

背景：推理模型有时只写出一部分条目（其余静默缺失）。原来的实现会让缺
的条目直接落回规整稿——内容没错，但等于没润色，且没有任何日志说明。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.llm import polish_segments  # noqa: E402


class FakeLLM:
    """假模型：按需返回前 n 条结果；每次调用都记下来。"""

    def __init__(self, keep: int | None = None, empty: bool = False):
        self.keep = keep          # 每批只返回前 keep 条
        self.empty = empty        # 整批不返回
        self.calls: list[int] = []  # 每批请求的条数

    @property
    def budget_extra(self) -> int:
        return 0

    def complete(self, prompt, system, max_tokens=2048, temperature=0.3) -> str:
        # 从 prompt 里数出请求了几条（"请逐条改写下面 N 条"）
        n = int(prompt.split("请逐条改写下面 ")[1].split(" 条")[0])
        self.calls.append(n)
        if self.empty:
            return ""
        keep = n if self.keep is None else min(self.keep, n)
        # 取出 prompt 里的原始条目，改写成贴近原文的版本（要过相似度校验）
        import re
        pairs = re.findall(r"^(\d+)\. (.+)$", prompt, re.M)
        out = []
        for num, text in pairs[:keep]:
            out.append(json.dumps({"i": int(num), "t": f"{text}已顺句"}, ensure_ascii=False))
        return "\n".join(out)


def segs(n: int):
    return [{"raw": f"原始{i}", "text": f"原始{i}"} for i in range(n)]


class EchoLLM(FakeLLM):
    """按提示词里给出的编号原样回填——真实模型就是这个行为。"""

    def complete(self, prompt, system, max_tokens=2048, temperature=0.3) -> str:
        import re
        pairs = re.findall(r"^(\d+)\. (.+)$", prompt, re.M)
        self.calls.append(len(pairs))
        out = []
        for num, text in pairs:
            out.append(json.dumps({"i": int(num), "t": f"{text}已顺句"}, ensure_ascii=False))
        return "\n".join(out)


def run() -> int:
    failed = []

    # 0) 超过一批（16 条）时，后面的批次也必须拿到结果——
    #    这条曾经整批丢失：提示词里的编号是段落的绝对位置，取值却按批内位置查，
    #    于是第 2 批往后对不上号，模型的改写全被丢弃。
    llm = EchoLLM()
    out = polish_segments(llm, segs(20))
    want = [f"原始{i}已顺句" for i in range(20)]
    if out != want:
        bad = [(i, o) for i, o in enumerate(out) if o != want[i]]
        failed.append(("跨批次都要润色", f"{len(bad)} 条没润色（例 {bad[:3]}）", out[:4]))

    # 1) 全部返回：每段都应被改写
    llm = FakeLLM()
    out = polish_segments(llm, segs(4))
    want = [f"原始{i}已顺句" for i in range(4)]
    if out != want:
        failed.append(("全量返回", want, out))

    # 2) 只返回前 2 条：后 2 条应拆小重试并拿到结果
    llm = FakeLLM(keep=2)
    out = polish_segments(llm, segs(4))
    want = [f"原始{i}已顺句" for i in range(4)]
    if out != want:
        failed.append(("部分返回", want, out))
    if not any(c < 4 for c in llm.calls):
        failed.append(("部分返回应触发拆小重试", "有更小的批次", llm.calls))

    # 3) 整批都空：拆半重试后仍应各拿到结果
    calls = {"n": 0}

    class EmptyThenOK(FakeLLM):
        def complete(self, prompt, system, max_tokens=2048, temperature=0.3) -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                self.calls.append(1)
                return ""
            return super().complete(prompt, system, max_tokens, temperature)

    llm2 = EmptyThenOK()
    out = polish_segments(llm2, segs(2))
    want = [f"原始{i}已顺句" for i in range(2)]
    if out != want:
        failed.append(("整批空后拆半", want, out))

    # 4) 缺的那条也没写出来：至少要落回规整稿，不能是空串
    class AlwaysEmpty(FakeLLM):
        def complete(self, prompt, system, max_tokens=2048, temperature=0.3) -> str:
            self.calls.append(1)
            return ""

    llm3 = AlwaysEmpty()
    out = polish_segments(llm3, segs(3))
    if any(not o.strip() for o in out):
        failed.append(("全失败时不能留空", "非空", out))

    print(f"用例 5 项")
    if failed:
        for name, want, got in failed:
            print(f"  ✗ {name}\n      期望 {want}\n      实际 {got}")
        print(f"\n失败 {len(failed)} 项")
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
