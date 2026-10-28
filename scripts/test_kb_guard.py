#!/usr/bin/env python3
"""自测：知识库生成必须 5 个小节齐全（防 max_tokens 截断）。

用假 LLM 模拟三种情况：完整输出、被截断、一直失败。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.llm import _kb_complete, build_knowledge_base  # noqa: E402

FULL_KB = """# 标题
## 摘要
这是一段摘要。
## 要点
- 要点一
## 术语表
- 术语：解释
## 待确认
- 无
"""

TRUNCATED_KB = """# 标题
## 摘要
这是一段摘要。
## 要点
- 要点一
- 要点二
"""

FAILED: list[str] = []


def check(label: str, got, want) -> None:
    ok = got == want
    print(f"  {'✓' if ok else '✗'} {label}: {got}")
    if not ok:
        print(f"      期望: {want}")
        FAILED.append(label)


class FakeLLM:
    """假 LLM：分段提炼返回要点，汇总返回预设文本（可设定前几次失败）。"""

    def __init__(self, summary: str, fail_summary_times: int = 0) -> None:
        self.summary = summary
        self.fail_times = fail_summary_times
        self.summary_calls = 0
        self.max_tokens_seen: list[int] = []

    def complete(self, prompt, system, max_tokens=2048, temperature=0.3):
        if "知识库" in system:
            self.summary_calls += 1
            self.max_tokens_seen.append(max_tokens)
            if self.summary_calls <= self.fail_times:
                return ""
            return self.summary
        return "- 分段要点一\n- 分段要点二"


class ContextLLM:
    """模拟有上下文上限的假 LLM：提示词超长就报 context_length_exceeded。

    内网引擎的上下文只有 32768 token。长稿的「要点合并」如果不切片，
    这一步自己就先超限，异常被吞掉后 merged 原样带回，最终汇总再超一次，
    整篇知识库就退化成 46 字符的空壳。这里用提示词长度上限复现这个坑。
    """

    def __init__(self, limit: int = 16000, summary: str = FULL_KB) -> None:
        self.limit = limit
        self.summary = summary
        self.longest = 0
        self.merge_prompts: list[int] = []

    def complete(self, prompt, system, max_tokens=2048, temperature=0.3):
        self.longest = max(self.longest, len(prompt))
        if len(prompt) > self.limit:
            raise RuntimeError("context_length_exceeded")
        if "知识库" in system:
            return self.summary
        if "去重合并" in prompt:
            self.merge_prompts.append(len(prompt))
            return "- 要点合并后的简明结论"
        return "- 分段要点" * 300


def main() -> int:
    print("· _kb_complete 判定")
    check("完整的算完整", _kb_complete(FULL_KB), True)
    check("截断的不算完整", _kb_complete(TRUNCATED_KB), False)
    check("空的不算完整", _kb_complete(""), False)

    print("· 完整输出：不重试")
    llm = FakeLLM(FULL_KB)
    out = build_knowledge_base(llm, "演讲正文。" * 500, title="测试")
    check("只调 1 次汇总", llm.summary_calls, 1)
    check("5 节齐全", _kb_complete(out), True)

    print("· 被截断：重试 + 兜底补节")
    llm = FakeLLM(TRUNCATED_KB)
    out = build_knowledge_base(llm, "演讲正文。" * 500, title="测试")
    check("重试到 2 次", llm.summary_calls, 2)
    check("最终 5 节齐全", _kb_complete(out), True)
    check("缺的小节被补上", "## 待确认" in out and "## 术语表" in out, True)

    print("· 汇总调用失败：兜底补节")
    llm = FakeLLM(TRUNCATED_KB, fail_summary_times=99)
    out = build_knowledge_base(llm, "演讲正文。" * 500, title="测试")
    check("仍 5 节齐全", _kb_complete(out), True)

    print("· max_tokens 足够")
    llm = FakeLLM(FULL_KB)
    build_knowledge_base(llm, "演讲正文。" * 500, title="测试")
    check("汇总 max_tokens >= 8192", llm.max_tokens_seen[0] >= 8192, True)

    print("· 上下文上限：要点合并要切片，不能自己就超限")
    llm = ContextLLM()
    out = build_knowledge_base(llm, "演讲正文。" * 6000, title="测试")
    check("走过分段提炼的合并", len(llm.merge_prompts) > 0, True)
    check("合并是切片的", max(llm.merge_prompts or [0]) <= 7000, True)
    check("没有任何提示词超限", llm.longest <= llm.limit, True)
    check("知识库有正文", "这是一段摘要" in out, True)
    check("5 节齐全", _kb_complete(out), True)

    print()
    if FAILED:
        print(f"✗ {len(FAILED)} 项未通过: {', '.join(FAILED)}")
        return 1
    print("✓ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
