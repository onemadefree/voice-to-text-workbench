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

    print()
    if FAILED:
        print(f"✗ {len(FAILED)} 项未通过: {', '.join(FAILED)}")
        return 1
    print("✓ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
