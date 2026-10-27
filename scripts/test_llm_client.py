#!/usr/bin/env python3
"""自测：正文为空且响应被截断时，不能把思考过程当成答案返回。

原实现：content 为空就回退到 reasoning_content。模型把 max_tokens 全花在
思考上时（finish_reason=length），reasoning_content 里是半截思考过程——
29k 字的思考过程曾被当成知识库写进交付物。这里锁住这个行为。
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app.llm as llm_mod  # noqa: E402

FAILED: list[str] = []


def check(label: str, got, want) -> None:
    ok = got == want
    print(f"  {'✓' if ok else '✗'} {label}: {got!r}")
    if not ok:
        print(f"      期望: {want!r}")
        FAILED.append(label)


class _FakeResp:
    def __init__(self, payload: dict) -> None:
        self._data = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _run(payload: dict) -> str:
    """用假响应调一次在线后端的 generate。"""
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        return _FakeResp(payload)

    orig = urllib.request.urlopen
    urllib.request.urlopen = fake_urlopen
    try:
        fn, _info = llm_mod._build_client(
            "custom",
            override={"base": "http://192.168.162.100:2353/v1",
                      "api_key": "x", "model": "qwen3.8-27b"})
        return fn("测试", "系统", max_tokens=16384)
    finally:
        urllib.request.urlopen = orig


def main() -> int:
    print("· 正文为空 + 被截断：思考过程不能当答案")
    payload = {
        "choices": [{"finish_reason": "length", "message": {
            "content": "", "reasoning_content": "我们需要先分析一下原文…（思考了一万六千字）"}}],
        "usage": {"completion_tokens": 16384,
                  "completion_tokens_details": {"reasoning_tokens": 16000}},
    }
    try:
        out = _run(payload)
        check("抛错而不是返回思考过程", isinstance(out, str) and out != "", False)
    except RuntimeError as exc:
        check("抛错而不是返回思考过程", "思考耗尽预算" in str(exc), True)

    print("· 正文为空但正常结束：仍可用 reasoning_content")
    payload2 = {
        "choices": [{"finish_reason": "stop", "message": {
            "content": "", "reasoning_content": "好的"}}],
        "usage": {},
    }
    try:
        out = _run(payload2)
        check("正常结束时仍回退到 reasoning_content", out, "好的")
    except RuntimeError as exc:
        check("正常结束时仍回退到 reasoning_content", f"抛错了: {exc}", "好的")

    print("· 正文正常：直接用正文")
    payload3 = {
        "choices": [{"finish_reason": "stop", "message": {
            "content": "这是正文", "reasoning_content": "思考"}}],
        "usage": {},
    }
    try:
        out = _run(payload3)
        check("正文优先", out, "这是正文")
    except RuntimeError as exc:
        check("正文优先", f"抛错了: {exc}", "这是正文")

    print()
    if FAILED:
        print(f"✗ {len(FAILED)} 项未通过: {', '.join(FAILED)}")
        return 1
    print("✓ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
