#!/usr/bin/env python3
"""自测：转写接口的源文件解析。

以前只按不带扩展名的 id 找源文件，找不到就回退到 source 目录本身，
结果探测音频直接失败——网页上点「开始转写」就是这么坏的。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.server import _resolve_source  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}　{detail}")
        FAILED.append(name)


def main() -> int:
    files = [p for p in config.SOURCE_DIR.iterdir() if p.is_file() and not p.name.startswith(".")]
    if not files:
        print("  ! source 目录里没有文件，跳过")
        return 0
    sample = files[0]
    doc_id = sample.stem

    print("· 解析源文件")
    got = _resolve_source(doc_id)
    check("不带扩展名的 id 能找到文件", got is not None and got.is_file(),
          f"{doc_id} → {got}")
    check("找到的是正确的那个", got and got.name == sample.name,
          f"{got} != {sample.name}")

    got = _resolve_source(doc_id, sample.name)
    check("带 name 也能找到", got is not None and got.name == sample.name, str(got))

    print("· 找不到的情况")
    check("不存在的 id 返回 None",
          _resolve_source("__根本没有这个文件__") is None, "")
    check("空 name 不会误命中目录",
          _resolve_source("__根本没有这个文件__", "") is None, "")
    # 传空 name 时不能把 source 目录本身当成源文件
    got = _resolve_source("__根本没有这个文件__", "不存在的.mp3")
    check("不存在的 name 返回 None", got is None, str(got))

    if FAILED:
        print(f"\n失败 {len(FAILED)} 项：{FAILED}")
        return 1
    print("\n全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
