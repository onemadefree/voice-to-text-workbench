#!/usr/bin/env python3
"""按原音核对结果，修掉知识库里被 ASR 错字带出来的假人名。

背景（2026-10-29 听原音核实）：
    逐字稿第 20 行「刚才郭老师介绍过。其实我蒋一元也做了。」——原音是
    「刚才郭老师介绍过我，其实我讲义也做了（讲义到时候发给大家）」。
    ASR 把「讲义」听成了「蒋一元」，知识库模型又把「蒋一元」当成讲师姓名，
    于是摘要写成「清华 MBA 导师蒋一元主讲」、要点写成「讲师蒋一元 1995 年参加工作」。

    讲者本人从头到尾没报过姓名；主持人一律称「陈老师」。文件名「程建刚」
    与「陈老师」不一致，但没有任何原音证据能判定是「程老师」还是「陈老师」。

用法：
    .venv/bin/python scripts/_fix_kb_fake_name.py "文档名"
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.store import DocumentStore  # noqa: E402

# 假人名 → 替换文案
FAKE_NAME = "蒋一元"


def main() -> int:
    doc_id = sys.argv[1]
    store = DocumentStore()
    doc = store.load(doc_id)
    if doc is None:
        print(f"✗ 找不到文档: {doc_id}")
        return 1
    kb = doc.get("knowledge_base") or ""
    if not kb.strip():
        print("✗ 知识库为空")
        return 1
    if FAKE_NAME not in kb:
        print(f"· 知识库里没有「{FAKE_NAME}」，无需处理")
        return 0

    lines = kb.splitlines()
    out: list[str] = []
    changed: list[str] = []
    for line in lines:
        if FAKE_NAME not in line:
            out.append(line)
            continue
        old = line
        # 「清华大学MBA导师蒋一元主讲」→「清华大学经管学院MBA导师（讲师姓名待核）」
        line = re.sub(rf"清华(?:大学)?(?:经济管理学院|经管学院)?MBA导师\s*{FAKE_NAME}",
                      "清华大学经管学院MBA导师（讲师姓名待核）", line)
        # 「讲师蒋一元1995年参加工作」→「讲师1995年参加工作」
        line = line.replace(f"讲师{FAKE_NAME}", "讲师")
        # 兜底：任何残留
        line = line.replace(FAKE_NAME, "")
        if line != old:
            changed.append(f"  - {old[:100]}\n  + {line[:100]}")
        out.append(line)

    fixed = "\n".join(out)
    doc["knowledge_base"] = fixed
    store.save(doc)
    kbmd = config.OUTPUT_DIR / doc_id / f"{doc_id}.kb.md"
    if kbmd.exists():
        kbmd.write_text(fixed if fixed.endswith("\n") else fixed + "\n",
                        encoding="utf-8")
        print(f"  ✓ 已改 {kbmd.name}")
    print(f"  改动 {len(changed)} 处：")
    for c in changed:
        print(c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
