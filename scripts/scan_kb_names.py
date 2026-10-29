#!/usr/bin/env python3
"""全库扫描：知识库里有没有「原文没有的人名」（模型臆造 / ASR 错字升级成人名）。

背景（2026-10-29）：程建刚那篇的知识库摘要写成「清华 MBA 导师蒋一元主讲」，
「蒋一元」来自逐字稿里的 ASR 误听「我讲义也做了」——模型把一个听错的词
当成了讲者姓名，还写进了摘要。

做法：抽出知识库里所有像人名的中文词（2~4 字），跟逐字稿原文比对，
原文里根本没出现的候选一律列出来（人工判断是不是臆造）。

用法：
    .venv/bin/python scripts/scan_kb_names.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.store import segment_final_text  # noqa: E402

# 明显不是人名的词（常见于知识库行文），命中就直接跳过
STOP = {
    "知识库", "摘要", "要点", "术语表", "待确认", "标题", "本课程", "本讲",
    "这堂", "这门", "课程", "老师", "学员", "企业", "组织", "变革", "领导",
    "管理", "战略", "文化", "人才", "系统", "思考", "体系", "构建", "建模",
    "演讲", "现场", "问题", "时候", "方面", "能力", "发展", "创新", "变化",
    "第二个", "第一", "第二", "第三", "整体", "非常", "真正", "不是", "没有",
    "可以", "因为", "所以", "如果", "这个", "那个", "我们", "他们", "自己",
    "一个", "一些", "这些", "那些", "什么", "怎么", "为了", "已经", "仍然",
    "昆仑", "上海", "北京", "深圳", "广州", "上海交", "清华", "中石油",
    "华为", "ibm", "ibm", "海尔", "诺基亚", "爱立信", "富士", "中集", "通用",
    "电气", "联通", "移动", "电信", "中煤", "科伦", "昆仑", "戴厚良",
}

# 人名常见结构：出现在「XX老师/XX教授/XX主任/XX博士/XX主讲/XX表示」
TITLE_AFTER = re.compile(
    r"([一-龥]{2,4})(老师|教授|主任|博士|讲师|导师|院长|校长|总|董|书记)"
)
# 「X说/X认为/X指出/X提到」——引述型人名
SPEECH = re.compile(r"([一-龥]{2,4})(说|认为|指出|提到|强调|坦言|介绍)")


def looks_like_name(word: str) -> bool:
    if word in STOP or len(word) < 2 or len(word) > 4:
        return False
    if not re.fullmatch(r"[一-龥]+", word):
        return False
    return True


def main() -> int:
    base = config.OUTPUT_DIR
    docs = sorted(p for p in base.iterdir() if (p / "transcript.json").exists())
    total = 0
    for d in docs:
        doc = json.loads((d / "transcript.json").read_text(encoding="utf-8"))
        kb = doc.get("knowledge_base") or ""
        if not kb.strip():
            continue
        src = "\n".join(segment_final_text(s) for s in (doc.get("segments") or []))
        cands: set[str] = set()
        for m in TITLE_AFTER.finditer(kb):
            if looks_like_name(m.group(1)):
                cands.add(m.group(1))
        for m in SPEECH.finditer(kb):
            if looks_like_name(m.group(1)):
                cands.add(m.group(1))
        missing = sorted(c for c in cands if c not in src)
        if missing:
            total += len(missing)
            print(f"  {d.name}")
            for c in missing:
                # 打印上下文，人工判断
                hit = [ln.strip() for ln in kb.splitlines() if c in ln][:2]
                print(f"      ? {c}  ←  {hit[0][:100] if hit else ''}")
    print(f"\n共 {len(docs)} 篇，待人工判断的候选 {total} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
