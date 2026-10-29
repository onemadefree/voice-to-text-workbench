#!/usr/bin/env python3
"""按定案结论，把本篇讲者姓氏统一为「程」。

定案依据（2026-10-29 核查）：
  1. 文件名与公开资料都作「程建刚」（另有讲师简介作「程建岗」）：
     濮阳工信局政府通稿「清华大学经济管理学院MBA实战导师程建刚老师专题讲授」；
     讲师简介「清华大学经济管理学院MBA实战老师、招生面试评委、清华继续教育学院特聘讲席」。
  2. 讲者自述履历与该公开画像高度吻合：1995年参加工作（至今年31年）、
     昆山起步、最高做到董事总经理、做过八十年历史大型跨国公司的战略转型、
     长期为央企管理赋能。
  3. 横向对照：其余 10 篇里，主持人念的称呼姓氏与文件名**全部**一致
     （张/赵/严(闫)/张/高/张/王），只有本篇对不上；而「陈 chén」与「程 chéng」
     只差一个 -ng 鼻音尾，正是 ASR 最典型的混淆方向（chéng→chén）。
  4. 原音听音：受控实验（正对照「流程/程度」能稳定输出 chéng，说明方法能分辨 -ng）
     对本篇称呼输出 chén（6/6），原音上无法完全定案——故在「待确认」里如实留痕。

用法：
    .venv/bin/python scripts/_fix_surname_cheng.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.store import DocumentStore, segment_final_text  # noqa: E402

DOC_ID = "组织变革管理与变革领导力-程建刚"

KB_SUMMARY_OLD = "本课程由清华大学经管学院MBA导师（讲师姓名待核）主讲"
KB_SUMMARY_NEW = "本课程由清华大学经管学院MBA导师程建刚主讲"
KB_POINT_OLD = "- 讲师1995年参加工作，拥有从初创企业到大型跨国公司战略转型的全生命周期管理经验，最高职位为董事总经理。"
KB_POINT_NEW = "- 讲师程建刚1995年参加工作，拥有从初创企业到大型跨国公司战略转型的全生命周期管理经验，最高职位为董事总经理。"

KB_PENDING_OLD = "演讲者姓名存疑：文件名写「程建刚」"
KB_PENDING_NEW = (
    "- 讲师姓氏：ASR 逐字稿把称呼记作「陈老师」（共 6 处），而文件名与公开资料都作「程建刚」"
    "（另有讲师简介作「程建岗」）。判断是 chéng 的 -ng 鼻音尾在语流中丢失、或被 ASR 与听音模型"
    "规范化成了 chén，已按公开资料与文件名统一为「程老师／程建刚」。"
    "核查过程：受控听音实验以同一录音中确定无疑的「流程」「程度」做正对照，拼音听音方法能稳定输出 chéng"
    "（说明该方法确实能分辨 -ng 尾音），但对本篇称呼输出 chén（6/6）；原音层面无法完全定案，"
    "如需彻底确认建议人工听原音或向课程方核对。"
    "另注：原逐字稿此处曾有「我蒋一元也做了」，经听原音核实实为「我讲义也做了」（讲义稍后发给大家），"
    "「蒋一元」是 ASR 误听，不是人名。"
)


def main() -> int:
    store = DocumentStore()
    doc = store.load(DOC_ID)
    if doc is None:
        print(f"✗ 找不到文档: {DOC_ID}")
        return 1
    segs = doc.get("segments") or []

    # ---- 1. 逐字稿：陈老师 → 程老师 ----
    n_seg = 0
    for seg in segs:
        hit = False
        for field in ("raw", "clean", "polished"):
            old = seg.get(field)
            if isinstance(old, str) and "陈老师" in old:
                seg[field] = old.replace("陈老师", "程老师")
                hit = True
        if hit:
            n_seg += 1
    print(f"  ✓ 逐字稿：{n_seg} 段「陈老师」→「程老师」")

    # ---- 2. 知识库 ----
    kb = doc.get("knowledge_base") or ""
    before = kb
    kb = kb.replace(KB_SUMMARY_OLD, KB_SUMMARY_NEW)
    kb = kb.replace(KB_POINT_OLD, KB_POINT_NEW)
    lines = kb.splitlines()
    hit = [i for i, l in enumerate(lines)
           if l.strip().startswith("-") and KB_PENDING_OLD in l]
    if len(hit) != 1:
        print(f"✗ 待确认条目定位失败（命中 {len(hit)} 条）")
        return 1
    lines[hit[0]] = KB_PENDING_NEW
    kb = "\n".join(lines)
    if kb == before:
        print("✗ 知识库没有任何改动")
        return 1
    doc["knowledge_base"] = kb
    print("  ✓ 知识库：摘要/要点/待确认 已更新")

    store.save(doc)

    # ---- 3. 同步导出 ----
    body = "\n".join(segment_final_text(s) for s in segs)
    txt = config.OUTPUT_DIR / DOC_ID / f"{DOC_ID}.txt"
    txt.write_text(body, encoding="utf-8")
    print(f"  ✓ 已同步导出 {txt.name}（{len(body.splitlines())} 行）")

    kbmd = config.OUTPUT_DIR / DOC_ID / f"{DOC_ID}.kb.md"
    kbmd.write_text(kb if kb.endswith("\n") else kb + "\n", encoding="utf-8")
    print(f"  ✓ 已同步导出 {kbmd.name}")

    # ---- 4. 残留检查 ----
    leftover = [i for i, s in enumerate(segs)
                if any("陈老师" in (s.get(f) or "") for f in ("raw", "clean", "polished"))]
    print(f"  {'✓' if not leftover else '✗'} 逐字稿残留「陈老师」：{leftover or '无'}")
    kb_left = [l for l in kb.splitlines() if "陈老师" in l]
    print(f"  {'✓' if not kb_left else '✗'} 知识库残留「陈老师」：{len(kb_left)} 处")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
