"""分辨 clean 档位：存储的 clean 是 light 还是 standard 生成的，
并用新代码算出「真正被数字保护改动」的段落集合。"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import textproc  # noqa: E402
from app.llm import _digit_weight, _numbers_preserved  # noqa: E402

CJK = textproc.CJK
RE_REPEAT_CJK = textproc.RE_REPEAT_CJK
RE_REPEAT_WORD = textproc.RE_REPEAT_WORD


def old_strip_stutter(text: str) -> str:
    """修复前的叠字压缩（整段直接压，不保护数字）。"""
    if not text:
        return ""
    t = RE_REPEAT_CJK.sub(r"\1", text)
    return RE_REPEAT_WORD.sub(r"\1", t)


def old_clean_text(text: str, level: str = "light") -> str:
    t = textproc.normalize(text)
    t = old_strip_stutter(t)
    return textproc.strip_fillers(t, level=level)


plan = {}
for tj in sorted((ROOT / "output").glob("*/transcript.json")):
    if tj.parent.name.startswith("_"):
        continue
    doc_id = tj.parent.name
    doc = json.loads(tj.read_text(encoding="utf-8"))
    segs = doc.get("segments") or []
    n_light = n_standard = n_other = 0
    changed = []
    for i, s in enumerate(segs):
        raw = s.get("raw") or ""
        stored = s.get("clean") or ""
        ol = old_clean_text(raw, "light")
        os_ = old_clean_text(raw, "standard")
        if stored == ol:
            n_light += 1
        elif stored == os_:
            n_standard += 1
        else:
            n_other += 1
        # 新代码（light 档，与存储档位一致）
        new = textproc.clean_text(raw, level="light")
        if new != stored:
            changed.append(i)
    plan[doc_id] = changed
    print(f"{doc_id}")
    print(f"    档位判定: light {n_light} / standard {n_standard} / 其它 {n_other}")
    print(f"    数字修复导致 clean 变化: {len(changed)} 段 → {changed[:20]}{'...' if len(changed) > 20 else ''}")

total = sum(len(v) for v in plan.values())
print(f"\n合计需重刷：{total} 段")
(ROOT / "output" / "_repolish_plan.json").write_text(
    json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
print("计划已写入 output/_repolish_plan.json")
