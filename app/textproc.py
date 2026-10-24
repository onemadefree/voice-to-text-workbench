"""文本规整：口语文本清理、错别字规范化、可选的术语纠正。

分三层：
1. `normalize`  —— 标点/空白/全半角规整，永远安全。
2. `strip_stutter` —— 去口吃重复（我我我 → 我）。
3. `strip_fillers` —— 去语气词与口水词，分 light / standard 两档。
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------- 词表

# 纯语气词/叹词：只在“独立成词”时删除，避免误伤实词（例如“唉声叹气”）
INTERJECTIONS = [
    "嗯嗯", "嗯", "呃呃", "呃", "额额", "额", "唔", "噢", "哦哦", "哦", "诶",
    "唉", "哎呀", "哎", "哈", "呵", "嘿", "嗨", "唔使",
]

# 口水词 / 话语标记
FILLERS_LIGHT = [
    "就是说就是说",
]

FILLERS_STANDARD = [
    "怎么说呢", "我跟你讲", "我跟你说", "你懂我意思吧", "你听懂了吗",
    "对不对", "对吧", "是吧", "好不好", "行不行", "可以吗",
    "你知道吧", "你知道吗", "大家知道吗", "我问一下",
    "怎么说", "这么讲", "这样讲", "那么讲",
    "那么接下来呢", "接下来呢",
    "咱们", "大家伙",
    "就是说", "也就是说", "换句话说", "所以说",
]

# 句末语气词（默认保留句中标点后的语气词，避免语义丢失）
TRAILING_PARTICLES = ["呢", "吧", "啊", "呀", "嘛", "哦", "噢", "嗯"]

CJK = r"一-鿿㐀-䶿"
RE_CJK_SPACE = re.compile(rf"(?<=[{CJK}])\s+(?=[{CJK}])")
RE_SPACE_BEFORE_PUNCT = re.compile(r"\s+([，。！？；：、,.!?;:])")
RE_PUNCT_SPACE = re.compile(r"([，。！？；：、,.!?;:])\s+")
RE_MULTI_PUNCT = re.compile(r"([，。！？；：、,.!?;:])\1{1,}")
RE_MULTI_PUNCT_MIX = re.compile(r"([，。！？；：])\s*[，,.]\s*")
RE_HALF_PUNCT = re.compile(rf"(?<=[{CJK}])\s*,\s*(?=[{CJK}])")
RE_TRAILING_JUNK = re.compile(r"[\s,，、;；:：]+$")
RE_LEADING_JUNK = re.compile(r"^[\s,，.。、;；:：!！?？]+")
RE_REPEAT_CJK = re.compile(rf"([{CJK}])\1{{1,}}")
RE_REPEAT_WORD = re.compile(rf"([{CJK}]{{2,4}})\1+")

# 中文数字串（年份、金额、口述数字）。叠字压缩会把它们误当口吃：
#   「二零二零年」→ RE_REPEAT_WORD 压成「二零年」（年份直接错一位）
#   「二零二二年」→ RE_REPEAT_CJK 压成「二零二年」
#   「三三五元」   → RE_REPEAT_CJK 压成「三五元」（335 变 35）
#   「二二年」     → RE_REPEAT_CJK 压成「二年」（2022 变 2）
# 这些是事实性数字，规整阶段一个都不能碰。两位以上的中文数字串整段保护；
# 口吃（我我我 / 统一一下）是否顺滑，交给后面有上下文的 LLM 润色判断——
# 它能分辨「统一一下」和「二二年」，正则不能。
CN_DIGIT_CHARS = "零〇一二三四五六七八九"
# 「幺」是读数字用的「1」（八八幺幺 = 88:11），同样属于不能压的数字读法
RE_CN_DIGIT_RUN = re.compile(rf"[{CN_DIGIT_CHARS}幺]{{2,}}")

FULLWIDTH_MAP = {
    "！": "！", "？": "？", "，": "，", "。": "。", "；": "；", "：": "：",
}


# ---------------------------------------------------------------- 基础规整

def normalize(text: str) -> str:
    """标点与空白规整，不改变用词。"""
    if not text:
        return ""
    t = text.replace("\u3000", " ").strip()
    t = t.replace("．", ".").replace("､", ",")
    t = RE_MULTI_PUNCT_MIX.sub(r"\1", t)
    t = RE_MULTI_PUNCT.sub(r"\1", t)
    t = RE_SPACE_BEFORE_PUNCT.sub(r"\1", t)
    t = RE_PUNCT_SPACE.sub(r"\1", t)
    t = RE_HALF_PUNCT.sub("，", t)
    t = RE_CJK_SPACE.sub("", t)
    t = RE_TRAILING_JUNK.sub("", t)
    t = RE_LEADING_JUNK.sub("", t)
    return t.strip()


def _compress_stutter(text: str) -> str:
    """对一段不含中文数字串的文本做叠字压缩。"""
    if not text:
        return text
    t = RE_REPEAT_CJK.sub(r"\1", text)
    t = RE_REPEAT_WORD.sub(r"\1", t)
    return t


def strip_stutter(text: str) -> str:
    """去掉明显的口吃重复：我我我 → 我；这个这个 → 这个。

    中文数字串（两位及以上）整段保护，不参与叠字压缩——年份、金额一旦被
    压掉一位就是事实性错误。数字相关的口吃（统一一下）保留原样，由后面
    有上下文的 LLM 润色顺带处理。
    """
    if not text:
        return ""
    if not RE_CN_DIGIT_RUN.search(text):
        return _compress_stutter(text)

    parts: List[str] = []
    last = 0
    for m in RE_CN_DIGIT_RUN.finditer(text):
        parts.append(_compress_stutter(text[last:m.start()]))
        parts.append(m.group(0))
        last = m.end()
    parts.append(_compress_stutter(text[last:]))
    return "".join(parts)


def _drop_token(text: str, token: str) -> str:
    """把 token 当作独立词删除（中英文边界安全）。"""
    if not token:
        return text
    pattern = re.compile(
        rf"(?<![{CJK}A-Za-z0-9]){re.escape(token)}(?![{CJK}A-Za-z0-9])"
    )
    out = pattern.sub("", text)
    return out


def _is_standalone_interjection(text: str, token: str) -> bool:
    """判断语气词是否独立成词：前面是标点/开头，后面是标点/结尾。"""
    idx = text.find(token)
    while idx != -1:
        before = text[idx - 1] if idx > 0 else ""
        after = text[idx + len(token)] if idx + len(token) < len(text) else ""
        boundary_before = (not before) or before in "，。！？；：、,.!?;:\n “”\"'（(【["
        boundary_after = (not after) or after in "，。！？；：、,.!?;:\n “”\"'）)】]"
        if boundary_before and boundary_after:
            return True
        idx = text.find(token, idx + 1)
    return False


def strip_fillers(text: str, level: str = "light") -> str:
    """去语气词 / 口水词。

    level:
      - none     : 不处理
      - light    : 只删独立叹词 + 强口水词
      - standard : 再删更多话语标记与重复句尾
    """
    if not text or level == "none":
        return text

    t = text
    if level in ("light", "standard"):
        for token in INTERJECTIONS:
            while _is_standalone_interjection(t, token):
                new = _drop_token(t, token)
                if new == t:
                    break
                t = new
        for token in FILLERS_LIGHT:
            t = t.replace(token, "")

    if level == "standard":
        for token in FILLERS_STANDARD:
            t = _drop_token(t, token)
        # 去掉“……，，”这种被掏空后残留的标点
        t = RE_MULTI_PUNCT_MIX.sub(r"\1", t)
        t = RE_MULTI_PUNCT.sub(r"\1", t)

    t = normalize(t)
    return t


def clean_text(text: str, level: str = "light") -> str:
    """完整规整流水线。"""
    t = normalize(text)
    t = strip_stutter(t)
    t = strip_fillers(t, level=level)
    return t


# ---------------------------------------------------------------- 术语纠正

DEFAULT_TERMS: Dict[str, str] = {}


def apply_terms(text: str, terms: Optional[Dict[str, str]] = None) -> str:
    """按用户词典做精确替换（长词优先），用于纠正专有名词。"""
    if not text:
        return text
    table = dict(DEFAULT_TERMS)
    if terms:
        table.update(terms)
    if not table:
        return text
    for wrong in sorted(table, key=len, reverse=True):
        right = table[wrong]
        if wrong and right and wrong in text:
            text = text.replace(wrong, right)
    return text


# ---------------------------------------------------------------- 句子切分

_SENT_END = "。！？!?…"


def split_sentences(text: str, min_len: int = 6) -> List[str]:
    """把一段文字切成句子（保留标点）。"""
    if not text:
        return []
    out: List[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in _SENT_END:
            out.append(buf)
            buf = ""
    if buf.strip():
        out.append(buf)

    # 合并过短的碎片
    merged: List[str] = []
    for s in out:
        s = s.strip()
        if not s:
            continue
        if merged and len(s) < min_len and len(merged[-1]) + len(s) < 40:
            merged[-1] = merged[-1] + s
        else:
            merged.append(s)
    return merged


def estimate_char_weights(sentences: Sequence[str]) -> List[float]:
    """按字符数估算句子权重，用于在没有词级时间戳时按比例分配时间。"""
    total = sum(max(1, len(s)) for s in sentences) or 1
    return [max(1, len(s)) / total for s in sentences]
