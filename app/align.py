"""把 ASR 文本与词级强制对齐结果合并成「带精确时间戳的句子」。

关键点：对齐器只输出实词（不含标点），所以**句子文本必须取自 ASR 原文**，
对齐结果只用来提供每个字的时间。做法是把对齐词展开成字符，
再和 ASR 文本逐字对齐，得到字符级时间轴，最后按标点切句。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

PUNCT = set(
    "，。！？；：、,.!?;:…—－-～~《》〈〉「」『』【】〔〕()（）“”\"'‘’·「」\n\r\t "
)

SENT_END = "。！？!?…"
PAUSE_GAP = 0.42          # 超过这个停顿视为一句结束
MAX_SENT_CHARS = 60       # 无标点时的最长句长


def _expand_words(words: Sequence[Dict[str, Any]]) -> List[Tuple[str, float, float]]:
    """把词级时间戳展开成字符级（英文单词内部线性插值）。"""
    out: List[Tuple[str, float, float]] = []
    for w in words:
        text = str(w.get("text", ""))
        if not text:
            continue
        try:
            start = float(w.get("start", 0.0))
            end = float(w.get("end", start))
        except (TypeError, ValueError):
            continue
        n = len(text)
        step = (end - start) / n if n > 1 else 0.0
        for i, ch in enumerate(text):
            out.append((ch, start + step * i, start + step * (i + 1)))
    return out


def char_timeline(text: str, words: Sequence[Dict[str, Any]]) -> List[Optional[Tuple[float, float]]]:
    """返回与 text 等长的字符时间轴，标点沿用邻近字符的时间。"""
    n = len(text)
    times: List[Optional[Tuple[float, float]]] = [None] * n
    expanded = _expand_words(words)
    if not expanded:
        return times

    ei = 0
    for i, ch in enumerate(text):
        if ch in PUNCT:
            continue
        hop = 0
        while ei < len(expanded) and expanded[ei][0] != ch and hop < 4:
            ei += 1
            hop += 1
        if ei < len(expanded) and expanded[ei][0] == ch:
            times[i] = (expanded[ei][1], expanded[ei][2])
            ei += 1

    # 向前继承
    last: Optional[Tuple[float, float]] = None
    for i in range(n):
        if times[i] is None:
            if last is not None:
                times[i] = (last[0], last[1])
        else:
            last = times[i]
    # 向后继承
    nxt: Optional[Tuple[float, float]] = None
    for i in range(n - 1, -1, -1):
        if times[i] is None:
            if nxt is not None:
                times[i] = (nxt[0], nxt[1])
        else:
            nxt = times[i]
    return times


def split_sentence_ranges(text: str) -> List[Tuple[int, int]]:
    """把文本切成句子，返回 [(起始下标, 结束下标)]，下标含标点。"""
    ranges: List[Tuple[int, int]] = []
    start = 0
    for i, ch in enumerate(text):
        if ch in SENT_END:
            ranges.append((start, i + 1))
            start = i + 1
    if start < len(text):
        ranges.append((start, len(text)))
    return [(a, b) for a, b in ranges if text[a:b].strip()]


def _split_by_pause(
    text: str,
    times: Sequence[Optional[Tuple[float, float]]],
) -> List[Tuple[int, int]]:
    """没有标点时，按停顿间隔与最长句长切分。"""
    ranges: List[Tuple[int, int]] = []
    start = 0
    prev_end: Optional[float] = None
    for i, ch in enumerate(text):
        t = times[i] if i < len(times) else None
        if t is None:
            continue
        gap = (t[0] - prev_end) if prev_end is not None else 0.0
        prev_end = t[1]
        long_enough = (i - start) >= MAX_SENT_CHARS
        if gap >= PAUSE_GAP or long_enough:
            ranges.append((start, i + 1))
            start = i + 1
            prev_end = None
    if start < len(text):
        ranges.append((start, len(text)))
    return [(a, b) for a, b in ranges if text[a:b].strip()]


def sentences_with_times(
    text: str,
    words: Optional[Sequence[Dict[str, Any]]],
    offset: float = 0.0,
    span: Optional[Tuple[float, float]] = None,
) -> List[Dict[str, Any]]:
    """返回 [{start, end, text}]，时间为相对整条音频的绝对秒。"""
    text = (text or "").strip()
    if not text:
        return []

    if words:
        times = char_timeline(text, words)
        ranges = split_sentence_ranges(text)
        if len(ranges) <= 1 and not any(ch in SENT_END for ch in text):
            ranges = _split_by_pause(text, times)
        out: List[Dict[str, Any]] = []
        for a, b in ranges:
            seg_text = text[a:b].strip()
            if not seg_text:
                continue
            t_start = times[a] if a < len(times) else None
            t_end = times[b - 1] if b - 1 < len(times) else None
            if t_start is None or t_end is None:
                continue
            out.append({
                "start": round(offset + t_start[0], 3),
                "end": round(offset + t_end[1], 3),
                "text": seg_text,
            })
        if out:
            return out

    # 没有对齐结果：按字符数比例分配到整块时间区间
    from .textproc import split_sentences

    sentences = split_sentences(text)
    if not sentences:
        sentences = [text]
    total = sum(max(1, len(s)) for s in sentences) or 1
    if span is None:
        span = (offset, offset)
    span_start, span_end = span
    width = max(0.0, span_end - span_start)
    out = []
    cursor = 0.0
    for s in sentences:
        w = max(1, len(s)) / total
        out.append({
            "start": round(span_start + width * cursor, 3),
            "end": round(span_start + width * (cursor + w), 3),
            "text": s,
        })
        cursor += w
    return out
