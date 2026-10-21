"""端到端管线：切段 → 识别 → 对齐 → 规则清理 → LLM 润色 → 知识库。

每个阶段都会落盘，中断后可以续跑（已识别段落不重复识别）。
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import audio as A
from . import config, textproc
from .store import DocumentStore, new_document, segment_final_text

log = logging.getLogger("stt.pipeline")

SENT_PUNCT = "。！？!?…"


# ---------------------------------------------------------------- 任务管理

@dataclass
class Job:
    doc_id: str
    stage: str = "queued"
    message: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    error: str = ""

    @property
    def running(self) -> bool:
        return self.finished_at is None


class Pipeline:
    """所有耗时操作的调度中心。"""

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.jobs: Dict[str, Job] = {}
        self._lock = threading.RLock()
        self._threads: Dict[str, threading.Thread] = {}
        self._llm = None

    # ------------------------------------------------------------ 公共入口
    def start_transcribe(
        self,
        doc_id: str,
        source_path: Path,
        engine: str = "qwen3",
        language: Optional[str] = None,
        align: bool = True,
        use_rules: bool = True,
        force: bool = False,
        hotwords: Optional[Sequence[str]] = None,
    ) -> Job:
        with self._lock:
            job = self.jobs.get(doc_id)
            if job and job.running:
                return job
            job = Job(doc_id=doc_id, stage="queued")
            self.jobs[doc_id] = job

        def runner():
            try:
                self._transcribe(doc_id, source_path, engine, language, align, use_rules, force, hotwords, job)
                job.stage = "done"
                job.message = "完成"
            except Exception as exc:
                log.exception("转写失败 %s", doc_id)
                job.stage = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.message = job.error
                self.store.update(doc_id, status="error", error=job.error)
            finally:
                job.finished_at = time.time()

        th = threading.Thread(target=runner, name=f"transcribe-{doc_id}", daemon=True)
        with self._lock:
            self._threads[doc_id] = th
        th.start()
        return job

    def get_job(self, doc_id: str) -> Optional[Job]:
        with self._lock:
            return self.jobs.get(doc_id)

    def get_llm(self, backend: Optional[str] = None):
        """取 LLM 客户端。backend 为 None 时用默认（本地 mlx-lm）。"""
        if backend:
            from .llm import LocalLLM

            return LocalLLM(backend=backend)
        if self._llm is None:
            from .llm import LocalLLM

            self._llm = LocalLLM()
        return self._llm

    # ------------------------------------------------------------ 转写主体
    def _transcribe(
        self,
        doc_id: str,
        source_path: Path,
        engine: str,
        language: Optional[str],
        align: bool,
        use_rules: bool,
        force: bool,
        hotwords: Optional[Sequence[str]],
        job: Job,
    ) -> None:
        from .asr import build_engine

        config.ensure_dirs()
        out_dir = config.OUTPUT_DIR / doc_id
        out_dir.mkdir(parents=True, exist_ok=True)
        wav = out_dir / "audio_16k.wav"
        # 每次运行使用独立的临时目录，避免并发运行时互相覆盖分块音频
        work = out_dir / "chunks" / f"run-{os.getpid()}"
        work.mkdir(parents=True, exist_ok=True)

        # ---- 媒体信息
        job.stage = "probing"
        job.message = "读取音频信息"
        info = A.probe(source_path)
        media = info.to_dict()

        doc = self.store.load(doc_id)
        if doc is None:
            doc = new_document(doc_id, source_path, media)
        doc["media"] = media
        doc["status"] = "running"
        doc["error"] = ""
        doc["progress"] = {"done": 0, "total": 0, "stage": "转码", "message": ""}
        self.store.save(doc)

        # ---- 转码
        job.stage = "converting"
        job.message = "转码为 16kHz 单声道"
        A.to_wav16k(source_path, wav)

        # ---- 切段
        job.stage = "segmenting"
        job.message = "静音检测与分段"
        silences = A.detect_silence(wav)
        regions = A.speech_regions(silences, info.duration)
        chunks = A.chunk_regions(regions, silences, config.SEG_MAX_SECONDS, config.SEG_MIN_MERGE)
        total = len(chunks)
        doc["progress"] = {"done": 0, "total": total, "stage": "识别中", "message": ""}
        doc["asr"] = {
            "engine": engine,
            "language": language or config.ASR_LANGUAGE,
            "aligned": bool(align),
            "chunk_count": total,
            "hotwords": list(hotwords or []),
        }
        self.store.save(doc)

        # ---- 识别
        asr = build_engine(engine, language=language, use_aligner=align,
                           hotwords=list(hotwords or []) if hotwords else None)
        job.stage = "loading_model"
        job.message = "加载模型"
        asr.load()

        # 断点续跑：按「块序号」归拢已有结果，块内句子整体复用。
        # 段落归属一律按时间中点重新推导（不信任旧文件里的 chunk 字段），
        # 落在块间隙时取最近的块，保证段落不会被丢弃。
        existing: Dict[int, List[Dict[str, Any]]] = {}
        if not force:
            for seg in doc.get("segments") or []:
                if not seg.get("raw"):
                    continue
                mid = (float(seg.get("start", 0.0)) + float(seg.get("end", 0.0))) / 2
                c = _chunk_index_for(chunks, mid)
                if c is None:
                    continue
                seg["chunk"] = c
                existing.setdefault(c, []).append(seg)

        segments: List[Dict[str, Any]] = []
        t0 = time.time()

        for idx, (start, end) in enumerate(chunks):
            cached = existing.get(idx)
            # 覆盖度不完整（历史数据缺失/损坏）时重新识别该块
            if cached and not _chunk_is_complete(cached, (start, end)):
                cached = None
            if cached:
                segments.extend(cached)
            else:
                chunk_wav = work / f"{idx:05d}.wav"
                A.cut(wav, start, end, chunk_wav)
                try:
                    res = asr.transcribe(chunk_wav, offset=0.0)
                    text = res.text
                    words = asr.align(chunk_wav, text, offset=0.0) if align else None
                    for seg in self._split_into_segments(text, words, start, end):
                        seg["chunk"] = idx
                        segments.append(seg)
                finally:
                    chunk_wav.unlink(missing_ok=True)

            elapsed = time.time() - t0
            done = idx + 1
            rate = elapsed / max(1, done)
            eta = rate * (total - done)
            for i, seg in enumerate(segments):
                seg["id"] = i
            doc["segments"] = segments
            doc["progress"] = {
                "done": done,
                "total": total,
                "stage": "识别中",
                "message": f"{A.fmt_ts(start)} → {A.fmt_ts(end)}",
                "elapsed": round(elapsed, 1),
                "eta": round(eta, 1),
            }
            self.store.save(doc)
            job.message = f"{done}/{total} 段"

        # ---- 去重（同一时间同一文本只保留一条）
        deduped = _dedupe_segments(segments)
        if len(deduped) != len(segments):
            log.info("去除重复段落 %d 条", len(segments) - len(deduped))
        segments = deduped

        # ---- 清理退化段落（零时长 + 幻觉重复串）
        clean = _drop_degenerate(segments)
        if len(clean) != len(segments):
            log.info("清理退化段落 %d 条", len(segments) - len(clean))
        segments = clean
        doc["segments"] = segments  # 后续规则清理/润色都基于这份列表

        # ---- 规则清理
        if use_rules:
            job.stage = "cleaning"
            job.message = "口语清理"
            self._apply_rules_to(doc)

        doc["status"] = "ready"
        doc["progress"] = {"done": total, "total": total, "stage": "完成", "message": ""}
        self.store.save(doc)

        # ---- 预览与波形（后台做，不阻塞）
        job.stage = "preview"
        job.message = "生成预览音频与波形"
        try:
            self._build_preview(source_path, doc_id)
        except Exception as exc:
            log.warning("预览生成失败: %s", exc)
        try:
            self._build_peaks(source_path, doc_id)
        except Exception as exc:
            log.warning("波形生成失败: %s", exc)

        job.stage = "done"
        job.message = "完成"

    # ------------------------------------------------------------ 切句
    def _split_into_segments(
        self,
        text: str,
        words: Optional[List[Dict[str, Any]]],
        start: float,
        end: float,
    ) -> List[Dict[str, Any]]:
        """把一段识别结果切成带时间戳的句子（文本取自 ASR 原文，对齐只提供时间）。"""
        from .align import sentences_with_times

        sentences = sentences_with_times(text, words, offset=start, span=(start, end))
        return [
            {
                "start": s["start"],
                "end": max(s["end"], s["start"]),
                "raw": s["text"],
            }
            for s in sentences
        ]

    # ------------------------------------------------------------ 后处理
    def _apply_rules_to(self, doc: Dict[str, Any]) -> None:
        segs = doc.get("segments") or []
        for i, seg in enumerate(segs):
            raw = seg.get("raw") or ""
            seg["clean"] = textproc.clean_text(raw, level="light")
            seg["id"] = i
        self.store.save(doc)

    def run_rules(self, doc_id: str, level: str = "light") -> Optional[Dict[str, Any]]:
        doc = self.store.load(doc_id)
        if doc is None:
            return None
        for i, seg in enumerate(doc.get("segments") or []):
            raw = seg.get("raw") or ""
            seg["clean"] = textproc.clean_text(raw, level=level)
            seg["id"] = i
        return self.store.save(doc)

    def run_polish(
        self,
        doc_id: str,
        indices: Optional[Sequence[int]] = None,
        level: str = "standard",
        use_context: bool = True,
        group_size: int = 96,
        force_polish: bool = False,
        backend: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """用 LLM 对指定段落（默认全部）做纠错+润色。

        backend=None 用默认后端（本地 mlx-lm）；也可传 "minimax"/"deepseek" 等
        预设名或 "openai"（配合 STT_LLM_API_* 配置）。
        按 group_size 分组处理并逐组落盘，中途失败不会丢掉已完成的部分。
        """
        doc = self.store.load(doc_id)
        if doc is None:
            return None
        segs = doc.get("segments") or []
        if not segs:
            return doc

        idxs = [i for i in (indices if indices is not None else range(len(segs)))
                if 0 <= i < len(segs)]
        # 已经润色过的段落默认跳过（支持中断续跑）
        if not force_polish:
            idxs = [i for i in idxs if not segs[i].get("polished")]
        if not idxs:
            return doc

        llm = self.get_llm(backend)
        from .llm import polish_segments

        for g_start in range(0, len(idxs), group_size):
            group = idxs[g_start:g_start + group_size]
            work = [
                {"raw": segs[i].get("clean") or segs[i].get("raw") or "", "text": segs[i].get("raw") or ""}
                for i in group
            ]
            head = ""
            if use_context and g_start > 0:
                head = " ".join(
                    (segs[j].get("clean") or segs[j].get("raw") or "")
                    for j in idxs[max(0, g_start - 3):g_start]
                )[:400]

            results = polish_segments(llm, work, context_head=head)
            for pos, i in enumerate(group):
                segs[i]["polished"] = results[pos]
                segs[i]["polish_level"] = level

            doc["polish_progress"] = {
                "done": min(g_start + group_size, len(idxs)),
                "total": len(idxs),
            }
            self.store.save(doc)

        doc.pop("polish_progress", None)
        return self.store.save(doc)

    def run_knowledge_base(self, doc_id: str, title: str = "",
                           backend: Optional[str] = None) -> Optional[Dict[str, Any]]:
        doc = self.store.load(doc_id)
        if doc is None:
            return None
        segs = doc.get("segments") or []
        text = "\n".join(segment_final_text(s) for s in segs)
        if not text.strip():
            return doc
        from .llm import build_knowledge_base

        kb = build_knowledge_base(self.get_llm(backend), text,
                                  title=title or doc.get("source", ""))
        doc["knowledge_base"] = kb
        return self.store.save(doc)

    # ------------------------------------------------------------ 预览资源
    def _build_preview(self, source_path: Path, doc_id: str) -> None:
        out = config.OUTPUT_DIR / doc_id / "preview.mp3"
        A.make_preview(source_path, out)

    def _build_peaks(self, source_path: Path, doc_id: str) -> None:
        out = config.OUTPUT_DIR / doc_id / "peaks.bin"
        if out.exists() and out.stat().st_size > 0:
            return
        data = A.compute_peaks(source_path)
        out.write_bytes(data)


# ---------------------------------------------------------------- 工具函数

def _dedupe_segments(segments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """去掉「时间完全相同且文本相同」的重复段落（并发/重跑可能产生）。"""
    seen = set()
    out: List[Dict[str, Any]] = []
    for seg in segments:
        key = (round(float(seg.get("start", 0.0)), 2), round(float(seg.get("end", 0.0)), 2),
               (seg.get("raw") or "").strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(seg)
    for i, seg in enumerate(out):
        seg["id"] = i
    return out


_HALLUC_RE = re.compile(r"[\s，。！？、；：,.!?;:\"'“”‘’（）()《》…·—\-~·]+")


def _strip_punct(text: str) -> str:
    return _HALLUC_RE.sub("", text or "")


def _is_hallucination(text: str) -> bool:
    """判断是否为 ASR 幻觉串。

    静音/噪声段里模型会反复吐同一句话（"一，一，一，…"或"就我这个事儿，
    就我这个事儿，…"），文本成倍冗余。判据：去掉标点后，尾部是某个 1~8 字
    单元的整倍重复（单字重复 3 次起，多字单元 4 次起，避免误伤口语复述）。
    """
    chars = _strip_punct(text)
    if len(chars) < 3:
        return False
    for unit in range(1, 9):
        k = len(chars) // unit
        need = 3 if unit == 1 else 4
        if k < need or (unit >= 2 and len(chars) < 8):
            continue
        if chars[-unit * k:] == chars[-unit:] * k:
            return True
    return False


def _interpolate_missing_times(segments: List[Dict[str, Any]]) -> None:
    """给「有文字但零时长」的段落补出时间跨度。

    对齐器在部分区域会给出零跨度，但文字是真实口语，不能丢。按前后
    有时间的邻居之间的空隙、按字数比例分配，让时间轴继续单调可用。
    """
    n = len(segments)
    i = 0
    while i < n:
        if segments[i].get("end", 0.0) - segments[i].get("start", 0.0) > 0.01:
            i += 1
            continue
        j = i
        while (j < n and segments[j].get("end", 0.0) - segments[j].get("start", 0.0) <= 0.01):
            j += 1
        # 前一个已知时间的段落
        prev_end = segments[i - 1]["end"] if i > 0 else None
        # 后一个已知时间的段落
        next_start = segments[j]["start"] if j < n else None
        run = segments[i:j]
        weights = [max(1, len(_strip_punct(s.get("raw") or ""))) for s in run]
        total_w = sum(weights) or 1
        if prev_end is not None and next_start is not None and next_start > prev_end:
            gap = next_start - prev_end
            cur = prev_end
            for s, w in zip(run, weights):
                span = gap * w / total_w
                s["start"], s["end"] = cur, cur + span
                cur += span
        elif prev_end is not None:
            cur = prev_end
            for s, w in zip(run, weights):
                span = max(0.3, w * 0.22)
                s["start"], s["end"] = cur, cur + span
                cur += span
        elif next_start is not None:
            cur = next_start
            for s, w in zip(run, weights):
                span = max(0.3, w * 0.22)
                s["start"], s["end"] = cur - span, cur
                cur -= span
        i = j


def _drop_degenerate(segments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """丢掉退化段落，并给零时长的真实内容补时间。

    三类要丢：
    1. 重复幻觉串——模型在静音区自问自答（"一，一，一，…"）；
    2. 零时长的对齐噪声——只剩 "嗯。"、"”" 这类语气碎片；
    3. 连续退化区——成片的同质碎片（模型卡住刷"线，线，…"），整片丢弃。
    零时长但有实义的口语保留，并按邻居空隙插值补时间。
    这类段落若不清理，会把段数放大数倍（实测 11000 段 → 2700 段），
    后续 LLM 润色会大量浪费时间并产出垃圾。
    """
    out: List[Dict[str, Any]] = []
    for seg in segments:
        raw = seg.get("raw") or ""
        if _is_hallucination(raw):
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        if end - start <= 0.01 and len(_strip_punct(raw)) <= 2:
            continue
        out.append(seg)

    # 按时间排序：退化区里对齐器会给出乱序的时间戳
    out.sort(key=lambda s: (float(s.get("start", 0.0)), float(s.get("end", 0.0))))
    out = _drop_degenerate_regions(out)
    _interpolate_missing_times(out)
    for i, seg in enumerate(out):
        seg["id"] = i
    return out


def _drop_degenerate_regions(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """整片丢弃「同质碎片」连续区（模型卡在同一个字上刷不出内容）。

    判据：按时间相邻的若干段，文字去掉标点后只有 <=2 种字、且每段都很短，
    连续达到 5 段以上即视为退化区（实测出现过 5 分半的连续刷屏）。
    """
    out: List[Dict[str, Any]] = []
    run: List[Dict[str, Any]] = []

    def flush() -> None:
        nonlocal run
        if len(run) < 5:
            out.extend(run)
        run = []

    for seg in segments:
        txt = _strip_punct(seg.get("raw") or "")
        span = float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))
        homogeneous = bool(txt) and len(set(txt)) <= 2 and span <= 1.5
        if homogeneous:
            run.append(seg)
        else:
            flush()
            out.append(seg)
    flush()
    return out


def _chunk_index_for(chunks: Sequence[Tuple[float, float]], t: float) -> Optional[int]:
    """找出时间 t 所属的块；落在块之间的空隙时取时间上最近的块。"""
    if not chunks:
        return None
    for i, (start, end) in enumerate(chunks):
        if start <= t < end:
            return i
    best = 0
    best_dist = None
    for i, (start, end) in enumerate(chunks):
        dist = (start - t) if t < start else (t - end)
        if dist < 0:
            dist = -dist
        if best_dist is None or dist < best_dist:
            best, best_dist = i, dist
    return best


def _chunk_is_complete(
    segs: Sequence[Dict[str, Any]],
    chunk: Tuple[float, float],
    tolerance: float = 2.5,
) -> bool:
    """判断某块的已有结果是否完整（覆盖整块且无异常空隙）。"""
    if not segs:
        return False
    cs, ce = chunk
    items = sorted(((float(s.get("start", 0.0)), float(s.get("end", 0.0))) for s in segs))
    if items[0][0] > cs + tolerance:
        return False
    if items[-1][1] < ce - tolerance:
        return False
    for (s1, e1), (s2, _) in zip(items, items[1:]):
        if s2 - e1 > tolerance:
            return False
    return True
