"""识别引擎抽象。

默认引擎 Qwen3ASREngine（mlx-audio + Qwen3-ASR，Apple Silicon 原生加速）。
可选强制对齐器 Qwen3-ForcedAligner，把一段音频里的文本切成带精确时间戳的句子。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import config

log = logging.getLogger("stt.asr")


def _require_local_model(model_id: str, kind: str) -> None:
    """模型必须已下载到 models/ 目录，避免运行时卡在自动下载上。"""
    if Path(model_id).exists():
        return
    if "/" in model_id:
        raise RuntimeError(
            f"{kind} 模型未下载：{model_id}\n"
            f"请先执行： .venv/bin/python scripts/fetch_models.py "
            f"{'hq' if '1.7B' in model_id else 'asr'}"
        )


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"start": round(self.start, 3), "end": round(self.end, 3), "text": self.text}
        if self.words:
            d["words"] = self.words
        return d


class BaseEngine:
    name = "base"

    def load(self) -> None:  # pragma: no cover - 抽象
        raise NotImplementedError

    def transcribe(self, wav_path: Path, offset: float = 0.0) -> Segment:  # pragma: no cover
        raise NotImplementedError

    def align(self, wav_path: Path, text: str, offset: float = 0.0) -> Optional[List[Dict[str, Any]]]:
        return None


class Qwen3ASREngine(BaseEngine):
    """Qwen3-ASR（MLX 量化版）识别 + Qwen3-ForcedAligner 词级对齐。"""

    name = "qwen3-asr"

    def __init__(
        self,
        model: Optional[str] = None,
        aligner_model: Optional[str] = None,
        language: Optional[str] = None,
        use_aligner: bool = True,
        hotwords: Optional[Sequence[str]] = None,
    ) -> None:
        self.model_id = config.local_model(model or config.ASR_MODEL)
        self.aligner_id = config.local_model(aligner_model or config.ALIGNER_MODEL)
        self.language = language or config.ASR_LANGUAGE
        self.use_aligner = use_aligner
        self.hotwords = list(hotwords or [])
        self._model = None
        self._aligner = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ 加载
    def load(self) -> None:
        if self._model is not None:
            return
        _require_local_model(self.model_id, "ASR")
        from mlx_audio.stt import load as stt_load  # 延迟导入，避免无模型时也加载 MLX

        log.info("加载 ASR 模型: %s", self.model_id)
        self._model = stt_load(self.model_id)
        if self.use_aligner:
            try:
                if Path(self.aligner_id).exists():
                    log.info("加载对齐模型: %s", self.aligner_id)
                    self._aligner = stt_load(self.aligner_id)
                else:
                    raise FileNotFoundError(self.aligner_id)
            except Exception as exc:  # 对齐是可选增强，失败不阻断
                log.warning("对齐模型不可用（将退化为按字数估时）: %s", exc)
                self._aligner = None

    @property
    def has_aligner(self) -> bool:
        return self._aligner is not None

    # ------------------------------------------------------------------ 识别
    def transcribe(self, wav_path: Path, offset: float = 0.0) -> Segment:
        self.load()
        with self._lock:
            kwargs: Dict[str, Any] = {"language": self.language}
            if self.hotwords:
                kwargs["hotwords"] = self.hotwords
            result = self._model.generate(str(wav_path), **kwargs)
        text = (getattr(result, "text", "") or "").strip()
        segs = getattr(result, "segments", None) or []
        end = offset
        if segs:
            try:
                end = offset + float(segs[-1].get("end", 0.0))
            except Exception:
                end = offset
        return Segment(start=offset, end=end, text=text)

    # ------------------------------------------------------------------ 对齐
    def align(self, wav_path: Path, text: str, offset: float = 0.0) -> Optional[List[Dict[str, Any]]]:
        """返回 [{text, start, end}]，时间为相对整段音频的绝对秒。"""
        if self._aligner is None or not text:
            return None
        self.load()
        with self._lock:
            try:
                items = self._aligner.generate(audio=str(wav_path), text=text, language=self.language)
            except Exception as exc:
                log.warning("对齐失败: %s", exc)
                return None
        out: List[Dict[str, Any]] = []
        for item in items or []:
            try:
                out.append({
                    "text": str(getattr(item, "text", "")),
                    "start": round(offset + float(getattr(item, "start_time", 0.0)), 3),
                    "end": round(offset + float(getattr(item, "end_time", 0.0)), 3),
                })
            except Exception:
                continue
        return out or None


def build_engine(engine: str = "qwen3", **kwargs) -> BaseEngine:
    """engine: qwen3 | qwen3-hq | whisper"""
    engine = (engine or "qwen3").lower()
    if engine in ("qwen3", "qwen3-asr"):
        return Qwen3ASREngine(model=config.ASR_MODEL, **kwargs)
    if engine in ("qwen3-hq", "hq", "1.7b"):
        return Qwen3ASREngine(model=config.ASR_MODEL_HQ, **kwargs)
    if engine in ("whisper", "whisper-turbo"):
        from .asr_whisper import WhisperEngine

        return WhisperEngine(**kwargs)
    raise ValueError(f"未知引擎: {engine}")
