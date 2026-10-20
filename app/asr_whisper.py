"""Whisper 备用引擎（多语言场景用；中文精度不如 Qwen3-ASR）。

通过 mlx-audio 运行 mlx-community/whisper-large-v3-turbo-asr-fp16。
Whisper 原生输出词级时间戳，因此本引擎自带句级切分，不依赖强制对齐模型。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import config

log = logging.getLogger("stt.asr.whisper")

MODEL_ID = "mlx-community/whisper-large-v3-turbo-asr-fp16"


class WhisperEngine:
    """与 Qwen3ASREngine 相同接口的 Whisper 实现。"""

    name = "whisper"

    def __init__(
        self,
        model: Optional[str] = None,
        language: Optional[str] = None,
        use_aligner: bool = True,
        hotwords: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> None:
        self.model_id = config.local_model(model or MODEL_ID)
        self.language = language or "chinese"
        self.hotwords = list(hotwords or [])
        self._model = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._model is not None:
            return
        from mlx_audio.stt import load as stt_load

        log.info("加载 Whisper 模型: %s", self.model_id)
        self._model = stt_load(self.model_id)

    @property
    def has_aligner(self) -> bool:
        return False

    def transcribe(self, wav_path: Path, offset: float = 0.0) -> Any:
        self.load()
        with self._lock:
            result = self._model.generate(str(wav_path), language=self.language)
        return result

    def align(self, wav_path: Path, text: str, offset: float = 0.0) -> Optional[List[Dict[str, Any]]]:
        """Whisper 自带 segments 时间戳，直接复用，不另跑对齐模型。"""
        self.load()
        with self._lock:
            result = self._model.generate(str(wav_path), language=self.language)
        out: List[Dict[str, Any]] = []
        for seg in getattr(result, "segments", None) or []:
            try:
                out.append({
                    "text": str(seg.get("text", "")),
                    "start": round(offset + float(seg.get("start", 0.0)), 3),
                    "end": round(offset + float(seg.get("end", 0.0)), 3),
                })
            except (TypeError, ValueError):
                continue
        return out or None
