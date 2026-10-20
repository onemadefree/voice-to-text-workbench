"""全局配置：路径、模型、参数。全部可通过环境变量覆盖。"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- 路径
SOURCE_DIR = Path(os.environ.get("STT_SOURCE_DIR", ROOT / "source")).expanduser()
OUTPUT_DIR = Path(os.environ.get("STT_OUTPUT_DIR", ROOT / "output")).expanduser()
MODELS_DIR = Path(os.environ.get("STT_MODELS_DIR", ROOT / "models")).expanduser()
CACHE_DIR = Path(os.environ.get("STT_CACHE_DIR", ROOT / ".cache")).expanduser()

# ---------------------------------------------------------------- 模型
# 主识别模型：中文 CER 最低的一档（Qwen3-ASR 0.6B 中文 CER 7.65%）
ASR_MODEL = os.environ.get("STT_ASR_MODEL", "mlx-community/Qwen3-ASR-0.6B-8bit")
# 更高精度模型（1.7B），内存吃紧时可换回 0.6B
ASR_MODEL_HQ = os.environ.get("STT_ASR_MODEL_HQ", "mlx-community/Qwen3-ASR-1.7B-8bit")
# 词级强制对齐模型（可选，用于把句子切得更细并给出精确时间戳）
ALIGNER_MODEL = os.environ.get("STT_ALIGNER_MODEL", "mlx-community/Qwen3-ForcedAligner-0.6B-8bit")
# 后处理用本地 LLM
LLM_MODEL = os.environ.get("STT_LLM_MODEL", "mlx-community/Qwen3-8B-4bit")

ASR_LANGUAGE = os.environ.get("STT_LANG", "Chinese")

# ---------------------------------------------------------------- 音频
SAMPLE_RATE = 16000
# 语音切段参数
SEG_MAX_SECONDS = float(os.environ.get("STT_SEG_MAX", "30"))
SEG_MIN_SECONDS = float(os.environ.get("STT_SEG_MIN", "0.4"))
# 小于该长度的块会与相邻块合并（给识别模型更多上下文）
SEG_MIN_MERGE = float(os.environ.get("STT_SEG_MIN_MERGE", "9"))
# 静音检测阈值（秒 / dB）
SILENCE_MIN_SEC = float(os.environ.get("STT_SILENCE_MIN", "0.45"))
SILENCE_NOISE_DB = float(os.environ.get("STT_SILENCE_DB", "-32"))
# 两段语音之间的最短停顿（秒），短于此则合并
MIN_GAP_MERGE = float(os.environ.get("STT_MIN_GAP", "0.35"))

# ---------------------------------------------------------------- 服务
HOST = os.environ.get("STT_HOST", "127.0.0.1")
PORT = int(os.environ.get("STT_PORT", "8787"))

# 预览音频（浏览器播放用，体积小、seek 快）
PREVIEW_BITRATE = os.environ.get("STT_PREVIEW_BR", "64k")
# 波形峰值密度（每秒桶数）
PEAK_BUCKETS_PER_SEC = int(os.environ.get("STT_PEAK_BPS", "20"))

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma", ".mp4", ".mov", ".mkv"}


def local_model(repo_id: str) -> str:
    """若模型已下载到 models/ 目录则返回本地路径，否则返回仓库 ID（触发自动下载）。"""
    candidate = MODELS_DIR / repo_id.replace("/", "__")
    if candidate.exists() and any(candidate.iterdir()):
        return str(candidate)
    return repo_id


def ensure_dirs() -> None:
    for d in (SOURCE_DIR, OUTPUT_DIR, MODELS_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
