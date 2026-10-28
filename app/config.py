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

# 构建标记：改动了服务行为（尤其是 LLM 实例复用/显存回收）时 +1，
# 用来确认「跑着的服务」加载的是不是最新代码。
APP_BUILD = os.environ.get("STT_APP_BUILD", "2026-10-28-dash-ctrl-model")

# ---------------------------------------------------------------- 润色后端
# local  = mlx-lm 本地推理（默认，全离线）
# openai = 任意 OpenAI 兼容网络 API；也可直接用下面的预设名（minimax/deepseek/…）
LLM_BACKEND = os.environ.get("STT_LLM_BACKEND", "local")
# OpenAI 兼容接口的自定义参数（预设名未覆盖时生效）
LLM_API_BASE = os.environ.get("STT_LLM_API_BASE", "")
LLM_API_KEY = os.environ.get("STT_LLM_API_KEY", "")
LLM_API_MODEL = os.environ.get("STT_LLM_API_MODEL", "")
# 推理模型（M2/M3 系列）单次调用要几十秒到几分钟，并发时更慢，超时给太小会白重试
LLM_API_TIMEOUT = float(os.environ.get("STT_LLM_API_TIMEOUT", "600"))
LLM_API_RETRIES = int(os.environ.get("STT_LLM_API_RETRIES", "3"))
# 单次调用的 token 上限。推理模型（M2/M3 系列）会先花掉几千 token 思考，
# 上限给小了会 finish_reason=length 把正文一起砍没，所以默认给得比较宽。
LLM_API_MAX_TOKENS = int(os.environ.get("STT_LLM_API_MAX_TOKENS", "16384"))
# 知识库「分段提炼」的并发数。长稿能切出几十段，串行跑慢模型要等很久。
LLM_KB_WORKERS = int(os.environ.get("STT_LLM_KB_WORKERS", "4"))

# 常见 OpenAI 兼容服务预设：设 STT_LLM_BACKEND=minimax 即可切换
LLM_PRESETS: dict = {
    "minimax": {
        "base": "https://api.minimax.chat/v1",
        "model": "MiniMax-Text-01",
        "key_env": "MINIMAX_API_KEY",
        "label": "MiniMax（国内 api.minimax.chat）",
    },
    "minimax-intl": {
        "base": "https://api.minimaxi.com/v1",
        "model": "MiniMax-Text-01",
        "key_env": "MINIMAX_API_KEY",
        "label": "MiniMax（国际 api.minimaxi.com）",
    },
    "deepseek": {
        "base": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "key_env": "DEEPSEEK_API_KEY",
        "label": "DeepSeek",
    },
    "dashscope": {
        "base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "key_env": "DASHSCOPE_API_KEY",
        "label": "阿里云百炼（通义千问）",
    },
    "kimi": {
        "base": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-32k",
        "key_env": "MOONSHOT_API_KEY",
        "label": "Kimi（月之暗面）",
    },
    "openai": {
        "base": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "key_env": "OPENAI_API_KEY",
        "label": "OpenAI",
    },
    "local-openai": {
        "base": "http://127.0.0.1:1234/v1",
        "model": "local-model",
        "key_env": "",
        "label": "本地 OpenAI 兼容服务（LM Studio / Ollama / llama.cpp）",
    },
}

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
# 按文件自适应：阈值 = 噪声底(p25) + SILENCE_OFFSET × (语音电平(p75) − 噪声底)，
# 再夹到 [SILENCE_MIN_DB, SILENCE_MAX_DB]。各录音响度实测相差 11 dB，
# 固定阈值必然对某些文件失效（轻声文件被误切 / 吵杂文件切不干净）。
SILENCE_ADAPTIVE = os.environ.get("STT_SILENCE_ADAPTIVE", "1") not in ("0", "false", "no")
SILENCE_OFFSET = float(os.environ.get("STT_SILENCE_OFFSET", "0.55"))
SILENCE_MIN_DB = float(os.environ.get("STT_SILENCE_MIN_DB", "-48"))
SILENCE_MAX_DB = float(os.environ.get("STT_SILENCE_MAX_DB", "-28"))
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
