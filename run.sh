#!/usr/bin/env bash
# 启动本地语音转文字工作台
set -euo pipefail

cd "$(dirname "$0")"

HOST="${STT_HOST:-127.0.0.1}"
PORT="${STT_PORT:-8787}"

echo "→ 语音转文字工作台: http://${HOST}:${PORT}"
echo "  源音频目录: ${STT_SOURCE_DIR:-./source}"
echo "  模型: ${STT_ASR_MODEL:-mlx-community/Qwen3-ASR-0.6B-8bit}"

exec .venv/bin/python -m uvicorn app.server:app --host "$HOST" --port "$PORT" "$@"
