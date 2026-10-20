#!/usr/bin/env bash
# 一键安装：Python 环境 + 依赖 + 模型
set -euo pipefail

cd "$(dirname "$0")"

PY_VERSION="3.11"
# 国内网络建议开启镜像（已默认开启，可按需关闭）
export UV_INDEX_URL="${UV_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "▸ 1/4 创建 Python ${PY_VERSION} 虚拟环境"
command -v uv >/dev/null 2>&1 || { echo "需要先安装 uv: https://docs.astral.sh/uv/"; exit 1; }
uv python install "$PY_VERSION"
uv venv --python "$PY_VERSION" .venv

echo "▸ 2/4 安装依赖"
uv pip install --python .venv/bin/python \
  mlx-audio mlx-lm fastapi "uvicorn[standard]" python-multipart

echo "▸ 3/4 下载模型（约 6 GB，用时取决于网速）"
.venv/bin/python scripts/fetch_models.py asr aligner llm

echo "▸ 4/4 自检"
.venv/bin/python scripts/selftest.py

echo ""
echo "完成。启动界面:  ./run.sh"
