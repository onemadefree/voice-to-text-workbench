#!/bin/bash
# 补做其余文档的润色 + 知识库 + 导出。
#
# 前提：服务已用新代码重启（/api/health 里的 build 标记为最新），
# 这样 LocalLLM 实例会复用、加载模型前会先回收 mlx 缓存里的显存，
# 16GB 机器上连续跑多篇不会再因为重复加载 8B 而内存不足。
set -u
cd "/Users/onemade/work/语音转文字" || exit 1
LOG="output/_polish.log"
exec >> "$LOG" 2>&1

echo "[$(date '+%F %T')] 等待服务以新代码重启…"
code=""
for _ in $(seq 1 1800); do
  body=$(curl -s -m 5 http://127.0.0.1:8787/api/health 2>/dev/null)
  if echo "$body" | grep -q '"build"'; then
    code="ok"
    echo "[$(date '+%F %T')] 服务已重启：$body"
    break
  fi
  sleep 10
done
if [ "$code" != "ok" ]; then
  echo "[$(date '+%F %T')] 等待超时（服务没重启），退出"
  exit 1
fi

echo "[$(date '+%F %T')] 开始补做润色 + 知识库（已润色的段落会自动跳过）"
.venv/bin/python scripts/batch_service.py --polish-only
echo "[$(date '+%F %T')] 润色批次结束（退出码 $?）"
