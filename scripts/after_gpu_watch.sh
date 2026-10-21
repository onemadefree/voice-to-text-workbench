#!/bin/bash
# 等 GPU 会话恢复后：重启服务（加载修复后的代码）→ 补做润色与知识库
#
# 背景：23:11 前后系统显示会话异常，新进程拿不到 Metal device（已持有
# device 的服务进程仍可用）。润色需要 LLM（也要 GPU），所以等服务能重启、
# 新进程能用 Metal 之后，再补做润色。
set -u
cd "/Users/onemade/work/语音转文字" || exit 1
LOG=/tmp/batch_polish.log

echo "[$(date '+%H:%M:%S')] 等待转写批次结束…" | tee -a "$LOG"
while ! grep -q "全部完成" /tmp/batch_asr.log 2>/dev/null; do
  if grep -q "失败\|Traceback" /tmp/batch_asr.log 2>/dev/null; then
    echo "[$(date '+%H:%M:%S')] 转写批次出错，中止" | tee -a "$LOG"
    exit 1
  fi
  sleep 60
done
echo "[$(date '+%H:%M:%S')] 转写批次已结束" | tee -a "$LOG"

echo "[$(date '+%H:%M:%S')] 等待 GPU 会话恢复（新进程可用 Metal）…" | tee -a "$LOG"
while true; do
  if .venv/bin/python -c "import mlx.core as mx; assert mx.metal.is_available()" 2>/dev/null; then
    echo "[$(date '+%H:%M:%S')] GPU 已恢复" | tee -a "$LOG"
    break
  fi
  sleep 120
done

echo "[$(date '+%H:%M:%S')] 重启服务（加载修复后的代码）…" | tee -a "$LOG"
pkill -f "uvicorn app.server:app" 2>/dev/null
sleep 3
(nohup ./run.sh > /tmp/语音转文字服务.log 2>&1 &)
sleep 15
curl -s --max-time 10 http://127.0.0.1:8787/api/health | head -c 200 | tee -a "$LOG"
echo "" | tee -a "$LOG"

echo "[$(date '+%H:%M:%S')] 开始补做润色与知识库" | tee -a "$LOG"
.venv/bin/python scripts/batch_service.py --polish-only 2>&1 | tee -a "$LOG"
echo "[$(date '+%H:%M:%S')] 润色批次结束" | tee -a "$LOG"
