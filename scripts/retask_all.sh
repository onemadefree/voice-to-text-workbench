#!/bin/bash
# 全量重转写：等服务就绪（必须是新代码）→ 备份旧结果 → 重转写 → 退化段清理 → 导出
#
# 为什么等服务：MCode 的 bash 工具在 seatbelt 沙箱里跑，拿不到 Metal（GPU）；
# 由用户在终端里启动的服务进程持有 GPU，ASR 由它执行，本脚本只做编排与后处理。
set -u
cd "/Users/onemade/work/语音转文字" || exit 1
LOG="output/_retask.log"
exec >> "$LOG" 2>&1

echo "[$(date '+%F %T')] 等待服务 127.0.0.1:8787（需新代码：/api/llm/backends 可用）…"
for _ in $(seq 1 720); do
  code=$(curl -s -m 5 -o /dev/null -w "%{http_code}" http://127.0.0.1:8787/api/llm/backends 2>/dev/null)
  if [ "$code" = "200" ]; then
    echo "[$(date '+%F %T')] 服务就绪（运行中的是新代码）"
    break
  fi
  sleep 10
done
if [ "${code:-}" != "200" ]; then
  echo "[$(date '+%F %T')] 等待超时，服务没起来，退出"
  exit 1
fi

echo "[$(date '+%F %T')] 备份现有转写结果 → output/_backup_$(date '+%m%d_%H%M')"
mkdir -p "output/_backup_$(date '+%m%d_%H%M')"
for f in output/*/transcript.json; do
  [ -e "$f" ] || continue
  d=$(basename "$(dirname "$f")")
  cp "$f" "output/_backup_$(date '+%m%d_%H%M')/${d}.transcript.json" 2>/dev/null
done

echo "[$(date '+%F %T')] 开始全量重转写（force + Qwen3-ASR 0.6B）"
.venv/bin/python scripts/batch_service.py --force --engine qwen3 --no-polish --no-kb
echo "[$(date '+%F %T')] 批次结束（退出码 $?）"
