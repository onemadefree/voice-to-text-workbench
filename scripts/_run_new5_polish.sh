#!/bin/bash
# 新增 5 篇（结业讲话 / 张涛 / 王峰 / 高太山 / 程建刚）：
# ASR 队列把哪一篇转写完（transcript.json status=ready），就接着把那篇
# 润色 + 建知识库 + 导出，全程不用人工介入。
#
# 走在线 MiniMax API，不吃 GPU，所以和 6 篇的全量重刷可以并行；
# 也不会和 ASR 抢 GPU。
cd "/Users/onemade/work/语音转文字" || exit 1
export STT_LLM_API_TIMEOUT=600
LOG=output/_new5_polish.log

DOCS=(
  "人力资源组织部-结业讲话"
  "AI 赋能企业项目管理的应用与实践-张涛"
  "以场景为导向的 AI 项目端到端应用落地方法论-王峰"
  "数字经济时代的产业创新：机遇与挑战-高太山"
  "组织变革管理与变革领导力-程建刚"
)

echo "=== 新5篇接力开始 $(date '+%m-%d %H:%M:%S') ===" >> "$LOG"

for doc in "${DOCS[@]}"; do
  echo "--- 等待转写完成：$doc $(date '+%m-%d %H:%M:%S') ---" >> "$LOG"
  ok=0
  # 最多等 8 小时；转写失败（status=error）直接跳过，别白等
  for i in $(seq 1 480); do
    if [ -f "output/$doc/transcript.json" ]; then
      state=$(.venv/bin/python -c "
import json
try:
    d = json.load(open('output/$doc/transcript.json'))
    segs = d.get('segments') or []
    if d.get('status') == 'ready' and segs:
        print('ready')
    elif d.get('status') == 'error':
        print('error')
    else:
        print('pending')
except Exception:
    print('pending')
" 2>/dev/null)
      case "$state" in
        ready) ok=1; break;;
        error) echo "--- 转写失败，跳过：$doc ---" >> "$LOG"; break;;
      esac
    fi
    sleep 60
  done

  if [ "$ok" != "1" ]; then
    echo "--- 未就绪，跳过：$doc $(date '+%m-%d %H:%M:%S') ---" >> "$LOG"
    continue
  fi

  echo "=== 开始润色：$doc $(date '+%m-%d %H:%M:%S') ===" >> "$LOG"
  .venv/bin/python scripts/_local_repolish.py "$doc" >> "$LOG" 2>&1
  echo "--- 结束：$doc $(date '+%m-%d %H:%M:%S') ---" >> "$LOG"
done

echo "=== 新5篇接力完成 $(date '+%m-%d %H:%M:%S') ===" >> "$LOG"
