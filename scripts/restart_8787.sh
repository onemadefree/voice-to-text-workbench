#!/usr/bin/env bash
# 一次性维护脚本：重启 8787 服务 + 清理遗留进程/文件。
#
# 沙箱里 kill 不到这些进程（EPERM），必须由用户在终端手动执行。
# 用法（在仓库根目录）：
#     bash scripts/restart_8787.sh
#
# 做三件事：
#   1. 停掉 8791 上的旧测试服务（白占内存）
#   2. 停掉 8787 主服务并用最新代码重启（no_think / 标题去后缀 才生效）
#   3. 删掉最后一个 ASR 中间产物 audio_16k.wav（291MB）

set -uo pipefail
cd "$(dirname "$0")/.."

echo "→ 停掉 8791 旧测试服务"
lsof -nP -iTCP:8791 -sTCP:LISTEN 2>/dev/null | awk 'NR>1 {print $2}' | while read -r pid; do
  [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null && echo "  已 kill 8791 (pid $pid)"
done

echo "→ 停掉 8787 主服务"
lsof -nP -iTCP:8787 -sTCP:LISTEN 2>/dev/null | awk 'NR>1 {print $2}' | while read -r pid; do
  [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null && echo "  已 kill 8787 (pid $pid)"
done
sleep 2

echo "→ 删除残留 audio_16k.wav"
find output -name audio_16k.wav -print -delete 2>/dev/null

echo "→ 用最新代码重启 8787"
nohup ./run.sh > output/server.log 2>&1 &
sleep 6
curl -s -m 5 http://127.0.0.1:8787/api/health | head -c 200
echo
echo "→ 完成。日志：output/server.log"
