#!/bin/bash
# 停止 Bonsai 2 27B 服务。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
. "$SCRIPT_DIR/env.sh"

if [ ! -f "$BONSAI_PID_FILE" ]; then
  echo "没有 pid 文件，服务未由 start.sh 启动"
  exit 0
fi

pid="$(cat "$BONSAI_PID_FILE")"
if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  echo "已停止 pid $pid"
else
  echo "进程 $pid 已经不在"
fi
rm -f "$BONSAI_PID_FILE"
