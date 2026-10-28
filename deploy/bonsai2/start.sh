#!/bin/bash
# 启动 Bonsai 2 27B 的 OpenAI 兼容接口。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
. "$SCRIPT_DIR/env.sh"

BIN="$BONSAI_BIN_DIR/llama-server"
MODEL="$BONSAI_MODEL_DIR/$BONSAI_MODEL_FILE"
MMPROJ="$BONSAI_MODEL_DIR/$BONSAI_MMPROJ_FILE"

[ -x "$BIN" ] || { echo "还没有推理程序，先执行 ./install.sh" >&2; exit 1; }
[ -s "$MODEL" ] || { echo "还没有模型文件，先执行 ./install.sh" >&2; exit 1; }

if [ -f "$BONSAI_PID_FILE" ] && kill -0 "$(cat "$BONSAI_PID_FILE")" 2>/dev/null; then
  echo "已经在运行，pid $(cat "$BONSAI_PID_FILE")"
  exit 0
fi

args=(
  -m "$MODEL"
  --host "$BONSAI_HOST"
  --port "$BONSAI_PORT"
  -ngl 0
  -t "$BONSAI_THREADS"
  -c "$BONSAI_CTX"
  -np 1
  -fa on
  --temp 1.0
  --top-p 0.95
  --top-k 20
  --min-p 0.05
  --jinja
  --chat-template-kwargs '{"reasoning_effort":"medium"}'
)
if [ -s "$MMPROJ" ]; then
  args+=(--mmproj "$MMPROJ")
fi

export OMP_NUM_THREADS="$BONSAI_THREADS"
export MKL_NUM_THREADS="$BONSAI_THREADS"
cd "$BONSAI_ROOT"
cmd=("$BIN")
if command -v numactl >/dev/null 2>&1 && [ -n "${BONSAI_NUMA}" ]; then
  cmd=(numactl "--cpunodebind=${BONSAI_NUMA}" "--membind=${BONSAI_NUMA}" "$BIN")
else
  echo "未找到 numactl，未绑定 NUMA。可执行: yum install -y numactl" >&2
fi
if [ "${1:-}" = "--foreground" ]; then
  exec "${cmd[@]}" "${args[@]}"
fi
nohup "${cmd[@]}" "${args[@]}" >>"$BONSAI_LOG" 2>&1 &
echo $! >"$BONSAI_PID_FILE"
echo "已启动 pid $(cat "$BONSAI_PID_FILE")"
echo "日志: $BONSAI_LOG"
echo "页面: http://${BONSAI_HOST}:${BONSAI_PORT}"
echo "线程 ${BONSAI_THREADS}，NUMA node ${BONSAI_NUMA}。CPU 上第一次回答会比较慢。思考强度是 medium。"
