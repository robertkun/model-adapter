#!/bin/bash
# 在终端里和 Bonsai 2 27B 对话。输入一行回车即发送，Ctrl+C 退出。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
. "$SCRIPT_DIR/env.sh"

BIN="$BONSAI_BIN_DIR/llama-cli"
MODEL="$BONSAI_MODEL_DIR/$BONSAI_MODEL_FILE"

[ -x "$BIN" ] || { echo "还没有 llama-cli。在源码目录执行: cmake --build src/llama.cpp/build -j ${BONSAI_THREADS} --target llama-cli && cp src/llama.cpp/build/bin/llama-cli bin/" >&2; exit 1; }
[ -s "$MODEL" ] || { echo "还没有模型文件" >&2; exit 1; }

if [ -f "$BONSAI_PID_FILE" ] && kill -0 "$(cat "$BONSAI_PID_FILE")" 2>/dev/null; then
  echo "网页服务还在跑，会和命令行抢 CPU。先执行 ./stop.sh" >&2
  exit 1
fi

export OMP_NUM_THREADS="$BONSAI_THREADS"
export MKL_NUM_THREADS="$BONSAI_THREADS"
cmd=("$BIN")
if command -v numactl >/dev/null 2>&1 && [ -n "${BONSAI_NUMA}" ]; then
  cmd=(numactl "--cpunodebind=${BONSAI_NUMA}" "--membind=${BONSAI_NUMA}" "$BIN")
fi

exec "${cmd[@]}" \
  -m "$MODEL" \
  -ngl 0 \
  -t "$BONSAI_THREADS" \
  -c "$BONSAI_CTX" \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.05 \
  --reasoning-effort medium \
  "$@"
