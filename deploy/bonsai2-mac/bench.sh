#!/bin/bash
# 用 llama-bench 测本机 Metal 速度。结果写到 bench-result.txt。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
. "$SCRIPT_DIR/env.sh"

model="$BONSAI_MODEL_DIR/$BONSAI_MODEL_FILE"
[ -s "$model" ] || { echo "先执行 ./download.sh" >&2; exit 1; }

out="$BONSAI_ROOT/bench-result.txt"
{
  echo "host: Apple M4 Pro, 24GB, Metal"
  echo "model: $BONSAI_MODEL_FILE"
  echo "ngl=$BONSAI_NGL threads=$BONSAI_THREADS ctx=$BONSAI_CTX fa=$BONSAI_FA"
  echo "time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo
  "$LLAMA_BENCH" \
    -m "$model" \
    -ngl "$BONSAI_NGL" \
    -t "$BONSAI_THREADS" \
    -fa "$BONSAI_FA" \
    -p 512 \
    -n 128
} | tee "$out"
echo "结果已写入 $out"
