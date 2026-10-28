#!/bin/bash
# 本机命令行对话。权重在 Metal 上，和服务器的 CPU 脚本不是同一套。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
. "$SCRIPT_DIR/env.sh"

model="$BONSAI_MODEL_DIR/$BONSAI_MODEL_FILE"
[ -s "$model" ] || { echo "先执行 ./download.sh" >&2; exit 1; }

exec "$LLAMA_CLI" \
  -m "$model" \
  -ngl "$BONSAI_NGL" \
  -t "$BONSAI_THREADS" \
  -c "$BONSAI_CTX" \
  -fa "$BONSAI_FA" \
  --temp "$BONSAI_TEMP" \
  --top-p "$BONSAI_TOP_P" \
  --top-k "$BONSAI_TOP_K" \
  --min-p "$BONSAI_MIN_P" \
  --reasoning-effort "$BONSAI_REASONING" \
  "$@"
