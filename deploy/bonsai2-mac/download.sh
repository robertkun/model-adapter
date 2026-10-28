#!/bin/bash
# 下载本机测速用的 Q2_0 权重。优先 ModelScope，失败再走 hf-mirror。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
. "$SCRIPT_DIR/env.sh"

dest="$BONSAI_MODEL_DIR/$BONSAI_MODEL_FILE"
mkdir -p "$BONSAI_MODEL_DIR"

is_gguf() {
  [ -s "$1" ] || return 1
  [ "$(dd if="$1" bs=4 count=1 2>/dev/null || true)" = "GGUF" ]
}
file_bytes() { stat -f%z "$1" 2>/dev/null || stat -c%s "$1"; }

if is_gguf "$dest" && [ "$(file_bytes "$dest")" -ge "$BONSAI_MIN_BYTES" ]; then
  echo "已存在: $dest"
  exit 0
fi

for url in \
  "https://www.modelscope.cn/models/${BONSAI_REPO}/resolve/master/${BONSAI_MODEL_FILE}" \
  "https://modelscope.cn/models/${BONSAI_REPO}/resolve/master/${BONSAI_MODEL_FILE}" \
  "https://hf-mirror.com/${BONSAI_REPO}/resolve/main/${BONSAI_MODEL_FILE}"
do
  echo "下载 $url"
  curl -4 -L --retry 3 --retry-delay 2 -C - -o "$dest.partial" "$url" || true
  if is_gguf "$dest.partial" && [ "$(file_bytes "$dest.partial")" -ge "$BONSAI_MIN_BYTES" ]; then
    mv -f "$dest.partial" "$dest"
    echo "完成: $dest ($(file_bytes "$dest") bytes)"
    exit 0
  fi
  echo "这个地址没有得到完整的 GGUF"
done
echo "下载失败" >&2
exit 1
