# Bonsai 2 27B，CPU 部署。供 install.sh / start.sh / stop.sh 共用。
BONSAI_ROOT="$(cd "$(dirname "$0")" && pwd)"
BONSAI_HOST="${BONSAI_HOST:-0.0.0.0}"
BONSAI_PORT="${BONSAI_PORT:-8088}"
# Xeon E5-2620 v4：2 路 × 8 核 × 2 线程，有 AVX2，没有 AVX-512。
# PQ2_0 权重大约 7GB，放得进单个 NUMA 节点。绑在 node0 的 8 个物理核上，
# 避免跨路读内存。node0 的 CPU 是 0-7,16-23。
BONSAI_THREADS="${BONSAI_THREADS:-8}"
BONSAI_NUMA="${BONSAI_NUMA:-0}"
# 默认会先思考再回答。32768 给思考留出空间；机器有 125GB 内存，可以再调大，上限 262144。
BONSAI_CTX="${BONSAI_CTX:-32768}"
BONSAI_RELEASE="${BONSAI_RELEASE:-prism-b10826-c9c34bc}"
BONSAI_REPO="${BONSAI_REPO:-prism-ml/Ternary-Bonsai-2-27B-gguf}"
BONSAI_MODEL_FILE="${BONSAI_MODEL_FILE:-Ternary-Bonsai-2-27B-PQ2_0.gguf}"
BONSAI_MMPROJ_FILE="${BONSAI_MMPROJ_FILE:-Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf}"
BONSAI_BIN_DIR="$BONSAI_ROOT/bin"
BONSAI_MODEL_DIR="$BONSAI_ROOT/models"
BONSAI_PID_FILE="$BONSAI_ROOT/bonsai2.pid"
BONSAI_LOG="$BONSAI_ROOT/bonsai2.log"
