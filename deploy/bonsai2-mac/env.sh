# 本机调优配置。和 deploy/bonsai2 的服务器脚本分开。
# 机器：Apple M4 Pro，14 核 CPU（10 性能 + 4 能效），20 核 GPU，24GB 统一内存，Metal。
# 磁盘紧张，只下一种权重。本机 llama.cpp 是 build 11669，能跑 Q2_0，不能跑服务器上的 PQ2_0。
BONSAI_ROOT="$(cd "$(dirname "$0")" && pwd)"
BONSAI_MODEL_DIR="$BONSAI_ROOT/models"
BONSAI_REPO="${BONSAI_REPO:-prism-ml/Ternary-Bonsai-2-27B-gguf-dev}"
BONSAI_MODEL_FILE="${BONSAI_MODEL_FILE:-Ternary-Bonsai-2-27B-Q2_0-prism-fork-required.gguf}"
BONSAI_MIN_BYTES="${BONSAI_MIN_BYTES:-7000000000}"
# 全部层放到 Metal。OpenClaw 的工具提示大约 1.7 万 token，上下文用 32768。
# 2026-10-27 基线（llama-bench，pp512 / tg128）：读入 121.4 t/s，生成 21.5 t/s。
# 对照服务器 Xeon E5-2620 v4、PQ2_0、8 线程：读入约 5 t/s，生成约 2.4 t/s。
BONSAI_NGL="${BONSAI_NGL:-99}"
BONSAI_THREADS="${BONSAI_THREADS:-10}"
BONSAI_CTX="${BONSAI_CTX:-32768}"
BONSAI_FA="${BONSAI_FA:-on}"
BONSAI_TEMP="${BONSAI_TEMP:-1.0}"
BONSAI_TOP_P="${BONSAI_TOP_P:-0.95}"
BONSAI_TOP_K="${BONSAI_TOP_K:-20}"
BONSAI_MIN_P="${BONSAI_MIN_P:-0.05}"
BONSAI_REASONING="${BONSAI_REASONING:-medium}"
LLAMA_CLI="${LLAMA_CLI:-llama-cli}"
LLAMA_BENCH="${LLAMA_BENCH:-llama-bench}"
