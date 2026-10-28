#!/bin/bash
# 在 Linux CPU 服务器上安装 Ternary Bonsai 2 27B。
# 用法：把 deploy/bonsai2 整个目录拷到服务器后执行 ./install.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
. "$SCRIPT_DIR/env.sh"

info() { printf '[OK] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*" >&2; }
die() { printf '[ERR] %s\n' "$*" >&2; exit 1; }

is_gguf() {
  [ -s "$1" ] || return 1
  [ "$(dd if="$1" bs=4 count=1 2>/dev/null || true)" = "GGUF" ]
}

file_bytes() {
  stat -c%s "$1" 2>/dev/null || stat -f%z "$1"
}

# ModelScope 经常在文件已经下完后仍让 curl 返回非 0，不能只看退出码。
accept_model() {
  local file="$1" min_bytes="$2"
  is_gguf "$file" || return 1
  [ "$(file_bytes "$file")" -ge "$min_bytes" ]
}

download() {
  url="$1"
  dest="$2"
  min_bytes="${3:-}"
  mkdir -p "$(dirname "$dest")"
  if [ -n "$min_bytes" ] && accept_model "$dest" "$min_bytes"; then
    return 0
  fi
  if command -v curl >/dev/null 2>&1; then
    # 不用 -f。镜像会先回一段很短的跳转内容，-f 会把已经下完的 7GB 判成失败。
    curl -4 -L --retry 3 --retry-delay 2 -C - -o "$dest" "$url" || true
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "$dest" "$url" || true
  else
    die "需要 curl 或 wget"
  fi
  if [ -n "$min_bytes" ]; then
    accept_model "$dest" "$min_bytes"
    return
  fi
  [ -s "$dest" ]
}

have_cxx17() {
  "${CXX:-g++}" -std=c++17 -x c++ -E - </dev/null >/dev/null 2>&1
}

yum_install() {
  yum --disableplugin=fastestmirror,subscription-manager install -y "$@"
}

# CentOS 7 已停更，mirror.centos.org 没有 baseurl。改到国内存档源。
fix_centos7_repos() {
  [ -f /etc/centos-release ] || return 0
  grep -q "release 7" /etc/centos-release || return 0
  local mirror
  mirror="${CENTOS_MIRROR:-https://mirrors.aliyun.com/centos-vault/7.9.2009}"
  mkdir -p /etc/yum.repos.d/backup
  cp -an /etc/yum.repos.d/CentOS-*.repo /etc/yum.repos.d/backup/ 2>/dev/null || true
  for repo in /etc/yum.repos.d/CentOS-Base.repo /etc/yum.repos.d/CentOS-SCLo-scl.repo /etc/yum.repos.d/CentOS-SCLo-scl-rh.repo; do
    [ -f "$repo" ] && sed -i 's/^enabled=1/enabled=0/' "$repo"
  done
  cat >/etc/yum.repos.d/CentOS-Vault-bonsai.repo <<EOF
[base]
name=CentOS-7.9 - Base
baseurl=${mirror}/os/\$basearch/
gpgcheck=0
enabled=1

[updates]
name=CentOS-7.9 - Updates
baseurl=${mirror}/updates/\$basearch/
gpgcheck=0
enabled=1

[extras]
name=CentOS-7.9 - Extras
baseurl=${mirror}/extras/\$basearch/
gpgcheck=0
enabled=1

[centos-sclo-rh]
name=CentOS-7.9 - SCLo rh
baseurl=${mirror}/sclo/\$basearch/rh/
gpgcheck=0
enabled=1

[centos-sclo-sclo]
name=CentOS-7.9 - SCLo sclo
baseurl=${mirror}/sclo/\$basearch/sclo/
gpgcheck=0
enabled=1
EOF
  info "已把 CentOS 7 源改到 $mirror"
}

enable_newer_gcc() {
  if have_cxx17; then
    return 0
  fi
  for enable in /opt/rh/devtoolset-11/enable /opt/rh/devtoolset-10/enable /opt/rh/devtoolset-9/enable /opt/rh/gcc-toolset-11/enable; do
    if [ -f "$enable" ]; then
      # shellcheck disable=SC1090
      . "$enable"
      have_cxx17 && return 0
    fi
  done
  if command -v yum >/dev/null 2>&1; then
    warn "系统 g++ 太旧，尝试安装 devtoolset-11"
    fix_centos7_repos || true
    yum_install centos-release-scl || true
    fix_centos7_repos || true
    yum_install devtoolset-11-gcc devtoolset-11-gcc-c++ devtoolset-11-binutils git || true
    if [ -f /opt/rh/devtoolset-11/enable ]; then
      # shellcheck disable=SC1091
      . /opt/rh/devtoolset-11/enable
      have_cxx17 && return 0
    fi
  fi
  return 1
}

binary_runs() {
  [ -x "$BONSAI_BIN_DIR/llama-server" ] || return 1
  "$BONSAI_BIN_DIR/llama-server" --version >/dev/null 2>&1
}

install_prebuilt() {
  local asset url tmp
  if binary_runs; then
    info "推理程序已可用，跳过下载"
    return 0
  fi
  asset="llama-${BONSAI_RELEASE}-bin-ubuntu-x64.tar.gz"
  url="https://github.com/PrismML-Eng/llama.cpp/releases/download/${BONSAI_RELEASE}/${asset}"
  tmp="$(mktemp)"
  info "下载 CPU 推理程序: $asset"
  if ! download "$url" "$tmp"; then
    rm -f "$tmp"
    warn "预编译包下载失败"
    return 1
  fi
  mkdir -p "$BONSAI_BIN_DIR"
  tar -xzf "$tmp" -C "$BONSAI_BIN_DIR" --strip-components=1 2>/dev/null \
    || tar -xzf "$tmp" -C "$BONSAI_BIN_DIR"
  rm -f "$tmp"
  chmod +x "$BONSAI_BIN_DIR"/llama-* 2>/dev/null || true
  if binary_runs; then
    info "预编译包可以运行"
    return 0
  fi
  warn "预编译包无法运行。这台系统的 glibc 通常低于 Ubuntu 构建版本。"
  ldd "$BONSAI_BIN_DIR/llama-server" 2>&1 | grep "not found" || true
  return 1
}

build_from_source() {
  local src cmake_bin jobs
  enable_newer_gcc || die "没有支持 C++17 的编译器。CentOS 7 可先安装 devtoolset-11 后再执行本脚本。"
  info "使用编译器: $($CXX --version 2>/dev/null | head -1 || g++ --version | head -1)"
  if [ -x /opt/miniconda3/bin/cmake ]; then
    cmake_bin=/opt/miniconda3/bin/cmake
  elif command -v cmake >/dev/null 2>&1; then
    cmake_bin="$(command -v cmake)"
  elif [ -x /opt/miniconda3/bin/python ]; then
    /opt/miniconda3/bin/python -m pip install -U cmake ninja
    cmake_bin=/opt/miniconda3/bin/cmake
  else
    die "没有 cmake。请先安装 cmake 3.16 以上。"
  fi
  command -v git >/dev/null 2>&1 || die "编译源码需要 git。可执行: yum --disableplugin=subscription-manager,fastestmirror install -y git"
  src="$BONSAI_ROOT/src/llama.cpp"
  mkdir -p "$BONSAI_ROOT/src"
  if [ ! -d "$src/.git" ]; then
    info "拉取 PrismML llama.cpp $BONSAI_RELEASE"
    git clone --depth 1 --branch "$BONSAI_RELEASE" https://github.com/PrismML-Eng/llama.cpp.git "$src"
  fi
  jobs="${BONSAI_THREADS}"
  info "编译 llama-server，大约需要十几分钟"
  "$cmake_bin" -S "$src" -B "$src/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=ON \
    -DGGML_CUDA=OFF \
    -DLLAMA_CURL=OFF
  "$cmake_bin" --build "$src/build" -j "$jobs" --target llama-server llama-cli
  mkdir -p "$BONSAI_BIN_DIR"
  cp -f "$src/build/bin/llama-server" "$BONSAI_BIN_DIR/llama-server"
  cp -f "$src/build/bin/llama-cli" "$BONSAI_BIN_DIR/llama-cli"
  chmod +x "$BONSAI_BIN_DIR/llama-server" "$BONSAI_BIN_DIR/llama-cli"
  binary_runs || die "源码编译完成，但 llama-server 仍无法启动"
  info "源码编译完成"
}

fetch_model() {
  local name dest min_bytes url
  name="$1"
  min_bytes="$2"
  dest="$BONSAI_MODEL_DIR/$name"
  mkdir -p "$BONSAI_MODEL_DIR"
  if accept_model "$dest" "$min_bytes"; then
    info "已存在: $dest"
    return 0
  fi
  if accept_model "$dest.partial" "$min_bytes"; then
    mv -f "$dest.partial" "$dest"
    info "已有完整文件: $dest"
    return 0
  fi
  # 国内优先走 ModelScope。Hugging Face 在这台机器上会连到不可达的 IPv6。
  for url in \
    "https://modelscope.cn/models/${BONSAI_REPO}/resolve/master/${name}" \
    "https://www.modelscope.cn/models/${BONSAI_REPO}/resolve/master/${name}" \
    "https://hf-mirror.com/${BONSAI_REPO}/resolve/main/${name}"
  do
    info "下载 $name"
    echo "  $url"
    if download "$url" "$dest.partial" "$min_bytes"; then
      mv -f "$dest.partial" "$dest"
      info "完成: $dest ($(file_bytes "$dest") bytes)"
      return 0
    fi
    warn "这个地址没有得到完整的 GGUF，换下一个"
  done
  return 1
}

free_kb="$(df -Pk "$SCRIPT_DIR" | awk 'NR==2 {print $4}')"
if [ "${free_kb:-0}" -lt $((12 * 1024 * 1024)) ]; then
  die "磁盘剩余不足 12GB。PQ2_0 权重大约 7.2GB，还要放推理程序。"
fi

mkdir -p "$BONSAI_BIN_DIR" "$BONSAI_MODEL_DIR"
if ! install_prebuilt; then
  build_from_source
fi

fetch_model "$BONSAI_MODEL_FILE" 6000000000 || die "模型下载失败: $BONSAI_MODEL_FILE"
if ! fetch_model "$BONSAI_MMPROJ_FILE" 100000000; then
  warn "视觉投影没有下下来。文本对话仍可用，图片输入不可用。"
fi

"$BONSAI_BIN_DIR/llama-server" --version || true
echo
info "安装完成"
echo "启动: $SCRIPT_DIR/start.sh"
echo "停止: $SCRIPT_DIR/stop.sh"
echo "页面: http://服务器IP:${BONSAI_PORT}"
echo "接口: http://服务器IP:${BONSAI_PORT}/v1/chat/completions"
