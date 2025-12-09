#!/usr/bin/env bash
set -euo pipefail

# === 可配置变量 ===
OFFLINE_ZIP_URL="${OFFLINE_ZIP_URL:-}"   # 离线依赖包直链
MODEL_ZIP_URL="${MODEL_ZIP_URL:-}"       # 模型包直链
PYTHON_VERSION="${PYTHON_VERSION:-3.10}" # 可按需调整
REPO_URL="https://github.com/BruceLee569/CosyVoice.git"
REPO_DIR="$HOME/CosyVoice"
PKG_DIR="$HOME/offline_packages"
MODEL_ZIP="pretrained_models.zip"

# 函数：打印错误并退出
abort() { echo "[ERR] $*" >&2; exit 1; }

# 函数：检查必要命令是否存在
need_cmd() { command -v "$1" >/dev/null 2>&1 || abort "缺少命令: $1"; }

# 函数：准备网络加速（如 /etc/network_turbo 可用则引入）
prepare_network() {
  if [[ -f /etc/network_turbo ]]; then
    # shellcheck disable=SC1091
    source /etc/network_turbo
  fi
}

# 函数：安装 uv（若未安装）
install_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
}

# 函数：克隆仓库并初始化子模块
clone_repo() {
  if [[ ! -d "$REPO_DIR" ]]; then
    git clone --recursive "$REPO_URL" "$REPO_DIR"
  else
    (cd "$REPO_DIR" && git submodule update --init --recursive)
  fi
}

# 函数：创建并激活虚拟环境
create_venv() {
  cd "$REPO_DIR"
  uv venv .venv --python "$PYTHON_VERSION"
  # shellcheck disable=SC1091
  source .venv/bin/activate
}

# 函数：安装系统依赖
install_sys_deps() {
  sudo apt-get update
  sudo apt-get install -y \
    git curl wget aria2 unzip git-lfs build-essential ffmpeg sox libsox-dev
  git lfs install
}

# 函数：下载并解压离线依赖包
download_offline_pkgs() {
  [[ -n "$OFFLINE_ZIP_URL" ]] || abort "未设置 OFFLINE_ZIP_URL"
  aria2c -c -x 16 -s 16 -o "offline_packages.zip" "$OFFLINE_ZIP_URL"
  rm -rf "$PKG_DIR"
  unzip offline_packages.zip -d "$PKG_DIR"
  rm -f offline_packages.zip
  rm -f "$PKG_DIR"/*cp38*.whl
}

# 函数：安装离线依赖（先 wheel 再源码包）
install_offline_pkgs() {
  cd "$REPO_DIR"
  uv pip install --no-index --find-links "$PKG_DIR" \*.whl
  uv pip install --no-index --no-build-isolation --no-deps \
    --find-links "$PKG_DIR" "$PKG_DIR"/*.tar.gz "$PKG_DIR"/*.zip
}

# 函数：下载并解压模型包
download_models() {
  [[ -n "$MODEL_ZIP_URL" ]] || abort "未设置 MODEL_ZIP_URL"
  cd "$REPO_DIR"
  aria2c -c -x 16 -s 16 -o "$MODEL_ZIP" "$MODEL_ZIP_URL"
  unzip -o "$MODEL_ZIP" -d pretrained_models
  rm -f "$MODEL_ZIP"
}

# 函数：安装 TTSFRD 相关依赖
install_ttsfrd() {
  cd "$REPO_DIR"
  uv pip install --no-index --find-links pretrained_models/CosyVoice-ttsfrd/ \
    pretrained_models/CosyVoice-ttsfrd/ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl \
    pretrained_models/CosyVoice-ttsfrd/ttsfrd_dependency-0.1-py3-none-any.whl
}

# 函数：启动服务
start_server() {
  cd "$REPO_DIR"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python server.py
}

# === 主流程 ===
prepare_network
need_cmd curl
need_cmd git
install_uv
clone_repo
create_venv
install_sys_deps
need_cmd aria2c
download_offline_pkgs
install_offline_pkgs
download_models
install_ttsfrd
start_server
