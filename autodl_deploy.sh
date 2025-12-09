#!/usr/bin/env bash
set -euo pipefail

# === 可配置变量 ===
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
REPO_URL="https://github.com/BruceLee569/CosyVoice.git"
REPO_DIR="$HOME/CosyVoice"
PKG_DIR="$HOME/offline_packages"
# 预下载的资源路径（Autodl默认数据盘）
AUTODL_TMP="/root/autodl-tmp"
OFFLINE_ZIP="$AUTODL_TMP/offline_packages.zip"
MODEL_ZIP="$AUTODL_TMP/pretrained_models.zip"

# 函数：打印错误并退出
abort() { echo "[ERR] $*" >&2; exit 1; }

# 函数：打印信息
info() { echo "[INFO] $*"; }

# 函数：打印成功信息
success() { echo "[SUCCESS] $*"; }

# 函数：检查必要命令是否存在
need_cmd() { command -v "$1" >/dev/null 2>&1 || abort "缺少命令: $1"; }

# 函数：准备网络加速（如 /etc/network_turbo 可用则引入）
prepare_network() {
  if [[ -f /etc/network_turbo ]]; then
    # shellcheck disable=SC1091
    source /etc/network_turbo
    success "网络加速已启用"
  fi
}

# 函数：检查预下载文件是否存在
check_predownloaded_files() {
  info "检查预下载文件..."
  [[ -f "$OFFLINE_ZIP" ]] || abort "离线依赖包不存在: $OFFLINE_ZIP\n请先将 offline_packages.zip 上传到 $AUTODL_TMP"
  [[ -f "$MODEL_ZIP" ]] || abort "模型包不存在: $MODEL_ZIP\n请先将 pretrained_models.zip 上传到 $AUTODL_TMP"
  success "预下载文件检查通过"
}

# 函数：安装 uv（若未安装）
install_uv() {
  if command -v uv >/dev/null 2>&1; then
    info "uv 已安装，跳过"
    export PATH="$HOME/.local/bin:$PATH"
    return 0
  fi
  
  info "安装 uv 包管理器..."
  curl -LsSf https://astral.sh/uv/install.sh | sh || abort "uv 安装失败"
  export PATH="$HOME/.local/bin:$PATH"
  success "uv 安装完成: $(uv --version)"
}

# 函数：克隆仓库并初始化子模块
clone_repo() {
  if [[ -d "$REPO_DIR/.git" ]]; then
    info "仓库已存在，更新子模块..."
    cd "$REPO_DIR"
    git submodule update --init --recursive || abort "子模块更新失败"
    return 0
  fi
  
  info "克隆 CosyVoice 仓库..."
  git clone --recursive "$REPO_URL" "$REPO_DIR" || abort "仓库克隆失败"
  success "仓库克隆完成"
}

# 函数：创建并激活虚拟环境
create_venv() {
  cd "$REPO_DIR"
  if [[ -f ".venv/bin/activate" ]]; then
    info "虚拟环境已存在，跳过创建"
  else
    info "创建 Python 虚拟环境..."
    uv venv .venv --python "$PYTHON_VERSION" || abort "虚拟环境创建失败"
    success "虚拟环境创建完成"
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  info "Python 版本: $(python --version)"
}

# 函数：安装系统依赖
install_sys_deps() {
  # 检查关键命令是否已安装
  if command -v ffmpeg >/dev/null 2>&1 && command -v sox >/dev/null 2>&1; then
    info "系统依赖已安装，跳过"
    return 0
  fi
  
  info "安装系统依赖..."
  apt-get update || abort "apt-get update 失败"
  apt-get install -y \
    git curl wget unzip git-lfs build-essential ffmpeg sox libsox-dev || abort "系统依赖安装失败"
  git lfs install || abort "git-lfs 初始化失败"
  success "系统依赖安装完成"
}

# 函数：解压离线依赖包
extract_offline_pkgs() {
  if [[ -d "$PKG_DIR" ]] && [[ -n "$(ls -A "$PKG_DIR"/*.whl 2>/dev/null)" ]]; then
    info "离线依赖包已解压，跳过"
    return 0
  fi
  
  info "解压离线依赖包..."
  rm -rf "$PKG_DIR"
  unzip -q "$OFFLINE_ZIP" -d "$HOME" || abort "离线依赖包解压失败"
  
  # 检查是否多了一层目录
  if [[ -d "$PKG_DIR/offline_packages" ]]; then
    mv "$PKG_DIR/offline_packages"/* "$PKG_DIR/" 2>/dev/null || true
    rmdir "$PKG_DIR/offline_packages"
  fi
  
  # 清理不兼容的包
  rm -f "$PKG_DIR"/*cp38*.whl
  
  success "离线依赖包已准备就绪: $PKG_DIR"
}

# 函数：安装离线依赖
install_offline_pkgs() {
  cd "$REPO_DIR"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  
  # 检查是否已安装关键包
  if python -c "import torch" 2>/dev/null; then
    info "离线依赖已安装，跳过"
    return 0
  fi
  
  info "安装 wheel 包..."
  # 使用 find-links 让 uv 自动发现并安装所有 wheel 包
  if ls "$PKG_DIR"/*.whl >/dev/null 2>&1; then
    for whl in "$PKG_DIR"/*.whl; do
      uv pip install --no-index --find-links="$PKG_DIR" "$whl" || true
    done
  fi
  
  # 安装源码包（如果存在）
  if ls "$PKG_DIR"/*.tar.gz >/dev/null 2>&1 || ls "$PKG_DIR"/*.zip >/dev/null 2>&1; then
    info "安装源码包..."
    uv pip install --no-index --no-build-isolation --no-deps \
      --find-links="$PKG_DIR" "$PKG_DIR"/*.tar.gz "$PKG_DIR"/*.zip 2>/dev/null || true
  fi
  # 清理缓存，避免系统盘爆满
  uv cache clean
  success "离线依赖安装完成"
}

# 函数：解压模型包
extract_models() {
  cd "$REPO_DIR"
  if [[ -d "pretrained_models/CosyVoice-ttsfrd" ]]; then
    info "模型包已解压，跳过"
    return 0
  fi
  
  info "解压模型包（较大，请稍候）..."
  unzip -q "$MODEL_ZIP" -d . || abort "模型包解压失败"
  
  [[ -d "pretrained_models/CosyVoice-ttsfrd" ]] || abort "模型目录验证失败"
  success "模型包已准备就绪"
}

# 函数：安装 TTSFRD
install_ttsfrd() {
  cd "$REPO_DIR"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  
  # 检查是否已安装
  if python -c "import ttsfrd" 2>/dev/null; then
    info "TTSFRD 已安装，跳过"
    return 0
  fi
  
  # 解压 resource.zip
  if [[ -f "pretrained_models/CosyVoice-ttsfrd/resource.zip" ]]; then
    info "解压 TTSFRD resource.zip..."
    cd pretrained_models/CosyVoice-ttsfrd/
    unzip -q resource.zip -d . || echo "[WARN] resource.zip 解压失败，继续"
    cd "$REPO_DIR"
  fi
  
  info "安装 TTSFRD..."
  uv pip install --no-index --find-links pretrained_models/CosyVoice-ttsfrd/ \
    pretrained_models/CosyVoice-ttsfrd/ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl \
    pretrained_models/CosyVoice-ttsfrd/ttsfrd_dependency-0.1-py3-none-any.whl || abort "TTSFRD 安装失败"
  
  success "TTSFRD 安装完成"
}

# 函数：启动服务
start_server() {
  cd "$REPO_DIR"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  
  [[ -f "server.py" ]] || abort "server.py 不存在"
  
  success "=== 部署完成！正在启动服务 ==="
  info "按 Ctrl+C 停止服务"
  .venv/bin/python server.py
}

# === 主流程 ===
main() {
  info "=== CosyVoice Autodl 自动部署脚本 ==="
  info "目标目录: $REPO_DIR"
  info "数据盘: $AUTODL_TMP"
  echo ""
  
  # 检查必要命令
  need_cmd curl
  need_cmd git
  need_cmd unzip
  
  # 执行部署步骤
  prepare_network
  check_predownloaded_files
  install_uv
  clone_repo
  create_venv
  install_sys_deps
  extract_offline_pkgs
  install_offline_pkgs
  extract_models
  install_ttsfrd
  
  # 启动服务
  start_server
}

# 运行主流程
main
