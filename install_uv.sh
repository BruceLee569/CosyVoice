#!/bin/bash
# CosyVoice 一键安装脚本（使用 uv）
# 适用于 Ubuntu Server

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

print_info "开始安装 CosyVoice..."

# 1. 检查并安装系统依赖
print_info "步骤 1/6: 检查系统依赖..."
if ! command_exists git; then
    print_warning "Git 未安装，正在安装..."
    sudo apt-get update
    sudo apt-get install -y git
fi

if ! command_exists sox; then
    print_warning "Sox 未安装，正在安装系统依赖..."
    sudo apt-get update
    sudo apt-get install -y git git-lfs sox libsox-dev build-essential curl wget ffmpeg unzip
    git lfs install
fi

# 2. 初始化 Git 子模块（Matcha-TTS）
print_info "步骤 2/6: 初始化 Git 子模块..."
if [ ! -f "third_party/Matcha-TTS/README.md" ]; then
    print_warning "Matcha-TTS 子模块未初始化，正在初始化..."
    git submodule init
    git submodule update
    
    if [ ! -f "third_party/Matcha-TTS/README.md" ]; then
        print_error "Matcha-TTS 子模块初始化失败"
        exit 1
    fi
    print_info "Matcha-TTS 子模块初始化成功！"
else
    print_info "Matcha-TTS 子模块已存在"
fi

# 3. 检查并安装 uv
print_info "步骤 3/6: 检查 uv 包管理器..."
if ! command_exists uv; then
    print_warning "uv 未安装，正在安装..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # 将 uv 添加到当前 shell 的 PATH
    export PATH="$HOME/.local/bin:$PATH"
    
    # 检查安装是否成功
    if ! command_exists uv; then
        print_error "uv 安装失败，请手动安装后重试"
        exit 1
    fi
    
    print_info "uv 安装成功！版本: $(uv --version)"
else
    print_info "uv 已安装，版本: $(uv --version)"
fi

# 4. 配置国内镜像源（可选）
print_info "步骤 4/6: 配置 pip 镜像源（可选）..."
if [ ! -f ~/.config/pip/pip.conf ]; then
    mkdir -p ~/.config/pip
    cat > ~/.config/pip/pip.conf << EOF
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
extra-index-url = https://download.pytorch.org/whl/cu121
EOF
    print_info "pip 镜像源配置完成（使用阿里云镜像）"
else
    print_info "pip 镜像源已配置"
fi

# 5. 使用 uv 安装 Python 依赖
print_info "步骤 5/6: 安装 Python 依赖（这可能需要几分钟）..."
if [ ! -d ".venv" ]; then
    # 创建不含 tensorrt 的临时 requirements 文件（tensorrt 在某些系统上可能安装失败）
    grep -v "tensorrt" requirements.txt > .requirements_temp.txt
    
    # 创建虚拟环境
    uv venv
    
    # 激活环境并安装依赖
    source .venv/bin/activate
    uv pip install --index-strategy unsafe-best-match -r .requirements_temp.txt
    
    # 清理临时文件
    rm -f .requirements_temp.txt
    
    print_info "Python 环境创建成功！"
    print_warning "注意: TensorRT 未安装（可选依赖）。如需使用 TensorRT 加速，请手动安装。"
else
    print_warning "虚拟环境已存在，跳过创建"
    source .venv/bin/activate
fi

# 6. 验证安装
print_info "步骤 6/6: 验证安装..."
source .venv/bin/activate

# 设置 PYTHONPATH（包含 Matcha-TTS）
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/third_party/Matcha-TTS"

# 检查 PyTorch
if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
    print_info "PyTorch 版本: $TORCH_VERSION"
    print_info "CUDA 可用: $CUDA_AVAILABLE"
else
    print_error "PyTorch 安装失败"
    exit 1
fi

# 检查其他关键依赖
if python -c "import torchaudio, transformers, modelscope" 2>/dev/null; then
    print_info "所有关键依赖安装成功！"
else
    print_warning "部分依赖可能未正确安装，请检查"
fi

# 检查 Matcha-TTS
if python -c "import sys; sys.path.append('third_party/Matcha-TTS'); from matcha.models.components.flow_matching import BASECFM" 2>/dev/null; then
    print_info "Matcha-TTS 模块加载成功！"
else
    print_warning "Matcha-TTS 模块加载失败，但这不影响基本使用"
fi

deactivate

# 创建启动脚本
print_info "创建启动脚本..."
cat > start_webui.sh << 'EOF'
#!/bin/bash
# CosyVoice WebUI 启动脚本

# 设置工作目录
cd "$(dirname "$0")"

# 激活虚拟环境
source .venv/bin/activate

# 设置 PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/third_party/Matcha-TTS"

# 检查模型目录
if [ ! -d "pretrained_models/CosyVoice2-0.5B" ]; then
    echo "错误: 模型目录不存在！"
    echo "请先下载模型："
    echo "  python -c \"from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')\""
    exit 1
fi

# 启动 WebUI
python webui.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B "$@"
EOF

chmod +x start_webui.sh
print_info "启动脚本已创建: start_webui.sh"

# 安装完成提示
echo ""
print_info "=========================================="
print_info "CosyVoice 安装完成！"
print_info "=========================================="
echo ""
echo -e "下一步操作："
echo -e "  1. 激活虚拟环境并设置环境变量:"
echo -e "     ${GREEN}source .venv/bin/activate${NC}"
echo -e "     ${GREEN}export PYTHONPATH=\"\${PYTHONPATH}:\$(pwd):\$(pwd)/third_party/Matcha-TTS\"${NC}"
echo ""
echo -e "  2. 下载预训练模型（选择一种方式）："
echo -e "     方式 A - 使用 Python SDK:"
echo -e "     ${GREEN}python -c \"from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')\"${NC}"
echo ""
echo -e "     方式 B - 使用 git:"
echo -e "     ${GREEN}mkdir -p pretrained_models && git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B${NC}"
echo ""
echo -e "  3. 启动 Web 界面（使用启动脚本）:"
echo -e "     ${GREEN}./start_webui.sh${NC}"
echo ""
echo -e "     或者手动启动:"
echo -e "     ${GREEN}source .venv/bin/activate${NC}"
echo -e "     ${GREEN}export PYTHONPATH=\"\${PYTHONPATH}:\$(pwd):\$(pwd)/third_party/Matcha-TTS\"${NC}"
echo -e "     ${GREEN}python webui.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B${NC}"
echo ""
echo -e "  4. 查看完整文档:"
echo -e "     ${GREEN}cat INSTALL_UV.md${NC}"
echo ""
print_info "如需帮助，请访问: https://github.com/FunAudioLLM/CosyVoice/issues"