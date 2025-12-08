# CosyVoice Ubuntu Server 部署指南（使用 uv）

本指南提供使用 uv 包管理器在 Ubuntu Server 上部署 CosyVoice 的完整步骤。

## 前置要求

- Ubuntu Server 20.04 或更高版本
- NVIDIA GPU（推荐）及相应的 CUDA 驱动
- 至少 16GB RAM
- 足够的磁盘空间（建议 50GB+）

## 完整安装步骤

### 1. 更新系统并安装基础依赖

```bash
# 更新包列表
sudo apt-get update && sudo apt-get upgrade -y

# 安装必要的系统工具
sudo apt-get install -y \
    git \
    git-lfs \
    sox \
    libsox-dev \
    build-essential \
    curl \
    wget \
    ffmpeg \
    unzip \
    ca-certificates

# 配置 git-lfs
git lfs install
```

### 2. 安装 CUDA（如果使用 GPU）

如果还未安装 CUDA，请按照以下步骤安装（以 CUDA 12.1 为例）：

```bash
# 下载并安装 CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-1

# 配置环境变量（添加到 ~/.bashrc）
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 3. 安装 uv 包管理器

```bash
# 安装 uv（快速的 Python 包管理器）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 将 uv 添加到 PATH（重新登录或执行以下命令）
source $HOME/.local/bin/env

# 验证安装
uv --version
```

### 4. 克隆 CosyVoice 仓库

```bash
# 克隆项目（包括子模块）
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git

# 如果克隆子模块失败，单独更新
cd CosyVoice
git submodule update --init --recursive
```

### 5. 配置国内镜像（可选，加速下载）

如果在国内服务器上部署，建议配置 pip 镜像源：

```bash
mkdir -p ~/.config/pip
cat > ~/.config/pip/pip.conf << EOF
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
extra-index-url = https://download.pytorch.org/whl/cu121
EOF
```

### 6. 使用 uv 安装 Python 环境和依赖

**方式 1: 使用安装脚本（推荐）**
```bash
# 进入项目目录
cd CosyVoice

# 运行自动安装脚本
./install_uv.sh

# 脚本会自动完成以下操作：
# - 创建虚拟环境
# - 安装所有必要依赖
# - 验证安装
# 注意: TensorRT 将被跳过（可选依赖，某些系统上可能安装失败）
```

**方式 2: 手动安装**
```bash
# 进入项目目录
cd CosyVoice

# 创建虚拟环境
uv venv

# 激活环境
source .venv/bin/activate

# 安装依赖（不包含 TensorRT）
grep -v "tensorrt" requirements.txt > requirements_temp.txt
uv pip install --index-strategy unsafe-best-match -r requirements_temp.txt
rm requirements_temp.txt

# 如需 TensorRT 加速（可选）
# uv pip install tensorrt-cu12==10.0.1 tensorrt-cu12-bindings==10.0.1 tensorrt-cu12-libs==10.0.1
```

### 7. 激活虚拟环境

有两种方式使用已安装的环境：

**方式 1：手动激活虚拟环境**
```bash
source .venv/bin/activate
```

**方式 2：使用 uv run（推荐）**
```bash
# 无需激活环境，直接运行命令
uv run python your_script.py
```

### 8. 验证安装

```bash
# 激活环境后，验证 PyTorch 和 CUDA
source .venv/bin/activate
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# 验证其他关键依赖
python -c "import torchaudio, transformers, modelscope; print('All key dependencies imported successfully!')"
```

### 9. 下载预训练模型

```bash
# 使用 Python SDK 下载模型
source .venv/bin/activate
python << EOF
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')
snapshot_download('iic/CosyVoice-300M-Instruct', local_dir='pretrained_models/CosyVoice-300M-Instruct')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
EOF
```

或使用 git 下载：

```bash
mkdir -p pretrained_models
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B
git clone https://www.modelscope.cn/iic/CosyVoice-300M.git pretrained_models/CosyVoice-300M
git clone https://www.modelscope.cn/iic/CosyVoice-300M-SFT.git pretrained_models/CosyVoice-300M-SFT
git clone https://www.modelscope.cn/iic/CosyVoice-300M-Instruct.git pretrained_models/CosyVoice-300M-Instruct
git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git pretrained_models/CosyVoice-ttsfrd
```

### 10. 安装 ttsfrd（可选，提升文本规范化性能）

```bash
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
uv pip install ttsfrd_dependency-0.1-py3-none-any.whl
uv pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
cd ../..
```

### 11. 启动 Web 演示

```bash
# 使用 uv run 启动 webui
uv run python webui.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B

# 或者激活环境后运行
source .venv/bin/activate
python webui.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B
```

访问 `http://your-server-ip:50000` 即可使用 Web 界面。

### 12. 测试推理

创建测试脚本 `test_inference.py`：

```python
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

# 加载模型
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)

# 零样本推理测试
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
for i, j in enumerate(cosyvoice.inference_zero_shot(
    '你好，这是一个测试。', 
    '希望你以后能够做的比我还好呦。', 
    prompt_speech_16k, 
    stream=False
)):
    torchaudio.save(f'test_output_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
    print(f"生成音频已保存到: test_output_{i}.wav")
```

运行测试：
```bash
uv run python test_inference.py
```

## 常见问题

### 1. 如果遇到 sox 相关错误

```bash
sudo apt-get install -y sox libsox-dev libsox-fmt-all
```

### 2. 如果 GPU 内存不足

在推理时使用 `fp16=True` 选项减少显存占用：
```python
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', fp16=True)
```

### 3. 如果需要更新依赖

```bash
# 更新所有依赖到最新兼容版本
uv sync --upgrade

# 更新特定包
uv pip install --upgrade package-name
```

### 4. 查看已安装的包

```bash
uv pip list
```

### 5. 卸载环境

```bash
# 删除虚拟环境
rm -rf .venv
```

## 生产部署建议

### 使用 systemd 服务

创建服务文件 `/etc/systemd/system/cosyvoice.service`：

```ini
[Unit]
Description=CosyVoice TTS Service
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/CosyVoice
ExecStart=/path/to/CosyVoice/.venv/bin/python webui.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启动服务：
```bash
sudo systemctl daemon-reload
sudo systemctl enable cosyvoice
sudo systemctl start cosyvoice
sudo systemctl status cosyvoice
```

### 使用 Nginx 反向代理

安装 Nginx：
```bash
sudo apt-get install -y nginx
```

配置文件 `/etc/nginx/sites-available/cosyvoice`：
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:50000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

启用配置：
```bash
sudo ln -s /etc/nginx/sites-available/cosyvoice /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## 性能优化

1. **使用 TensorRT 加速**（需要单独配置）
2. **启用 fp16 推理**减少显存占用
3. **使用流式推理**降低延迟
4. **配置适当的批处理大小**

## 总结

使用 uv 管理 Python 环境相比 conda 的优势：
- ✅ 安装速度快 10-100 倍
- ✅ 环境隔离更彻底
- ✅ 依赖解析更准确
- ✅ 占用磁盘空间更小
- ✅ 与现代 Python 生态系统更好集成

如有问题，请访问 [GitHub Issues](https://github.com/FunAudioLLM/CosyVoice/issues)。
