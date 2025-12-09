# 离线环境部署指南 (WSL2 -> AutoDL)

由于网络环境限制，推荐在本地 (WSL2) 下载好所有依赖包，打包上传到 AutoDL 服务器进行离线安装。

## 第一阶段：本地打包 (在 WSL2 终端执行)

### 1. 准备依赖列表
首先生成纯净的 `requirements.txt`，并剔除本地路径引用。

```bash
# 1. 生成锁定文件
uv pip compile pyproject.toml -o requirements.txt

# 2. 清理本地路径引用 (防止 pip 下载时报错)
grep -v "file:///" requirements.txt > requirements_clean.txt
```

### 2. 下载依赖包
使用 `uv` 管理的 Python 3.10 环境来运行 `pip download`。
**注意**：由于 `uv` 环境默认不带 `pip`，需要添加 `--with pip` 参数。

```bash
# 创建存放目录
mkdir -p offline_packages

# 使用 Python 3.10 下载所有依赖 (自动注入 pip)
# 注意：这一步会下载 PyTorch (2GB+)，请确保网络通畅
uv run --python 3.10 --with pip python -m pip download \
    -r requirements_clean.txt \
    -d offline_packages \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

### 3. 补充本地 Wheel 包
将项目自带的 `ttsfrd` 等本地包复制到下载目录。

```bash
cp pretrained_models/CosyVoice-ttsfrd/*.whl offline_packages/
```

### 4. 打包
```bash
tar -zcvf offline_packages.tar.gz offline_packages
```

---

## 第二阶段：服务器安装 (在 AutoDL 终端执行)

### 1. 上传与解压
将 `offline_packages.tar.gz` 上传到 AutoDL 服务器的 `/root/CosyVoice/` (或其他项目目录) 下。

```bash
# 解压离线包
tar -zxvf offline_packages.tar.gz
```

### 2. 创建 Python 3.10 环境（放在项目目录内，更易隔离）
```bash
cd ~/CosyVoice  # 先进入克隆的项目目录
pip install uv  # 如果未安装

# 创建干净环境（建议放在项目根目录下）
uv venv .venv --python 3.10
source .venv/bin/activate
```

### 3. 离线安装（whl + tar.gz + zip）
使用 `--no-index` 强制只从本地装，按文件类型分两步：

```bash
# 0. 准备：离线包解压在 ~/offline_packages
cd ~/CosyVoice

# 1) 删除不兼容的 cp38 包
rm -f ~/offline_packages/*cp38*.whl

# 2) 先装所有 wheel
uv pip install --no-index --find-links ~/offline_packages \*.whl

# 3) 再装源码包 (tar.gz/zip)，关闭 build isolation 避免联网
uv pip install --no-index --no-build-isolation --no-deps \
    --find-links ~/offline_packages ~/offline_packages/*.tar.gz ~/offline_packages/*.zip
```

---

## 第三阶段：部署项目代码

### 1. 打包代码 (WSL2)
排除大文件，只打包源码。
```bash
cd /home/cz/ai/CosyVoice
tar -zcvf cosyvoice_code.tar.gz \
    --exclude=offline_packages \
    --exclude=.venv \
    --exclude=.git \
    --exclude=pretrained_models \
    --exclude=asset \
    --exclude=__pycache__ \
    .
```

### 2. 上传与安装 (AutoDL)
上传 `cosyvoice_code.tar.gz` 到 `/root/`。

```bash
# 解压代码
mkdir -p ~/CosyVoice
tar -zxvf cosyvoice_code.tar.gz -C ~/CosyVoice

# 进入项目目录
cd ~/CosyVoice

# 激活之前创建的环境 (关键步骤！)
source ~/offline_packages/.venv/bin/activate

# 直接运行 (不要用 uv run，防止触发联网更新)
python server.py --port 50000 --model_dir pretrained_models/CosyVoice-300M
```
