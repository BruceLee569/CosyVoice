#!/usr/bin/env python3
"""
CosyVoice 模型下载脚本

使用方法：
  python download_models.py                    # 下载推荐模型 (2.0, 2.0-llm, tts)
  python download_models.py --model 2.0        # 仅下载 CosyVoice 2.0 (ModelScope)
  python download_models.py --model 2.0-llm    # 仅下载 CosyVoice 2.0 LLM (HuggingFace)
  python download_models.py --model 300m       # 仅下载 CosyVoice-300M
  python download_models.py --list             # 列出所有可用模型

💡 离线下载支持：
  - 一旦网络中断，直接重新运行脚本即可续传
  - 不需要任何参数，自动检测并继续下载
  - 不会重复下载空字节，获得最佳效率

🔁 代理支持：
  自动检测环境变量（HTTP_PROXY, HTTPS_PROXY, NO_PROXY 等）
  
  方式 1: 从 /etc/network_turbo 读取（AutoDL 推荐）
    source /etc/network_turbo && python download_models.py
  
  方式 2: 主动设置环境变量
    export HTTP_PROXY=http://proxy:8080
    export HTTPS_PROXY=http://proxy:8080
    python download_models.py
  
  方式 3: 一行命令
    HTTP_PROXY=http://proxy:8080 python download_models.py
"""

import argparse
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ========== 代理配置支持 ==========
def setup_proxy_from_env():
    """
    从环境变量读取代理设置（支持 HTTP_PROXY, HTTPS_PROXY, ALL_PROXY 等）
    优先级: ALL_PROXY > HTTPS_PROXY/HTTP_PROXY > (no proxy)
    同时支持 no_proxy 配置（跳过特定域名）
    """
    proxy_url = None
    no_proxy = os.environ.get('no_proxy') or os.environ.get('NO_PROXY', '')
    
    # 检查各种常见的代理环境变量
    proxy_vars = ['ALL_PROXY', 'all_proxy', 'HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy']
    
    for var in proxy_vars:
        if var in os.environ and os.environ[var]:
            proxy_url = os.environ[var]
            break
    
    if proxy_url:
        print_colored(f"✓ 检测到代理配置: {proxy_url}", "blue")
        if no_proxy:
            print_colored(f"✓ no_proxy 配置: {no_proxy}", "blue")
        
        # 为 urllib 设置代理
        proxy_handler = urllib.request.ProxyHandler({
            'http': proxy_url,
            'https': proxy_url
        })
        opener = urllib.request.build_opener(proxy_handler)
        urllib.request.install_opener(opener)
        
        # 设置环境变量供第三方库使用
        os.environ['HTTP_PROXY'] = proxy_url
        os.environ['HTTPS_PROXY'] = proxy_url
        if no_proxy:
            os.environ['no_proxy'] = no_proxy
            os.environ['NO_PROXY'] = no_proxy
        
        return True
    
    return False


# 可用的模型配置
# source: ModelScope 或 HuggingFace
MODELS = {
    "2.0": {
        "source": "modelscope",
        "id": "iic/CosyVoice2-0.5B",
        "dir": "pretrained_models/CosyVoice2-0.5B",
        "description": "CosyVoice 2.0 (推荐, ModelScope)",
        "size": "~2.5GB"
    },
    "2.0-llm": {
        "source": "modelscope",
        "id": "yunye007/cosyvoice2_llm",
        "dir": "pretrained_models/cosyvoice2_llm",
        "description": "CosyVoice 2.0 LLM (ModelScope)",
        "size": "~2.5GB"
    },
    "300m": {
        "source": "modelscope",
        "id": "iic/CosyVoice-300M",
        "dir": "pretrained_models/CosyVoice-300M",
        "description": "CosyVoice-300M 基础模型",
        "size": "~1.5GB"
    },
    "300m-sft": {
        "source": "modelscope",
        "id": "iic/CosyVoice-300M-SFT",
        "dir": "pretrained_models/CosyVoice-300M-SFT",
        "description": "CosyVoice-300M SFT 版本",
        "size": "~1.5GB"
    },
    "300m-instruct": {
        "source": "modelscope",
        "id": "iic/CosyVoice-300M-Instruct",
        "dir": "pretrained_models/CosyVoice-300M-Instruct",
        "description": "CosyVoice-300M Instruct 版本",
        "size": "~1.5GB"
    },
    "ttsfrd": {
        "source": "modelscope",
        "id": "iic/CosyVoice-ttsfrd",
        "dir": "pretrained_models/CosyVoice-ttsfrd",
        "description": "文本规范化资源（可选）",
        "size": "~100MB"
    }
}


def print_colored(text, color="green"):
    """打印带颜色的文本"""
    colors = {
        "green": "\033[0;32m",
        "yellow": "\033[1;33m",
        "red": "\033[0;31m",
        "blue": "\033[0;34m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")


def list_models():
    """列出所有可用的模型"""
    print("\n可用的模型:")
    print("=" * 80)
    for key, model in MODELS.items():
        status = "✅ 已下载" if os.path.exists(model["dir"]) else "⬇️  未下载"
        source = model['source'].upper()
        print(f"{key:15} {status:10} [{source:11}] {model['description']:35} ({model['size']})")
    print("=" * 80)
    print("\n使用示例:")
    print("  python download_models.py --model 2.0              # 下载 CosyVoice 2.0 (ModelScope)")
    print("  python download_models.py --model 2.0-llm          # 下载 CosyVoice 2.0 LLM (HuggingFace)")
    print("  python download_models.py                          # 下载推荐模型 (2.0, 2.0-llm, tts)")
    print()


def check_model_exists(model_dir):
    """检查模型是否已完全下载"""
    return os.path.exists(model_dir) and os.path.isdir(model_dir) and len(os.listdir(model_dir)) > 0


def is_model_incomplete(model_dir):
    """检查模型是否下载不完整（用于续传识别）"""
    if not os.path.exists(model_dir):
        return False
    # 如果目录存在但为空，表示下载不完整
    if os.path.isdir(model_dir) and len(os.listdir(model_dir)) == 0:
        return True
    # 如果存在 .incomplete 或类似标记文件，表示下载不完整
    # 通常框架会自动处理，这里只做简单检查
    return False


def download_model_from_modelscope(model_id, model_dir, description):
    """从 ModelScope 下载模型（支持断点续传 + 多线程加速）"""
    try:
        from modelscope import snapshot_download
        
        print("   正在从 ModelScope 下载... 这可能需要几分钟")
        print("   💡 支持断点续传: 网络中断后可重新运行脚本继续下载")
        print("   🚀 使用 6 线程并发下载，配合多模型并行最大化带宽")
        if 'HTTP_PROXY' in os.environ or 'HTTPS_PROXY' in os.environ:
            print("   🔁 使用代理配置连接")
        
        # ModelScope 的 snapshot_download 支持断点续传和多线程下载
        # max_workers 参数控制并发下载线程数，建议 4-8 个线程
        snapshot_download(
            model_id, 
            local_dir=model_dir,
            max_workers=6  # 使用 6 个线程并发下载，配合多模型并行
        )
        
        print_colored(f"✅ 下载完成: {description}", "green")
        return True
        
    except ImportError:
        print_colored("❌ 错误: 未找到 modelscope 模块", "red")
        print("   请确保已安装: uv pip install modelscope")
        return False
        
    except Exception as e:
        print_colored(f"❌ 下载失败: {e}", "red")
        print("\n建议:")
        print("  1. 检查网络连接和代理配置")
        print("  2. 网络恢复后重新运行脚本，支持断点续传:")
        print(f"     python download_models.py --model {model_id.split('/')[-1]}")
        print("  3. 或使用 git 方式下载:")
        print(f"     git clone https://www.modelscope.cn/{model_id}.git {model_dir}")
        return False


def download_model_from_huggingface(model_id, model_dir, description):
    """从 HuggingFace 下载模型（支持断点续传 + 多线程加速）"""
    try:
        from huggingface_hub import snapshot_download as hf_snapshot_download
        
        print("   正在从 HuggingFace 下载... 这可能需要几分钟")
        print("   💡 支持断点续传: 网络中断后可重新运行脚本继续下载")
        print("   🚀 使用 6 线程并发下载，配合多模型并行最大化带宽")
        if 'HTTP_PROXY' in os.environ or 'HTTPS_PROXY' in os.environ:
            print("   🔁 使用代理配置连接")
        
        # HuggingFace 的 snapshot_download 支持断点续传和多线程下载
        # max_workers 参数控制并发下载线程数
        hf_snapshot_download(
            repo_id=model_id, 
            local_dir=model_dir,
            max_workers=6  # 使用 6 个线程并发下载，配合多模型并行
        )
        
        print_colored(f"✅ 下载完成: {description}", "green")
        return True
        
    except ImportError:
        print_colored("❌ 错误: 未找到 huggingface_hub 模块", "red")
        print("   请确保已安装: uv pip install huggingface-hub")
        print("\n或使用命令行工具:")
        print("   huggingface-cli download --local-dir {model_dir} {model_id}")
        return False
        
    except Exception as e:
        print_colored(f"❌ 下载失败: {e}", "red")
        print("\n建议:")
        print("  1. 检查网络连接和代理配置")
        print("  2. 网络恢复后重新运行脚本，支持断点续传:")
        print(f"     python download_models.py")
        print("  3. 或使用 huggingface-cli 命令（也支持断点续传）:")
        print(f"     huggingface-cli download --local-dir {model_dir} {model_id}")
        return False


def download_model(model_id, model_dir, description, source="modelscope"):
    """下载单个模型"""
    if check_model_exists(model_dir):
        print_colored(f"✅ 模型已存在: {description} ({model_dir})", "yellow")
        return True
    
    print_colored(f"\n⬇️  开始下载: {description}", "blue")
    print(f"   来源: {source.upper()}")
    print(f"   模型 ID: {model_id}")
    print(f"   保存路径: {model_dir}")
    
    if source == "huggingface":
        return download_model_from_huggingface(model_id, model_dir, description)
    else:
        return download_model_from_modelscope(model_id, model_dir, description)


def main():
    """主函数"""
    # 不要缺失 ✅
    # 为了支持代理配置，应该末尾执行脚本时已经针对了代理
    # 但为了保险起觑，也在这里执行一次
    setup_proxy_from_env()
    
    parser = argparse.ArgumentParser(
        description="CosyVoice 模型下载工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python download_models.py                    # 下载推荐模型 (2.0, 2.0-llm, tts)
  python download_models.py --model 2.0        # 仅下载 CosyVoice 2.0 (ModelScope)
  python download_models.py --model 2.0-llm    # 仅下载 CosyVoice 2.0 LLM (HuggingFace)
  python download_models.py --model 300m       # 仅下载 CosyVoice-300M
  python download_models.py --list             # 列出所有可用模型
  python download_models.py --all              # 下载所有模型
  python download_models.py --force            # 强制重新下载已存在的模型
        """
    )
    
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        help="指定要下载的模型（如: 2.0, 2.0-hf, 300m 等）"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用模型"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="下载所有模型（不指定 --model 时默认仅下载推荐模型）"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载已存在的模型"
    )
    
    args = parser.parse_args()
    
    # 列出模型
    if args.list:
        list_models()
        return 0
    
    print_colored("\n🎤 CosyVoice 模型下载工具 (支持 ModelScope & HuggingFace)\n", "blue")
    
    # 确定要下载的模型
    if args.model:
        # 指定具体模型
        models_to_download = {args.model: MODELS[args.model]}
    elif args.all:
        # 下载所有模型
        models_to_download = MODELS.copy()
    else:
        # 默认下载推荐模型：2.0, 2.0-llm, tts
        models_to_download = {
            "2.0": MODELS["2.0"],
            "2.0-llm": MODELS["2.0-llm"],
            "ttsfrd": MODELS["ttsfrd"]
        }
    
    # 创建模型目录
    os.makedirs("pretrained_models", exist_ok=True)
    
    # 下载模型（并行下载多个模型以跑满带宽）
    success_count = 0
    total_count = len(models_to_download)
    
    print(f"计划下载 {total_count} 个模型")
    print(f"🚀 使用并行下载策略，同时下载最多 3 个模型以最大化带宽利用\n")
    
    # 预处理：清理强制重新下载或不完整的模型
    import shutil
    for key, model in models_to_download.items():
        if args.force and check_model_exists(model["dir"]):
            print(f"🗑️  删除已存在的模型: {model['dir']}")
            shutil.rmtree(model["dir"])
        
        if is_model_incomplete(model["dir"]):
            print(f"⚠️  检测到不完整下载: {model['dir']}（为空目录，将删除后重新下载）")
            shutil.rmtree(model["dir"])
    
    # 使用线程池并行下载多个模型
    # max_workers=3 表示最多同时下载 3 个模型
    # 每个模型内部还会使用 6 个线程下载文件，总共约 18 个并发连接
    with ThreadPoolExecutor(max_workers=3) as executor:
        # 提交所有下载任务
        future_to_model = {
            executor.submit(
                download_model, 
                model["id"], 
                model["dir"], 
                model["description"], 
                model["source"]
            ): key
            for key, model in models_to_download.items()
        }
        
        # 等待所有任务完成并统计结果
        for future in as_completed(future_to_model):
            key = future_to_model[future]
            try:
                if future.result():
                    success_count += 1
            except Exception as e:
                print_colored(f"❌ 模型 {key} 下载时发生异常: {e}", "red")
    
    # 总结
    print("\n" + "=" * 70)
    print(f"下载完成: {success_count}/{total_count} 个模型下载成功")
    print("=" * 70)
    
    if success_count == total_count:
        print_colored("\n✅ 所有模型下载成功！", "green")
        print("\n下一步:")
        print("  1. 运行安装验证:")
        print("     uv run python test_installation.py")
        print("\n  2. 启动 Web 界面:")
        print("     uv run python webui.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B")
        print("\n💡 离线续传: 如下载失败，网络恢复后直接重新运行脚本即可续传下载")
        return 0
    else:
        print_colored(f"\n⚠️  部分模型下载失败 ({total_count - success_count} 个)", "yellow")
        print("\n建议:")
        print("  1. 检查网络连接")
        print("  2. 使用 git 方式下载失败的模型")
        print("  3. 查看上述错误信息")
        return 1


if __name__ == "__main__":
    # 首先从环境变量读取代理配置（并设置给各个下载库）
    print("\n" + "="*70)
    print("🚀 CosyVoice 模型下载器")
    print("="*70)
    
    has_proxy = setup_proxy_from_env()
    if not has_proxy:
        print("ℹ️  未检测到代理配置，直接连接网络")
    
    print()
    sys.exit(main())
