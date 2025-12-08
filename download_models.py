#!/usr/bin/env python3
"""
CosyVoice 模型下载脚本

使用方法：
  python download_models.py                    # 下载所有模型
  python download_models.py --model 2.0        # 仅下载 CosyVoice 2.0
  python download_models.py --model 300m       # 仅下载 CosyVoice-300M
  python download_models.py --list             # 列出所有可用模型
"""

import argparse
import os
import sys

# 可用的模型配置
MODELS = {
    "2.0": {
        "id": "iic/CosyVoice2-0.5B",
        "dir": "pretrained_models/CosyVoice2-0.5B",
        "description": "CosyVoice 2.0 (推荐)",
        "size": "~2.5GB"
    },
    "300m": {
        "id": "iic/CosyVoice-300M",
        "dir": "pretrained_models/CosyVoice-300M",
        "description": "CosyVoice-300M 基础模型",
        "size": "~1.5GB"
    },
    "300m-sft": {
        "id": "iic/CosyVoice-300M-SFT",
        "dir": "pretrained_models/CosyVoice-300M-SFT",
        "description": "CosyVoice-300M SFT 版本",
        "size": "~1.5GB"
    },
    "300m-instruct": {
        "id": "iic/CosyVoice-300M-Instruct",
        "dir": "pretrained_models/CosyVoice-300M-Instruct",
        "description": "CosyVoice-300M Instruct 版本",
        "size": "~1.5GB"
    },
    "ttsfrd": {
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
    print("=" * 70)
    for key, model in MODELS.items():
        status = "✅ 已下载" if os.path.exists(model["dir"]) else "⬇️  未下载"
        print(f"{key:15} {status:10} {model['description']:35} ({model['size']})")
    print("=" * 70)
    print("\n使用示例:")
    print("  python download_models.py --model 2.0              # 下载 CosyVoice 2.0")
    print("  python download_models.py                          # 下载所有模型")
    print()


def check_model_exists(model_dir):
    """检查模型是否已下载"""
    return os.path.exists(model_dir) and os.path.isdir(model_dir) and len(os.listdir(model_dir)) > 0


def download_model(model_id, model_dir, description):
    """下载单个模型"""
    if check_model_exists(model_dir):
        print_colored(f"✅ 模型已存在: {description} ({model_dir})", "yellow")
        return True
    
    print_colored(f"\n⬇️  开始下载: {description}", "blue")
    print(f"   模型 ID: {model_id}")
    print(f"   保存路径: {model_dir}")
    
    try:
        from modelscope import snapshot_download
        
        print("   正在下载... 这可能需要几分钟")
        snapshot_download(model_id, local_dir=model_dir)
        
        print_colored(f"✅ 下载完成: {description}", "green")
        return True
        
    except ImportError:
        print_colored("❌ 错误: 未找到 modelscope 模块", "red")
        print("   请确保已安装: uv pip install modelscope")
        return False
        
    except Exception as e:
        print_colored(f"❌ 下载失败: {e}", "red")
        print("\n建议:")
        print("  1. 检查网络连接")
        print("  2. 使用 git 方式下载:")
        print(f"     git clone https://www.modelscope.cn/{model_id}.git {model_dir}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="CosyVoice 模型下载工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python download_models.py                    # 下载所有模型
  python download_models.py --model 2.0        # 仅下载 CosyVoice 2.0
  python download_models.py --model 300m       # 仅下载 CosyVoice-300M
  python download_models.py --list             # 列出所有可用模型
  python download_models.py --skip-ttsfrd      # 下载所有模型但跳过 ttsfrd
        """
    )
    
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        help="指定要下载的模型"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用模型"
    )
    
    parser.add_argument(
        "--skip-ttsfrd",
        action="store_true",
        help="跳过下载 ttsfrd 资源"
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
    
    print_colored("\n🎤 CosyVoice 模型下载工具\n", "blue")
    
    # 确定要下载的模型
    if args.model:
        models_to_download = {args.model: MODELS[args.model]}
    else:
        models_to_download = MODELS.copy()
        if args.skip_ttsfrd:
            models_to_download.pop("ttsfrd", None)
    
    # 创建模型目录
    os.makedirs("pretrained_models", exist_ok=True)
    
    # 下载模型
    success_count = 0
    total_count = len(models_to_download)
    
    print(f"计划下载 {total_count} 个模型\n")
    
    for key, model in models_to_download.items():
        # 如果强制下载，先删除已存在的模型
        if args.force and check_model_exists(model["dir"]):
            print(f"🗑️  删除已存在的模型: {model['dir']}")
            import shutil
            shutil.rmtree(model["dir"])
        
        if download_model(model["id"], model["dir"], model["description"]):
            success_count += 1
    
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
        return 0
    else:
        print_colored(f"\n⚠️  部分模型下载失败 ({total_count - success_count} 个)", "yellow")
        print("\n建议:")
        print("  1. 检查网络连接")
        print("  2. 使用 git 方式下载失败的模型")
        print("  3. 查看上述错误信息")
        return 1


if __name__ == "__main__":
    sys.exit(main())
