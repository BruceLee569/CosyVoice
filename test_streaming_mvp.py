#!/usr/bin/env python3
"""
MVP 流式服务测试脚本
用于排查音频生成问题：
1. 提示音频泄露（生成的音频包含提示音频内容）
2. 音调异常（提示词部分音调不正常）
"""
import sys
sys.path.append('third_party/Matcha-TTS')
import os
import time
import torch
import torchaudio
import argparse
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Monkey patch load_wav 函数（必须在导入 CosyVoice 之前）
import cosyvoice.utils.file_utils

def patched_load_wav(wav, target_sr, min_sr=16000):
    """使用 soundfile 替代 torchaudio.load 以兼容 PyTorch 2.9.x"""
    import soundfile as sf
    speech, sample_rate = sf.read(wav, dtype='float32')
    if len(speech.shape) == 1:
        speech = torch.from_numpy(speech).unsqueeze(0)
    else:
        speech = torch.from_numpy(speech).T
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate >= min_sr, f'wav sample rate {sample_rate} must be greater than {target_sr}'
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech

cosyvoice.utils.file_utils.load_wav = patched_load_wav

# 导入 vLLM 和 CosyVoice
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice3 import CosyVoice3ForCausalLM
from cosyvoice.cli.cosyvoice import AutoModel

# 注册模型
ModelRegistry.register_model("CosyVoice3ForCausalLM", CosyVoice3ForCausalLM)


def save_audio(audio_tensor, sample_rate, filename):
    """保存音频文件"""
    import soundfile as sf
    audio_data = audio_tensor.squeeze().cpu().numpy()
    sf.write(filename, audio_data, sample_rate)
    logging.info(f"✓ 保存音频: {filename} ({audio_data.shape[0] / sample_rate:.2f}s)")


def test_case_4_streaming_chunks(cosyvoice, test_text, prompt_text, prompt_wav, spk_id='test_speaker_stream'):
    """
    测试用例 4: 流式推理（检查每个音频块的质量）
    """
    logging.info("=" * 70)
    logging.info(f"测试用例 4: 流式推理（检查每个音频块）")
    logging.info("=" * 70)
    
    # 先注册说话人
    logging.info(f"注册说话人: {spk_id}")
    cosyvoice.add_zero_shot_spk(prompt_text, prompt_wav, spk_id)
    
    # 流式推理
    logging.info(f"使用 zero_shot_spk_id='{spk_id}' 进行流式推理...")
    output_chunks = []
    chunk_count = 0
    
    for i, output in enumerate(cosyvoice.inference_zero_shot(
        test_text,
        '',
        None,
        zero_shot_spk_id=spk_id,
        stream=True  # 启用流式
    )):
        chunk_audio = output['tts_speech']
        chunk_duration = chunk_audio.shape[1] / cosyvoice.sample_rate
        logging.info(f"  音频块 {chunk_count}: {chunk_duration*1000:.1f}ms ({chunk_audio.shape[1]} samples)")
        
        # 保存每个音频块
        save_audio(chunk_audio, cosyvoice.sample_rate, f'test_case4_chunk_{chunk_count:02d}.wav')
        output_chunks.append(chunk_audio)
        chunk_count += 1
    
    # 拼接所有音频块
    if output_chunks:
        full_audio = torch.cat(output_chunks, dim=1)
        save_audio(full_audio, cosyvoice.sample_rate, 'test_case4_streaming_full.wav')
        logging.info(f"流式推理完成，共 {chunk_count} 个音频块")
    
    logging.info("✓ 测试用例 4 完成\n")


def main():
    parser = argparse.ArgumentParser(description='MVP 流式服务测试脚本')
    parser.add_argument(
        '--model_dir',
        type=str,
        default='/home/cz/ai/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B',
        help='模型目录（绝对路径或相对路径）'
    )
    parser.add_argument(
        '--test_text',
        type=str,
        default=' 哼，你还知道回来呀？饭菜都凉透了！ ',
        help='测试文本'
    )
    parser.add_argument(
        '--prompt_text',
        type=str,
        default='You are a helpful assistant.<|endofprompt|>说得好像您带我以来我考好过几次一样。',
        help='提示文本'
    )
    parser.add_argument(
        '--prompt_wav',
        type=str,
        default='/home/cz/ai/CosyVoice/asset/speakers/[jok老师]说得好像您带我以来我考好过几次一样.wav',
        help='提示音频文件路径（绝对路径或相对路径）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./test_output',
        help='输出目录'
    )
    args = parser.parse_args()
    
    # 转换相对路径为绝对路径
    import os
    if not os.path.isabs(args.model_dir):
        args.model_dir = os.path.abspath(args.model_dir)
    if not os.path.isabs(args.prompt_wav):
        args.prompt_wav = os.path.abspath(args.prompt_wav)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.abspath(args.output_dir)
    
    logging.info(f"模型目录: {args.model_dir}")
    logging.info(f"提示音频: {args.prompt_wav}")
    logging.info(f"输出目录: {args.output_dir}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)
    
    # 加载模型
    logging.info("=" * 70)
    logging.info("开始加载 CosyVoice3 模型（vLLM 0.12.0）")
    logging.info("=" * 70)
    
    cosyvoice = AutoModel(
        model_dir=args.model_dir,
        load_trt=True,
        load_vllm=True,
        fp16=False
    )
    
    logging.info("✓ 模型加载完成\n")
    
    # 运行测试用例
    logging.info("🔍 开始运行测试用例...")
    logging.info("")
    
    # 测试用例 4: 流式推理
    test_case_4_streaming_chunks(
        cosyvoice,
        args.test_text,
        args.prompt_text,
        args.prompt_wav,
        spk_id='test_speaker_stream'
    )

    logging.info("=" * 70)
    logging.info("🎉 所有测试用例完成！")
    logging.info(f"输出文件保存在: {os.getcwd()}")
    logging.info("=" * 70)
    logging.info("")


if __name__ == '__main__':
    main()
