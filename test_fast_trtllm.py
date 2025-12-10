#!/usr/bin/env python3
"""
TensorRT-LLM 推理质量测试脚本
测试修复后的 TensorRT-LLM 集成是否能正确生成与输入文本匹配的音频
"""

import sys
import os
import torch
import time
import numpy as np

sys.path.append("third_party/Matcha-TTS")

def test_trtllm_quality():
    """测试 TensorRT-LLM 推理质量"""
    print("=" * 60)
    print("📝 TensorRT-LLM 推理质量测试")
    print("=" * 60)
    
    # 检查 TensorRT-LLM 是否可用
    try:
        import tensorrt_llm
        from tensorrt_llm.runtime import ModelRunnerCpp
        print(f"✅ TensorRT-LLM 版本: {tensorrt_llm.__version__}")
        trtllm_available = True
    except ImportError as e:
        print(f"❌ TensorRT-LLM 不可用: {e}")
        trtllm_available = False
        return
    
    # 检查引擎
    engine_path = "runtime/triton_trtllm/trt_engines_bfloat16/rank0.engine"
    engine_dir = "runtime/triton_trtllm/trt_engines_bfloat16"
    
    # Tokenizer 目录
    tokenizer_dir = "runtime/triton_trtllm/cosyvoice2_llm"
    if not os.path.exists(tokenizer_dir):
        # 尝试备用路径
        alt_tokenizer_dir = "pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN"
        if os.path.exists(alt_tokenizer_dir):
            tokenizer_dir = alt_tokenizer_dir
            print(f"📡 使用备用 tokenizer: {tokenizer_dir}")
        else:
            print(f"❌ Tokenizer 目录不存在: {tokenizer_dir}")
            return
    
    if os.path.exists(engine_path):
        size_mb = os.path.getsize(engine_path) / (1024 * 1024)
        print(f"✅ 引擎存在: {size_mb:.2f} MB")
    else:
        print(f"❌ 引擎不存在: {engine_path}")
        return
    
    # 加载模型
    print("\n加载 CosyVoice2 模型...")
    from cosyvoice.cli.cosyvoice import CosyVoice2
    cosyvoice = CosyVoice2("pretrained_models/CosyVoice2-0.5B", load_jit=True, load_trt=True, fp16=True)
    print("✅ CosyVoice2 加载成功")
    
    # 加载 FastCosyVoice2
    print("\n初始化 FastCosyVoice2 (TensorRT-LLM 模式)...")
    from fast_server import FastCosyVoice2
    
    fast_cosyvoice = FastCosyVoice2(
        cosyvoice_model=cosyvoice,
        trtllm_engine_dir=engine_dir,
        trtllm_tokenizer_dir=tokenizer_dir,
        use_trtllm=True
    )
    
    if fast_cosyvoice.use_trtllm:
        print("✅ FastCosyVoice2 初始化成功 (TensorRT-LLM)")
    else:
        print("❌ FastCosyVoice2 TensorRT-LLM 初始化失败")
        return
    
    # 加载说话人
    print("\n加载说话人...")
    from cosyvoice.utils.file_utils import load_wav
    import glob
    import re
    
    wav_files = glob.glob("asset/speakers/*.wav")
    if not wav_files:
        print("❌ 未找到说话人文件")
        return
    
    test_wav = wav_files[0]
    filename = os.path.basename(test_wav)
    match = re.match(r'\[(.+?)\](.+)\.wav$', filename)
    
    if not match:
        print(f"❌ 文件名格式不正确: {filename}")
        return
    
    speaker_name = match.group(1)
    prompt_text = match.group(2)
    
    print(f"加载说话人: {speaker_name}")
    print(f"Prompt 文本: {prompt_text[:50]}...")
    
    prompt_speech_16k = load_wav(test_wav, 16000)
    
    # 归一化
    max_val = torch.abs(prompt_speech_16k).max().item()
    if max_val > 0.95:
        prompt_speech_16k = (prompt_speech_16k / max_val) * 0.95
    
    fast_cosyvoice.add_zero_shot_spk(prompt_text, prompt_speech_16k, speaker_name)
    print(f"✅ 说话人添加成功: {speaker_name}")
    
    # 测试推理
    print("\n" + "=" * 60)
    print("🎯 测试推理 (TensorRT-LLM 模式)")
    print("=" * 60)
    
    test_texts = [
        "你好，欢迎使用这个语音合成系统！",
        "今天天气真不错，适合出去走走。",
        "收到好友从远方寄来的生日礼物，很感动。",
    ]
    
    # 创建输出目录
    output_dir = "output_test_trtllm"
    os.makedirs(output_dir, exist_ok=True)
    
    import soundfile as sf
    
    for i, test_text in enumerate(test_texts):
        print(f"\n测试 {i+1}/{len(test_texts)}: {test_text}")
        print("-" * 40)
        
        start_time = time.time()
        audio_chunks = []
        
        try:
            for output in fast_cosyvoice.inference_zero_shot(
                test_text, "", None,
                zero_shot_spk_id=speaker_name,
                stream=True
            ):
                audio_chunks.append(output['tts_speech'].numpy())
            
            if audio_chunks:
                full_audio = np.concatenate(audio_chunks, axis=1).flatten()
                duration = len(full_audio) / 24000
                total_time = time.time() - start_time
                
                # 保存音频
                output_path = os.path.join(output_dir, f"test_{i+1}.wav")
                sf.write(output_path, full_audio, 24000)
                
                print(f"✅ 生成成功!")
                print(f"   音频时长: {duration:.2f}s")
                print(f"   推理时间: {total_time:.2f}s")
                print(f"   RTF: {total_time/duration:.2f}x")
                print(f"   输出: {output_path}")
            else:
                print("❌ 未生成任何音频")
                
        except Exception as e:
            print(f"❌ 推理失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🎉 测试完成!")
    print(f"📁 音频文件保存在: {output_dir}/")
    print("🎧 请播放音频文件检查内容是否与输入文本匹配")
    print("=" * 60)


if __name__ == "__main__":
    test_trtllm_quality()
