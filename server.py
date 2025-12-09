import os
import pickle
import time
import sys
import argparse
import logging
import re
import glob

import requests

logging.getLogger("matplotlib").setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import torch
from tqdm import tqdm

# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append("{}/../../..".format(ROOT_DIR))
# sys.path.append("{}/../../../third_party/Matcha-TTS".format(ROOT_DIR))
sys.path.append("third_party/Matcha-TTS")

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

def generate_data(model_output):
    """生成音频数据流，对输出进行削波处理防止爆音"""
    for i in model_output:
        tts_speech = i["tts_speech"].numpy()
        
        # 2. 输出端削波：防止 float -> int16 转换时的整数溢出
        # 即使输入正常，模型生成的浮点数也可能略微超出 [-1, 1]
        # 如果不进行削波，转换时会发生整数溢出（Wrap around），产生严重的爆音
        # 使用 clip 进行硬削波是音频处理的标准做法
        tts_speech = np.clip(tts_speech, -1.0, 1.0)
        
        # 转换为 int16 格式
        # 注意：int16 的范围是 [-32768, 32767]
        # 如果乘以 32768，当值为 1.0 时会得到 32768，导致 int16 溢出变成 -32768（爆音）
        # 因此应该乘以 32767
        tts_audio = (tts_speech * 32767).astype(np.int16).tobytes()
        yield tts_audio

def load_speakers_from_directory(speaker_dir="asset/speakers"):
    """从目录加载所有说话人"""
    speakers = {}
    
    if not os.path.exists(speaker_dir):
        logging.warning(f"说话人目录 {speaker_dir} 不存在")
        return speakers
    
    wav_files = glob.glob(os.path.join(speaker_dir, "*.wav"))
    
    for wav_path in wav_files:
        filename = os.path.basename(wav_path)
        # 解析文件名格式：[说话人名称]文本内容.wav
        match = re.match(r'\[(.+?)\](.+)\.wav$', filename)
        
        if match:
            speaker_name = match.group(1)
            prompt_text = match.group(2)
            
            try:
                prompt_speech_16k = load_wav(wav_path, 16000)
                
                # 1. 输入端归一化：确保输入音频在安全范围内
                # 用户的输入音频来源不确定，可能存在振幅过大的情况
                # 即使音频在 [-1, 1] 范围内，如果接近边界（如 0.99），后续的重采样（Resampling）
                # 可能会产生 Gibbs 现象导致数值溢出（Overshoot），从而引发爆音
                # 因此，我们将峰值限制在 0.95，留出 5% 的 Headroom
                max_val = torch.abs(prompt_speech_16k).max().item()
                target_peak = 0.95
                
                if max_val > target_peak:
                    logging.warning(f"说话人 {speaker_name} 音频峰值 {max_val:.4f} 超出安全范围，归一化到 {target_peak}")
                    prompt_speech_16k = (prompt_speech_16k / max_val) * target_peak
                
                speakers[speaker_name] = {
                    'prompt_text': prompt_text,
                    'prompt_speech_16k': prompt_speech_16k,
                    'wav_path': wav_path
                }
                logging.info(f"加载说话人: {speaker_name}")
            except Exception as e:
                logging.error(f"加载说话人 {speaker_name} 失败: {e}")
        else:
            logging.warning(f"文件名格式不正确，跳过: {filename}")
    
    return speakers


@app.get("/")
async def index():
    """主页路由，返回前端页面"""
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "CosyVoice TTS Server is running. Visit /static/index.html for the web interface."}


@app.get("/api/speakers")
async def get_speakers():
    """获取所有可用的说话人列表"""
    try:
        speakers = cosyvoice.list_available_spks()
        return JSONResponse(content={"speakers": speakers})
    except Exception as e:
        logging.error(f"获取说话人列表失败: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/tts")
@app.post("/tts")
async def inference_zero_shot(text: str = Form(), speaker: str = Form(default="")):
    """文本转语音接口"""
    try:
        # 默认使用jok老师，如果没有则使用第一个说话人
        default_speaker = "jok老师" if "jok老师" in speakers_data else (list(speakers_data.keys())[0] if speakers_data else "")
        selected_speaker = speaker if speaker else default_speaker
        
        if not selected_speaker:
            return JSONResponse(content={"error": "没有可用的说话人"}, status_code=400)
        
        # 使用指定的说话人ID进行推理
        model_output = cosyvoice.inference_zero_shot(
            text, "", None, 
            zero_shot_spk_id=selected_speaker,
            stream=True
        )
        return StreamingResponse(generate_data(model_output))
    except Exception as e:
        logging.error(f"TTS推理失败: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50000)
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/CosyVoice2-0.5B",
        # default="/root/autodl-tmp/llm/CosyVoice2-0.5B",
        help="模型本地路径或 modelscope 仓库 id",
    )
    parser.add_argument(
        "--speaker_dir",
        type=str,
        default="asset/speakers",
        help="说话人音频文件目录",
    )
    args = parser.parse_args()

    try:
        # cosyvoice = CosyVoice2(args.model_dir, load_jit=False, load_trt=True, fp16=True)
        # 1、JIT编译对速度影响不大（开了之后前面几次推理会卡慢几秒，十次之后稳定？）
        # 2、tensorrt对推理加速明显（首次推理还是会慢0.3秒左右），首次加载trt要等个3分钟左右编译
        # 3、fp16会影响后续推理速度，开启后会快个0.3秒左右
        # 实测 3090 首包延迟 0.8 秒，后续延迟 0.55 秒，24G 显存占用 5G，tts前后的文本处理速度也比4060ti快，前端首包快了将近1秒
        # 实测 4060 Ti 首包延迟 1.15，后续延迟 0.85，16G 显存占用 7.5G
        # 实测 4090 总体推理速度只比 3090 快个0.2秒，24G 显存占用 8G
        # cosyvoice = CosyVoice2(args.model_dir, load_jit=False, load_trt=False, fp16=True)
        cosyvoice = CosyVoice2(args.model_dir, load_jit=False, load_trt=True, fp16=True)
    except Exception as e:
        raise TypeError(f"导入{args.model_dir}失败，模型类型有误！错误: {e}")

    # 加载所有说话人
    print("正在加载说话人...")
    speakers_data = load_speakers_from_directory(args.speaker_dir)
    
    if not speakers_data:
        print(f"警告：未在 {args.speaker_dir} 目录找到任何说话人文件")
    else:
        print(f"成功加载 {len(speakers_data)} 个说话人")
        
        # 将所有说话人添加到模型
        for speaker_name, speaker_info in speakers_data.items():
            try:
                cosyvoice.add_zero_shot_spk(
                    speaker_info['prompt_text'],
                    speaker_info['prompt_speech_16k'],
                    speaker_name
                )
                print(f"  ✓ {speaker_name}")
            except Exception as e:
                print(f"  ✗ {speaker_name}: {e}")
        
        # 保存说话人信息
        try:
            cosyvoice.save_spkinfo()
            print("说话人信息已保存")
        except Exception as e:
            print(f"保存说话人信息失败: {e}")
    
    # 模型预热
    print("\n正在预热模型...")
    if speakers_data:
        # 默认使用jok老师进行预热，如果没有则使用第一个说话人
        warmup_speaker = "jok老师" if "jok老师" in speakers_data else list(speakers_data.keys())[0]
        print(f"使用 '{warmup_speaker}' 进行预热")
        warmup_texts = [
            '收到好友从远方寄来的生日礼物，', 
            '那份意外的惊喜与深深的祝福', 
            '让我心中充满了甜蜜的快乐，', 
        ]
        
        for t in warmup_texts:
            try:
                for _ in cosyvoice.inference_zero_shot(
                        t, "", None, zero_shot_spk_id=warmup_speaker, stream=True):
                    pass
            except Exception as e:
                print(f"预热失败: {e}")
                break
    
    print("预热完毕\n")

    uvicorn.run(app, host="0.0.0.0", port=args.port)
