import os
import pickle
import time
import sys
import argparse
import logging

import requests

logging.getLogger("matplotlib").setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse, FileResponse
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
    for i in model_output:
        tts_audio = (i["tts_speech"].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio

liuyifei_prompt_speech_16k = load_wav(
    "asset/lyf.wav",
    16000,
)


@app.get("/")
async def index():
    """主页路由，返回前端页面"""
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "CosyVoice TTS Server is running. Visit /static/index.html for the web interface."}


@app.get("/tts")
@app.post("/tts")
async def inference_zero_shot(text: str = Form(), speaker: str = Form()):
    prompt_text = "现在正在去酒店准备化妆的路上，今天刚刚从合肥到北京，大家好吗？"
    prompt_speech_16k = liuyifei_prompt_speech_16k
    model_output = cosyvoice.inference_zero_shot(
        text, prompt_text, prompt_speech_16k, stream=True
    )
    return StreamingResponse(generate_data(model_output))


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
    args = parser.parse_args()

    try:
        # cosyvoice = CosyVoice2(args.model_dir, load_jit=False, load_trt=True, fp16=True)
        # 1、JIT编译对速度影响不大（开了之后前面几次推理会卡慢几秒，十次之后稳定？）
        # 2、tensorrt对推理加速明显（首次推理还是会慢0.3秒左右），首次加载trt要等个3分钟左右编译
        # 3、fp16会影响后续推理速度，开启后会快个0.3秒左右
        # 实测 3090 首包延迟 0.8 秒，后续延迟 0.55 秒，24G 显存占用 5G，tts前后的文本处理速度也比4060ti快，前端首包快了将近1秒
        # 实测 4060 Ti 首包延迟 1.15，后续延迟 0.85，16G 显存占用 7.5G
        # 实测 4090 总体推理速度只比 3090 快个0.2秒，24G 显存占用 8G
        cosyvoice = CosyVoice2(args.model_dir, load_jit=False, load_trt=True, fp16=True)
        # cosyvoice = CosyVoice2(args.model_dir, load_jit=True, load_trt=False, fp16=True)
    except Exception:
        raise TypeError(f"导入{args.model_dir}失败，模型类型有误！")

    # 模型预热，不然初次加载时会卡
    prompt_text = "现在正在去酒店准备化妆的路上，今天刚刚从合肥到北京，大家好吗？"
    prompt_speech_16k = liuyifei_prompt_speech_16k

    # for _ in range(30):
    #     for i in cosyvoice.inference_zero_shot(
    #             '老公，想我了没？人家晚上很寂寞的呢！',
    #             # '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
    #             prompt_text, prompt_speech_16k, stream=True):
    #         pass

    # '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
    for t in ['收到好友从远方寄来的生日礼物，', '那份意外的惊喜与深深的祝福', '让我心中充满了甜蜜的快乐，', '笑容如花儿般绽放。']:
        for _ in cosyvoice.inference_zero_shot(
                t, prompt_text, prompt_speech_16k, stream=True):
            pass
    print("预热完毕")

    uvicorn.run(app, host="0.0.0.0", port=args.port)
