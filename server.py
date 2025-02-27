import os
import pickle
import time
import sys
import argparse
import logging

import requests

logging.getLogger("matplotlib").setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse
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


def generate_data(model_output):
    for i in model_output:
        tts_audio = (i["tts_speech"].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio

def download_file_gitee(repo_file_path, local_file_path, repo="BruceLee569/public_resource"):
    # 下载音色文件到本地
    url = f"https://gitee.com/{repo}/raw/master/{repo_file_path}"  # 替换为实际的文件下载链接
    with requests.get(url, stream=True) as response:
        if response.status_code == 200:
            with open(local_file_path, "wb") as file:
                # 分块写入文件，避免内存占用过高
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:  # 如果有数据
                        file.write(chunk)
            print(f"文件已成功下载到 {local_file_path}")
        else:
            print(f"下载失败，状态码：{response.status_code}")


if not os.path.exists("asset/【刘亦菲2015可爱】现在正在去酒店准备化妆的路上，今天刚刚从合肥到北京，大家好吗？.wav"):
    download_file_gitee("Musics/【刘亦菲2015可爱】现在正在去酒店准备化妆的路上，今天刚刚从合肥到北京，大家好吗？.wav",
                        "asset/【刘亦菲2015可爱】现在正在去酒店准备化妆的路上，今天刚刚从合肥到北京，大家好吗？.wav")
if not os.path.exists("asset/刘亦菲.pt"):
    download_file_gitee("Musics/刘亦菲.pt",
                        "asset/刘亦菲.pt")

liuyifei_prompt_speech_16k = load_wav(
    "asset/【刘亦菲2015可爱】现在正在去酒店准备化妆的路上，今天刚刚从合肥到北京，大家好吗？.wav",
    16000,
)

# speaker_embedding = torch.load("asset/刘亦菲.pt")['embedding']


@app.get("/tts")
@app.post("/tts")
async def inference_zero_shot(text: str = Form(), speaker: str = Form()):
    prompt_text = "现在正在去酒店准备化妆的路上，今天刚刚从合肥到北京，大家好吗？"
    prompt_speech_16k = liuyifei_prompt_speech_16k
    model_output = cosyvoice.inference_zero_shot(
        text, prompt_text, prompt_speech_16k, stream=True
    )
    return StreamingResponse(generate_data(model_output))


# @app.get("/tts")
# @app.post("/tts")
# async def inference_zero_shot(text: str = Form(), speaker: str = Form(), ):
#     # text = "宝玉听了，喜跃非常，便忘了秦氏在何处，竟随了仙姑至一处所在。此处有石牌坊横建，上书“太虚幻境”四个大字，两边一副对联，乃是：假作真时真亦假，无为有处有还无。"
#     prompt_text = "现在正在去酒店准备化妆的路上，今天刚刚从合肥到北京，大家好吗？"
#
#     prompt_speech_16k = liuyifei_prompt_speech_16k
#
#     stream = True
#     speed = 1.0,
#     text_frontend = True
#
#     prompt_text = cosyvoice.frontend.text_normalize(
#         prompt_text, split=False, text_frontend=text_frontend
#     )
#     # print(prompt_text)
#
#     def gen_tts():
#         nonlocal prompt_speech_16k, stream, speed, text_frontend
#
#         for i in tqdm(
#                 cosyvoice.frontend.text_normalize(
#                     text, split=True, text_frontend=text_frontend
#                 )
#         ):
#             print(len(i), i)
#             print(len(prompt_text), prompt_text)
#             if (not isinstance(i, Generator)) and len(i) < 0.5 * len(prompt_text):
#                 logging.warning(
#                     "synthesis text {} too short than prompt text {}, this may lead to bad performance".format(
#                         i, prompt_text
#                     )
#                 )
#             model_input = cosyvoice.frontend.frontend_zero_shot(
#                 i, prompt_text, prompt_speech_16k, cosyvoice.sample_rate
#             )
#             start_time = time.time()
#             logging.info("synthesis text {}".format(i))
#             # logging.info(f"请求合成：{model_input}")
#             for model_output in cosyvoice.model.tts(
#                     **model_input, stream=stream, speed=speed
#             ):
#                 speech_len = model_output["tts_speech"].shape[1] / cosyvoice.sample_rate
#                 logging.info(
#                     "yield speech len {}, rtf {}".format(
#                         speech_len, (time.time() - start_time) / speech_len
#                     )
#                 )
#                 yield model_output
#                 start_time = time.time()
#
#     # model_output = cosyvoice.inference_zero_shot(
#     #     text, prompt_text, prompt_speech_16k, stream=True
#     # )
#     model_output = gen_tts()
#     return StreamingResponse(generate_data(model_output))

# with open("speaker_embedding.pkl", "rb") as f:
#     speaker_embedding = pickle.load(f)
#
#
# @app.get("/tts1")
# @app.post("/tts1")
# async def inference_zero_shot1(text: str = Form(), speaker: str = Form(), ):
#     stream = True
#     speed = 1.0,
#     text_frontend = True
#
#     def gen_tts():
#         for i in tqdm(
#                 cosyvoice.frontend.text_normalize(
#                     text, split=True, text_frontend=text_frontend
#                 )
#         ):
#             tts_text_token, tts_text_token_len = cosyvoice.frontend._extract_text_token(i)
#             model_input = {'text': tts_text_token, 'text_len': tts_text_token_len, 'llm_embedding': speaker_embedding,
#                            'flow_embedding': speaker_embedding}
#
#             start_time = time.time()
#             logging.info("synthesis text {}".format(i))
#             for model_output in cosyvoice.model.tts(
#                     **model_input, stream=stream, speed=speed
#             ):
#                 speech_len = model_output["tts_speech"].shape[1] / cosyvoice.sample_rate
#                 logging.info(
#                     "yield speech len {}, rtf {}".format(
#                         speech_len, (time.time() - start_time) / speech_len
#                     )
#                 )
#                 yield model_output
#                 start_time = time.time()
#
#     model_output = gen_tts()
#     return StreamingResponse(generate_data(model_output))

# speakers = torch.load("asset/刘亦菲.pt", map_location="cuda")
# speakers = torch.load("pretrained_models/CosyVoice2-0.5B/spk2info.pt", map_location="cuda")

# @app.get("/inference_sft")
# @app.post("/inference_sft")
# async def inference_sft(tts_text: str = Form()):
#
#     def gen_tts():
#         for i in tqdm(
#             cosyvoice.frontend.text_normalize(
#                 tts_text, split=True, text_frontend=True
#             )
#         ):
#             tts_text_token, tts_text_token_len = cosyvoice.frontend._extract_text_token(tts_text)
#
#             embedding = speakers["中文女"]['embedding']
#             # 加载自定义音色
#             # embedding = speaker_embedding
#             model_input = {'text': tts_text_token, 'text_len': tts_text_token_len, 'llm_embedding': embedding,
#                            'flow_embedding': embedding}
#
#             start_time = time.time()
#             logging.info("synthesis text {}".format(i))
#             for model_output in cosyvoice.model.tts(
#                 **model_input, stream=True, speed=1.0
#             ):
#                 speech_len = model_output["tts_speech"].shape[1] / cosyvoice.sample_rate
#                 logging.info(
#                     "yield speech len {}, rtf {}".format(
#                         speech_len, (time.time() - start_time) / speech_len
#                     )
#                 )
#                 yield model_output
#                 start_time = time.time()
#
#     return StreamingResponse(generate_data(gen_tts()))

@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    model_output = cosyvoice.inference_sft(tts_text, spk_id, stream=True)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(
        tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()
):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    print(111, prompt_wav.file)
    model_output = cosyvoice.inference_zero_shot(
        tts_text, prompt_text, prompt_speech_16k
    )
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(
        tts_text: str = Form(), prompt_wav: UploadFile = File()
):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(
        tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()
):
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(
        tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()
):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_instruct2(
        tts_text, instruct_text, prompt_speech_16k
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
        # cosyvoice = CosyVoice2(args.model_dir, load_jit=True, load_trt=True, fp16=True)
        # 1、JIT编译对速度影响不大（开了之后前面几次推理会卡慢几秒，十次之后稳定？）
        # 2、tensorrt对推理加速明显（首次推理还是会慢0.3秒左右），首次加载trt要等个3分钟左右编译
        # 3、fp16会影响后续推理速度，开启后会快个0.3秒左右
        # 实测 3090 首包延迟 0.8 秒，后续延迟 0.55 秒，24G 显存占用 5G，tts前后的文本处理速度也比4060ti快，前端首包快了将近1秒
        # 实测 4060 Ti 首包延迟 1.15，后续延迟 0.85，16G 显存占用 7.5G
        # 实测 4090 总体推理速度只比 3090 快个0.2秒，24G 显存占用 8G
        cosyvoice = CosyVoice2(args.model_dir, load_jit=False, load_trt=True, fp16=True)
    except Exception:
        raise TypeError(f"导入{args.model_dir}失败，模型类型有误！")

    # print(cosyvoice.frontend.spk2info)
    # spk2info = torch.load("pretrained_models/CosyVoice2-0.5B/spk2info.pt", map_location="cuda")
    # print(spk2info)
    # spk2info = torch.load("asset/刘亦菲.pt", map_location="cuda")
    # print(list(spk2info.keys()))

    # cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M')
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
    for t in ['收到好友从远方寄来的生日礼物，', '那份意外的惊喜与深深的祝福', '让我心中充满了甜蜜的快乐，', '笑容如花儿般绽放。', '老公想我了没？我想你想得都要发疯啦！']:
        for _ in cosyvoice.inference_zero_shot(
                t, prompt_text, prompt_speech_16k, stream=True):
            pass
    print("预热完毕")

    uvicorn.run(app, host="0.0.0.0", port=args.port)
