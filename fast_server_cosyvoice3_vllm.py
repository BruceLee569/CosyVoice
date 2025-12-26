import os
import time
import sys
import argparse
import logging
import re
import glob
import json
from functools import partial
import inflect

# 配置日志格式，确保显示毫秒级时间戳
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

from fastapi import FastAPI, Form, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn
import numpy as np
import torch
import torchaudio

# 添加 Matcha-TTS 路径
sys.path.append("third_party/Matcha-TTS")

# Monkey patch load_wav 函数（必须在导入 CosyVoice 之前）
import cosyvoice.utils.file_utils

def patched_load_wav(wav, target_sr, min_sr=16000):
    """使用 soundfile 替代 torchaudio.load 以兼容 PyTorch 2.9.x"""
    import soundfile as sf
    speech, sample_rate = sf.read(wav, dtype='float32')
    # soundfile返回 (samples,) 或 (samples, channels)，转为 (channels, samples)
    if len(speech.shape) == 1:
        speech = torch.from_numpy(speech).unsqueeze(0)  # (samples,) -> (1, samples)
    else:
        speech = torch.from_numpy(speech).T  # (samples, channels) -> (channels, samples)
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate >= min_sr, f'wav sample rate {sample_rate} must be greater than {target_sr}'
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech

# 应用 patch（在导入其他模块前）
cosyvoice.utils.file_utils.load_wav = patched_load_wav

# 导入 vLLM 和 CosyVoice
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice3 import CosyVoice3ForCausalLM
from cosyvoice.utils.frontend_utils import (
    contains_chinese,
    replace_blank,
    replace_corner_mark,
    remove_bracket,
    spell_out_number,
    split_paragraph,
    is_only_punctuation,
)
import uuid as uuid_module

# 注册 CosyVoice3 模型到 vLLM
ModelRegistry.register_model("CosyVoice3ForCausalLM", CosyVoice3ForCausalLM)

# 导入 ttsfrd 模块（用于文本规范化）
try:
    import ttsfrd
    USE_TTSFRD = True
    logging.info("已导入 ttsfrd 模块用于文本规范化")
except ImportError:
    USE_TTSFRD = False
    logging.warning("ttsfrd 不可用，将使用 wetext 进行文本规范化")
    try:
        from wetext import Normalizer as ZhNormalizer
        from wetext import Normalizer as EnNormalizer
    except ImportError:
        logging.warning("wetext 也不可用，文本规范化功能将受限")

app = FastAPI()


class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request.state.start_time = time.perf_counter()
        response = await call_next(request)
        return response


app.add_middleware(TimingMiddleware)
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


class TextNormalizer:
    """文本规范化工具类，复用 CosyVoice frontend 的成熟实现"""

    def __init__(self, tokenizer, use_ttsfrd=True):
        self.tokenizer = tokenizer
        self.use_ttsfrd = use_ttsfrd
        if self.use_ttsfrd:
            self.frd = ttsfrd.TtsFrontendEngine()
            resource_dir = os.path.join(
                os.path.dirname(__file__), "pretrained_models/CosyVoice-ttsfrd/resource"
            )
            if os.path.exists(resource_dir):
                if not self.frd.initialize(resource_dir):
                    logging.warning("ttsfrd 初始化失败，将使用 wetext")
                    self.use_ttsfrd = False
                    self._init_wetext()
                else:
                    self.frd.set_lang_type("pinyinvg")
                    logging.info("✅ ttsfrd 初始化成功")
            else:
                logging.warning(
                    f"ttsfrd 资源目录不存在: {resource_dir}，将使用 wetext"
                )
                self.use_ttsfrd = False
                self._init_wetext()
        else:
            self._init_wetext()

    def _init_wetext(self):
        if "ZhNormalizer" in globals():
            self.zh_tn_model = ZhNormalizer(remove_erhua=False)
            self.en_tn_model = EnNormalizer()
            self.inflect_parser = inflect.engine()
            logging.info("使用 wetext 作为文本规范化工具")
        else:
            logging.warning("wetext 不可用，将跳过文本规范化")

    def normalize_and_split(self, text, token_max_n=80, token_min_n=60):
        if not text or text.strip() == "":
            return []
        text = text.strip()
        if self.use_ttsfrd:
            try:
                result = self.frd.do_voicegen_frd(text)
                texts = [i["text"] for i in json.loads(result)["sentences"]]
                text = "".join(texts)
            except Exception as e:
                logging.warning(f"ttsfrd 处理失败: {e}，使用原始文本")
        else:
            if "zh_tn_model" in dir(self):
                if contains_chinese(text):
                    text = self.zh_tn_model.normalize(text)
                    text = text.replace("\n", "")
                    text = replace_blank(text)
                    text = replace_corner_mark(text)
                    text = text.replace(".", "。")
                    text = text.replace(" - ", "，")
                    text = remove_bracket(text)
                    text = re.sub(r"[，,、]+$", "。", text)
                else:
                    text = self.en_tn_model.normalize(text)
                    text = spell_out_number(text, self.inflect_parser)

        tokenize_fn = partial(self.tokenizer.encode, allowed_special="all")
        if contains_chinese(text):
            texts = list(
                split_paragraph(
                    text,
                    tokenize_fn,
                    "zh",
                    token_max_n=token_max_n,
                    token_min_n=token_min_n,
                    merge_len=20,
                    comma_split=False,
                )
            )
        else:
            texts = list(
                split_paragraph(
                    text,
                    tokenize_fn,
                    "en",
                    token_max_n=token_max_n,
                    token_min_n=token_min_n,
                    merge_len=20,
                    comma_split=False,
                )
            )
        texts = [i for i in texts if not is_only_punctuation(i)]
        if texts:
            logging.info(
                f"[文本规范化] 原始长度: {len(text)} 字符 → 分成 {len(texts)} 段"
            )
            for idx, seg in enumerate(texts):
                logging.info(
                    f"  段{idx+1}: {len(seg)}字符 - '"
                    f"{seg[:30]}{'...' if len(seg) > 30 else ''}'"
                )
        return texts


def convert_speech_tokens_to_str(speech_tokens):
    if isinstance(speech_tokens, torch.Tensor):
        speech_tokens = speech_tokens.flatten().tolist()
    return "".join([f"<|s_{token}|>" for token in speech_tokens])


def extract_speech_ids_from_str(speech_tokens_str_list):
    speech_ids = []
    for token_str in speech_tokens_str_list:
        if token_str.startswith("<|s_") and token_str.endswith("|>"):
            try:
                num_str = token_str[4:-2]
                num = int(num_str)
                speech_ids.append(num)
            except ValueError:
                logging.warning(f"无法解析 speech token: {token_str}")
    return speech_ids


class FastCosyVoice3VLLM:
    """集成 vLLM 的 CosyVoice3 推理类"""

    def __init__(self, cosyvoice_model, vllm_tokenizer):
        """
        Args:
            cosyvoice_model: 使用 AutoModel(load_vllm=True) 加载的模型
            vllm_tokenizer: vLLM 使用的 tokenizer（用于文本规范化）
        """
        self.cosyvoice = cosyvoice_model
        self.vllm_tokenizer = vllm_tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_normalizer = TextNormalizer(
            tokenizer=self.vllm_tokenizer, use_ttsfrd=USE_TTSFRD
        )
        logging.info("✅ FastCosyVoice3VLLM 初始化成功")

    def inference_zero_shot(self, text, prompt_text, prompt_speech_16k, zero_shot_spk_id="", stream=True, request_start_time=None):
        """
        使用 vLLM 加速的 zero-shot 推理
            
        注意：此方法直接使用 cosyvoice.inference_zero_shot，其内部已经集成了 vLLM
        关键修复：当使用 zero_shot_spk_id 时，必须将 prompt_text 和 prompt_speech_16k 设置为空/None
        """
        if request_start_time is None:
            request_start_time = time.perf_counter()
            
        text_segments = self.text_normalizer.normalize_and_split(
            text, token_max_n=80, token_min_n=60
        )
        if len(text_segments) == 0:
            logging.warning("[文本分段] 输入文本为空，跳过推理")
            return
            
        logging.info(
            f"[长文本处理] 原始文本 {len(text)} 字符 → 分成 {len(text_segments)} 段进行流式推理"
        )
            
        for segment_idx, text_segment in enumerate(text_segments):
            segment_start_time = time.perf_counter()
            logging.info(
                f"[段落推理 {segment_idx+1}/{len(text_segments)}] 开始处理: '"
                f"{text_segment[:50]}{'...' if len(text_segment) > 50 else ''}"
            )
                
            # 🔥 关键修复：按照 test_streaming_mvp.py 的逻辑，使用 zero_shot_spk_id 时必须清空 prompt 参数
            # 因为说话人特征已经通过 add_zero_shot_spk 注册，再传入 prompt 会导致音频异常
            for output in self.cosyvoice.inference_zero_shot(
                text_segment,
                '',  # 使用空字符串而不是原始 prompt_text
                None,  # 使用 None 而不是原始 prompt_speech_16k
                zero_shot_spk_id=zero_shot_spk_id,
                stream=stream
            ):
                yield output
                
            segment_time = (time.perf_counter() - segment_start_time) * 1000
            logging.info(
                f"[段落推理 {segment_idx+1}/{len(text_segments)}] 完成，耗时: {segment_time:.2f}ms"
            )

    def list_available_spks(self):
        return self.cosyvoice.list_available_spks()

    def add_zero_shot_spk(self, prompt_text, prompt_speech_16k, zero_shot_spk_id):
        return self.cosyvoice.add_zero_shot_spk(
            prompt_text, prompt_speech_16k, zero_shot_spk_id
        )


def generate_data(model_output, request_start_time):
    is_first = True
    chunk_count = 0
    for i in model_output:
        if is_first:
            first_chunk_time = time.perf_counter()
            ttfb = (first_chunk_time - request_start_time) * 1000
            logging.info(
                f"[TTS统计] HTTP响应首包生成完毕! HTTP TTFB: {ttfb:.2f}ms"
            )
            is_first = False
        tts_speech = i["tts_speech"].numpy()
        tts_speech = np.clip(tts_speech, -1.0, 1.0)
        tts_audio = (tts_speech * 32767.0).astype(np.int16).tobytes()
        chunk_count += 1
        yield tts_audio
    total_time = (time.perf_counter() - request_start_time) * 1000
    logging.info(
        f"[TTS统计] 流式传输结束. 总耗时: {total_time:.2f}ms, 共发送 {chunk_count} 个数据块"
    )


def load_speakers_from_directory(speaker_dir="asset/speakers"):
    speakers = {}
    if not os.path.exists(speaker_dir):
        logging.warning(f"说话人目录 {speaker_dir} 不存在")
        return speakers
    wav_files = glob.glob(os.path.join(speaker_dir, "*.wav"))
    for wav_path in wav_files:
        filename = os.path.basename(wav_path)
        match = re.match(r"\[(.+?)\](.+)\.wav$", filename)
        if match:
            speaker_name = match.group(1)
            prompt_text_raw = match.group(2)
            # 🔥 关键修复：为 CosyVoice3 添加系统提示词前缀
            prompt_text = f"You are a helpful assistant.<|endofprompt|>{prompt_text_raw}"
            try:
                prompt_speech_16k = patched_load_wav(wav_path, 16000)
                max_val = torch.abs(prompt_speech_16k).max().item()
                target_peak = 0.95
                if max_val > target_peak:
                    logging.warning(
                        f"说话人 {speaker_name} 音频峰值 {max_val:.4f} 超出安全范围，归一化到 {target_peak}"
                    )
                    prompt_speech_16k = (prompt_speech_16k / max_val) * target_peak
                speakers[speaker_name] = {
                    "prompt_text": prompt_text,
                    "prompt_speech_16k": prompt_speech_16k,
                    "wav_path": wav_path,
                }
                logging.info(f"加载说话人: {speaker_name} (prompt_text='{prompt_text}')")
            except Exception as e:
                logging.error(f"加载说话人 {speaker_name} 失败: {e}")
        else:
            logging.warning(f"文件名格式不正确，跳过: {filename}")
    return speakers



@app.get("/")
async def index():
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": "FastCosyVoice3 vLLM TTS Server is running. Visit /static/index.html for the web interface.",
    }


@app.get("/api/speakers")
async def get_speakers():
    try:
        speakers = fast_cosyvoice.list_available_spks()
        return JSONResponse(content={"speakers": speakers})
    except Exception as e:
        logging.error(f"获取说话人列表失败: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/tts")
async def inference_zero_shot(request: Request, text: str = Form(), speaker: str = Form(default="")):
    request_start_time = request.state.start_time
    logging.info(
        f"[FastTTS请求] 收到请求: text='{text[:50] if len(text) > 50 else text}', speaker='{speaker}'"
    )
    try:
        available_spks = fast_cosyvoice.list_available_spks()
        default_speaker = (
            "jok老师" if "jok老师" in available_spks else (available_spks[0] if available_spks else "")
        )
        selected_speaker = speaker if speaker else default_speaker
        if not selected_speaker:
            return JSONResponse(content={"error": "没有可用的说话人"}, status_code=400)
        logging.info(f"[FastTTS推理] 开始推理, 说话人: {selected_speaker}")
        model_output = fast_cosyvoice.inference_zero_shot(
            text,
            "",
            None,
            zero_shot_spk_id=selected_speaker,
            stream=True,
            request_start_time=request_start_time,
        )
        return StreamingResponse(generate_data(model_output, request_start_time))
    except Exception as e:
        logging.error(f"FastTTS推理失败: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50002, help="服务端口")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/Fun-CosyVoice3-0.5B",
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
        logging.info("=" * 60)
        logging.info("启动 FastCosyVoice3 vLLM TTS Server")
        logging.info(f"模型目录: {args.model_dir}")
        logging.info(f"vLLM 版本: 0.12.0 (适配 PyTorch 2.9.x + RTX 50 系列)")
        logging.info("=" * 60)

        # 导入 AutoModel
        from cosyvoice.cli.cosyvoice import AutoModel
        
        # 加载 CosyVoice3 模型（启用 vLLM）
        logging.info("开始加载 CosyVoice3 模型（vLLM 0.12.0 + PyTorch 2.9.1）...")
        cosyvoice = AutoModel(
            model_dir=args.model_dir,
            load_trt=True,
            load_vllm=True,
            fp16=False
        )
        logging.info("✅ 模型加载配置: vLLM=True, TRT=True, FP16=False")

        # 创建 FastCosyVoice3VLLM 实例
        # 获取 vLLM tokenizer（用于文本规范化）
        if hasattr(cosyvoice, 'frontend') and hasattr(cosyvoice.frontend, 'tokenizer'):
            vllm_tokenizer = cosyvoice.frontend.tokenizer
        else:
            # 如果获取不到，使用 transformers AutoTokenizer
            from transformers import AutoTokenizer
            tokenizer_path = os.path.join(args.model_dir, "llm")
            vllm_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            logging.info(f"从 {tokenizer_path} 加载 tokenizer")
        
        fast_cosyvoice = FastCosyVoice3VLLM(
            cosyvoice_model=cosyvoice,
            vllm_tokenizer=vllm_tokenizer,
        )
        logging.info("✅ FastCosyVoice3VLLM 初始化成功，使用 vLLM 加速")

    except Exception as e:
        raise TypeError(f"导入{args.model_dir}失败，模型类型有误！错误: {e}")

    # 🔥 强制每次启动都重新提取特征（不使用 spk2info.pt 缓存）
    print("=" * 60)
    print("⚠️  禁用 spk2info.pt 缓存，每次启动都重新提取说话人特征")
    print("=" * 60)
    
    print("正在从 wav 文件提取说话人音频特征...")
    speakers_data = load_speakers_from_directory(args.speaker_dir)
    if not speakers_data:
        print(f"警告：未在 {args.speaker_dir} 目录找到任何说话人文件")
    else:
        print(f"成功加载 {len(speakers_data)} 个说话人")
        for speaker_name, speaker_info in speakers_data.items():
            try:
                # 使用 wav_path 而不是 tensor
                fast_cosyvoice.add_zero_shot_spk(
                    speaker_info["prompt_text"],
                    speaker_info["wav_path"],  # 传入文件路径
                    speaker_name,
                )
                print(f"  ✓ {speaker_name}")
            except Exception as e:
                print(f"  ✗ {speaker_name}: {e}")

    print("\n正在预热模型...")
    available_spks = fast_cosyvoice.list_available_spks()
    if available_spks:
        warmup_speaker = "jok老师" if "jok老师" in available_spks else available_spks[0]
        print(f"使用 '{warmup_speaker}' 进行预热")
        warmup_texts = [
            "你好。",
            "这是一个用于预热模型的测试句子，确保服务响应速度。",
        ]
        for t in warmup_texts:
            try:
                for _ in fast_cosyvoice.inference_zero_shot(
                    t, "", None, zero_shot_spk_id=warmup_speaker, stream=True
                ):
                    pass
            except Exception as e:
                print(f"预热失败: {e}")
                break
    else:
        print("未找到可用说话人，跳过预热")

    print("预热完毕\n")
    print("=" * 60)
    print(f"🚀 FastCosyVoice3 vLLM TTS Server 启动在端口 {args.port}")
    print(f"   使用 vLLM 0.12.0 加速 LLM 推理")
    print(f"   支持 RTX 50 系列 GPU (Blackwell sm_120)")
    print("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        timeout_keep_alive=60,
        limit_concurrency=100,
        backlog=2048,
    )
