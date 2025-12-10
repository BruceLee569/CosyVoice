import os
import time
import sys
import argparse
import logging
import re
import glob

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

# 添加 Matcha-TTS 路径
sys.path.append("third_party/Matcha-TTS")

# TensorRT-LLM 相关
try:
    import tensorrt_llm
    from tensorrt_llm.runtime import ModelRunnerCpp
    from transformers import AutoTokenizer
    TRTLLM_AVAILABLE = True
except ImportError as e:
    TRTLLM_AVAILABLE = False
    logging.warning(f"TensorRT-LLM 不可用: {e}. 将使用原始 PyTorch 推理")

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import threading
import uuid as uuid_module

app = FastAPI()

# 添加请求计时中间件（必须在 CORS 之前，确保最早开始计时）
class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 在请求刚到达时立即开始计时
        request.state.start_time = time.time()
        response = await call_next(request)
        return response

app.add_middleware(TimingMiddleware)

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


# Chat template for TensorRT-LLM (CosyVoice2)
TRTLLM_CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if message['role'] == 'user' %}"
    "{{- '<|sos|>' + message['content'] + '<|task_id|>' }}"
    "{%- elif message['role'] == 'assistant' %}"
    "{{- message['content']}}"
    "{%- endif %}"
    "{%- endfor %}"
)


def convert_speech_tokens_to_str(speech_tokens):
    """将 speech token IDs 转换为 <|s_XXXXX|> 格式的字符串"""
    if isinstance(speech_tokens, torch.Tensor):
        speech_tokens = speech_tokens.flatten().tolist()
    return ''.join([f"<|s_{token}|>" for token in speech_tokens])


def extract_speech_ids_from_str(speech_tokens_str_list):
    """从 <|s_XXXXX|> 格式的字符串列表中提取 speech token IDs"""
    speech_ids = []
    for token_str in speech_tokens_str_list:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            try:
                num_str = token_str[4:-2]
                num = int(num_str)
                speech_ids.append(num)
            except ValueError:
                logging.warning(f"无法解析 speech token: {token_str}")
        # 忽略其他 token（如 <|eos1|> 等）
    return speech_ids


class FastCosyVoice2:
    """集成 TensorRT-LLM 的 CosyVoice2 推理类"""
    
    def __init__(self, cosyvoice_model, trtllm_engine_dir=None, trtllm_tokenizer_dir=None, use_trtllm=True):
        self.cosyvoice = cosyvoice_model
        self.use_trtllm = use_trtllm and TRTLLM_AVAILABLE
        self.trtllm_runner = None
        self.trtllm_tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 存储原始的 prompt_text 字符串，用于 TensorRT-LLM
        self.spk_prompt_text_raw = {}
        
        if self.use_trtllm:
            try:
                self._init_trtllm(trtllm_engine_dir, trtllm_tokenizer_dir)
                logging.info("✅ TensorRT-LLM 初始化成功")
            except Exception as e:
                logging.error(f"TensorRT-LLM 初始化失败: {e}. 回退到 PyTorch 推理")
                self.use_trtllm = False
    
    def _init_trtllm(self, engine_dir, tokenizer_dir):
        """初始化 TensorRT-LLM 引擎"""
        if not engine_dir or not os.path.exists(engine_dir):
            raise ValueError(f"TensorRT-LLM 引擎目录不存在: {engine_dir}")
        
        runtime_rank = tensorrt_llm.mpi_rank()
        
        # 初始化 tokenizer
        self.trtllm_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        
        # 设置正确的 chat template
        if 'system' in self.trtllm_tokenizer.chat_template:
            self.trtllm_tokenizer.chat_template = TRTLLM_CHAT_TEMPLATE
            logging.info("已设置 CosyVoice2 专用 chat template")
        
        # EOS token ID
        self.eos_token_id = self.trtllm_tokenizer.convert_tokens_to_ids("<|eos1|>")
        
        # 初始化 TensorRT-LLM ModelRunner
        runner_kwargs = dict(
            engine_dir=engine_dir,
            rank=runtime_rank,
            max_output_len=2048,
            enable_context_fmha_fp32_acc=False,
            max_batch_size=1,
            max_input_len=512,
            kv_cache_free_gpu_memory_fraction=0.6,
            cuda_graph_mode=False,
            gather_generation_logits=False,
        )
        self.trtllm_runner = ModelRunnerCpp.from_dir(**runner_kwargs)
        logging.info(f"TensorRT-LLM 引擎已加载: {engine_dir}")
    
    def _prepare_llm_input(self, tts_text, prompt_text, prompt_speech_tokens):
        """准备 LLM 输入（使用 chat template）
        
        Args:
            tts_text: 要合成的目标文本（原始字符串）
            prompt_text: 提示文本（原始字符串）
            prompt_speech_tokens: 提示语音的 speech token IDs（tensor 或 list）
        
        Returns:
            input_ids: tokenized 后的输入 tensor
        """
        # 1. 拼接完整文本：prompt_text + tts_text
        full_text = prompt_text + tts_text
        
        # 2. 将 prompt_speech_tokens 转换为 <|s_XXXXX|> 格式字符串
        prompt_speech_str = convert_speech_tokens_to_str(prompt_speech_tokens)
        
        # 3. 构建 chat 格式
        chat = [
            {"role": "user", "content": full_text},
            {"role": "assistant", "content": prompt_speech_str}
        ]
        
        # 4. 使用 chat template 进行 tokenization
        input_ids = self.trtllm_tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            return_tensors='pt',
            continue_final_message=True  # 继续生成 assistant 的回复
        )
        
        return input_ids
    
    def _trtllm_generate_streaming(self, input_ids):
        """使用 TensorRT-LLM 流式生成 speech tokens
        
        Args:
            input_ids: tokenized 输入 tensor
        
        Yields:
            Tuple[List[int], bool]: (当前累积的 speech_ids, 是否完成)
        """
        try:
            input_length = input_ids.shape[1]
            
            # TensorRT-LLM 流式生成
            outputs_iter = self.trtllm_runner.generate(
                batch_input_ids=[input_ids[0]],
                max_new_tokens=2048,
                end_id=self.eos_token_id,
                pad_id=self.eos_token_id,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.1,
                streaming=True,  # 启用流式生成
                output_sequence_lengths=True,
                return_dict=True,
            )
            
            # 迭代流式输出
            for outputs in outputs_iter:
                torch.cuda.synchronize()
                
                # 提取生成的 token IDs
                output_ids = outputs["output_ids"]
                sequence_lengths = outputs["sequence_lengths"]
                
                # 获取实际生成的部分（排除输入）
                actual_length = sequence_lengths[0][0].item()
                generated_ids = output_ids[0][0][input_length:actual_length].tolist()
                
                # 将 token IDs 解码为字符串，然后解析 <|s_XXXXX|> 格式
                generated_tokens_str = self.trtllm_tokenizer.batch_decode(
                    [[tid] for tid in generated_ids],
                    skip_special_tokens=False
                )
                
                # 提取真实的 speech token IDs
                speech_ids = extract_speech_ids_from_str(generated_tokens_str)
                
                # 检查是否是最后一个响应
                is_final = outputs.get("finished", False)
                if isinstance(is_final, torch.Tensor):
                    is_final = is_final.item()
                
                yield speech_ids, is_final
                
                if is_final:
                    break
            
        except Exception as e:
            logging.error(f"TensorRT-LLM 流式生成失败: {e}")
            raise
    
    def _trtllm_generate(self, input_ids):
        """使用 TensorRT-LLM 生成 speech tokens（非流式，用于回退）
        
        Args:
            input_ids: tokenized 输入 tensor
        
        Returns:
            speech_ids: 生成的 speech token IDs 列表
        """
        try:
            input_length = input_ids.shape[1]
            
            # TensorRT-LLM 生成
            outputs = self.trtllm_runner.generate(
                batch_input_ids=[input_ids[0]],
                max_new_tokens=2048,
                end_id=self.eos_token_id,
                pad_id=self.eos_token_id,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.1,
                streaming=False,
                output_sequence_lengths=True,
                return_dict=True,
            )
            
            torch.cuda.synchronize()
            
            # 提取生成的 token IDs
            output_ids = outputs["output_ids"]
            sequence_lengths = outputs["sequence_lengths"]
            
            # 获取实际生成的部分（排除输入）
            actual_length = sequence_lengths[0][0].item()
            generated_ids = output_ids[0][0][input_length:actual_length].tolist()
            
            # 将 token IDs 解码为字符串，然后解析 <|s_XXXXX|> 格式
            generated_tokens_str = self.trtllm_tokenizer.batch_decode(
                [[tid] for tid in generated_ids],
                skip_special_tokens=False
            )
            
            # 提取真实的 speech token IDs
            speech_ids = extract_speech_ids_from_str(generated_tokens_str)
            
            logging.info(f"TensorRT-LLM 生成了 {len(speech_ids)} 个 speech tokens")
            return speech_ids
            
        except Exception as e:
            logging.error(f"TensorRT-LLM 生成失败: {e}")
            raise
    
    def inference_zero_shot(self, text, prompt_text, prompt_speech_16k, zero_shot_spk_id='', stream=True):
        """零样本推理（集成 TensorRT-LLM 流式生成）"""
        if not self.use_trtllm:
            # 回退到原始 CosyVoice2 推理
            for output in self.cosyvoice.inference_zero_shot(
                text, prompt_text, prompt_speech_16k, 
                zero_shot_spk_id=zero_shot_spk_id, 
                stream=stream
            ):
                yield output
            return
        
        # 使用 TensorRT-LLM 加速的推理流程（流式生成 + 流式 token2wav）
        try:
            # 1. 获取 speaker info（包含 prompt_speech_tokens）
            spk_info = self.cosyvoice.frontend.spk2info.get(zero_shot_spk_id)
            if spk_info is None:
                raise ValueError(f"Speaker {zero_shot_spk_id} 不存在")
            
            # 2. 获取原始的 prompt_text 字符串
            prompt_text_raw = self.spk_prompt_text_raw.get(zero_shot_spk_id, '')
            if not prompt_text_raw:
                logging.warning(f"Speaker {zero_shot_spk_id} 缺少原始 prompt_text，回退到 PyTorch 推理")
                raise ValueError("缺少原始 prompt_text")
            
            # 3. 获取 spk_info 中的数据
            llm_prompt_speech_token = spk_info['llm_prompt_speech_token']
            flow_prompt_speech_token = spk_info['flow_prompt_speech_token']
            prompt_speech_feat = spk_info['prompt_speech_feat']
            flow_embedding = spk_info['flow_embedding']
            
            # 4. 准备 LLM 输入（使用原始字符串 + chat template）
            input_ids = self._prepare_llm_input(text, prompt_text_raw, llm_prompt_speech_token)
            
            logging.info(f"[FastTTS] TensorRT-LLM 输入长度: {input_ids.shape[1]} tokens")
            
            # 5. 初始化流式参数
            this_uuid = str(uuid_module.uuid1())
            model = self.cosyvoice.model
            model.hift_cache_dict[this_uuid] = None
            
            token_hop_len = 25  # 每块处理的 token 数
            pre_lookahead_len = model.flow.pre_lookahead_len  # 前瞻长度 (3)
            
            # 计算 prompt_token_pad（对齐到 token_hop_len 的倍数）
            prompt_token_pad = int(np.ceil(flow_prompt_speech_token.shape[1] / token_hop_len) * token_hop_len - flow_prompt_speech_token.shape[1])
            
            token_offset = 0
            chunk_idx = 0
            first_chunk_tokens_needed = token_hop_len + prompt_token_pad + pre_lookahead_len  # ~28
            
            logging.info(f"[流式生成] 开始流式生成+token2wav: first_chunk_needed={first_chunk_tokens_needed}")
            
            # 6. 流式生成 + 流式 token2wav
            speech_tokens = []
            generation_done = False
            
            for current_tokens, is_final in self._trtllm_generate_streaming(input_ids):
                speech_tokens = current_tokens
                generation_done = is_final
                
                # 检查是否有足够的 tokens 生成第一块音频
                while True:
                    this_token_hop_len = token_hop_len + prompt_token_pad if token_offset == 0 else token_hop_len
                    tokens_needed = token_offset + this_token_hop_len + pre_lookahead_len
                    
                    if tokens_needed <= len(speech_tokens):
                        # 有足够的 tokens，生成一块音频
                        chunk_start_time = time.time()
                        
                        this_tts_speech_token = torch.tensor(
                            speech_tokens[:tokens_needed]
                        ).unsqueeze(0)
                        
                        tts_speech = model.token2wav(
                            token=this_tts_speech_token,
                            prompt_token=flow_prompt_speech_token,
                            prompt_feat=prompt_speech_feat,
                            embedding=flow_embedding,
                            token_offset=token_offset,
                            uuid=this_uuid,
                            stream=True,
                            finalize=False,
                            speed=1.0
                        )
                        
                        chunk_time = (time.time() - chunk_start_time) * 1000
                        logging.info(f"[流式token2wav] 块{chunk_idx}: offset={token_offset}, tokens_in={tokens_needed}, total_generated={len(speech_tokens)}, 耗时={chunk_time:.1f}ms")
                        
                        token_offset += this_token_hop_len
                        chunk_idx += 1
                        yield {'tts_speech': tts_speech.cpu()}
                    else:
                        # 不够 tokens，等待更多生成
                        break
                
                if generation_done:
                    break
            
            # 7. 处理剩余的 tokens（最后一块）
            if token_offset < len(speech_tokens):
                chunk_start_time = time.time()
                
                this_tts_speech_token = torch.tensor(speech_tokens).unsqueeze(0)
                
                tts_speech = model.token2wav(
                    token=this_tts_speech_token,
                    prompt_token=flow_prompt_speech_token,
                    prompt_feat=prompt_speech_feat,
                    embedding=flow_embedding,
                    token_offset=token_offset,
                    uuid=this_uuid,
                    stream=True,
                    finalize=True,
                    speed=1.0
                )
                
                chunk_time = (time.time() - chunk_start_time) * 1000
                logging.info(f"[流式token2wav] 最终块{chunk_idx}: offset={token_offset}, tokens_in={len(speech_tokens)}, 耗时={chunk_time:.1f}ms (finalize)")
                yield {'tts_speech': tts_speech.cpu()}
            
            logging.info(f"[FastTTS] 流式生成完成: 共 {len(speech_tokens)} 个 speech tokens, {chunk_idx + 1} 个音频块")
            
            # 清理缓存
            if this_uuid in model.hift_cache_dict:
                model.hift_cache_dict.pop(this_uuid)
            
        except Exception as e:
            logging.error(f"TensorRT-LLM 推理失败: {e}，回退到 PyTorch 推理")
            import traceback
            traceback.print_exc()
            # 回退到原始推理
            for output in self.cosyvoice.inference_zero_shot(
                text, prompt_text, prompt_speech_16k,
                zero_shot_spk_id=zero_shot_spk_id,
                stream=stream
            ):
                yield output
    
    def list_available_spks(self):
        """获取可用说话人列表"""
        return self.cosyvoice.list_available_spks()
    
    def add_zero_shot_spk(self, prompt_text, prompt_speech_16k, zero_shot_spk_id):
        """添加零样本说话人"""
        # 保存原始的 prompt_text 字符串（用于 TensorRT-LLM）
        self.spk_prompt_text_raw[zero_shot_spk_id] = prompt_text
        return self.cosyvoice.add_zero_shot_spk(prompt_text, prompt_speech_16k, zero_shot_spk_id)
    
    def save_spkinfo(self):
        """保存说话人信息"""
        return self.cosyvoice.save_spkinfo()


def generate_data(model_output, request_start_time):
    """生成音频数据流，对输出进行削波处理防止爆音"""
    is_first = True
    chunk_count = 0
    
    for i in model_output:
        if is_first:
            first_chunk_time = time.time()
            ttfb = (first_chunk_time - request_start_time) * 1000
            logging.info(f"[TTS统计] 首包生成完毕! 服务端TTFB: {ttfb:.2f}ms")
            is_first = False

        tts_speech = i["tts_speech"].numpy()
        
        # 2. 输出端削波：防止 float -> int16 转换时的整数溢出
        tts_speech = np.clip(tts_speech, -1.0, 1.0)
        
        # 转换为 int16 格式
        tts_audio = (tts_speech * 32767).astype(np.int16).tobytes()
        chunk_count += 1
        yield tts_audio
    
    total_time = (time.time() - request_start_time) * 1000
    logging.info(f"[TTS统计] 流式传输结束. 总耗时: {total_time:.2f}ms, 共发送 {chunk_count} 个数据块")


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
                
                # 输入端归一化
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
    return {"message": "FastCosyVoice TTS Server (TensorRT-LLM Accelerated) is running. Visit /static/index.html for the web interface."}


@app.get("/api/speakers")
async def get_speakers():
    """获取所有可用的说话人列表"""
    try:
        speakers = fast_cosyvoice.list_available_spks()
        return JSONResponse(content={"speakers": speakers})
    except Exception as e:
        logging.error(f"获取说话人列表失败: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/tts")
async def inference_zero_shot(request: Request, text: str = Form(), speaker: str = Form(default="")):
    """文本转语音接口（集成 TensorRT-LLM 加速）"""
    # 使用中间件记录的开始时间，确保与前端对齐
    request_start_time = request.state.start_time
    logging.info(f"[FastTTS请求] 收到请求: text='{text[:50] if len(text) > 50 else text}', speaker='{speaker}'")

    try:
        # 默认使用jok老师，如果没有则使用第一个说话人
        default_speaker = "jok老师" if "jok老师" in speakers_data else (list(speakers_data.keys())[0] if speakers_data else "")
        selected_speaker = speaker if speaker else default_speaker
        
        if not selected_speaker:
            return JSONResponse(content={"error": "没有可用的说话人"}, status_code=400)
        
        logging.info(f"[FastTTS推理] 开始推理, 说话人: {selected_speaker}")

        # 使用 FastCosyVoice2 进行推理（自动选择 TensorRT-LLM 或 PyTorch）
        model_output = fast_cosyvoice.inference_zero_shot(
            text, "", None, 
            zero_shot_spk_id=selected_speaker,
            stream=True
        )
        return StreamingResponse(generate_data(model_output, request_start_time))
    except Exception as e:
        logging.error(f"FastTTS推理失败: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=6008, help="服务端口")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/CosyVoice2-0.5B",
        help="模型本地路径或 modelscope 仓库 id",
    )
    parser.add_argument(
        "--trtllm_engine_dir",
        type=str,
        default="runtime/triton_trtllm/trt_engines_bfloat16",
        help="TensorRT-LLM 引擎目录",
    )
    parser.add_argument(
        "--trtllm_tokenizer_dir",
        type=str,
        default="runtime/triton_trtllm/cosyvoice2_llm",
        help="TensorRT-LLM tokenizer 目录",
    )
    parser.add_argument(
        "--speaker_dir",
        type=str,
        default="asset/speakers",
        help="说话人音频文件目录",
    )
    args = parser.parse_args()

    try:
        # 初始化基础 CosyVoice2 模型
        logging.info("=" * 60)
        logging.info("启动 FastCosyVoice TTS Server")
        logging.info(f"模型目录: {args.model_dir}")
        logging.info(f"TensorRT-LLM 引擎目录: {args.trtllm_engine_dir}")
        logging.info(f"TensorRT-LLM Tokenizer: {args.trtllm_tokenizer_dir}")
        logging.info("=" * 60)
        
        # 加载原始 CosyVoice2 模型
        cosyvoice = CosyVoice2(args.model_dir, load_jit=True, load_trt=True, fp16=True)
        
        # 创建 FastCosyVoice2 实例（集成 TensorRT-LLM）
        fast_cosyvoice = FastCosyVoice2(
            cosyvoice_model=cosyvoice,
            trtllm_engine_dir=args.trtllm_engine_dir,
            trtllm_tokenizer_dir=args.trtllm_tokenizer_dir,
        )
        
        logging.info("✅ FastCosyVoice2 初始化成功，使用 TensorRT-LLM 加速")

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
                fast_cosyvoice.add_zero_shot_spk(
                    speaker_info['prompt_text'],
                    speaker_info['prompt_speech_16k'],
                    speaker_name
                )
                print(f"  ✓ {speaker_name}")
            except Exception as e:
                print(f"  ✗ {speaker_name}: {e}")
        
        # 保存说话人信息
        try:
            fast_cosyvoice.save_spkinfo()
            print("说话人信息已保存")
        except Exception as e:
            print(f"保存说话人信息失败: {e}")
    
    # 模型预热
    print("\n正在预热模型...")
    if speakers_data:
        warmup_speaker = "jok老师" if "jok老师" in speakers_data else list(speakers_data.keys())[0]
        print(f"使用 '{warmup_speaker}' 进行预热")
        warmup_texts = [
            '收到好友从远方寄来的生日礼物，', 
        ]
        
        for t in warmup_texts:
            try:
                for _ in fast_cosyvoice.inference_zero_shot(
                        t, "", None, zero_shot_spk_id=warmup_speaker, stream=True):
                    pass
            except Exception as e:
                print(f"预热失败: {e}")
                break
    
    print("预热完毕\n")
    print("=" * 60)
    print(f"🚀 FastCosyVoice TTS Server 启动在端口 {args.port}")

    # 配置 uvicorn 以支持 HTTP keep-alive
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=args.port,
        timeout_keep_alive=60,  # keep-alive 超时时间（秒）
        limit_concurrency=100,  # 最大并发连接数
        backlog=2048,  # TCP backlog 队列大小
    )
