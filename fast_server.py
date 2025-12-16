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
from cosyvoice.utils.frontend_utils import (
    contains_chinese, replace_blank, replace_corner_mark, 
    remove_bracket, spell_out_number, split_paragraph, is_only_punctuation
)
import uuid as uuid_module

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

# 添加请求计时中间件（必须在 CORS 之前，确保最早开始计时）
class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 在请求刚到达时立即开始计时
        request.state.start_time = time.perf_counter()
        response = await call_next(request)
        return response

app.add_middleware(TimingMiddleware)

# 设置同源策略
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


class TextNormalizer:
    """文本规范化工具类，复用 CosyVoice frontend 的成熟实现"""
    
    def __init__(self, tokenizer, use_ttsfrd=True):
        self.tokenizer = tokenizer
        self.use_ttsfrd = use_ttsfrd
        
        if self.use_ttsfrd:
            self.frd = ttsfrd.TtsFrontendEngine()
            # 初始化 ttsfrd 资源（使用项目中的资源目录）
            resource_dir = os.path.join(os.path.dirname(__file__), 'pretrained_models/CosyVoice-ttsfrd/resource')
            if os.path.exists(resource_dir):
                if not self.frd.initialize(resource_dir):
                    logging.warning(f"ttsfrd 初始化失败，将使用 wetext")
                    self.use_ttsfrd = False
                    self._init_wetext()
                else:
                    self.frd.set_lang_type('pinyinvg')
                    logging.info("✅ ttsfrd 初始化成功")
            else:
                logging.warning(f"ttsfrd 资源目录不存在: {resource_dir}，将使用 wetext")
                self.use_ttsfrd = False
                self._init_wetext()
        else:
            self._init_wetext()
    
    def _init_wetext(self):
        """初始化 wetext 规范化器（回退方案）"""
        if 'ZhNormalizer' in globals():
            self.zh_tn_model = ZhNormalizer(remove_erhua=False)
            self.en_tn_model = EnNormalizer()
            self.inflect_parser = inflect.engine()
            logging.info("使用 wetext 作为文本规范化工具")
        else:
            logging.warning("wetext 不可用，将跳过文本规范化")
    
    def normalize_and_split(self, text, token_max_n=80, token_min_n=60):
        """文本规范化与智能分段（复用 frontend.text_normalize 逻辑）
        
        Args:
            text: 输入文本
            token_max_n: 最大 token 数（默认 80）
            token_min_n: 最小 token 数（默认 60）
        
        Returns:
            List[str]: 规范化并切分后的文本段落列表
        """
        if not text or text.strip() == '':
            return []
        
        text = text.strip()
        
        # 使用 ttsfrd 进行文本规范化
        if self.use_ttsfrd:
            try:
                result = self.frd.do_voicegen_frd(text)
                texts = [i["text"] for i in json.loads(result)["sentences"]]
                text = ''.join(texts)
            except Exception as e:
                logging.warning(f"ttsfrd 处理失败: {e}，使用原始文本")
        else:
            # 使用 wetext 进行规范化（与 frontend.py 逻辑一致）
            if 'zh_tn_model' in dir(self):
                if contains_chinese(text):
                    text = self.zh_tn_model.normalize(text)
                    text = text.replace("\n", "")
                    text = replace_blank(text)
                    text = replace_corner_mark(text)
                    text = text.replace(".", "。")
                    text = text.replace(" - ", "，")
                    text = remove_bracket(text)
                    text = re.sub(r'[，,、]+$', '。', text)
                else:
                    text = self.en_tn_model.normalize(text)
                    text = spell_out_number(text, self.inflect_parser)
        
        # 使用 split_paragraph 进行智能分段（与 frontend.py 逻辑一致）
        tokenize_fn = partial(self.tokenizer.encode, allowed_special='all')
        
        if contains_chinese(text):
            texts = list(split_paragraph(
                text, tokenize_fn, "zh", 
                token_max_n=token_max_n,
                token_min_n=token_min_n, 
                merge_len=20, 
                comma_split=False
            ))
        else:
            texts = list(split_paragraph(
                text, tokenize_fn, "en", 
                token_max_n=token_max_n,
                token_min_n=token_min_n, 
                merge_len=20, 
                comma_split=False
            ))
        
        # 过滤纯标点段落
        texts = [i for i in texts if not is_only_punctuation(i)]
        
        if texts:
            logging.info(f"[文本规范化] 原始长度: {len(text)} 字符 → 分成 {len(texts)} 段")
            for idx, seg in enumerate(texts):
                logging.info(f"  段{idx+1}: {len(seg)}字符 - '{seg[:30]}{'...' if len(seg) > 30 else ''}'")
        
        return texts


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
    
    def __init__(self, cosyvoice_model, trtllm_engine_dir, trtllm_tokenizer_dir, spk2info_path=None):
        self.cosyvoice : CosyVoice2 = cosyvoice_model
        self.trtllm_runner = None
        self.trtllm_tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.spk2info_path = spk2info_path  # 保存 spk2info.pt 的路径
        
        # 存储原始的 prompt_text 字符串，用于 TensorRT-LLM
        self.spk_prompt_text_raw = {}
        
        # 初始化文本规范化器（使用 frontend 的成熟实现）
        self.text_normalizer = None
        
        # 强制初始化 TensorRT-LLM（追求极致流式推理性能）
        self._init_trtllm(trtllm_engine_dir, trtllm_tokenizer_dir)
        logging.info("✅ TensorRT-LLM 初始化成功")
        
        # 初始化文本规范化器（在 TensorRT-LLM tokenizer 初始化后）
        self.text_normalizer = TextNormalizer(
            tokenizer=self.trtllm_tokenizer,
            use_ttsfrd=USE_TTSFRD
        )
    
    def _init_trtllm(self, engine_dir, tokenizer_dir):
        """初始化 TensorRT-LLM 引擎"""
        if not engine_dir or not os.path.exists(engine_dir):
            raise ValueError(f"TensorRT-LLM 引擎目录不存在: {engine_dir}")
        
        # 获取当前进程的 MPI 排名
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
            max_output_len=2048,    # 最大输出长度，支持长文本生成
            enable_context_fmha_fp32_acc=False,
            max_batch_size=1,
            max_input_len=2048,     # 🔴 增大到 2048，支持长文本输入（prompt + tts_text）
            kv_cache_free_gpu_memory_fraction=0.25,  # 降低 KV 缓存占用，平衡显存
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
    
    def inference_zero_shot(self, text, prompt_text, prompt_speech_16k, zero_shot_spk_id='', stream=True, request_start_time=None):
        """零样本推理（集成 TensorRT-LLM 流式生成 + 长文本分段）"""
        # 记录推理开始时间（如果未传入）
        if request_start_time is None:
            request_start_time = time.perf_counter()
        
        # ========== 长文本分段预处理 ==========
        # 使用 frontend 的成熟实现：文本规范化 + 智能分段
        text_segments = self.text_normalizer.normalize_and_split(
            text, 
            token_max_n=80,  # 与 frontend.py 保持一致
            token_min_n=60
        )
        
        if len(text_segments) == 0:
            logging.warning("[文本分段] 输入文本为空，跳过推理")
            return
        
        logging.info(f"[长文本处理] 原始文本 {len(text)} 字符 → 分成 {len(text_segments)} 段进行流式推理")
        
        # ========== 逐段推理并流式返回 ==========
        for segment_idx, text_segment in enumerate(text_segments):
            segment_start_time = time.perf_counter()
            logging.info(f"[段落推理 {segment_idx+1}/{len(text_segments)}] 开始处理: '{text_segment[:50]}{'...' if len(text_segment) > 50 else ''}'")
            
            # 调用单段推理（内部逻辑保持不变）
            for output in self._inference_single_segment(
                text_segment, prompt_text, prompt_speech_16k,
                zero_shot_spk_id=zero_shot_spk_id,
                stream=stream,
                request_start_time=request_start_time if segment_idx == 0 else segment_start_time,
                is_first_segment=(segment_idx == 0)
            ):
                yield output
            
            segment_time = (time.perf_counter() - segment_start_time) * 1000
            logging.info(f"[段落推理 {segment_idx+1}/{len(text_segments)}] 完成，耗时: {segment_time:.2f}ms")
    
    def _inference_single_segment(self, text, prompt_text, prompt_speech_16k, zero_shot_spk_id='', stream=True, request_start_time=None, is_first_segment=True):
        """单段文本推理（TensorRT-LLM 流式生成 + 流式 token2wav，原 inference_zero_shot 的核心逻辑）"""
        try:
            # ========== 阶段 1: 上下文加载 ==========
            context_start = time.perf_counter()
            
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
            
            context_load_time = (time.perf_counter() - context_start) * 1000
            if is_first_segment:
                logging.info(f"[延迟分析-01] 上下文加载: {context_load_time:.2f}ms (spk_info检索+数据解析)")
            
            # ========== 阶段 2: LLM 输入准备 ==========
            prepare_start = time.perf_counter()
            
            # 4. 准备 LLM 输入（使用原始字符串 + chat template）
            input_ids = self._prepare_llm_input(text, prompt_text_raw, llm_prompt_speech_token)
            
            prepare_time = (time.perf_counter() - prepare_start) * 1000
            if is_first_segment:
                logging.info(f"[延迟分析-02] LLM输入准备: {prepare_time:.2f}ms (text:{len(text)} chars, prompt:{len(prompt_text_raw)} chars, input_tokens:{input_ids.shape[1]})")
            
            # ========== 阶段 3: 推理参数初始化 ==========
            init_start = time.perf_counter()
            
            # 5. 初始化流式参数
            this_uuid = str(uuid_module.uuid1())
            model = self.cosyvoice.model
            model.hift_cache_dict[this_uuid] = None
            
            # 核心参数：保持原始对齐逻辑不变
            token_hop_len = 25  # 标准 hop 长度（保持原始设置）
            token_hop_len_first = 15  # 首块使用更小的 hop，减少等待
            pre_lookahead_len = model.flow.pre_lookahead_len  # 前瞻长度 (3)
            
            # 关键：prompt_token_pad 必须基于标准 hop_len 计算，不能改变
            prompt_token_pad = int(np.ceil(flow_prompt_speech_token.shape[1] / token_hop_len) * token_hop_len - flow_prompt_speech_token.shape[1])
            
            token_offset = 0
            chunk_idx = 0
            
            # RTF 统计
            total_audio_duration = 0.0  # 累计生成的音频时长（秒）
            total_processing_time = 0.0  # 累计处理时间（秒）
            sample_rate = 22050  # CosyVoice2 的采样率
            
            init_time = (time.perf_counter() - init_start) * 1000
            # 首块实际需要的 tokens 数量
            first_chunk_tokens_needed = token_hop_len_first + prompt_token_pad + pre_lookahead_len
            if is_first_segment:
                logging.info(f"[延迟分析-03] 推理参数初始化: {init_time:.2f}ms (hop_first={token_hop_len_first}, hop_normal={token_hop_len}, prompt_pad={prompt_token_pad}, lookahead={pre_lookahead_len}, first_needed={first_chunk_tokens_needed})")
                logging.info(f"[流式生成] 开始流式生成+token2wav: first_chunk_needed={first_chunk_tokens_needed} tokens (原始配置需28)")
            
            # ========== 阶段 4: Token 生成 ==========
            token_gen_start = time.perf_counter()
            first_token_gen_time = None
            
            # 6. 流式生成 + 流式 token2wav
            speech_tokens = []
            generation_done = False
            first_chunk_generated = False
            
            for current_tokens, is_final in self._trtllm_generate_streaming(input_ids):
                # 记录首个 token 生成完毕的时间
                if first_token_gen_time is None:
                    first_token_gen_time = (time.perf_counter() - token_gen_start) * 1000
                    if is_first_segment:
                        logging.info(f"[延迟分析-04a] 首个Token生成完毕: {first_token_gen_time:.2f}ms (首包tokens: {len(current_tokens)})")
                
                speech_tokens = current_tokens
                generation_done = is_final
                
                # 检查是否有足够的 tokens 生成第一块音频
                while True:
                    # 首块使用小 hop，后续块使用正常hop
                    if token_offset == 0:
                        this_token_hop_len = token_hop_len_first + prompt_token_pad
                    else:
                        this_token_hop_len = token_hop_len
                    
                    tokens_needed = token_offset + this_token_hop_len + pre_lookahead_len
                    
                    if tokens_needed <= len(speech_tokens):
                        if not first_chunk_generated:
                            # ========== 阶段 5: 首块音频合成 ==========
                            first_chunk_start = time.perf_counter()
                            accumulated_tokens_time = (first_chunk_start - token_gen_start) * 1000
                            if is_first_segment:
                                logging.info(f"[延迟分析-04b] Token累积到首块需量: {accumulated_tokens_time:.2f}ms (已累积tokens: {len(speech_tokens)}/{tokens_needed})")
                        
                        # 有足够的 tokens，生成一块音频
                        chunk_start_time = time.perf_counter()
                        
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
                        
                        chunk_time = (time.perf_counter() - chunk_start_time) * 1000
                        
                        # 计算当前块的音频时长和 RTF
                        chunk_audio_duration = tts_speech.shape[-1] / sample_rate  # 秒
                        chunk_rtf = (chunk_time / 1000) / chunk_audio_duration if chunk_audio_duration > 0 else 0
                        total_audio_duration += chunk_audio_duration
                        total_processing_time += (chunk_time / 1000)
                        cumulative_rtf = total_processing_time / total_audio_duration if total_audio_duration > 0 else 0
                        
                        if not first_chunk_generated:
                            if is_first_segment:
                                logging.info(f"[延迟分析-05] 首块音频合成(token2wav): {chunk_time:.2f}ms (tokens: {tokens_needed}, hop: {this_token_hop_len})")
                                logging.info(f"[首块RTF] 音频时长: {chunk_audio_duration*1000:.1f}ms, 处理耗时: {chunk_time:.1f}ms, RTF: {chunk_rtf:.3f}")
                            first_chunk_generated = True
                            
                            # 计算从请求到首包的总延迟（仅首段）
                            if is_first_segment:
                                total_ttfb = (time.perf_counter() - request_start_time) * 1000
                                logging.info(f"\n{'='*70}")
                                logging.info(f"[首包延迟汇总 TTFB] 总耗时: {total_ttfb:.2f}ms")
                                logging.info(f"  ├─ 上下文加载: {context_load_time:.2f}ms (step 1)")
                                logging.info(f"  ├─ LLM输入准备: {prepare_time:.2f}ms (step 2)")
                                logging.info(f"  ├─ 参数初始化: {init_time:.2f}ms (step 3)")
                                logging.info(f"  ├─ Token生成(首个): {first_token_gen_time:.2f}ms (step 4a)")
                                logging.info(f"  ├─ Token累积等待: {accumulated_tokens_time - first_token_gen_time:.2f}ms (step 4b)")
                                logging.info(f"  └─ 音频合成(token2wav): {chunk_time:.2f}ms (step 5)")
                                logging.info(f"[延迟分解] Model:{context_load_time+prepare_time+init_time:.1f}ms + LLMGen:{first_token_gen_time:.1f}ms + TTW:{chunk_time:.1f}ms + Wait:{accumulated_tokens_time - first_token_gen_time:.1f}ms = {total_ttfb:.1f}ms")
                                logging.info(f"[性能指标] 首块RTF: {chunk_rtf:.3f}, 音频: {chunk_audio_duration*1000:.0f}ms, 目标RTF: <0.2")
                                logging.info(f"{'='*70}\n")
                        else:
                            logging.info(f"[流式token2wav] 块{chunk_idx}: 音频={chunk_audio_duration*1000:.0f}ms, 耗时={chunk_time:.1f}ms, RTF={chunk_rtf:.3f}, 累积RTF={cumulative_rtf:.3f}")
                        
                        token_offset += this_token_hop_len
                        chunk_idx += 1
                        yield {'tts_speech': tts_speech.cpu()}
                    else:
                        # 不够 tokens，等待更多生成
                        break
                
                if generation_done:
                    break
            
            # ========== 阶段 6: 处理剩余Tokens ==========
            # 7. 处理剩余的 tokens（最后一块）
            if token_offset < len(speech_tokens):
                chunk_start_time = time.perf_counter()
                
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
                
                chunk_time = (time.perf_counter() - chunk_start_time) * 1000
                
                # 计算最终块的音频时长和 RTF
                chunk_audio_duration = tts_speech.shape[-1] / sample_rate
                chunk_rtf = (chunk_time / 1000) / chunk_audio_duration if chunk_audio_duration > 0 else 0
                total_audio_duration += chunk_audio_duration
                total_processing_time += (chunk_time / 1000)
                cumulative_rtf = total_processing_time / total_audio_duration if total_audio_duration > 0 else 0
                
                logging.info(f"[流式token2wav] 最终块{chunk_idx}: 音频={chunk_audio_duration*1000:.0f}ms, 耗时={chunk_time:.1f}ms, RTF={chunk_rtf:.3f} (finalize)")
                yield {'tts_speech': tts_speech.cpu()}  # 🔴 关键：必须 yield 最后一块音频
            
            # 输出总体 RTF 统计（仅首段详细输出）
            overall_rtf = total_processing_time / total_audio_duration if total_audio_duration > 0 else 0
            if is_first_segment:
                logging.info(f"[FastTTS] 流式生成完成: 共 {len(speech_tokens)} 个 speech tokens, {chunk_idx + 1} 个音频块")
                logging.info(f"[整体RTF统计] 总音频时长: {total_audio_duration:.2f}s, 总处理时间: {total_processing_time:.2f}s, 整体RTF: {overall_rtf:.3f} (目标: <0.2)")
            
            # 清理缓存
            if this_uuid in model.hift_cache_dict:
                model.hift_cache_dict.pop(this_uuid)
            
        except Exception as e:
            logging.error(f"TensorRT-LLM 推理失败: {e}")
            import traceback
            traceback.print_exc()
            raise  # 直接抛出异常，不回退到 PyTorch
    
    def list_available_spks(self):
        """获取可用说话人列表"""
        return self.cosyvoice.list_available_spks()
    
    def add_zero_shot_spk(self, prompt_text, prompt_speech_16k, zero_shot_spk_id):
        """添加零样本说话人"""
        # 保存原始的 prompt_text 字符串（用于 TensorRT-LLM）
        self.spk_prompt_text_raw[zero_shot_spk_id] = prompt_text
        return self.cosyvoice.add_zero_shot_spk(prompt_text, prompt_speech_16k, zero_shot_spk_id)
    
    def save_spkinfo(self):
        """保存说话人信息到指定路径"""
        if self.spk2info_path:
            torch.save(self.cosyvoice.frontend.spk2info, self.spk2info_path)
            logging.info(f"说话人信息已保存到: {self.spk2info_path}")
        else:
            return self.cosyvoice.save_spkinfo()
    
    def load_spk_prompt_text_raw(self, spk_prompt_text_raw_dict):
        """从外部加载原始 prompt_text 映射"""
        self.spk_prompt_text_raw.update(spk_prompt_text_raw_dict)


def generate_data(model_output, request_start_time):
    """生成音频数据流，对输出进行削波处理防止爆音"""
    is_first = True
    chunk_count = 0
    
    for i in model_output:
        if is_first:
            first_chunk_time = time.perf_counter()
            ttfb = (first_chunk_time - request_start_time) * 1000
            # 注意：此处 TTFB 是 HTTP 响应级别的首包时间（从请求到 generate_data 首次产出数据）
            # 实际的推理 TTFB 已在 inference_zero_shot 中详细打印
            logging.info(f"[TTS统计] HTTP响应首包生成完毕! HTTP TTFB: {ttfb:.2f}ms")
            is_first = False

        tts_speech = i["tts_speech"].numpy()
        
        # 输出端削波：防止 float -> int16 转换时的整数溢出
        tts_speech = np.clip(tts_speech, -1.0, 1.0)
        
        # 转换为 int16 格式
        tts_audio = (tts_speech * 32767.0).astype(np.int16).tobytes()
        chunk_count += 1
        yield tts_audio
    
    total_time = (time.perf_counter() - request_start_time) * 1000
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


def extract_spk_prompt_text_from_directory(speaker_dir="asset/speakers"):
    """从说话人目录提取 spk_id -> prompt_text 映射（不加载音频）"""
    spk_prompt_text_raw = {}
    
    if not os.path.exists(speaker_dir):
        logging.warning(f"说话人目录 {speaker_dir} 不存在")
        return spk_prompt_text_raw
    
    wav_files = glob.glob(os.path.join(speaker_dir, "*.wav"))
    
    for wav_path in wav_files:
        filename = os.path.basename(wav_path)
        # 解析文件名格式：[说话人名称]文本内容.wav
        match = re.match(r'\[(.+?)\](.+)\.wav$', filename)
        
        if match:
            speaker_name = match.group(1)
            prompt_text = match.group(2)
            spk_prompt_text_raw[speaker_name] = prompt_text
    
    return spk_prompt_text_raw


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
        # 获取可用说话人列表
        available_spks = fast_cosyvoice.list_available_spks()
        # 默认使用jok老师，如果没有则使用第一个说话人
        default_speaker = "jok老师" if "jok老师" in available_spks else (available_spks[0] if available_spks else "")
        selected_speaker = speaker if speaker else default_speaker
        
        if not selected_speaker:
            return JSONResponse(content={"error": "没有可用的说话人"}, status_code=400)
        
        logging.info(f"[FastTTS推理] 开始推理, 说话人: {selected_speaker}")

        # 使用 FastCosyVoice2 进行推理（自动选择 TensorRT-LLM 或 PyTorch）
        model_output = fast_cosyvoice.inference_zero_shot(
            text, "", None, 
            zero_shot_spk_id=selected_speaker,
            stream=True,
            request_start_time=request_start_time
        )
        return StreamingResponse(generate_data(model_output, request_start_time))
    except Exception as e:
        logging.error(f"FastTTS推理失败: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50000, help="服务端口")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/CosyVoice2-0.5B",
        help="模型本地路径或 modelscope 仓库 id",
    )
    parser.add_argument(
        "--trtllm_engine_dir",
        type=str,
        default="pretrained_models/cosyvoice2_llm/trt_engines_bfloat16",
        help="TensorRT-LLM 引擎目录",
    )
    parser.add_argument(
        "--trtllm_tokenizer_dir",
        type=str,
        default="pretrained_models/cosyvoice2_llm",
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
        
        # 加载原始 CosyVoice2 模型（确保启用所有加速选项）
        cosyvoice = CosyVoice2(
            args.model_dir, 
            load_jit=True,   # ✅ JIT编译加速
            load_trt=True,   # ✅ TensorRT优化
            fp16=True        # ✅ FP16混合精度
        )
        logging.info("✅ 模型加载配置: JIT=True, TRT=True, FP16=True")
        
        # 创建 FastCosyVoice2 实例（集成 TensorRT-LLM）
        spk2info_path = os.path.join(args.speaker_dir, 'spk2info.pt')
        fast_cosyvoice = FastCosyVoice2(
            cosyvoice_model=cosyvoice,
            trtllm_engine_dir=args.trtllm_engine_dir,
            trtllm_tokenizer_dir=args.trtllm_tokenizer_dir,
            spk2info_path=spk2info_path,
        )
        try:
            # 把 PyTorch LLM 移到 CPU，释放 GPU 显存
            fast_cosyvoice.cosyvoice.model.llm.to("cpu")
            torch.cuda.empty_cache()
            logging.info("已将 CosyVoice2 PyTorch LLM 移至 CPU，释放 GPU 显存")
        except Exception as e:
            logging.warning(f"移动 CosyVoice2 LLM 到 CPU 失败: {e}")
        logging.info("✅ FastCosyVoice2 初始化成功，使用 TensorRT-LLM 加速")

    except Exception as e:
        raise TypeError(f"导入{args.model_dir}失败，模型类型有误！错误: {e}")

    # 加载说话人信息
    spk2info_path = os.path.join(args.speaker_dir, 'spk2info.pt')
    spk_prompt_text_raw_map = extract_spk_prompt_text_from_directory(args.speaker_dir)
    
    # 检查 speakers 目录下的 spk2info.pt 是否存在且包含所有说话人
    need_regenerate = False
    
    # 如果 speakers 目录下有 spk2info.pt，加载它
    if os.path.exists(spk2info_path):
        spk2info_data = torch.load(spk2info_path, map_location=fast_cosyvoice.device)
        fast_cosyvoice.cosyvoice.frontend.spk2info.update(spk2info_data)
        logging.info(f"已加载 spk2info.pt: {spk2info_path}")
    
    existing_spks = set(fast_cosyvoice.cosyvoice.frontend.spk2info.keys())
    required_spks = set(spk_prompt_text_raw_map.keys())
    
    if not os.path.exists(spk2info_path):
        print(f"未找到 spk2info.pt，将生成新文件")
        need_regenerate = True
    elif required_spks - existing_spks:
        missing_spks = required_spks - existing_spks
        print(f"检测到新增说话人: {missing_spks}，需要重新生成 spk2info.pt")
        need_regenerate = True
    else:
        print(f"spk2info.pt 已存在，包含 {len(existing_spks)} 个说话人，跳过特征提取")
    
    if need_regenerate:
        print("正在提取说话人音频特征...")
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
                # 确保 speaker_dir 存在
                os.makedirs(args.speaker_dir, exist_ok=True)
                fast_cosyvoice.save_spkinfo()
                print(f"说话人信息已保存到 {spk2info_path}")
            except Exception as e:
                print(f"保存说话人信息失败: {e}")
    else:
        # 直接从文件名加载 prompt_text 映射
        fast_cosyvoice.load_spk_prompt_text_raw(spk_prompt_text_raw_map)
        print(f"已从文件名加载 {len(spk_prompt_text_raw_map)} 个说话人的 prompt_text 映射")
    
    # 模型预热
    print("\n正在预热模型...")
    available_spks = fast_cosyvoice.list_available_spks()
    if available_spks:
        warmup_speaker = "jok老师" if "jok老师" in available_spks else available_spks[0]
        print(f"使用 '{warmup_speaker}' 进行预热")
        warmup_texts = [
            "你好。",  # 短句
            "这是一个用于预热模型的测试句子，确保服务响应速度。", # 中句
            "语音合成服务正在启动中，请稍候，系统正在进行初始化操作。", # 长句
            "好的，没问题。" # 短句
        ]
        
        for t in warmup_texts:
            try:
                for _ in fast_cosyvoice.inference_zero_shot(
                        t, "", None, zero_shot_spk_id=warmup_speaker, stream=True):
                    pass
            except Exception as e:
                print(f"预热失败: {e}")
                break
    else:
        print("未找到可用说话人，跳过预热")
    
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
