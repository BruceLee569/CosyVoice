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


class CosyVoiceServer:
    """CosyVoice2 推理类（纯 PyTorch 实现）"""
    
    def __init__(self, cosyvoice_model, spk2info_path=None):
        self.cosyvoice: CosyVoice2 = cosyvoice_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.spk2info_path = spk2info_path
        
        # 初始化文本规范化器
        self.text_normalizer = TextNormalizer(
            tokenizer=self.cosyvoice.frontend.tokenizer,
            use_ttsfrd=USE_TTSFRD
        )
    
    def inference_zero_shot(self, text, prompt_text, prompt_speech_16k, zero_shot_spk_id='', stream=True, request_start_time=None):
        """零样本推理（PyTorch 原生实现 + 长文本分段）"""
        # 记录推理开始时间（如果未传入）
        if request_start_time is None:
            request_start_time = time.perf_counter()
        
        # ========== 长文本分段预处理 ==========
        text_segments = self.text_normalizer.normalize_and_split(
            text, 
            token_max_n=80,
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
            
            # 调用单段推理
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
        """单段文本推理（PyTorch 原生实现）"""
        try:
            # ========== 阶段 1: 上下文加载 ==========
            context_start = time.perf_counter()
            
            # 使用 CosyVoice2 原生的流式推理接口
            model_output = self.cosyvoice.inference_zero_shot(
                text, 
                prompt_text, 
                prompt_speech_16k, 
                zero_shot_spk_id=zero_shot_spk_id,
                stream=stream
            )
            
            context_load_time = (time.perf_counter() - context_start) * 1000
            if is_first_segment:
                logging.info(f"[延迟分析-01] 模型推理启动: {context_load_time:.2f}ms")
            
            # ========== 阶段 2: 流式输出 ==========
            is_first = True
            chunk_count = 0
            total_audio_duration = 0.0
            total_processing_time = 0.0
            sample_rate = 22050
            
            for output in model_output:
                chunk_start = time.perf_counter()
                
                if is_first and is_first_segment:
                    ttfb = (chunk_start - request_start_time) * 1000
                    logging.info(f"\n{'='*70}")
                    logging.info(f"[首包延迟汇总 TTFB] 总耗时: {ttfb:.2f}ms")
                    logging.info(f"{'='*70}\n")
                    is_first = False
                
                tts_speech = output['tts_speech']
                
                # 计算音频时长和 RTF
                chunk_audio_duration = tts_speech.shape[-1] / sample_rate
                chunk_processing_time = (time.perf_counter() - chunk_start)
                chunk_rtf = chunk_processing_time / chunk_audio_duration if chunk_audio_duration > 0 else 0
                
                total_audio_duration += chunk_audio_duration
                total_processing_time += chunk_processing_time
                cumulative_rtf = total_processing_time / total_audio_duration if total_audio_duration > 0 else 0
                
                chunk_count += 1
                logging.info(f"[流式输出] 块{chunk_count}: 音频={chunk_audio_duration*1000:.0f}ms, RTF={chunk_rtf:.3f}, 累积RTF={cumulative_rtf:.3f}")
                
                yield {'tts_speech': tts_speech.cpu()}
            
            # 输出总体 RTF 统计
            overall_rtf = total_processing_time / total_audio_duration if total_audio_duration > 0 else 0
            if is_first_segment:
                logging.info(f"[整体RTF统计] 总音频时长: {total_audio_duration:.2f}s, 总处理时间: {total_processing_time:.2f}s, 整体RTF: {overall_rtf:.3f}")
            
        except Exception as e:
            logging.error(f"PyTorch 推理失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def list_available_spks(self):
        """获取可用说话人列表"""
        return self.cosyvoice.list_available_spks()
    
    def add_zero_shot_spk(self, prompt_text, prompt_speech_16k, zero_shot_spk_id):
        """添加零样本说话人"""
        return self.cosyvoice.add_zero_shot_spk(prompt_text, prompt_speech_16k, zero_shot_spk_id)
    
    def save_spkinfo(self):
        """保存说话人信息到指定路径"""
        if self.spk2info_path:
            torch.save(self.cosyvoice.frontend.spk2info, self.spk2info_path)
            logging.info(f"说话人信息已保存到: {self.spk2info_path}")
        else:
            return self.cosyvoice.save_spkinfo()


def generate_data(model_output, request_start_time):
    """生成音频数据流，对输出进行削波处理防止爆音"""
    is_first = True
    chunk_count = 0
    
    for i in model_output:
        if is_first:
            first_chunk_time = time.perf_counter()
            ttfb = (first_chunk_time - request_start_time) * 1000
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
        speakers = cosyvoice_server.list_available_spks()
        return JSONResponse(content={"speakers": speakers})
    except Exception as e:
        logging.error(f"获取说话人列表失败: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/tts")
async def inference_zero_shot(request: Request, text: str = Form(), speaker: str = Form(default="")):
    """文本转语音接口（PyTorch 原生实现）"""
    # 使用中间件记录的开始时间，确保与前端对齐
    request_start_time = request.state.start_time
    logging.info(f"[TTS请求] 收到请求: text='{text[:50] if len(text) > 50 else text}', speaker='{speaker}'")

    try:
        # 获取可用说话人列表
        available_spks = cosyvoice_server.list_available_spks()
        # 默认使用jok老师，如果没有则使用第一个说话人
        default_speaker = "jok老师" if "jok老师" in available_spks else (available_spks[0] if available_spks else "")
        selected_speaker = speaker if speaker else default_speaker
        
        if not selected_speaker:
            return JSONResponse(content={"error": "没有可用的说话人"}, status_code=400)
        
        logging.info(f"[TTS推理] 开始推理, 说话人: {selected_speaker}")

        # 使用 CosyVoiceServer 进行推理
        model_output = cosyvoice_server.inference_zero_shot(
            text, "", None, 
            zero_shot_spk_id=selected_speaker,
            stream=True,
            request_start_time=request_start_time
        )
        return StreamingResponse(generate_data(model_output, request_start_time))
    except Exception as e:
        logging.error(f"TTS推理失败: {e}")
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
        "--speaker_dir",
        type=str,
        default="asset/speakers",
        help="说话人音频文件目录",
    )
    args = parser.parse_args()

    try:
        # 初始化 CosyVoice2 模型
        logging.info("=" * 60)
        logging.info("启动 CosyVoice TTS Server")
        logging.info(f"模型目录: {args.model_dir}")
        logging.info("=" * 60)
        
        # 加载 CosyVoice2 模型（启用所有加速选项）
        cosyvoice = CosyVoice2(
            args.model_dir, 
            load_jit=True,   # ✅ JIT编译加速 flow.encoder
            load_trt=True,   # ✅ TensorRT优化
            fp16=True        # ✅ FP16混合精度
        )
        logging.info("✅ 模型加载配置: JIT=True, TRT=True, FP16=True")
        
        # 创建 CosyVoiceServer 实例
        spk2info_path = os.path.join(args.speaker_dir, 'spk2info.pt')
        cosyvoice_server = CosyVoiceServer(
            cosyvoice_model=cosyvoice,
            spk2info_path=spk2info_path,
        )
        logging.info("✅ CosyVoiceServer 初始化成功")

    except Exception as e:
        raise TypeError(f"导入{args.model_dir}失败，模型类型有误！错误: {e}")

    # 加载说话人信息
    spk2info_path = os.path.join(args.speaker_dir, 'spk2info.pt')
    
    # 检查 speakers 目录下的 spk2info.pt 是否存在且包含所有说话人
    need_regenerate = False
    
    # 如果 speakers 目录下有 spk2info.pt，加载它
    if os.path.exists(spk2info_path):
        spk2info_data = torch.load(spk2info_path, map_location=cosyvoice_server.device)
        cosyvoice_server.cosyvoice.frontend.spk2info.update(spk2info_data)
        logging.info(f"已加载 spk2info.pt: {spk2info_path}")
        print(f"spk2info.pt 已存在，包含 {len(spk2info_data)} 个说话人")
    else:
        print(f"未找到 spk2info.pt，将生成新文件")
        need_regenerate = True
    
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
                    cosyvoice_server.add_zero_shot_spk(
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
                cosyvoice_server.save_spkinfo()
                print(f"说话人信息已保存到 {spk2info_path}")
            except Exception as e:
                print(f"保存说话人信息失败: {e}")
    
    # 模型预热
    print("\n正在预热模型...")
    available_spks = cosyvoice_server.list_available_spks()
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
                for _ in cosyvoice_server.inference_zero_shot(
                        t, "", None, zero_shot_spk_id=warmup_speaker, stream=True):
                    pass
            except Exception as e:
                print(f"预热失败: {e}")
                break
    else:
        print("未找到可用说话人，跳过预热")
    
    print("预热完毕\n")
    print("=" * 60)
    print(f"🚀 CosyVoice TTS Server 启动在端口 {args.port}")

    # 配置 uvicorn 以支持 HTTP keep-alive
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=args.port,
        timeout_keep_alive=60,  # keep-alive 超时时间（秒）
        limit_concurrency=100,  # 最大并发连接数
        backlog=2048,  # TCP backlog 队列大小
    )
