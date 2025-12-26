#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export CosyVoice3 LLM to HuggingFace format for TensorRT-LLM conversion

Usage:
    python export_cosyvoice3_to_huggingface.py \
        --pretrained-cosyvoice3-path pretrained_models/Fun-CosyVoice3-0.5B \
        --save-path pretrained_models/cosyvoice3_llm
"""

import sys
import os
from argparse import ArgumentParser
from transformers import AutoTokenizer
import torch

# Add third_party path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "third_party/Matcha-TTS"))

from cosyvoice.cli.cosyvoice import CosyVoice3


def get_args():
    parser = ArgumentParser(description="Export CosyVoice3 LLM to HuggingFace format")
    parser.add_argument(
        "--pretrained-cosyvoice3-path",
        type=str,
        default="pretrained_models/Fun-CosyVoice3-0.5B",
        help="CosyVoice3 model directory (contains cosyvoice3.yaml, llm.pt, etc.)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="pretrained_models/cosyvoice3_llm",
        help="Output directory for HuggingFace-compatible LLM",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    
    print("=" * 80)
    print("CosyVoice3 → HuggingFace LLM Export Tool")
    print("=" * 80)
    print(f"Source model: {args.pretrained_cosyvoice3_path}")
    print(f"Target path:  {args.save_path}")
    print()
    
    # Step 1: Load CosyVoice3 model
    print("[1/6] Loading CosyVoice3 model...")
    cosy3_model = CosyVoice3(
        args.pretrained_cosyvoice3_path, 
        load_trt=False, 
        load_vllm=False, 
        fp16=False
    )
    print("      ✓ CosyVoice3 model loaded")
    
    # Step 2: Extract LLM components
    print("[2/6] Extracting LLM components...")
    llm = cosy3_model.model.llm.llm.model  # Qwen2ForCausalLM
    speech_embedding = cosy3_model.model.llm.speech_embedding
    llm_decoder = cosy3_model.model.llm.llm_decoder
    
    print(f"      - LLM type: {type(llm).__name__}")
    print(f"      - speech_embedding shape: {speech_embedding.weight.shape}")
    print(f"      - llm_decoder shape: {llm_decoder.weight.shape}")
    print(f"      ✓ Components extracted")
    
    # Step 3: Load and configure tokenizer
    print("[3/6] Configuring tokenizer...")
    qwen_base_path = os.path.join(args.pretrained_cosyvoice3_path, "CosyVoice-BlankEN")
    if not os.path.exists(qwen_base_path):
        raise ValueError(f"CosyVoice-BlankEN not found at {qwen_base_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(qwen_base_path)
    
    # CosyVoice3 special tokens (from CosyVoice3Tokenizer)
    special_tokens = {
        'eos_token': '<|endoftext|>',
        'pad_token': '<|endoftext|>',
        'additional_special_tokens': [
            '<|im_start|>', '<|im_end|>', '<|endofprompt|>',
            '[breath]', '<strong>', '</strong>', '[noise]',
            '[laughter]', '[cough]', '[clucking]', '[accent]',
            '[quick_breath]',
            "<laughter>", "</laughter>",
            "[hissing]", "[sigh]", "[vocalized-noise]",
            "[lipsmack]", "[mn]", "<|endofsystem|>",
            # Phoneme tokens for CosyVoice3 (English ARPAbet + Chinese pinyin)
            "[AA]", "[AA0]", "[AA1]", "[AA2]", "[AE]", "[AE0]", "[AE1]", "[AE2]", 
            "[AH]", "[AH0]", "[AH1]", "[AH2]", "[AO]", "[AO0]", "[AO1]", "[AO2]", 
            "[AW]", "[AW0]", "[AW1]", "[AW2]", "[AY]", "[AY0]", "[AY1]", "[AY2]",
            "[B]", "[CH]", "[D]", "[DH]", "[EH]", "[EH0]", "[EH1]", "[EH2]", 
            "[ER]", "[ER0]", "[ER1]", "[ER2]", "[EY]", "[EY0]", "[EY1]", "[EY2]", 
            "[F]", "[G]", "[HH]", "[IH]", "[IH0]", "[IH1]", "[IH2]", "[IY]", 
            "[IY0]", "[IY1]", "[IY2]", "[JH]", "[K]", "[L]", "[M]", "[N]", "[NG]", 
            "[OW]", "[OW0]", "[OW1]", "[OW2]", "[OY]", "[OY0]", "[OY1]", "[OY2]", 
            "[P]", "[R]", "[S]", "[SH]", "[T]", "[TH]", "[UH]", "[UH0]", "[UH1]", 
            "[UH2]", "[UW]", "[UW0]", "[UW1]", "[UW2]", "[V]", "[W]", "[Y]", "[Z]", "[ZH]",
            "[a]", "[ai]", "[an]", "[ang]", "[ao]", "[b]", "[c]", "[ch]", "[d]", 
            "[e]", "[ei]", "[en]", "[eng]", "[f]", "[g]", "[h]", "[i]", "[ian]", 
            "[in]", "[ing]", "[iu]", "[ià]", "[iàn]", "[iàng]", "[iào]", "[iá]", 
            "[ián]", "[iáng]", "[iáo]", "[iè]", "[ié]", "[iòng]", "[ióng]", "[iù]", 
            "[iú]", "[iā]", "[iān]", "[iāng]", "[iāo]", "[iē]", "[iě]", "[iōng]", 
            "[iū]", "[iǎ]", "[iǎn]", "[iǎng]", "[iǎo]", "[iǒng]", "[iǔ]", "[j]", 
            "[k]", "[l]", "[m]", "[n]", "[o]", "[ong]", "[ou]", "[p]", "[q]", "[r]", 
            "[s]", "[sh]", "[t]", "[u]", "[uang]", "[ue]", "[un]", "[uo]", "[uà]", 
            "[uài]", "[uàn]", "[uàng]", "[uá]", "[uái]", "[uán]", "[uáng]", "[uè]", 
            "[ué]", "[uì]", "[uí]", "[uò]", "[uó]", "[uā]", "[uāi]", "[uān]", 
            "[uāng]", "[uē]", "[uě]", "[uī]", "[uō]", "[uǎ]", "[uǎi]", "[uǎn]", 
            "[uǎng]", "[uǐ]", "[uǒ]", "[vè]", "[w]", "[x]", "[y]", "[z]", "[zh]", 
            "[à]", "[ài]", "[àn]", "[àng]", "[ào]", "[á]", "[ái]", "[án]", "[áng]", 
            "[áo]", "[è]", "[èi]", "[èn]", "[èng]", "[èr]", "[é]", "[éi]", "[én]", 
            "[éng]", "[ér]", "[ì]", "[ìn]", "[ìng]", "[í]", "[ín]", "[íng]", "[ò]", 
            "[òng]", "[òu]", "[ó]", "[óng]", "[óu]", "[ù]", "[ùn]", "[ú]", "[ún]", 
            "[ā]", "[āi]", "[ān]", "[āng]", "[āo]", "[ē]", "[ēi]", "[ēn]", "[ēng]", 
            "[ě]", "[ěi]", "[ěn]", "[ěng]", "[ěr]", "[ī]", "[īn]", "[īng]", "[ō]", 
            "[ōng]", "[ōu]", "[ū]", "[ūn]", "[ǎ]", "[ǎi]", "[ǎn]", "[ǎng]", "[ǎo]", 
            "[ǐ]", "[ǐn]", "[ǐng]", "[ǒ]", "[ǒng]", "[ǒu]", "[ǔ]", "[ǔn]", "[ǘ]", 
            "[ǚ]", "[ǜ]"
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    print(f"      ✓ Added {len(special_tokens['additional_special_tokens'])} special tokens")
    
    # Step 4: Add speech tokens and control tokens
    print("[4/6] Adding speech tokens...")
    original_tokenizer_vocab_size = len(tokenizer)
    print(f"      - Original vocab size: {original_tokenizer_vocab_size}")
    
    # CosyVoice3: speech_token_size=6561, with extended control tokens
    # From cosyvoice3.yaml: speech_token_size: 6561
    # From CosyVoice3LM.__init__:
    #   self.sos = speech_token_size + 0
    #   self.eos_token = speech_token_size + 1
    #   self.task_id = speech_token_size + 2
    #   self.fill_token = speech_token_size + 3
    #   speech_embedding: Embedding(speech_token_size + 200, ...)
    #   llm_decoder: Linear(..., speech_token_size + 200, bias=False)
    
    cosyvoice3_token_size = 6561
    extended_control_tokens = 200  # CosyVoice3 uses +200 for extended control
    
    # Add speech tokens: <|s_0|> to <|s_6560|>
    speech_tokens = [f"<|s_{i}|>" for i in range(cosyvoice3_token_size)]
    
    # Add control tokens mapped to specific positions
    # sos=6561, eos=6562, task_id=6563, fill=6564
    control_tokens = ["<|sos|>", "<|eos1|>", "<|task_id|>", "<|fill|>"]
    
    # Add remaining extended tokens (6565~6760) as generic tokens
    remaining_tokens = [f"<|ext_{i}|>" for i in range(4, extended_control_tokens)]
    
    new_tokens = speech_tokens + control_tokens + remaining_tokens
    num_added_tokens = tokenizer.add_tokens(new_tokens)
    print(f"      ✓ Added {num_added_tokens} tokens (speech: {cosyvoice3_token_size}, control: {extended_control_tokens})")
    
    # Step 5: Resize embeddings and rebuild lm_head
    print("[5/6] Rebuilding model architecture...")
    llm.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128)
    vocab_size = llm.get_input_embeddings().weight.shape[0]
    print(f"      - New vocab size (padded): {vocab_size}")
    
    feature_size = speech_embedding.embedding_dim
    print(f"      - Feature size: {feature_size}")
    
    # Create new lm_head without bias (CosyVoice3 uses bias=False)
    new_lm_head = torch.nn.Linear(
        in_features=feature_size, 
        out_features=vocab_size, 
        bias=False
    )
    
    with torch.no_grad():
        # Initialize to zero
        new_lm_head.weight.data.zero_()
        
        # Copy llm_decoder weights to speech token positions
        # llm_decoder shape: (speech_token_size + 200, feature_size)
        decoder_vocab_size = llm_decoder.weight.shape[0]
        target_start = original_tokenizer_vocab_size
        target_end = original_tokenizer_vocab_size + decoder_vocab_size
        
        print(f"      - Copying decoder weights [{target_start}:{target_end}]")
        new_lm_head.weight[target_start:target_end] = llm_decoder.weight
    
    llm.lm_head = new_lm_head
    
    # Update input embeddings
    input_embeddings = llm.get_input_embeddings()
    
    with torch.no_grad():
        # Copy speech_embedding weights
        # speech_embedding shape: (speech_token_size + 200, feature_size)
        emb_vocab_size = speech_embedding.weight.shape[0]
        emb_start = original_tokenizer_vocab_size
        emb_end = original_tokenizer_vocab_size + emb_vocab_size
        
        print(f"      - Copying speech embeddings [{emb_start}:{emb_end}]")
        input_embeddings.weight[emb_start:emb_end] = speech_embedding.weight
    
    print("      ✓ Model architecture updated")
    
    # Step 6: Configure and save
    print("[6/6] Configuring generation and saving...")
    
    # Set eos_token_ids (CosyVoice3 has extended stop tokens)
    # From CosyVoice3LM: self.stop_token_ids = [speech_token_size + i for i in range(200)]
    # Primary eos: speech_token_size + 1 (eos1)
    eos_token_id = original_tokenizer_vocab_size + cosyvoice3_token_size + 1
    
    # Generate list of all stop token IDs
    stop_token_ids = [
        original_tokenizer_vocab_size + cosyvoice3_token_size + i 
        for i in range(extended_control_tokens)
    ]
    
    llm.generation_config.eos_token_id = stop_token_ids
    llm.generation_config.temperature = 1.0
    llm.generation_config.top_p = 0.8
    llm.generation_config.top_k = 25
    
    llm.config.eos_token_id = eos_token_id
    llm.config.vocab_size = vocab_size
    llm.config.tie_word_embeddings = False
    llm.config.use_bias = False  # CosyVoice3 decoder has no bias
    
    # Convert to bfloat16 for TensorRT-LLM compatibility
    llm.to(torch.bfloat16)
    
    print(f"      - EOS token ID: {eos_token_id}")
    print(f"      - Stop token IDs: {len(stop_token_ids)} tokens")
    print(f"      - Data type: bfloat16")
    
    # Save model
    os.makedirs(args.save_path, exist_ok=True)
    llm.save_pretrained(args.save_path)
    print(f"      ✓ Model saved to {args.save_path}")
    
    # Configure and save tokenizer with chat template
    TEMPLATE = (
        "{%- for message in messages %}"
        "{%- if message['role'] == 'user' %}"
        "{{- '<|sos|>' + message['content'] + '<|task_id|>' }}"
        "{%- elif message['role'] == 'assistant' %}"
        "{{- message['content']}}"
        "{%- endif %}"
        "{%- endfor %}"
    )
    tokenizer.chat_template = TEMPLATE
    tokenizer.save_pretrained(args.save_path)
    print(f"      ✓ Tokenizer saved to {args.save_path}")
    
    print()
    print("=" * 80)
    print("Export completed successfully!")
    print("=" * 80)
    print()
    print("Next steps:")
    print(f"  1. Verify the exported model: ls -lh {args.save_path}")
    print(f"  2. Convert to TensorRT-LLM using:")
    print(f"     python runtime/triton_trtllm/scripts/convert_checkpoint.py \\")
    print(f"       --model_dir {args.save_path} \\")
    print(f"       --output_dir {args.save_path}/trt_weights \\")
    print(f"       --dtype bfloat16")
    print()
