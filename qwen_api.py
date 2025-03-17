
import argparse, os, sys, platform, random, time, tiktoken
from copy import deepcopy
from threading import Thread

# import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.trainer_utils import set_seed

def _load_model_tokenizer(model_path, max_tokens=None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        resume_download=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype= "auto", #torch.bfloat16,
        # attn_implementation='flash_attention_2',
        device_map="auto",
        resume_download=True,
    ).eval()
    
    model.generation_config.max_new_tokens = 2048  # For chat.
    model.generation_config.temperature = 0.6  # Set temperature for generation

    return model, tokenizer

def _chat_stream(model, tokenizer, system_prompt, user_prompt):
    conversation = []
    conversation.append({"role": "system", "content": system_prompt})
    conversation.append({"role": "user", "content": user_prompt})
    input_text = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True
    )
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text

def qwen_api(model, tokenizer, model_path, user_prompt, system_prompt="", temperature=0.6, max_tokens=None):
    # 加载模型会有延迟（例如0.5B-FP16是1s），如果需要自动化执行，最好在上层函数中加载模型然后作为参数传入
    if model is None or tokenizer is None:
        model, tokenizer = _load_model_tokenizer(model_path)
    
    partial_text = ""
    for new_text in _chat_stream(model, tokenizer, model_path, user_prompt, system_prompt, temperature, max_tokens):
        partial_text += new_text
    return partial_text

