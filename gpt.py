import base64
import requests
import os
import json
import logging
from PIL import Image

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, AutoModelForCausalLM, Qwen3_5ForConditionalGeneration
from qwen_vl_utils import process_vision_info

# OpenAI API Key
API_KEY = ""
USE_LOCAL_QWEN = os.environ.get("USE_LOCAL_QWEN", "0") == "1"
USE_LOCAL_GEMMA = os.environ.get("USE_LOCAL_GEMMA", "0") == "1"
USE_LOCAL_VLLM = os.environ.get("USE_LOCAL_VLLM", "0") == "1"
QWEN_QUANTIZATION = os.environ.get("QWEN_QUANTIZATION_OVERRIDE", "1") == "1"

# QWEN_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
QWEN_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
QWEN_PROCESSOR = None

GEMMA_MODEL = "google/gemma-4-E4B-it"
GEMMA_PROCESSOR = None

def load_qwen():
    global QWEN_MODEL, QWEN_PROCESSOR
    if isinstance(QWEN_MODEL, str):
        model_name = QWEN_MODEL
        
        # Select the correct architecture class dynamically based on the model string!
        if "3.6" in model_name:
            ModelClass = Qwen3_5ForConditionalGeneration
        else:
            ModelClass = Qwen3VLForConditionalGeneration

        if QWEN_QUANTIZATION:
            print(f"Loading {model_name} into GPU with 8-bit quantization...")
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            QWEN_MODEL = ModelClass.from_pretrained(
                model_name, torch_dtype="auto", device_map="auto", quantization_config=quantization_config
            )
        else:
            print(f"Loading {model_name} into GPU with bfloat16 (NO quantization)...")
            QWEN_MODEL = ModelClass.from_pretrained(
                model_name, torch_dtype="auto", device_map="auto"
            )
        QWEN_PROCESSOR = AutoProcessor.from_pretrained(model_name)

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def prompt_make(prompt_path, ex_prompt):
    with open(prompt_path, "r", encoding='utf-8') as f:
        txt = f.readlines()
        prompt_system = txt[1]
        prompt = txt[3]
        if len(txt) > 4:
            for i in range(4, len(txt)):
                prompt = prompt + txt[i]
        prompt = prompt + ex_prompt
        return prompt_system, prompt

def qwen3_vl_ask(prompt_path, ex_prompt, img_path=None, force_think_tag=False, max_new_tokens=2048):
    load_qwen()
    prompt_system, prompt_user = prompt_make(prompt_path, ex_prompt)
    
    content = [{"type": "text", "text": prompt_user}]
    if img_path:
        content.insert(0, {"type": "image", "image": img_path})
        
    messages = [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": content}
    ]
    
    text = QWEN_PROCESSOR.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if force_think_tag:
        text += "<think>\n"
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = QWEN_PROCESSOR(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    
    inputs = inputs.to(QWEN_MODEL.device)
    
    with torch.inference_mode():
        generated_ids = QWEN_MODEL.generate(**inputs, max_new_tokens=max_new_tokens)
        
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = QWEN_PROCESSOR.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    del inputs
    del generated_ids
    del generated_ids_trimmed
    import gc
    gc.collect()
    
    logging.info(f"QWEN Model raw output:\n{output_text}")
    print(f"RAW QWEN OUTPUT: {repr(output_text)}")
    if "</think>" in output_text:
        after_think = output_text.split("</think>")[-1].strip()
        if after_think:
            output_text = after_think
        else:
            # Model put the final answer inside the think tags or didn't output anything after
            output_text = output_text.replace("</think>", "").replace("<think>", "").strip()
            
    return output_text.strip()
def load_gemma():
    global GEMMA_MODEL, GEMMA_PROCESSOR
    if isinstance(GEMMA_MODEL, str):
        model_name = GEMMA_MODEL
        print(f"Loading {model_name} into GPU without quantization (E4B fits in VRAM)...")
        GEMMA_MODEL = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        GEMMA_PROCESSOR = AutoProcessor.from_pretrained(model_name)

def gemma_ask(prompt_path, ex_prompt, img_path=None, force_think_tag=False, max_new_tokens=2048):
    load_gemma()
    prompt_system, prompt_user = prompt_make(prompt_path, ex_prompt)
    
    # (Removed GEMMA_THINKING tag injection to allow native thinking)
    content = []
    images = []
    
    content.append({"type": "text", "text": prompt_user})
    
    if img_path:
        content.insert(0, {"type": "image"})
        with Image.open(img_path) as img:
            images.append(img.convert("RGB"))
        
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt_system}] + content}
    ]
    
    text = GEMMA_PROCESSOR.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if force_think_tag:
        text += "<think>\n"
    
    if images:
        inputs = GEMMA_PROCESSOR(text=text, images=images, padding=True, return_tensors="pt")
    else:
        inputs = GEMMA_PROCESSOR(text=text, padding=True, return_tensors="pt")
        
    inputs = inputs.to(GEMMA_MODEL.device)
    
    with torch.inference_mode():
        generated_ids = GEMMA_MODEL.generate(**inputs, max_new_tokens=max_new_tokens)
        
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = GEMMA_PROCESSOR.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    del inputs
    del generated_ids
    del generated_ids_trimmed
    import gc
    gc.collect()
    
    logging.info(f"Model raw output:\n{output_text}")
    print(f"RAW GEMMA OUTPUT: {repr(output_text)}")
    
    # Parse out the thinking tags to return only the final answer
    if "</think>" in output_text:
        after_think = output_text.split("</think>")[-1].strip()
        if after_think:
            output_text = after_think
        else:
            output_text = output_text.replace("</think>", "").replace("<think>", "").strip()
    elif "thought\nThinking Process:" in output_text:
        # Fallback for its previous unstructured output behavior
        if "Final Output Generation:" in output_text:
            output_text = output_text.split("Final Output Generation:")[-1].replace('"', '')
        else:
            output_text = output_text.split("\n")[-1]
            
    return output_text.strip()


def gpt_4o_mini(prompt_path, ex_prompt, img_path=None, force_think_tag=False, max_new_tokens=2048):
    api_key = API_KEY
    if not api_key:
        raise ValueError("OpenAI API key is missing. Please provide an API key or set USE_LOCAL_QWEN=True to run locally.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    prompt_system, prompt = prompt_make(prompt_path, ex_prompt)
    content = [{"type": "text", "text": prompt}]
    if img_path:
        base64_image = encode_image(img_path)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
    payload = {
        "model": "gpt-4o-mini",       
        "messages": [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": content}
        ],
        "max_tokens": max_new_tokens
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=120)
    output = response.json()
    return output["choices"][0]['message']["content"]


def local_vllm_ask(prompt_path, ex_prompt, img_path=None, force_think_tag=False, max_new_tokens=2048):
    # Determine model string (fallback to default if not set)
    model_name = GEMMA_MODEL if isinstance(GEMMA_MODEL, str) else "nvidia/Gemma-4-31B-IT-NVFP4"
    
    headers = {
        "Content-Type": "application/json"
    }
    prompt_system, prompt = prompt_make(prompt_path, ex_prompt)
    content = [{"type": "text", "text": prompt}]
    if img_path:
        base64_image = encode_image(img_path)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
    payload = {
        "model": model_name,       
        "messages": [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": content}
        ],
        "max_tokens": max_new_tokens
    }
    response = requests.post("http://localhost:8000/v1/chat/completions", headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        raise ValueError(f"vLLM server error {response.status_code}: {response.text}")
        
    output = response.json()
    return output["choices"][0]['message']["content"]

def ask_model(prompt_path, ex_prompt, img_path=None, force_think_tag=False, max_new_tokens=2048):
    if USE_LOCAL_VLLM:
        return local_vllm_ask(prompt_path, ex_prompt, img_path, force_think_tag, max_new_tokens)
    elif USE_LOCAL_GEMMA:
        return gemma_ask(prompt_path, ex_prompt, img_path, force_think_tag, max_new_tokens)
    elif USE_LOCAL_QWEN:
        return qwen3_vl_ask(prompt_path, ex_prompt, img_path, force_think_tag, max_new_tokens)
    else:
        return gpt_4o_mini(prompt_path, ex_prompt, img_path, force_think_tag, max_new_tokens)
