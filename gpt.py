import base64
import requests
import os

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# OpenAI API Key
API_KEY = ""
USE_LOCAL_QWEN = True

# QWEN_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
QWEN_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
QWEN_PROCESSOR = None

def load_qwen():
    global QWEN_MODEL, QWEN_PROCESSOR
    if isinstance(QWEN_MODEL, str):
        model_name = QWEN_MODEL
        print(f"Loading {model_name} into GPU with 8-bit quantization...")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        QWEN_MODEL = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto", quantization_config=quantization_config
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

def qwen3_vl_ask(prompt_path, ex_prompt, img_path=None):
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
        generated_ids = QWEN_MODEL.generate(**inputs, max_new_tokens=128)
        
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = QWEN_PROCESSOR.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return output_text.strip()


def gpt_4o_mini(prompt_path, ex_prompt, img_path=None):
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
        ]}
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    output = response.json()
    return output["choices"][0]['message']["content"]


def ask_model(prompt_path, ex_prompt, img_path=None):
    if USE_LOCAL_QWEN:
        return qwen3_vl_ask(prompt_path, ex_prompt, img_path)
    else:
        return gpt_4o_mini(prompt_path, ex_prompt, img_path)
