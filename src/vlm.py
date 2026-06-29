import time
import logging
import torch
import numpy as np

from prismatic import load


class VLM:
    def __init__(self, cfg):
        start_time = time.time()
        self.model = load(cfg.model_id, hf_token=cfg.get("hf_token", None))
        self.model.to(cfg.device, dtype=torch.bfloat16)
        logging.info(f"Loaded VLM in {time.time() - start_time:.3f}s")

    def generate(self, prompt, image, T=0.4, max_tokens=512):
        prompt_builder = self.model.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=prompt)
        prompt_text = prompt_builder.get_prompt()
        generated_text = self.model.generate(
            image,
            prompt_text,
            do_sample=True,
            temperature=T,
            max_new_tokens=max_tokens,
            min_length=1,
        )
        return generated_text

    def get_loss(self, image, prompt, tokens, get_smx=True, T=1):
        "Get unnormalized losses (negative logits) of the tokens"
        prompt_builder = self.model.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=prompt)
        prompt_text = prompt_builder.get_prompt()
        
        image_transform, tokenizer = self.model.vision_backbone.image_transform, self.model.llm_backbone.tokenizer
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.model.device)
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.model.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.model.device) for k, v in pixel_values.items()}

        autocast_dtype = self.model.llm_backbone.half_precision_dtype
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.model.enable_mixed_precision_training):
                output = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    use_cache=False
                )
                next_token_logits = output.logits[0, -1, :]
                
                losses = []
                for token_str in tokens:
                    if token_str in getattr(self.model, 'string2idx', {}):
                        token_id = self.model.string2idx[token_str]
                    else:
                        token_id_list = tokenizer.encode(token_str, add_special_tokens=False)
                        token_id = token_id_list[-1]
                    losses.append(-next_token_logits[token_id].item())
                    
        losses = np.array(losses)
        if get_smx:
            return np.exp(-losses / T) / np.sum(np.exp(-losses / T))
        return losses
