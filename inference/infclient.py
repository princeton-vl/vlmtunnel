import os
import sys
import random
import base64
import math
import re

import requests

from io import BytesIO
from typing import Optional, Sequence, Union, List

from PIL import Image
import torch


from transformers import Qwen2_5_VLForConditionalGeneration  # NEW
from qwen_vl_utils import process_vision_info                 # NEW

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    Pix2StructProcessor,
    GenerationConfig,
    get_scheduler,
    AutoModelForImageTextToText,
    Pix2StructForConditionalGeneration, # Pix2StructProcessor is already here
    AutoModel,
    Gemma3ForConditionalGeneration  # <-- Add this import
)

from transformers import GenerationConfig

import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_diff = float('inf')
    best = (1, 1)
    area = width * height
    for i, j in target_ratios:
        tr = i / j
        diff = abs(aspect_ratio - tr)
        if diff < best_diff or (diff == best_diff and area > 0.5 * image_size**2 * i * j):
            best_diff = diff
            best = (i, j)
    return best

def dynamic_preprocess(image: Image.Image,
                       min_num=1, max_num=12,
                       image_size=448, use_thumbnail=False):
    w, h = image.size
    ar = w / h

    # fixed: parenthesize the generator expression
    ratios = sorted(
        ((i, j)
         for n in range(min_num, max_num + 1)
         for i in range(1, n + 1)
         for j in range(1, n + 1)
         if min_num <= i*j <= max_num),
        key=lambda x: x[0] * x[1]
    )

    tgt_i, tgt_j = find_closest_aspect_ratio(ar, ratios, w, h, image_size)
    blocks = tgt_i * tgt_j
    tw, th = image_size * tgt_i, image_size * tgt_j
    resized = image.resize((tw, th))

    tiles = []
    per_row = tw // image_size
    for idx in range(blocks):
        x0 = (idx % per_row) * image_size
        y0 = (idx // per_row) * image_size
        box = (x0, y0, x0 + image_size, y0 + image_size)
        tiles.append(resized.crop(box))

    if use_thumbnail and len(tiles) > 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles
def encode(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def decode(b64_png: str) -> Image.Image:
    """
    Inverse of `encode`.  Convert a Base‑64–encoded PNG string back
    into a fully‑loaded PIL image.

    Parameters
    ----------
    b64_png : str
        Output of `encode` – i.e., the PNG bytes of an image,
        Base‑64–encoded and UTF‑8–decoded.

    Returns
    -------
    PIL.Image.Image
        The reconstructed image (already loaded into memory).
    """
    byte_data = base64.b64decode(b64_png)
    img = Image.open(BytesIO(byte_data))
    img.load()                    # force actual pixel data into memory
    return img




class LocalFetcher:
    def __init__(self, model_path, model_name):

        if not model_path:
            if "molmo" in model_name.lower():
                self.processor = AutoProcessor.from_pretrained(
                'allenai/Molmo-7B-D-0924',
                trust_remote_code=True,
                torch_dtype='auto',
                device_map='auto'
                )

                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.client = AutoModelForCausalLM.from_pretrained(
                    'allenai/Molmo-7B-D-0924',
                    trust_remote_code=True,
                    torch_dtype='auto',
                    device_map='auto'
                    ).eval()
               
            elif model_name == "pix2struct":

                self.client = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-ai2d-base")
                self.processor = Pix2StructProcessor.from_pretrained("google/pix2struct-ai2d-base")
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.client = self.client.to(self.device).eval()
            elif "mistral" in model_name.lower():
                hf_id = model_name  # e.g. "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

                self.tokenizer = AutoTokenizer.from_pretrained(
                    hf_id, trust_remote_code=True, use_auth_token=True
                )
                self.processor = AutoProcessor.from_pretrained(
                    hf_id, trust_remote_code=True, use_auth_token=True
                )


                self.client = AutoModelForImageTextToText.from_pretrained(
                    hf_id,
                    trust_remote_code=True,
                    use_auth_token=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                ).eval()
                self.device = next(self.client.parameters()).device
            
            elif "phi-4" in model_name.lower():

                phi_path = model_name
                self.processor = AutoProcessor.from_pretrained(
                    phi_path,
                    trust_remote_code=True
                )

                self.client = AutoModelForCausalLM.from_pretrained(
                    phi_path,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    _attn_implementation="flash_attention_2",
                )
                self.client.load_adapter(
                    phi_path,
                    adapter_name="vision",
                    device_map="auto",
                    adapter_kwargs={"subfolder": "vision-lora"},
                )
                self.client.set_adapter("vision")
                self.client = self.client.half()  # ensure all weights & activations are FP16
                self.device = next(self.client.parameters()).device
                # ─── Magma-8B branch ──────────────────────────────────────────────────────
            
            elif "qwen2.5-vl-32b" in model_name.lower() or "qwen-2.5-vl-32b" in model_name.lower():
                hf_id = "Qwen/Qwen2.5-VL-32B-Instruct"

                self.processor = AutoProcessor.from_pretrained(hf_id)


                self.client = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    hf_id,
                    torch_dtype="auto",
                    device_map="auto"
                ).eval()

                self.device = next(self.client.parameters()).device
            elif "magma" in model_name.lower():
                # ─── Magma-8B Multimodal support is not fully tested
                dtype = torch.bfloat16

                # load model & processor
                self.client = AutoModelForCausalLM.from_pretrained(
                    "microsoft/Magma-8B",
                    trust_remote_code=True,
                    torch_dtype=dtype
                )
                self.processor = AutoProcessor.from_pretrained(
                    "microsoft/Magma-8B",
                    trust_remote_code=True
                )

                # move to device
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.client = self.client.to(self.device).eval()
            
            
            
            elif "internvl" in model_name.lower():
                self.client = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    use_flash_attn=True,
                    trust_remote_code=True
                ).eval().cuda()
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    use_fast=False
                )
                self.processor = None
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.client.tokenizer = self.tokenizer
            elif "gemma" in model_name.lower():
                model_id_gemma = model_name # e.g., "google/gemma-3-27b-it"
                
                print(f"INFO [GemmaLoader]: Initializing Gemma model: {model_id_gemma}", file=sys.stderr)
                print(f"INFO [GemmaLoader]: Loading processor for {model_id_gemma}", file=sys.stderr)
                self.processor = AutoProcessor.from_pretrained(model_id_gemma)
                
                print(f"INFO [GemmaLoader]: Loading model {model_id_gemma} with device_map='auto' and torch_dtype=torch.bfloat16", file=sys.stderr)
                self.client = Gemma3ForConditionalGeneration.from_pretrained(
                    model_id_gemma,
                    device_map="auto",     
                    torch_dtype=torch.bfloat16,
                    attn_implementation="eager"
                ).eval()
                
                # self.device is for reference; model parts are on devices determined by device_map.
                # For operations, use model.device or tensor.to(model.device).
                if hasattr(self.client, 'device') and self.client.device is not None:
                    self.device = self.client.device
                else:
                    # Fallback: if model.device is not set, infer from a parameter.
                    # This should generally not be needed with device_map="auto".
                    try:
                        self.device = next(self.client.parameters()).device
                    except StopIteration: 
                        print("WARNING [GemmaLoader]: Model has no parameters. Defaulting device.", file=sys.stderr)
                        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                print(f"INFO [GemmaLoader]: Gemma model {model_id_gemma} loaded.", file=sys.stderr)
                print(f"INFO [GemmaLoader]: Effective model device (self.device): {self.device}", file=sys.stderr)
                if hasattr(self.client, 'config') and self.client.config is not None :
                    print(f"INFO [GemmaLoader]: Model config torch_dtype: {self.client.config.torch_dtype}", file=sys.stderr)
                try:
                    param_dtype = next(self.client.parameters()).dtype
                    print(f"INFO [GemmaLoader]: Actual dtype of a model parameter: {param_dtype}", file=sys.stderr)
                except StopIteration:
                    print("INFO [GemmaLoader]: Could not retrieve parameter dtype (model might be empty or on meta device before dispatch).", file=sys.stderr)
            
            else:
                # hf_id = get_authoritative_model_path(model_name)
                # ModelClass = get_model_implementation_from_path(hf_id)
                # wrapped_pretrained = ModelClass(hf_id)
                # self.client = wrapped_pretrained.get_base_model()
                # self.processor = wrapped_pretrained.get_processor()
                # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # self.client = self.client.to(self.device).eval()
                pass
        else:       
            # train_config = TrainConfig()
            # fsdp_config = FSDPConfig()
            # wrapped_model, processor, _, _, _, _ = load_model_checkpoint_shared_shard(
            #     train_config=train_config,
            #     fsdp_config=fsdp_config,
            #     fsdp_checkpoint_path=model_path,
            #     model_nickname= model_name
            # )
            # self.client = wrapped_model.get_base_model()
            # self.processor = processor
            # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # self.client = self.client.to(self.device).eval()
            pass
    def get_model_params(self):
        return self.client, self.processor,self.device


def run_inference(
    model,
    processor,
    model_name: str = "",
    *,
    images: Sequence[Image.Image] | Image.Image | None = None,
    query: str | None = None,
    messages: list[dict] | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0001,
    **generate_kwargs,
    ) -> str:
    """
    Unified generation helper covering three architectures:

        • Pix2Struct  – single seq2seq call with *all* images in one batch
        • Molmo       – chat‑style; use `.process` + `generate_from_batch`
        • Generic chat‑style vision models
    """
    # 1️⃣  Collect images & prompt text
    if messages is None and (images is not None or query):
        # ensure we have a list of images
        imgs = images if isinstance(images, (list, tuple)) else ([images] if images is not None else [])
        content: List[dict] = []
        # embed each image
        for img in imgs:
            content.append({"type": "image", "image": img})
        # then the text
        if query:
            content.append({"type": "text", "text": query})
        messages = [{"role": "user", "content": content}]
    
    img_list: list[Image.Image] = []
    text_parts: list[str]       = []
    if messages:
        for msg in messages:
            for part in msg.get("content", []):
                typ = part.get("type", "")
                if typ in ("text", "input_text"):
                    text_parts.append(part["text"])
                elif typ in ("image", "input_image", "image_url"):
                    payload = part.get("image") or part.get("image_url")
                    if isinstance(payload, str) and payload.startswith("data:image"):
                        b64 = payload.split(",", 1)[1]
                        payload = Image.open(BytesIO(base64.b64decode(b64)))
                    if isinstance(payload, Image.Image):
                        img_list.append(payload)
        prompt = "\n".join(text_parts).strip()
    else:
        prompt = query or ""
        if images is not None:
            img_list = list(images) if isinstance(images, (list, tuple)) else [images]

    
    # 2️⃣  Pix2Struct branch --------------------------------------------------
    if "pix2struct" in model_name.lower():
        if not img_list:
            raise ValueError("Pix2Struct needs at least one image.")
        batch = processor(images=img_list, text=prompt, return_tensors="pt")
        device = next(model.parameters()).device
        for k, v in batch.items():
            batch[k] = v.to(device)
        out_ids = model.generate(
            **batch, max_new_tokens=max_new_tokens, **generate_kwargs
        )
        return processor.decode(out_ids[0], skip_special_tokens=True)

    elif "mistral" in model_name.lower():
        # build the full prompt via the processor’s chat template
        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        inputs = processor(
            text=prompt,
            images=img_list,
            return_tensors="pt"
        )

        # move tensors to the model's device; cast only floats to the model's dtype
        device = next(model.parameters()).device
        target_dtype = next(model.parameters()).dtype
        for k, v in inputs.items():
            if not isinstance(v, torch.Tensor):
                continue
            v = v.to(device)
            if v.dtype.is_floating_point:
                v = v.to(target_dtype)
            inputs[k] = v

        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            **generate_kwargs,
        )

        seq_len = inputs["input_ids"].shape[1]
        gen_ids = out_ids[0, seq_len:]
        return processor.decode(gen_ids, skip_special_tokens=True).strip()
    
    
    
    elif "molmo" in model_name.lower():
        molmo_use_only_last_image_turn_strict = True # Default. See Appendix of paper

        active_messages_to_process = messages
        active_img_list_to_process = img_list 

        if molmo_use_only_last_image_turn_strict:
            last_msg_idx_with_img = -1
            the_last_image_obj = None

            for i in range(len(messages) - 1, -1, -1):
                current_message_node = messages[i]
                current_message_content = current_message_node.get("content")
                if isinstance(current_message_content, list):
                    for part in current_message_content:
                        # Ensure it's an image part and the 'image' key holds a loaded PIL.Image object
                        if part.get("type") in ("image", "input_image", "image_url") and isinstance(part.get("image"), Image.Image):
                            last_msg_idx_with_img = i
                            the_last_image_obj = part.get("image")
                            break 
                if last_msg_idx_with_img != -1:
                    break 
            
            if last_msg_idx_with_img != -1:

                active_messages_to_process = [messages[last_msg_idx_with_img]]
                active_img_list_to_process = [the_last_image_obj] 
            else:

                active_messages_to_process = messages 
                active_img_list_to_process = [] # Explicitly no images

        _conversation_for_template = []
        for msg in active_messages_to_process:
            current_turn_text_parts = []
            raw_content = msg.get("content")

            if isinstance(raw_content, list):
                for part in raw_content:
                    if part.get("type") in ("text", "input_text"):
                        current_turn_text_parts.append(part["text"])
            elif isinstance(raw_content, str):
                current_turn_text_parts.append(raw_content)
            
            current_content_str = " ".join(current_turn_text_parts).strip()
            current_role = msg.get("role")

            if _conversation_for_template and _conversation_for_template[-1]["role"] == current_role:
                if current_content_str: 
                    if _conversation_for_template[-1]["content"]:
                        _conversation_for_template[-1]["content"] += "\n" + current_content_str
                    else:
                        _conversation_for_template[-1]["content"] = current_content_str
            else:
                _conversation_for_template.append({
                    "role": current_role,
                    "content": current_content_str
                })

        templated_prompt_str = processor.tokenizer.apply_chat_template(
            conversation=_conversation_for_template,
            tokenize=False,
            add_generation_prompt=True
        )
        

        proc_inputs = processor.process(images=active_img_list_to_process, text=templated_prompt_str)
        
        device = next(model.parameters()).device
        for k, v in proc_inputs.items():
            if isinstance(v, torch.Tensor):
                proc_inputs[k] = v.unsqueeze(0).to(device)

        plen = proc_inputs["input_ids"].shape[1]
        max_pos = getattr(model.config, "max_position_embeddings", 4096)
        total_max = min(plen + max_new_tokens, max_pos)

        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            max_length=total_max
        )

        out = model.generate_from_batch(
            proc_inputs,
            gen_cfg,
            tokenizer=processor.tokenizer
        )

        tokens = out[0, plen:]
        return processor.tokenizer.decode(tokens, skip_special_tokens=True)
    elif "qwen" in model_name.lower() and "vl" in model_name.lower():
        #messages already built exactly like in the docs
        if messages is None:
            raise ValueError("Qwen-2.5-VL-32B needs `messages` structured as "
                             "[{'role':'user','content':[{'type':'image',...},{'type':'text',...}]}].")


        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Extract images/videos from messages → helper bundled with Qwen
        image_inputs, video_inputs = process_vision_info(messages)


        inputs = processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)


        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
        out_text = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        return out_text[0].strip()
    elif "internvl" in model_name.lower():

        #
        prompt_text = prompt               
        pixel_values = None
        if img_list:
            base_img   = img_list[-1]                      # use last image
            tiles      = dynamic_preprocess(base_img,
                                            image_size=448,
                                            use_thumbnail=True,
                                            max_num=12)
            transform  = build_transform(448)
            pixel_vals = [transform(t) for t in tiles]     # list[3×448×448]
            pixel_values = torch.stack(pixel_vals, dim=0)  # (N,3,448,448)
            pixel_values = pixel_values.to(model.device, torch.bfloat16)

        #
        # 3.  Call InternVL’s built-in chat helper.
        #
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("InternVL model must have `.tokenizer` attribute.")

        gen_cfg = dict(max_new_tokens=max_new_tokens,
                        do_sample      = temperature > 0,
                        temperature    = temperature)

        response = model.chat(tokenizer,
                                pixel_values,
                                prompt_text,
                                gen_cfg)
        return response.strip()

    elif "magma" in model_name.lower():
        # build conversation list
        convs = [
            {"role": "system", "content": "You are agent that can see, talk and act."},
        ]
        # append all user/assistant exchanges
        for msg in messages:
            convs.append({
                "role": msg["role"],
                "content": "".join(
                    part.get("text", "") if part.get("type") in ("text", "input_text") else "<image>"
                    for part in msg["content"]
                )
            })

        # render chat template
        prompt = processor.tokenizer.apply_chat_template(
            convs, tokenize=False, add_generation_prompt=True
        )

        # prepare pixel inputs
        imgs = [
            part["image"]
            for msg in messages
            for part in msg["content"]
            if part.get("type") in ("image", "input_image") and isinstance(part.get("image"), Image.Image)
        ]
        inputs = processor(images=imgs, texts=prompt, return_tensors="pt")
        # match shape expected by Magma
        inputs["pixel_values"] = inputs["pixel_values"].unsqueeze(0)
        inputs["image_sizes"]  = inputs["image_sizes"].unsqueeze(0)

        # move to device & dtype
        target_dtype = next(model.parameters()).dtype  # should be torch.bfloat16
        # move tensors to device:
        model_dtype = next(model.parameters()).dtype
        for k, v in inputs.items():
            if not isinstance(v, torch.Tensor):
                continue
            # always move to GPU/CPU
            v = v.to(model.device)
            # floats → bfloat16 (or whatever the model uses)
            if v.dtype.is_floating_point:
                v = v.to(model_dtype)

            elif v.dtype == torch.bool:
                v = v.long()
            # leave int/long tensors alone
            inputs[k] = v
        # generate
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **generate_kwargs,
            )

        start = inputs["input_ids"].shape[-1]
        gen_ids = out_ids[:, start:]
        return processor.decode(gen_ids[0], skip_special_tokens=True).strip()

    elif "gemma" in model_name.lower():


        if not messages: 
            raise ValueError("Gemma [Tutorial]: Input 'messages' list is empty.")

        merged_intermediate_messages = []
        if messages: 
            current_processing_message = {
                "role": messages[0]["role"],
                "content": list(messages[0].get("content", [])) 
            }

            for i in range(1, len(messages)):
                next_message = messages[i]
                next_message_content = list(next_message.get("content", []))

                if next_message["role"] == current_processing_message["role"]:

                    current_processing_message["content"].extend(next_message_content)
                else:

                    merged_intermediate_messages.append(current_processing_message)

                    current_processing_message = {
                        "role": next_message["role"],
                        "content": next_message_content
                    }

            if current_processing_message:
                merged_intermediate_messages.append(current_processing_message)

        has_user_message = any(msg.get("role") == "user" for msg in merged_intermediate_messages)
        if has_user_message and not any(msg.get("role") == "system" for msg in merged_intermediate_messages):

            final_processed_messages = [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]}] + merged_intermediate_messages
        else:
            final_processed_messages = merged_intermediate_messages
        

        if not final_processed_messages and messages:
            raise ValueError("Gemma: Message processing resulted in an empty list unexpectedly.")
        elif not final_processed_messages: # If original messages was also empty, this is caught by the first check.
            pass


        inputs = processor.apply_chat_template(
            final_processed_messages, # Use the fully processed messages
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16) # model.device comes from LocalFetcher

        if inputs["input_ids"].dtype == torch.bfloat16: # This should NOT happen
            raise RuntimeError("CRITICAL ERROR: input_ids became bfloat16. This should not happen with tensor.to() on an int tensor.")
        if "attention_mask" in inputs and inputs["attention_mask"].dtype == torch.bfloat16: # This should also NOT happen
            raise RuntimeError("CRITICAL ERROR: attention_mask became bfloat16. This should not happen with tensor.to() on an int tensor.")
        if "pixel_values" in inputs and inputs["pixel_values"].dtype != torch.bfloat16:
                print(f"WARNING [GemmaRun Tutorial]: pixel_values is {inputs['pixel_values'].dtype}, not bfloat16, after .to() call. This is unexpected if it was float32.", file=sys.stderr)

        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation_output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens, 
                do_sample=False
            )
            
        generated_ids = generation_output[0][input_len:]

        # Step 3: Decode
        decoded_text = processor.decode(generated_ids, skip_special_tokens=True)
        return decoded_text.strip()
    
    
    elif "phi-4" in model_name.lower():
        model.set_adapter("vision")
        device = next(model.parameters()).device

        # 1) flatten out all images & text in order
        prompt = "<|user|>"
        img_list = []
        img_counter = 1

        for msg in messages:
            if msg["role"] != "user":
                continue
            for part in msg["content"]:
                t = part.get("type")
                if t in ("image", "input_image"):
                    # insert image token
                    prompt += f"<|image_{img_counter}|>"
                    img = part.get("image") or part.get("image_url")
                    if not isinstance(img, Image.Image):
                        # decode base64 if needed
                        b64 = img.split(",",1)[1]
                        img = Image.open(BytesIO(base64.b64decode(b64)))
                    img_list.append(img)
                    img_counter += 1
                elif t in ("text", "input_text"):
                    prompt += part["text"]
        prompt += "<|end|><|assistant|>"

        # 2) call processor with that prompt string + images
        batch = processor(
            text=prompt,
            images=img_list,
            return_tensors="pt"
        ).to(device, torch.float16)
        seq_len = batch["input_ids"].shape[1]
        # clone and tweak the model’s own config
        gen_cfg = GenerationConfig.from_pretrained("microsoft/Phi-4-multimodal-instruct")
        gen_cfg.max_new_tokens = max_new_tokens
        # ensure max_length covers prompt+new
        gen_cfg.max_length = seq_len + max_new_tokens
        gen_cfg.num_logits_to_keep = seq_len

        with torch.no_grad():
            out_ids = model.generate(
                **batch,
                generation_config=gen_cfg,
                use_cache=False,         
            )
        gen_ids = out_ids[:, seq_len:]
        return processor.decode(gen_ids[0], skip_special_tokens=True).strip()
    
    
# 4️⃣  Generic chat vision branch ----------------------------------------
    if messages is None:
        stub = [{"type":"image"} for _ in img_list]
        stub.append({"type":"text","text": prompt})
        messages = [{"role":"user","content": stub}]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    # re-collect images
    imgs = []
    for msg in messages:
        for part in msg.get("content", []):
            if part.get("type")=="image" and isinstance(part.get("image"), Image.Image):
                imgs.append(part["image"])
    if not imgs and img_list:
        imgs = img_list

    # preprocess & move to device
    if imgs:
        inputs = processor(images=imgs, text=prompt, return_tensors="pt")
    else:
        inputs = processor(text=prompt, return_tensors="pt")

    device = next(model.parameters()).device
    for k,v in inputs.items():
        inputs[k] = v.to(device)
        if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
            inputs[k] = inputs[k].to(next(model.parameters()).dtype)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **generate_kwargs,
        )

    # ─── decode ONLY the newly generated tokens ────────────────────────
    start         = inputs["input_ids"].shape[1]
    gen_token_ids = out_ids[0, start:]
    # most processors expose a .tokenizer attribute; fall back to AutoTokenizer if not
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model.config._name_or_path)
    return tok.decode(gen_token_ids, skip_special_tokens=True)