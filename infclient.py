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

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    Pix2StructForConditionalGeneration,
    Pix2StructProcessor,
    GenerationConfig,
)


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
        sys.path.append('/n/fs/penciller/finetuning/llama-heads')
        from utilities.utils import load_model_checkpoint_shared_shard
        from safetensors.torch import save_file
        # Imports from training modules.
        from config import TrainConfig, FSDPConfig
        from utilities.utils import get_policies, freeze_visual_only, save_fsdp_model_checkpoint_full,  load_model_checkpoint_shared_shard, apply_fsdp_checkpointing
        from transformers import AutoProcessor, get_scheduler, AutoTokenizer
        from factories import get_model_implementation_from_path, get_authoritative_model_path
        if not model_path:
            if model_name == "molmo":
                self.processor = AutoProcessor.from_pretrained(
                'allenai/Molmo-7B-D-0924',
                trust_remote_code=True,
                torch_dtype='auto',
                device_map='auto'
                )

                # load the model
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.client = AutoModelForCausalLM.from_pretrained(
                    'allenai/Molmo-7B-D-0924',
                    trust_remote_code=True,
                    torch_dtype='auto',
                        device_map={'': self.device},
                ).eval()
               
            elif model_name == "pix2struct":
                from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

                self.client = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-ai2d-base")
                self.processor = Pix2StructProcessor.from_pretrained("google/pix2struct-ai2d-base")
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.client = self.client.to(self.device).eval()
            elif "phi-4" in model_name.lower():
                from transformers import AutoProcessor, AutoModelForCausalLM

                phi_path = model_name
                # load processor + base model in FP16
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
                # load only the vision LoRA adapter, then cast entire model to half
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
            elif "magma" in model_name.lower():
                # ─── Magma-8B Multimodal support ──────────────────────────────────────
                from transformers import AutoModelForCausalLM, AutoProcessor
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
            
            
            
            if "internvl" in model_name.lower():
                from transformers import AutoModel, AutoTokenizer
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
            
            else:
                hf_id = get_authoritative_model_path(model_name)
                ModelClass = get_model_implementation_from_path(hf_id)
                wrapped_pretrained = ModelClass(hf_id)
                self.client = wrapped_pretrained.get_base_model()
                self.processor = wrapped_pretrained.get_processor()
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.client = self.client.to(self.device).eval()
            
        else:       
            train_config = TrainConfig()
            fsdp_config = FSDPConfig()
            wrapped_model, processor, _, _, _, _ = load_model_checkpoint_shared_shard(
                train_config=train_config,
                fsdp_config=fsdp_config,
                fsdp_checkpoint_path=model_path,
                model_nickname= model_name
            )
            self.client = wrapped_model.get_base_model()
            self.processor = processor
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.client = self.client.to(self.device).eval()
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

    # 3️⃣  Molmo branch -------------------------------------------------------
    elif "molmo" in model_name.lower():
        if not img_list:
            raise ValueError("Molmo needs at least one image.")
        proc_inputs = processor.process(images=img_list, text=prompt)
        # Make a batch dimension and move to the model’s device
        device = next(model.parameters()).device
        for k, v in proc_inputs.items():
            if isinstance(v, torch.Tensor):
                proc_inputs[k] = v.unsqueeze(0).to(device)
        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            stop_strings="<|endoftext|>",
        )
        out = model.generate_from_batch(
            proc_inputs, gen_cfg, tokenizer=processor.tokenizer
        )
        plen   = proc_inputs["input_ids"].size(1)
        tokens = out[0, plen:]
        return processor.tokenizer.decode(tokens, skip_special_tokens=True)
        # 4️⃣  Phi-4 Multimodal branch -----------------------------------------
    # ——— Phi-4-Multimodal branch ——————————————————————————————
        # ─── Magma-8B branch ──────────────────────────────────────────────────────
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
            # bool masks → long (so torch.sum(mask, dim=…) works inside Magma)
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

        # strip off prompt tokens and decode
        start = inputs["input_ids"].shape[-1]
        gen_ids = out_ids[:, start:]
        return processor.decode(gen_ids[0], skip_special_tokens=True).strip()

    
    elif "internvl" in model_name.lower():
        if messages is None:
            raise ValueError("InternVL requires `messages` with embedded images/text.")

        # 1) extract raw PIL images from messages
        imgs = []
        for msg in messages:
            for part in msg["content"]:
                if part.get("type") in ("image", "input_image") and isinstance(part.get("image"), Image.Image):
                    imgs.append(part["image"])

        if not imgs:
            raise ValueError("InternVL needs at least one image in messages.")

        # 2) prepare tiles & num_patches
        all_tiles = []
        num_patches_list = []
        for img in imgs:
            tiles = dynamic_preprocess(img, image_size=448, use_thumbnail=False, max_num=12)
            num_patches_list.append(len(tiles))
            all_tiles.extend(tiles)

        # 3) transform & cast
        transform = build_transform(input_size=448)
        tensor = torch.stack([transform(tile) for tile in all_tiles])  # float32
        pixel_values = tensor.to(model.device, dtype=next(model.parameters()).dtype)

        # 4) build the internvl‐style prompt from messages
        prompt_lines = []
        for msg in messages:
            if msg["role"] == "user":
                for part in msg["content"]:
                    if part.get("type") in ("image", "input_image"):
                        prompt_lines.append("<image>\n")
                    elif part.get("type") in ("text", "input_text"):
                        prompt_lines.append(part["text"] + "\n")
            elif msg["role"] == "assistant":
                for part in msg["content"]:
                    if part.get("type") in ("text", "input_text"):
                        prompt_lines.append(part["text"] + "\n")
        internvl_prompt = "".join(prompt_lines).strip()

        # 5) generate
        generation_config = dict(max_new_tokens=max_new_tokens, temperature=temperature, **generate_kwargs)
        response = model.chat(
            model.tokenizer,
            pixel_values,
            internvl_prompt,
            generation_config,
            history=None,
            num_patches_list=num_patches_list
        )
        return response
    
    
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
                use_cache=False,          # optional, but recommended
            )
        gen_ids = out_ids[:, seq_len:]
        # decode with the same processor
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
