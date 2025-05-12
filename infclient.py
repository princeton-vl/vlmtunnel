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
    if "molmo" in model_name.lower():
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
