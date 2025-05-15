#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_visual_attention.py

Creates synthetic “find-the-number-on-the-target-shape-or-color” problems:
  - A single image with an invisible GRID_SIZE×GRID_SIZE grid of numbers.
  - Behind exactly one number is the target: either a shape of a specific COLOR
    (trial "color") or a specific SHAPE_TYPE (trial "shape").
  - The model’s task is to output the correct grid number, formatted inside curly braces.

Also supports the *reverse* lookup:
  - trial "color_of_number": “What color is on number N?”
  - trial "shape_of_number": “What shape is on number N?”

And a *chain-following* trial:
  - trial "chain": Every cell has a unique (shape, color) pair.
    You start from a given (shape, color), then repeatedly move to the cell
    whose pair matches the label overlaid in your current cell—for X steps.
    Each cell in the chain shows the *next* pair name (color + shape).
    The question: what is the color of the final shape? Answer in curly braces.

Supports:
  • Local models (vision-capable) via --use-local
  • OpenAI Vision-capable API (requires OPENAI_API_KEY env var)
  • OpenRouter Vision-capable API (requires OPENROUTER_API_KEY env var)

Reports:
  • Per-trial and aggregate accuracy
  • Δ log-odds of accuracy by target CATEGORY (color or shape)

Usage:
------
    export OPENAI_API_KEY=...
    export OPENROUTER_API_KEY=...
    python test_visual_attention.py \
        --models openai openrouter local \
        --trials color shape color_of_number shape_of_number chain \
        --grid-size 5 \
        --cell-size 80 \
        --num 20 \
        --chain-length 3 \
        [--few-shot] [--use-local] [--test-image]
"""

import os
import sys
import re
import gc
import random
import math
import base64
import argparse
import requests
from io import BytesIO
from typing import Optional, Sequence, Tuple, List
import itertools
from collections import Counter, defaultdict
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoProcessor, get_scheduler, AutoTokenizer, AutoModelForCausalLM
from infclient import run_inference, LocalFetcher

# ─────────── Hyperparameters ───────────
SHAPE_TYPES = ['circle', 'square', 'triangle']
COLORS      = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'magenta', 'pink', "cyan", "gray", "lime"]
MODEL_CODE  = "o3-2025-04-16 "#"o4-mini-2025-04-16" # o3-2025-04-16 

# ─────────── Utility Functions ───────────

def encode(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def decode(b64_png: str) -> Image.Image:
    byte_data = base64.b64decode(b64_png)
    img = Image.open(BytesIO(byte_data))
    img.load()
    return img

_CURLY_RE = re.compile(r"\{([^{}]+)\}")

def extract_braced(text: str) -> str:
    """
    Extracts the first substring inside curly braces {}.
    If none found, returns the original text.
    """
    m = _CURLY_RE.search(text)
    return m.group(1).strip() if m else text.strip()

# ─────────── Updated: analyze_presence_effects ───────────
def analyze_presence_effects(
    records: List[dict],
    categories: List[str],
    key: str,
    label: str
):
    """
    Computes Δ-log-odds, 95 % CI, and p-value (Wald z-test)
    of accuracy when target category == cat vs != cat.
    """
    eps = 1e-6  # small constant to avoid p=0 or 1
    def _safe_logit(p):
        return math.log(p / (1 - p))

    def _norm_cdf(z):
        return (1.0 + math.erf(z / math.sqrt(2.0))) / 2.0

    rows = []
    for cat in categories:
        n1 = acc1 = n0 = acc0 = 0
        for r in records:
            if r[key] == cat:
                n1  += 1
                acc1 += r['correct']
            else:
                n0  += 1
                acc0 += r['correct']

        if n1 == 0 or n0 == 0:
            rows.append((cat, float('nan'), float('nan'), float('nan'), n1, n0))
            continue

        # raw empirical rates
        p1_raw, p0_raw = acc1 / n1, acc0 / n0
        # clamp to [eps, 1-eps]
        p1 = max(min(p1_raw, 1 - eps), eps)
        p0 = max(min(p0_raw, 1 - eps), eps)

        # Δ-log-odds
        coeff  = _safe_logit(p1) - _safe_logit(p0)

        # variance of Δ-log-odds: var = 1/(n1*p1*(1-p1)) + 1/(n0*p0*(1-p0))
        var   = 1/(n1 * p1 * (1 - p1)) + 1/(n0 * p0 * (1 - p0))
        se    = math.sqrt(var)
        z     = coeff / se
        ci_lo = coeff - 1.96 * se
        ci_hi = coeff + 1.96 * se
        pval  = 2 * (1 - _norm_cdf(abs(z)))

        rows.append((cat, coeff, (ci_lo, ci_hi), pval, n1, n0))

    # sort and print as before…
    rows.sort(key=lambda x: (not math.isnan(x[1]), abs(x[1]) if not math.isnan(x[1]) else 0),
              reverse=True)

    print(f"\n=== Δ-log-odds by {key} for {label} ===")
    print(f"{'category':<10} {'Δ log-odds':>12} {'95% CI':>26} {'p':>8} {'#target':>8} {'#others':>8}")
    for cat, coeff, ci, pval, n1, n0 in rows:
        if math.isnan(coeff):
            print(f"{cat:<10} {'—':>12} {'—':>26} {'—':>8} {n1:>8} {n0:>8}")
        else:
            ci_txt = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
            print(f"{cat:<10} {coeff:+12.3f} {ci_txt:>26} {pval:>8.4f} {n1:>8} {n0:>8}")

def generate_two_shot_examples(
    grid_size: int,
    cell_size: int,
    odir: str,
    one_canvas: bool = False,
    chain_length: int = 3
) -> List[Tuple[Image.Image, str, str]]:
    """
    Build two few-shot examples for the chain trial, with full chain description.
    Returns list of tuples: (image, demonstration_text, final_color)
    """
    os.makedirs(odir, exist_ok=True)
    examples = []
    for i in range(2):
        img, start_pair, final_color, chain, _ = generate_chain_image(
            grid_size, cell_size, chain_length
        )
        # construct full chain demonstration text
        desc = f"Start at {chain[0][1]} {chain[0][0]}"
        for shape, color in chain[1:]:
            desc += f", then go to the cell with {color} {shape}"
        desc += f". Answer in {{{final_color}}}."
        # save example image
        img.save(f"{odir}/two_shot_{i}.png")
        examples.append((img, desc, final_color))
    return examples
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

class InferenceClient:
    """
    Two back-ends only:
      • name == "openai"   → OpenAI Completion API (vision-capable model)
      • anything else      → OpenRouter Completion API (vision-capable model)

    Expected environment variables
        OPENAI_API_KEY        – for the OpenAI back-end
        OPENROUTER_API_KEY    – for the OpenRouter back-end
    """
    # ──────────────────────────────────────────── init
    def __init__(self,
                 name: str,
                 imgpair1,
                 fs_examples: Optional[List[Tuple[Image.Image,str,str]]] = None,
                 *,
                 api_key: Optional[str] = None,
                 openrouter_model: str = "grok-1",
                 use_local: bool = False,
                 model_name: str = "",
                 model_path: str = None):
        self.name = name.lower()
        self.or_model = openrouter_model.lower()
        if imgpair1 is None or imgpair1[0] is None:
            # no demo image available (e.g. non-few-shot load-dataset)
            self.img1 = None
            self.start_point = None
            self.color1 = None
        else:
            self.img1       = encode(imgpair1[0])
            self.start_point = imgpair1[1]
            self.color1     = imgpair1[2]

        self.use_local = use_local
        self.model_name = model_name
        self.model_path = model_path
        self.fs_examples = fs_examples or []
        self.max_retries = 3  # Or your preferred number of retries
        self.initial_wait_time = 5  # Initial wait time in seconds
        if use_local:
            self.client, self.processor, self.device = LocalFetcher(
                model_path=self.model_path,
                model_name=self.model_name
            ).get_model_params()
            

        elif self.name == "openai": 
            import openai# ─── OpenAI
            key = api_key or os.getenv("OPENAI_LAB_KEY")
            if key is None:
                raise RuntimeError("OPENAI_LAB_KEY is not set")
            self.client = openai.OpenAI(api_key=key,timeout=2000)

        else:
            import openai
            # ─── OpenRouter
            key = api_key or os.getenv("OPENROUTER_LAB_TOK")
            if key is None:
                raise RuntimeError("OPENROUTER_LAB_TOK is not set")
            self._or_endpoint = "https://openrouter.ai/api/v1/chat/completions"
            self._or_headers  = {
                "Authorization": f"Bearer {key}",
                "Content-Type":  "application/json",
            }

    # ──────────────────────────────────────────── helpers
    
    @staticmethod
    def _data_url(b64: str) -> dict:
        # OpenRouter uses {"image_url": {"url": "data:image/png;base64,..." } }
        return {"url": f"data:image/png;base64,{b64}"}

    def _or_post(self, messages: list[dict]) -> str:
        """Call OpenRouter and return the assistant’s text content.
        On failure, print the error and return a random fallback reply."""
        payload = {
            "model":     self.or_model,
            "messages":  messages,
            "reasoning": {"effort": "high"},
        }
        try:
            r = requests.post(
                self._or_endpoint,
                headers=self._or_headers,
                json=payload,
                timeout=120
            )
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            # Log the error and return a random fallback response
            print(f"[OpenRouter error] {e}")
            return random.choice(["{{no}}", "{{yes}}" ])

    
    def ask_single(self, prompt: str, b: str, few_shot: bool) -> str:
        backend, model_spec = self.name, self.model_name

        # decode our test image
        img = decode(b)

        if not few_shot:
            # no-shot: exactly as before
            if backend == "local":
                return run_inference(self.client, self.processor, self.model_name,
                                          images=[img], query=prompt,
                                          temperature=0.0001)
            elif backend == "openai":
                import openai
                key = os.getenv("OPENAI_LAB_KEY")
                if key is None:
                    raise RuntimeError("OPENAI_LAB_KEY not set")
                resp = self.client.responses.create(
                    model=model_spec,
                    reasoning={"effort":"high"},
                    input=[{
                        "role":"user","content":[
                            {"type":"input_text","text":prompt},
                            {"type":"input_image","image_url":f"data:image/png;base64,{b}"}
                        ]
                    }]
                )
                return resp.output_text
            else:  # openrouter
                usr = {"role":"user","content":[
                    {"type":"text","text":prompt},
                    {"type":"image_url","image_url":self._data_url(b)}
                ]}
                return self._or_post([usr])

        # --- few_shot branch: build demos from self.fs_examples ---
        if backend == "local":
            demo_msgs = []
            for ex_img, ex_prompt, ex_ans in self.fs_examples:
                demo_msgs.append({"role":"user","content":[
                    {"type":"image","image":ex_img},
                    {"type":"text","text":ex_prompt}
                ]})
                demo_msgs.append({"role":"assistant","content":[
                    {"type":"text","text":ex_ans}
                ]})
                demo_msgs.append({"role":"user","content":[
                {"type":"text","text":"That is correct."}
                ]})
            # finally our new query
            demo_msgs.append({"role":"user","content":[
                {"type":"image","image":img},
                {"type":"text","text":prompt}
            ]})
            return run_inference(self.client, self.processor,self.model_name,
                                      messages=demo_msgs,
                                      temperature=0.0001)

        if backend == "openai":
            import openai, base64
            key = os.getenv("OPENAI_LAB_KEY")
            if key is None:
                raise RuntimeError("OPENAI_LAB_KEY not set")
            client = self.client
            demo = []
            for ex_img, ex_prompt, ex_ans in self.fs_examples:
                ex_b64 = encode(ex_img)
                demo.extend([
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text",  "text": ex_prompt},
                            {"type": "input_image", "image_url": f"data:image/png;base64,{ex_b64}"},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": ex_ans},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "That is correct."},
                        ],
                    },
                ])

            # append our actual query
            demo.append({
                "role": "user",
                "content": [
                    {"type": "input_text",  "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{b}"},
                ],
            })

            num_total_attempts = self.max_retries
            for attempt_idx in range(num_total_attempts):
                try:
                    resp = self.client.responses.create(
                        model=self.model_name,
                        reasoning={"effort": "high"},
                        input=demo,
                    )
                    return resp.output_text

                except (openai.APIError,
                        openai.APIConnectionError,
                        openai.RateLimitError) as e:

                    if attempt_idx == num_total_attempts - 1:
                        raise  # bubble up after last failure

                    # exponential back-off, honouring Retry-After when present
                    wait_time = self.initial_wait_time * (2 ** attempt_idx)
                    if isinstance(e, openai.RateLimitError):
                        ra = getattr(getattr(e, "response", None), "headers", {}).get("retry-after")
                        if ra and ra.isdigit():
                            wait_time = max(wait_time, int(ra))

                    print(f"[OpenAI] {e} — retry {attempt_idx+1}/{num_total_attempts} in {wait_time}s")
                    time.sleep(wait_time)
                except openai.APIError as e:
                    print(f"OpenAI API error (few-shot, non-retryable or unhandled for retry): {e}")
                    raise
            # This line should ideally not be reached.
            raise RuntimeError(f"OpenAI API call (few-shot) failed exhaustively after {num_total_attempts} attempts.")

        # OPENROUTER few_shot
        usr = {"role":"user","content":[
            {"type":"text","text":prompt},
            {"type":"image_url","image_url":self._data_url(b)}
        ]}
        demo = []
        for ex_img, ex_prompt, ex_ans in self.fs_examples:
            ex_b64 = encode(ex_img)
            demo.append({"role":"user","content":[
                {"type":"text","text":ex_prompt},
                {"type":"image_url","image_url":self._data_url(ex_b64)}
            ]})
            demo.append({"role":"assistant","content":[
                {"type":"text","text":ex_ans}
            ]})
            demo.append({"role":"user","content":[
            {"type":"text","text":"That is correct."}
            ]})
        demo.append(usr)
        return self._or_post(demo)



# ─────────── Chain Image Generation ───────────
# ─────────── Updated: generate_chain_image ───────────
def generate_chain_image(
    grid_size: int,
    cell_size: int,
    chain_length: int
) -> Tuple[
        Image.Image,        # rendered grid
        Tuple[str, str],    # start_pair  (shape, color)
        str,                # final_color
        List[Tuple[str,str]],# full chain of pairs
        dict[str, int]      # color_counts over entire grid
]:
    """
    Generates a chain‑following puzzle.
    Returns img, start_pair, final_color, chain, color_counts.
    """
    # build grid of unique pairs
    all_pairs = [(s, c) for s in SHAPE_TYPES for c in COLORS]
    needed    = grid_size * grid_size
    if needed > len(all_pairs):
        raise ValueError(f"Grid too large: need {needed}, max {len(all_pairs)}")

    sampled   = random.sample(all_pairs, needed)
    positions = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    random.shuffle(positions)
    grid_map  = {pos: pair for pos, pair in zip(positions, sampled)}
    pos_map   = {pair: pos for pos, pair in grid_map.items()}
    color_counts = Counter(col for (_, col) in grid_map.values())

    # draw shapes
    img  = Image.new('RGB', (grid_size*cell_size, grid_size*cell_size), 'white')
    draw = ImageDraw.Draw(img)
    for (r, c), (shape, colr) in grid_map.items():
        x, y = c*cell_size + cell_size//2, r*cell_size + cell_size//2
        sz = cell_size * 0.6
        if shape == 'circle':
            draw.ellipse([x-sz/2, y-sz/2, x+sz/2, y+sz/2], fill=colr)
        elif shape == 'square':
            draw.rectangle([x-sz/2, y-sz/2, x+sz/2, y+sz/2], fill=colr)
        else:
            pts = [(x, y-sz/2), (x-sz/2, y+sz/2), (x+sz/2, y+sz/2)]
            draw.polygon(pts, fill=colr)

    # sample chain
    chain = random.sample(sampled, chain_length + 1)
    start_pair  = chain[0]
    final_color = chain[-1][1]

    font = ImageFont.load_default()

    # label chain cells with next pair
    for i in range(chain_length):
        cur_pair, next_pair = chain[i], chain[i+1]
        r, c = pos_map[cur_pair]
        label = f"{next_pair[1]} {next_pair[0]}"
        x, y = c*cell_size + cell_size//2, r*cell_size + cell_size//2
        bbox = draw.textbbox((0, 0), label, font=font)
        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        draw.text((x-w/2, y-h/2), label, fill='black', font=font)

    # random labels on all other cells incl. final
    for pos, pair in grid_map.items():
        if pair in chain:
            continue
        r, c = pos
        dummy_shape, dummy_color = random.choice(all_pairs)
        label = f"{dummy_color} {dummy_shape}"
        x, y = c*cell_size + cell_size//2, r*cell_size + cell_size//2
        bbox = draw.textbbox((0, 0), label, font=font)
        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        draw.text((x-w/2, y-h/2), label, fill='black', font=font)

    # random label on final cell
    r, c = pos_map[chain[-1]]
    dummy_shape, dummy_color = random.choice(all_pairs)
    label = f"{dummy_color} {dummy_shape}"
    x, y = c*cell_size + cell_size//2, r*cell_size + cell_size//2
    bbox = draw.textbbox((0, 0), label, font=font)
    w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    draw.text((x-w/2, y-h/2), label, fill='black', font=font)

    return img, start_pair, final_color, chain, color_counts

# ─────────── Standard Image Generation ───────────
# ─────────── Updated: generate_trial_image ───────────
def generate_trial_image(
    trial_type: str,
    grid_size: int,
    cell_size: int
) -> Tuple[Image.Image, int, str, dict[str, int]]:
    """
    trial_type ∈ {color, shape, color_of_number, shape_of_number}
    Returns
    -------
    img           : PIL.Image
    target_number : int
    target_feat   : str   (color or shape the query is about)
    color_counts  : dict  mapping color → count in the grid
    """
    total = grid_size * grid_size
    idx = random.randrange(total)
    row, col = divmod(idx, grid_size)
    target_number = idx + 1

    colors_grid: list[list[str]] = [[None]*grid_size for _ in range(grid_size)]

    if trial_type in ("color", "shape"):
        if trial_type == "color":
            target_feat   = random.choice(COLORS)
            others        = [c for c in COLORS if c != target_feat]
            shape_choices = SHAPE_TYPES
        else:
            target_feat   = random.choice(SHAPE_TYPES)
            others        = [s for s in SHAPE_TYPES if s != target_feat]
            color_choices = COLORS

        img  = Image.new('RGB', (grid_size*cell_size, grid_size*cell_size), 'white')
        draw = ImageDraw.Draw(img)

        for r in range(grid_size):
            for c in range(grid_size):
                x, y = c*cell_size + cell_size//2, r*cell_size + cell_size//2
                if (r, c) == (row, col):
                    feat = target_feat
                else:
                    feat = random.choice(others)
                if trial_type == "color":
                    shape, colr = random.choice(shape_choices), feat
                else:
                    shape, colr = feat, random.choice(color_choices)
                colors_grid[r][c] = colr
                sz = cell_size * 0.6
                if shape == 'circle':
                    rr = sz/2
                    draw.ellipse([x-rr, y-rr, x+rr, y+rr], fill=colr)
                elif shape == 'square':
                    hh = sz/2
                    draw.rectangle([x-hh, y-hh, x+hh, y+hh], fill=colr)
                else:
                    hh = sz/2
                    pts = [(x, y-hh), (x-hh, y+hh), (x+hh, y+hh)]
                    draw.polygon(pts, fill=colr)

    else:
        # "color_of_number" or "shape_of_number"
        img  = Image.new('RGB', (grid_size*cell_size, grid_size*cell_size), 'white')
        draw = ImageDraw.Draw(img)
        shapes = [[random.choice(SHAPE_TYPES) for _ in range(grid_size)]
                  for _ in range(grid_size)]
        colors = [[random.choice(COLORS)      for _ in range(grid_size)]
                  for _ in range(grid_size)]
        for r in range(grid_size):
            for c in range(grid_size):
                x, y = c*cell_size + cell_size//2, r*cell_size + cell_size//2
                shape, colr = shapes[r][c], colors[r][c]
                colors_grid[r][c] = colr
                sz = cell_size * 0.6
                if shape == 'circle':
                    rr = sz/2
                    draw.ellipse([x-rr, y-rr, x+rr, y+rr], fill=colr)
                elif shape == 'square':
                    hh = sz/2
                    draw.rectangle([x-hh, y-hh, x+hh, y+hh], fill=colr)
                else:
                    hh = sz/2
                    pts = [(x, y-hh), (x-hh, y+hh), (x+hh, y+hh)]
                    draw.polygon(pts, fill=colr)
        target_feat = colors[row][col] if trial_type == "color_of_number" else shapes[row][col]

    # overlay numbers
    font = ImageFont.load_default()
    for r in range(grid_size):
        for c in range(grid_size):
            num = r*grid_size + c + 1
            x, y = c*cell_size + cell_size//2, r*cell_size + cell_size//2
            txt = str(num)
            bbox = draw.textbbox((0, 0), txt, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            draw.text((x-w/2, y-h/2), txt, fill='black', font=font)

    flat_colors = [c for rowc in colors_grid for c in rowc]
    color_counts = Counter(flat_colors)
    return img, target_number, target_feat, color_counts

# ─────────── Main ───────────

from collections import defaultdict, Counter

def cooccurrence_counts(
    records: list[dict],
    *,
    row_key: str,
    col_key: str,
    title: str
):
    """
    Print a simple contingency table of counts for records[row_key] × records[col_key].
    """
    rows = sorted({r[row_key] for r in records})
    cols = sorted({r[col_key] for r in records})
    mat = defaultdict(lambda: defaultdict(int))
    for r in records:
        mat[r[row_key]][r[col_key]] += 1

    print(f"\n=== {title} ===")
    hdr = " " * 12 + "".join(f"{c:>8}" for c in cols)
    print(hdr)
    for row in rows:
        row_line = f"{row:<12}" + "".join(f"{mat[row][c]:>8}" for c in cols)
        print(row_line)

def analyze_chain_presence_effects(
    records: list[dict],
    *,
    title: str
):
    """
    For chain trials, compute Δ-log-odds of correct when a given
    start-color or start-shape is present vs not.
    Assumes each record has 'start_color' and 'start_shape' fields.
    """
    chain_recs = [r for r in records if r['trial']=='chain']
    print(f"\n=== {title} ===")
    analyze_presence_effects(chain_recs, COLORS,    'start_color', f"{title} (by start color)")
    analyze_presence_effects(chain_recs, SHAPE_TYPES,'start_shape', f"{title} (by start shape)")

def _safe_logit(p: float, eps: float = 1e-6) -> float:
    p = max(min(p, 1 - eps), eps)
    return math.log(p / (1.0 - p))

def analyze_prediction_distribution(
    records: list[dict],
    categories: list[str],
    *,
    field: str,
    title: str
):
    """
    Frequency‑and‑bias analysis of categorical predictions.

    For every `cat ∈ categories` we print:
        • count of times it appeared in `record[field]`
        • empirical probability
        • Δ‑log‑odds relative to a uniform prior (1/|categories|)
    """
    total = len(records)
    cnt   = Counter(r[field] for r in records)
    prior = 1.0 / len(categories)

    print(f"\n=== {title} (N={total}) ===")
    print(f"{'cat':<10} {'count':>7} {'p̂':>8} {'Δ log‑odds':>11}")
    for cat in categories:
        c = cnt.get(cat, 0)
        p = c / total if total else 0.0
        delta = _safe_logit(p) - _safe_logit(prior)
        print(f"{cat:<10} {c:>7} {p:>8.3f} {delta:>+11.3f}")

def accuracy_by_category(
    records: list[dict],
    categories: list[str],
    *,
    label_key: str,
    title: str
):
    """
    Per‑category accuracy of *ground‑truth* labels.
    """
    print(f"\n=== {title} ===")
    print(f"{'cat':<10} {'acc':>6} {'n':>6}")
    for cat in categories:
        rel = [r for r in records if r[label_key] == cat]
        n   = len(rel)
        if n == 0:
            acc = float('nan')
        else:
            acc = sum(r['correct'] for r in rel) / n
        print(f"{cat:<10} {acc:>6.2%} {n:>6}")

def grid_prediction_heatmap(
    records: list[dict],
    *,
    grid_size: int,
    title: str
):
    """
    Builds a grid‑size×grid‑size numpy array where (r,c) holds
    the empirical probability of the model *predicting* that cell.
    (Only considers records whose predictions are integers.)
    """
    mat = np.zeros((grid_size, grid_size), dtype=np.int64)
    total = 0
    for r in records:
        p = r["pred"]
        if isinstance(p, int) and 1 <= p <= grid_size * grid_size:
            rr, cc = divmod(p - 1, grid_size)
            mat[rr, cc] += 1
            total += 1
    if total:
        mat = mat / total

    print(f"\n=== {title} (probability map, rows then cols) ===")
    with np.printoptions(precision=3, suppress=True):
        print(mat)

def chain_color_difficulty(
    records: list[dict],
    *,
    title: str
):
    """
    For chain-trial records: Δ-log-odds of correctness for each *final* color,
    with 95% CI and p-value.
    """
    # reuse analyze_presence_effects on the ‘label’ field
    chain_recs = [r for r in records if r["trial"] == "chain"]
    if not chain_recs:
        return
    print()
    analyze_presence_effects(
        chain_recs,
        COLORS,
        key="label",
        label=f"{title} (by final color)"
    )


def analyze_chain_any_presence(
    records: List[dict],
    *,
    title: str,
    eps: float = 1e-6
):
    """
    Δ-log-odds of accuracy when a given color/shape appears anywhere in the chain,
    with 95 % CI and p-value.
    """
    def _safe_logit(p: float, eps: float = 1e-6) -> float:
        p = max(min(p, 1 - eps), eps)
        return math.log(p / (1 - p))

    def _norm_cdf(z: float) -> float:
        return (1.0 + math.erf(z / math.sqrt(2.0))) / 2.0

    chain_recs = [r for r in records if r['trial']=='chain']
    for cats, key, sub in [
        (COLORS,      'chain_colors', "color"),
        (SHAPE_TYPES, 'chain_shapes', "shape"),
    ]:
        rows = []
        for cat in cats:
            # count and correct
            in_mask  = [r for r in chain_recs if cat in r[key]]
            out_mask = [r for r in chain_recs if cat not in r[key]]
            n1, acc1 = len(in_mask), sum(r['correct'] for r in in_mask)
            n0, acc0 = len(out_mask), sum(r['correct'] for r in out_mask)
            if n1 == 0 or n0 == 0:
                rows.append((cat, float('nan'), float('nan'), float('nan'), n1, n0))
                continue

            p1_raw, p0_raw = acc1 / n1, acc0 / n0
            p1 = max(min(p1_raw, 1-eps), eps)
            p0 = max(min(p0_raw, 1-eps), eps)
            coeff = _safe_logit(p1) - _safe_logit(p0)
            var   = 1/(n1*p1*(1-p1)) + 1/(n0*p0*(1-p0))
            se    = math.sqrt(var)
            ci_lo = coeff - 1.96*se
            ci_hi = coeff + 1.96*se
            z     = coeff / se
            pval  = 2*(1 - _norm_cdf(abs(z)))

            rows.append((cat, coeff, (ci_lo, ci_hi), pval, n1, n0))

        # sort by magnitude
        rows.sort(key=lambda x: abs(x[1]) if not math.isnan(x[1]) else -1, reverse=True)
        print(f"\n=== Δ-log-odds by any-chain-{sub} for {title} ===")
        print(f"{'category':<10} {'Δ log-odds':>12} {'95% CI':>24} {'p':>8} {'#with':>8} {'#without':>8}")
        for cat, coeff, ci, pval, n1, n0 in rows:
            if math.isnan(coeff):
                print(f"{cat:<10} {'—':>12} {'—':>24} {'—':>8} {n1:>8} {n0:>8}")
            else:
                ci_txt = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
                print(f"{cat:<10} {coeff:+12.3f} {ci_txt:>24} {pval:>8.4f} {n1:>8} {n0:>8}")
# ───────────  New: analyze_excess_popular_color ───────────


def analyze_excess_popular_color(
    records: list[dict],
    *,
    title: str
):
    """
    For trials where the *prediction is a color*, compute the mean excess
    probability of picking the majority‑color in that image:
        Δ =  mean_i [ 𝟙{predicted_majority_i} − majority_freq_i ].
    Prints Δ, its 95 % CI, and p‑value for H0: Δ = 0.
    Requires each record to contain:
        'majority_freq'     : float
        'predicted_majority': bool
    """
    def _norm_cdf(z):  # N(0,1) cdf
        return (1.0 + math.erf(z / math.sqrt(2.0))) / 2.0

    diffs = [ (1.0 if r['predicted_majority'] else 0.0) - r['majority_freq']
              for r in records
              if 'majority_freq' in r ]
    N = len(diffs)
    if N == 0:
        print(f"\n=== {title} ===\n(no color predictions)")
        return
    mean_diff = sum(diffs)/N
    var = sum((d - mean_diff)**2 for d in diffs)/(N-1) if N > 1 else 0.0
    se  = math.sqrt(var / N)
    if se == 0:
        ci_lo = ci_hi = pval = float('nan')
    else:
        ci_lo = mean_diff - 1.96*se
        ci_hi = mean_diff + 1.96*se
        z     = mean_diff / se
        pval  = 2*(1 - _norm_cdf(abs(z)))

    print(f"\n=== {title} ===")
    print(f"N                                : {N}")
    print(f"Excess probability Δ             : {mean_diff:+.4f}")
    print(f"95 % CI                          : [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"p‑value (H₀: Δ = 0)              : {pval:.4f}")


def generate_chain_description_string(num_steps):
    """
    Generates a descriptive string for a chain of colored shapes.

    Args:
        num_steps (int): The number of steps in the chain (len(chain)-1). 
                         Must be between 1 and 5.

    Returns:
        str: The formatted descriptive string.
    """
    if not 1 <= num_steps <= 5:
        raise ValueError("Number of steps must be between 1 and 5.")

    items = [
        ("blue", "triangle"),  # Item 0 (start)
        ("red", "square"),    # Item 1
        ("blue", "circle"),    # Item 2
        ("magenta", "triangle"),  # Item 3
        ("green", "circle"),   # Item 4
        ("purple", "square")     # Item 5
    ]

    path_parts = []
    
    # Starting item (item 0)
    path_parts.append(f"you might start at a {items[0][0]} {items[0][1]}")

    # First step to item 1
    if num_steps >= 1:
        path_parts.append(f"then go to a {items[1][0]} {items[1][1]}")

    # Subsequent steps to item 2, 3, ..., num_steps
    for i in range(2, num_steps + 1):
        path_parts.append(f"then a {items[i][0]} {items[i][1]}")
    
    path_description = ", ".join(path_parts)
    
    steps_str = f"{num_steps} step"
    if num_steps != 1:
        steps_str += "s"
        
    final_color = items[num_steps][0] # The color of the item landed on after num_steps
    
    return f"{steps_str}, {path_description}. The answer would be {final_color}."

# Example usage:
# print(generate_chain_description_string(1))
# print(generate_chain_description_string(2))
# print(generate_chain_description_string(3))
# print(generate_chain_description_string(4))
# print(generate_chain_description_string(5))
def main():
    import argparse, os, glob, json, base64, torch, re
    from collections import defaultdict
    from tabulate import tabulate

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=
                    [
                    #  "openai:o4-mini-2025-04-16",
                    #  "openai:o3-2025-04-16",
                    #"openrouter:anthropic/claude-3.7-sonnet:thinking",
                    #"openrouter:google/gemini-2.5-pro-preview"
                    # "local:google/gemma-3-27b-it",
                    # "local:allenai/Molmo-7B-D-0924",
                    # "local:mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                    # "local:qwen",
                    # "local:qwen2.5-vl-32b",
                    # "local:llama",
                     "local:OpenGVLab/InternVL3-14B",
                     "local:microsoft/Phi-4-multimodal-instruct"
                    ])
    parser.add_argument("--trials", type=str,
                        choices=["chain"],
                        default=["chain"])
    parser.add_argument("--grid-size",    type=int, default=5)
    parser.add_argument("--cell-size",    type=int, default=80)
    parser.add_argument("--num",          type=int, default=10)
    parser.add_argument("--chain-length", type=int, default=3)
    parser.add_argument("--few-shot",     action="store_true")
    parser.add_argument("--all-settings", action="store_true")
    parser.add_argument("--make-dataset", type=str,
                        help="Directory to save generated dataset (no inference)")
    parser.add_argument("--load-dataset", nargs="+", type=str,
                        help="Directories of saved datasets to load and run inference")
    parser.add_argument("--verbose",      action="store_true")
    parser.add_argument("--use-local",    action="store_true")
    parser.add_argument("--local-model-name", type=str, default="")
    parser.add_argument("--local-model-path", type=str, default=None)
    parser.add_argument("--test-image",   action="store_true")
    args = parser.parse_args()

    # Prepare few-shot demos
    fs_examples = []
    if not args.load_dataset:
        for _ in range(1):
            img, start_pair, final_color, chain, _ = generate_chain_image(
                args.grid_size, args.cell_size, args.chain_length
            )
            prompt = (
                f"Starting at the {start_pair[1]} {start_pair[0]}, follow the labels for "
                f"{len(chain)-1} steps. (For instance, in a different example of {generate_chain_description_string(len(chain)-1)}) "
                "After those steps, what color are you on? Answer with the color in curly braces, e.g. {red}."
            )
            ans = f"We start at the {chain[0][1]} {chain[0][0]}"
            for shape, color in chain[1:-1]:
                ans += f", and then go to the {color} {shape}"
            last_shape, last_color = chain[-1]
            ans += f", and then end at the {last_color} {last_shape}. {{{last_color}}}"
            fs_examples.append((img, prompt, ans))

    # Collect model specs
    models = []
    for spec in args.models:
        if ":" in spec:
            backend, mdl = spec.split(":", 1)
        else:
            backend, mdl = spec, ""
        models.append((spec, backend.lower(), mdl))

    # Build settings
    settings = [(False,), (True,)] if args.all_settings else [(args.few_shot,)]

    # TEST-IMAGE
    if args.test_image:
        os.makedirs("test_images", exist_ok=True)
        for t in args.trials:
            if t == "chain":
                img, spc, fc, chain, _ = generate_chain_image(
                    args.grid_size, args.cell_size, args.chain_length
                )
                img.save("test_images/test_chain.png")
                print(f"Saved chain → start={spc}, final={fc}")
            else:
                img, num, feat, _ = generate_trial_image(
                    t, args.grid_size, args.cell_size
                )
                img.save(f"test_images/test_{t}.png")
                print(f"Saved {t} → number={num}, feature={feat}")
        return

    # MAKE-DATASET
    if args.make_dataset:
        ds_dir = args.make_dataset
        os.makedirs(ds_dir, exist_ok=True)
        for (few_flag,) in settings:
            for t in args.trials:
                trial_dir = os.path.join(ds_dir, f"{t}_fs{int(few_flag)}")
                os.makedirs(trial_dir, exist_ok=True)
                if few_flag:
                    few_dir = os.path.join(trial_dir, "few_shot_examples")
                    os.makedirs(few_dir, exist_ok=True)
                    for idx, (img, prompt, ans) in enumerate(fs_examples):
                        img.save(os.path.join(few_dir, f"fs_{idx}.png"))
                        with open(os.path.join(few_dir, f"fs_{idx}.json"), "w") as f:
                            json.dump({"prompt": prompt, "answer": ans}, f, indent=2)
                for i in range(args.num):
                    if t == "chain":
                        img, spc, fc, chain, cc = generate_chain_image(
                            args.grid_size, args.cell_size, args.chain_length
                        )
                        prompt = (
                            f"Starting at the {spc[1]} {spc[0]}, follow the labels for " # Use spc here
                            f"{len(chain)-1} steps. (For instance, in a different example of {generate_chain_description_string(len(chain)-1)}) "
                            "After those steps, what color are you on? Answer with the color in curly braces, e.g. {red}."
                        )
                        meta = {
                            "trial": t, "few_shot": few_flag,
                            "start_pair": spc, "final_color": fc,
                            "chain": chain, "color_counts": cc,
                            "prompt": prompt
                        }
                    else:
                        img, num, feat, cc = generate_trial_image(
                            t, args.grid_size, args.cell_size
                        )
                        if t == "color":
                            prompt = f"Which number is on the colored {feat}? Answer in {{{{5}}}}."
                        elif t == "shape":
                            prompt = f"Which number is on the shape {feat}? Answer in {{{{5}}}}."
                        elif t == "color_of_number":
                            prompt = f"What color is on number {num}? Answer in {{red}}."
                        else:
                            prompt = f"What shape is on number {num}? Answer in {{circle}}."
                        meta = {
                            "trial": t, "few_shot": few_flag,
                            "target_number": num, "target_feat": feat,
                            "color_counts": cc, "prompt": prompt
                        }
                    img.save(os.path.join(trial_dir, f"img_{i}.png"))
                    with open(os.path.join(trial_dir, f"meta_{i}.json"), "w") as f:
                        json.dump(meta, f, indent=2)
        print(f"[make-dataset] saved to {ds_dir}")
        return

    # LOAD-DATASET
    if args.load_dataset:
        import json
        from PIL import Image

        # 1) Load few-shot examples from the first dataset directory
        fs_examples = []
        if args.few_shot:
            first_ds = args.load_dataset[0]
            for t in args.trials:
                fs_dir = os.path.join(first_ds, f"{t}_fs1", "few_shot_examples")
                if not os.path.isdir(fs_dir):
                    continue
                for fname in sorted(os.listdir(fs_dir)):
                    if not fname.endswith(".png"):
                        continue
                    idx = fname.split("_")[1].split(".")[0]
                    img_path = os.path.join(fs_dir, fname)
                    json_path = os.path.join(fs_dir, f"fs_{idx}.json")
                    img = Image.open(img_path)
                    with open(json_path, "r") as jf:
                        meta = json.load(jf)
                    fs_examples.append((img, meta["prompt"], meta["answer"]))

        prev_spec = None
        client = None

        # 2) Run inference per model, reusing `client` if the spec is identical
        for spec, backend, mdl in models:
            if spec != prev_spec:
                # teardown previous client
                if client is not None:
                    try:
                        if hasattr(client, "client") and isinstance(client.client, torch.nn.Module):
                            client.client.to("cpu")
                    except:
                        pass
                    del client
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # initialize new client
                demo = fs_examples[0] if fs_examples else (None, None, None)
                client = InferenceClient(
                    name=backend,
                    imgpair1=demo,
                    fs_examples=fs_examples,
                    api_key=None,
                    openrouter_model=mdl,
                    use_local=(backend == "local"),
                    model_name=mdl,
                    model_path=args.local_model_path
                )
                prev_spec = spec

            # for this model, run *all* trials
            for (few_flag,) in settings:
                title = f"SETTING few_shot={few_flag} | MODEL={spec}"
                print(f"\n=== {title} ===")

                # collect all examples across every trial
                examples = []
                for ds_dir in args.load_dataset:
                    for t in args.trials:
                        pattern = os.path.join(ds_dir, f"{t}_fs{int(few_flag)}", "meta_*.json")
                        for mf in glob.glob(pattern):
                            info = json.load(open(mf))
                            if info.get("few_shot") != few_flag:
                                continue
                            img_file = mf.replace("meta_", "img_").replace(".json", ".png")
                            with open(img_file, "rb") as f_img:
                                b64 = base64.b64encode(f_img.read()).decode()
                            examples.append((info, b64, mf, img_file))

                records = []
                # now iterate every trial in turn
                for t in args.trials:
                    rec_t = []
                    for info, b64, meta_file, img_file in examples:
                        if info["trial"] != t:
                            continue
                        raw = client.ask_single(info["prompt"], b64, few_flag)
                        # build `rec` exactly as before:
                        if t == "chain":
                            label = info["final_color"].lower()
                            pred  = extract_braced(raw).lower()
                            rec = {
                                "trial": t,
                                "label": label,
                                "pred": pred,
                                "correct": pred == label,
                                "start_color": info["start_pair"][1].lower(),
                                "start_shape": info["start_pair"][0].lower(),
                                "chain_colors": [c for (_, c) in info["chain"]],
                                "chain_shapes": [s for (s, _) in info["chain"]],
                                "majority_freq": max(info["color_counts"].values())/(args.grid_size**2),
                                "predicted_majority": pred == max(info["color_counts"], key=info["color_counts"].get)
                            }
                        else:
                            if t in ("color", "shape"):
                                num  = info["target_number"]
                                match = re.search(r"\d+", raw)
                                pnum = int(match.group()) if match else None
                                rec = {"trial": t, "label": num, "pred": pnum, "correct": pnum == num, "feature": info["target_feat"]}
                            else:
                                feat    = info["target_feat"]
                                pred_txt = extract_braced(raw).lower()
                                rec     = {"trial": t, "label": feat.lower(), "pred": pred_txt, "correct": pred_txt == feat.lower(), "feature": feat}

                        rec_t.append(rec)
                        records.append(rec)

                        if args.verbose:
                            print("VERBOSE_RESPONSE:", json.dumps({
                                "model": spec,
                                "trial": t,
                                "meta_file": meta_file,
                                "img_file": img_file,
                                "raw": raw,
                                "label": rec["label"],
                                "pred": rec.get("pred"),
                                "correct": rec["correct"]
                            }, indent=2), flush=True)

                    if rec_t:
                        c, n = sum(r["correct"] for r in rec_t), len(rec_t)
                        print(f"{t:<20} accuracy: {c}/{n} = {c/n:.2%}")
                        if t != "chain":
                            cats = COLORS if "color" in t else SHAPE_TYPES
                            analyze_presence_effects(rec_t, cats, "feature", f"{title} {t}")
                        else:
                            analyze_chain_presence_effects(rec_t, title=f"{title} chain presence effects")
                            chain_color_difficulty(rec_t, title=f"{title} chain color difficulty")
                            analyze_chain_any_presence(rec_t, title=f"{title} chain any-presence")

                if not records:
                    print(f"[load-dataset] no examples found for few_shot={few_flag}, MODEL={spec}, skipping overall analysis.")
                    continue

                total = len(records)
                corr  = sum(r["correct"] for r in records)
                print(f"{title} overall accuracy: {corr}/{total} = {corr/total:.2%}")

                analyze_excess_popular_color(records, title=f"{title} excess popular color")

        # teardown final client
        if client is not None:
            try:
                if hasattr(client, "client") and isinstance(client.client, torch.nn.Module):
                    client.client.to("cpu")
            except:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return
    # LIVE-GENERATION
    for spec, backend, mdl in models:
        demo = fs_examples[0]
        client = InferenceClient(
            name=backend,
            imgpair1=demo,
            fs_examples=fs_examples,             # <<< new
            api_key=None,
            openrouter_model=mdl,
            use_local=(backend=="local"),
            model_name=mdl,
            model_path=args.local_model_path
        )

        for (few_flag,) in settings:
            title = f"LIVE few_shot={few_flag} | MODEL={spec}"
            print(f"\n=== {title} ===")
            records = []
            for t in args.trials:
                rec_t=[]
                for _ in range(args.num):
                    if t=="chain":
                        img,spc,fc,chain,cc = generate_chain_image(
                            args.grid_size,args.cell_size,args.chain_length
                        )
                        b64=encode(img)
                        prompt = (
                            f"Starting at the {spc[1]} {spc[0]}, follow the labels for " # Use spc here
                            f"{len(chain)-1} steps. (For instance, in a different example of {generate_chain_description_string(len(chain)-1)}) "
                            "After those steps, what color are you on? Answer with the color in curly braces, e.g. {red}."
                        )
                        raw=client.ask_single(prompt,b64,few_flag)
                        pred=extract_braced(raw).lower()
                        ok=(pred==fc.lower())
                        rec={
                            "trial":t,"label":fc.lower(),"pred":pred,"correct":ok,
                            "start_color":spc[1].lower(),"start_shape":spc[0].lower(),
                            "chain_colors":[c for (_,c) in chain],
                            "chain_shapes":[s for (s,_) in chain],
                            "majority_freq":max(cc.values())/(args.grid_size**2),
                            "predicted_majority":pred==max(cc,key=cc.get)
                        }
                    else:
                        img,num,feat,cc=generate_trial_image(
                            t,args.grid_size,args.cell_size
                        )
                        b64=encode(img)
                        if t=="color":
                            prompt=f"Which number is on the colored {feat}? Answer in {{{{5}}}}."
                        elif t=="shape":
                            prompt=f"Which number is on the shape {feat}? Answer in {{{{5}}}}."
                        elif t=="color_of_number":
                            prompt=f"What color is on number {num}? Answer in {{red}}."
                        else:
                            prompt=f"What shape is on number {num}? Answer in {{circle}}."
                        raw=client.ask_single(prompt,b64,few_flag)
                        ptxt=extract_braced(raw)
                        if t in ("color","shape"):
                            pnum=int(re.search(r"\d+",ptxt).group()) if re.search(r"\d+",ptxt) else None
                            ok=(pnum==num)
                            rec={"trial":t,"label":num,"pred":pnum,"correct":ok}
                        else:
                            ok=(ptxt.lower()==feat.lower())
                            rec={"trial":t,"label":feat.lower(),"pred":ptxt.lower(),"correct":ok}
                    if args.verbose:
                        record = {
                            "model": spec,
                            "trial": t,
                            "few_shot": few_flag,
                            "raw": raw,
                            "label": rec["label"],
                            "correct": rec["correct"]
                        }
                        print("VERBOSE_RESPONSE:", json.dumps(record))

                    rec_t.append(rec)
                    records.append(rec)

                c,n = sum(r["correct"] for r in rec_t), len(rec_t)
                print(f"{t:<20} acc {c}/{n} = {c/n:.2%}")
                if t!="chain":
                    cats=COLORS if "color" in t else SHAPE_TYPES
                    analyze_presence_effects(rec_t,cats,"feature",f"{title} {t}")
                else:
                    analyze_chain_presence_effects(rec_t,title=f"{title} chain presence effects")
                    chain_color_difficulty(rec_t,title=f"{title} chain color difficulty")
                    analyze_chain_any_presence(rec_t,title=f"{title} chain any-presence")

            total, corr = len(records), sum(r["correct"] for r in records)
            print(f"{title} overall accuracy: {corr}/{total} = {corr/total:.2%}")

            # Excess popular-color analysis
            analyze_excess_popular_color(records, title=f"{title} excess popular color")

        # teardown
        try:
            if hasattr(client,"client") and isinstance(client.client,torch.nn.Module):
                client.client.to("cpu")
        except:
            pass
        del client
        if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
