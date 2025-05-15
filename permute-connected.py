#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
permute‑connected.py  (fully self‑contained)

Creates synthetic “does the object re‑appear under global affine?”
problems, queries a vision LLM, and prints accuracies.

Five trials
-----------
1  baseline (no connectivity guarantee, separate images)
2  baseline (concatenated prompt)
7  Image 1 connected
8  Image 1 connected (concatenated prompt)
9  BOTH images connected (pre‑ and post‑jitter)

Hard constraints
----------------
* Every pixel lies in‑canvas after *all* transforms.
* Connectivity satisfied whenever requested (trials 7‑9).
* No shape fully occludes another.
* Per‑shape jitter **always** applied.
* Circles never get rotation jitter (it’s invisible).

CLI flags
---------
--no-distractors            disable distractors
--allow-distractor-overlap  let distractors touch the object cluster
--few-shot                  prepend two worked image examples

Usage
-----
    python permute-connected.py --models openai --num 10 --few-shot --verbose
"""

import os, random, math, base64, re, copy, argparse, requests, gc
from collections import deque
from typing import Optional, List, Tuple
import math
from PIL import Image, ImageDraw
from tabulate import tabulate
import numpy as np
from io import BytesIO
from datasets import load_dataset
from collections import defaultdict
import sys
from typing import Optional, Sequence
import torch
from PIL import Image
import torch
from io import BytesIO
import base64
import tempfile
# for Pix2Struct
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
# for Molmo
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import statsmodels.api as sm          # logistic‐regression w/ SE & p‑values
from scipy.stats import norm          # Gaussian CDF for Wald tests
from statsmodels.tools.sm_exceptions import PerfectSeparationError
#import permute-connectedtrial10

import warnings
import numpy as np
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError, ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

# ── suppress the “Perfect separation” warnings from statsmodels ───────────
warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
from infclient import run_inference, LocalFetcher
# ═════════════════════════════ Hyper‑parameters ════════════════════════════════
_IMAGENET_DS = None        # lazy‑loaded cache
USE_TORCH = True
_IMAGENET_IDX_BY_LABEL   = None      # ← NEW
YES_PROB                       = 0.5
GLOBAL_ROT_RANGE               = (-math.pi,  math.pi)
GLOBAL_SCALE_RANGE             = (0.3,      2.0)

NO_MIN_TRANSFORMS              = 1
NO_MAX_TRANSFORMS              = 2
NO_ADDITIONAL_TRANSFORM_PROB   = 0.1

PROB_TRANSLATE                 = 0.15
PROB_ROTATE                    = 0.15
PROB_SCALE                     = 0.15
PROB_RESHAPE                   = 0.05

RESCALE_MIN                    = 0.1
RESCALE_MAX                    = 0.33
NO_MIN_TRANSLATE_PX            = 15
NO_MAX_TRANSLATE_PX            = 30
NO_MIN_ROT                     = -math.pi/5
NO_MAX_ROT                     =  math.pi/5
JITTER_MIN_ROT                 = math.radians(6)          # ≥6°

LABEL_HEIGHT                   = 30
BAR_WIDTH                      = 10

MODEL_CODE                     = "o4-mini-2025-04-16" #"o3-2025-04-16"
_CURLY_RE                      = re.compile(r"\{([^{}]+)\}")
CANVAS_SIZE = 512
COLORS  = ['red','green','blue','orange','purple','teal','magenta','gold']
SHAPES  = ['circle','oval','square','rectangle','triangle','polygon','line']

# ═════ runtime switches (set in main) ═════
ENABLE_DISTRACTORS      = True
DISTRACTOR_CAN_OVERLAP  = False
FEW_SHOT                = False

if USE_TORCH:
    import torch

# ═════════════════════════════ Inference client ════════════════════════════════
def generate_two_shot_examples(
        canvas: int,
        odir:   str,
        *,
        one_canvas: bool = False,
        with_distractors: bool = False,
        allow_distractor_overlap: bool = False,
        conn1: bool = False,
        conn2: bool = False,
        no_affine: bool = False,
) -> Tuple[Tuple[Image.Image, ...], Tuple[Image.Image, ...]]:
    """
    Build YES / NO worked examples whose generation pipeline matches the
    *exact* settings of the current trial.

    Parameters
    ----------
    canvas      : side length in pixels.
    odir        : output folder – PNGs are written here for inspection.
    one_canvas  : if True, fuse views with a vertical bar (trial‑2 / trial‑8 style).
    with_distractors           : add distractors to the *second* image.
    allow_distractor_overlap   : allow distractors to touch the object cluster.
    conn1, conn2               : connectivity constraints for img‑1 / img‑2.
    no_affine                  : if True, skip the global affine on img‑2
                                 (i.e. identical composite before per‑shape jitter).

    Returns
    -------
    (yes_pair , no_pair) where each element is a tuple of PIL images.
      • len(tuple) == 2  when one_canvas == False   (img1, img2)
      • len(tuple) == 1  when one_canvas == True    (combined_img,)
    """
    os.makedirs(odir, exist_ok=True)
    BAR = 10                                             # separator width

    def _combine(a: Image.Image, b: Image.Image) -> Image.Image:
        """Return a concatenated canvas: image‑1 ▐ bar ▌ image‑2."""
        w, h = a.size
        out  = Image.new("RGB", (w * 2 + BAR, h), "white")
        out.paste(a, (0, 0))
        out.paste(b, (w + BAR, 0))
        ImageDraw.Draw(out).rectangle([w, 0, w + BAR - 1, h - 1], fill="black")
        return out

    def _single_example(tag: str, make_no: bool) -> Tuple[Image.Image, ...]:
        # Base composite for image 1
        shapes, offs = gen_shapes_offsets(canvas, require_conn=conn1)
        cx = cy = canvas // 2
        img1 = draw_composite(shapes, offs, canvas, (cx, cy))

        # Decide the second image
        if no_affine:
            ang, sc, offs2 = 0.0, 1.0, offs
        else:
            while True:
                ang = random.uniform(*GLOBAL_ROT_RANGE)
                sc  = random.uniform(*GLOBAL_SCALE_RANGE)
                offs2 = [
                    (dx * sc * math.cos(ang) - dy * sc * math.sin(ang),
                     dx * sc * math.sin(ang) + dy * sc * math.cos(ang))
                    for dx, dy in offs
                ]
                if composite_fits_canvas(shapes, offs2, canvas) and \
                   (not conn2 or composite_connected(shapes, offs2, canvas)):
                    break

        img2 = apply_affine(img1, sc, ang, (cx, cy), (cx, cy), canvas)

        # If NO example, jitter one or more shapes
        if make_no:
            while True:
                ns, no, _, _ = jitter_shapes(shapes, offs2, canvas)
                if composite_fits_canvas(ns, no, canvas) and \
                   (not conn2 or composite_connected(ns, no, canvas)):
                    img2 = draw_composite(ns, no, canvas, (cx, cy))
                    break

        # Optional distractors
        if with_distractors:
            R = max(
                math.hypot(dx, dy) + compute_shape_radius(s)
                for (dx, dy), s in zip(offs2, shapes)
            )
            img2 = _add_distractors(
                img2, (cx, cy), R, canvas,
                allow_overlap=allow_distractor_overlap
            )

        if not one_canvas:
            return (img1, img2)

        combo = _combine(img1, img2)
        return (combo, combo)                        # keep tuple‑length == 2

    yes_pair = _single_example("two_shot_yes", make_no=False)
    no_pair  = _single_example("two_shot_no",  make_no=True)
    return yes_pair, no_pair
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
                 yes_imgs,
                 no_imgs,
                 *,
                 api_key: Optional[str] = None,
                 openrouter_model: str = "grok-1",
                 use_local: bool = False,
                 model_name: str = "",
                 model_path: str = None):
        self.name = name.lower()
        self.or_model = openrouter_model.lower()
        self.yes0, self.yes1 = encode(yes_imgs[0]), encode(yes_imgs[1])
        self.no0,  self.no1  = encode(no_imgs[0]),  encode(no_imgs[1])
        self.use_local = use_local
        self.model_name = model_name
        self.model_path = model_path
        
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
            self.client = openai.OpenAI(api_key=key,timeout=1000)

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

    # ───────────────────────────────── ask_pair / ask_single
    def ask_pair(self, prompt: str, b1: str, b2: str, few_shot: bool) -> str:
        if self.use_local:
            # decode once
            img1, img2 = decode(b1), decode(b2)
            if not few_shot:
                # single-shot
                return run_inference(
                    self.client,
                    self.processor,
                    self.model_name,
                    images=[img1, img2],
                    query=prompt,
                    temperature=0.0001,
                )
            # few-shot: build a 5-turn convo
            msgs = [
                # 1) demo YES
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": decode(self.yes0)},
                        {"type": "image", "image": decode(self.yes1)},
                        {"type": "text",  "text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "{{yes}}"}],
                },
                # 2) demo NO
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": decode(self.no0)},
                        {"type": "image", "image": decode(self.no1)},
                        {"type": "text",  "text": "That is correct. " + prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "{{no}}"}],
                },
                # 3) actual query
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img1},
                        {"type": "image", "image": img2},
                        {"type": "text",  "text": "Yes, that is correct. " + prompt},
                    ],
                },
            ]
            return run_inference(
                self.client,
                self.processor,
                self.model_name,
                messages=msgs,
                temperature=0.0001,
            )
            
        elif self.name == "openai":                          # ─── OpenAI
            try:
                if not few_shot:
                    resp = self.client.responses.create(
                        model=self.model_name,
                        reasoning={"effort": "high"},
                        input=[{
                            "role":    "user",
                            "content": [
                                {"type": "input_text",  "text":  prompt},
                                {"type": "input_image", "image_url": f"data:image/png;base64,{b1}"},
                                {"type": "input_image", "image_url": f"data:image/png;base64,{b2}"},
                            ],
                        }],
                    )
                else:
                    resp = self.client.responses.create(
                        model=self.model_name,
                        reasoning={"effort": "high"},
                        input=[{
                            "role":    "user",
                            "content": [
                                {"type": "input_text",  "text":  prompt},
                                {"type": "input_image", "image_url": f"data:image/png;base64,{self.yes0}"},
                                {"type": "input_image", "image_url": f"data:image/png;base64,{self.yes1}"},
                            ],
                        },{
                            "role":    "assistant",
                            "content": [
                                {"type": "output_text",  "text":  "{{yes}}"},
                            ],
                        },{
                            "role":    "user",
                            "content": [
                                {"type": "input_text",  "text":  "That is correct. " + prompt},
                                {"type": "input_image", "image_url": f"data:image/png;base64,{self.no0}"},
                                {"type": "input_image", "image_url": f"data:image/png;base64,{self.no1}"},
                            ],
                        },{
                            "role":    "assistant",
                            "content": [
                                {"type": "output_text",  "text":  "{{no}}"},
                            ],
                        },{
                            "role":    "user",
                            "content": [
                                {"type": "input_text",  "text":  "Yes, that is correct. " + prompt},
                                {"type": "input_image", "image_url": f"data:image/png;base64,{b1}"},
                                {"type": "input_image", "image_url": f"data:image/png;base64,{b2}"},
                            ],
                        }],
                    )
                return resp.output_text
            except Exception as e:
                # catch ANY HTTP / InternalServerError / timeout / etc.
                print(f"[OpenAI error] {e}", file=sys.stderr)
                # fallback to random yes/no
                return random.choice(["{yes}", "{no}"])

        # ────────────────────────────── OpenRouter
        if not few_shot:
            messages = [{
                "role":    "user",
                "content": [
                    {"type": "text",  "text": prompt},
                    {"type": "image_url", "image_url": self._data_url(b1)},
                    {"type": "image_url", "image_url": self._data_url(b2)}
                ]
            }]
        else:
            messages =  messages = [
                # YES demo
                {"role": "user",      "content": [
                    {"type": "text",      "text": prompt},
                    {"type": "image_url", "image_url": f"data:image/png;base64,{self.yes0}"},
                    {"type": "image_url", "image_url": f"data:image/png;base64,{self.yes1}"},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "{{yes}}"},
                ]},
                # NO demo
                {"role": "user",      "content": [
                    {"type": "text",      "text": "That is correct. " + prompt},
                    {"type": "image_url", "image_url": f"data:image/png;base64,{self.no0}"},
                    {"type": "image_url", "image_url": f"data:image/png;base64,{self.no1}"},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "{{no}}"},
                ]},
                # Query
                {"role": "user",      "content": [
                    {"type": "text",      "text": "Yes, that is correct. " + prompt},
                    {"type": "image_url", "image_url": self._data_url(b1)},
                    {"type": "image_url", "image_url": self._data_url(b2)},
                ]},
            ]
        
        
        return self._or_post(messages)

    def ask_single(self, prompt: str, b: str, few_shot:bool) -> str:
        if self.use_local:
            img = decode(b)
            if not few_shot:
                return run_inference(
                    self.client, self.processor, self.model_name,
                    images=img,
                    query=prompt,
                    temperature=0.0001,
                )

            # Build YES/NO demos + query
            msgs = [
                {"role": "user",      "content": [
                    {"type": "image", "image": decode(self.yes0)},
                    {"type": "text", "text": prompt},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "{{yes}}"},
                ]},
                {"role": "user",      "content": [
                    {"type": "image", "image": decode(self.no0)},
                    {"type": "text",  "text": "That is correct. " + prompt},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "{{no}}"},
                ]},
                {"role": "user",      "content": [
                    {"type": "image", "image": img},
                    {"type": "text",  "text": "Yes, that is correct. " + prompt},
                ]},
            ]
            return run_inference(
                self.client, self.processor, self.model_name,
                messages=msgs,
                temperature=0.0001,
            )


        elif self.name == "openai":
            
            try:
                if not few_shot:
                    resp = self.client.responses.create(
                        model=self.model_name,
                        reasoning={"effort": "high"},
                        input=[{
                            "role":    "user",
                            "content": [
                                {"type": "input_text",  "text":  prompt},
                                {"type": "input_image", "image_url": f"data:image/png;base64,{b}"},
                            ],
                        }],
                    )
                else:
                    resp = self.client.responses.create(
                        model=self.model_name,
                        reasoning={"effort": "high"},
                        input=[
                            {"role": "user",      "content": [
                                {"type": "input_text",  "text": prompt},
                                {"type": "input_image", "image_url": f"data:image/png;base64,{self.yes0}"},
                            ]},
                            {"role": "assistant",  "content": [
                                {"type": "output_text", "text": "{{yes}}"},
                            ]},
                            {"role": "user",      "content": [
                                {"type": "input_text",  "text": "That is correct. " + prompt},
                                {"type": "input_image", "image_url": f"data:image/png;base64,{self.no0}"},
                            ]},
                            {"role": "assistant",  "content": [
                                {"type": "output_text", "text": "{{no}}"},
                            ]},
                            {"role": "user",      "content": [
                                {"type": "input_text",  "text": "Yes, that is correct. " + prompt},
                                {"type": "input_image", "image_url": f"data:image/png;base64,{b}"},
                            ]},
                        ],
                    )
                return resp.output_text
            except Exception as e:
                # catch ANY HTTP / InternalServerError / timeout / etc.
                print(f"[OpenAI error] {e}", file=sys.stderr)
                # fallback to random yes/no
                return random.choice(["{yes}", "{no}"])
        # ────────────────────────────── OpenRouter
        if not few_shot:
            messages = [{
                "role":    "user",
                "content": [
                    {"type": "text",      "text": prompt},
                    {"type": "image_url", "image_url": self._data_url(b)},
                ],
            }]
        else:
            messages = [
                {"role": "user",      "content": [
                    {"type": "text",      "text": prompt},
                    {"type": "image_url", "image_url": f"data:image/png;base64,{self.yes0}"},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "{{yes}}"},
                ]},
                {"role": "user",      "content": [
                    {"type": "text",      "text": "That is correct. " + prompt},
                    {"type": "image_url", "image_url": f"data:image/png;base64,{self.no0}"},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text",  "text": "{{no}}"},
                ]},
                {"role": "user",      "content": [
                    {"type": "text",      "text": "Yes, that is correct. " + prompt},
                    {"type": "image_url", "image_url": self._data_url(b)},
                ]},
            ]
        return self._or_post(messages)

# ═════════════════════════════ Geometry & drawing ══════════════════════════════
def rotate_point(x,y,cx,cy,ang):
    dx,dy=x-cx,y-cy; ca,sa=math.cos(ang),math.sin(ang)
    return cx+dx*ca-dy*sa, cy+dx*sa+dy*ca



def compute_shape_radius(info):
    t=info['type']
    if t=='circle':     return info['size']/2
    if t in ('square','rectangle','oval'):
        return math.hypot(info['width']/2, info['height']/2)
    if t=='triangle':   return info['size']*math.sqrt(3)/3
    if t=='polygon':    return info['size']
    if t=='line':       return info['size']/2
    return 0

def draw_composite(shapes, offs, canvas, center):
    cx,cy=center
    img=Image.new('RGB',(canvas,canvas),'white')
    d=ImageDraw.Draw(img)
    for info,(dx,dy) in sorted(zip(shapes,offs), key=lambda x:x[0]['z']):
        x0,y0=cx+dx, cy+dy
        typ,col,ang=info['type'],info['color'],info.get('angle',0)
        if typ=='circle':
            r=info['size']/2; d.ellipse([x0-r,y0-r,x0+r,y0+r],fill=col)
        elif typ=='oval':
            w,h=info['width'],info['height']
            d.ellipse([x0-w/2,y0-h/2,x0+w/2,y0+h/2],fill=col)
        elif typ in ('square','rectangle'):
            w,h=info['width'],info['height']
            pts=[(x0-w/2,y0-h/2),(x0+w/2,y0-h/2),
                 (x0+w/2,y0+h/2),(x0-w/2,y0+h/2)]
            if ang: pts=[rotate_point(px,py,x0,y0,ang) for px,py in pts]
            d.polygon(pts,fill=col)
        elif typ=='triangle':
            s,ht=info['size'],info['size']*math.sqrt(3)/2
            verts=[(x0,      y0-2/3*ht),
                   (x0-s/2,  y0+1/3*ht),
                   (x0+s/2,  y0+1/3*ht)]
            if ang: verts=[rotate_point(px,py,x0,y0,ang) for px,py in verts]
            d.polygon(verts,fill=col)
        elif typ=='polygon':
            rad,n=info['size'],info['n']
            verts=[(x0+rad*math.cos(2*math.pi*i/n),
                    y0+rad*math.sin(2*math.pi*i/n)) for i in range(n)]
            if ang: verts=[rotate_point(px,py,x0,y0,ang) for px,py in verts]
            d.polygon(verts,fill=col)
        elif typ=='line':
            L=info['size']; p1,p2=(x0-L/2,y0),(x0+L/2,y0)
            if ang: p1=rotate_point(*p1,x0,y0,ang); p2=rotate_point(*p2,x0,y0,ang)
            d.line([p1,p2],fill=col,width=max(1,int(info['size']/10)))
    return img

def composite_bounds(shapes, offs):
    xs,ys=[],[]
    for info,(dx,dy) in zip(shapes,offs):
        r=compute_shape_radius(info)
        xs.extend([dx-r,dx+r]); ys.extend([dy-r,dy+r])
    return min(xs),max(xs),min(ys),max(ys)

def composite_fits_canvas(shapes, offs, canvas):
    mnx,mxx,mny,mxy = composite_bounds(shapes,offs)
    return max(abs(mnx),abs(mxx),abs(mny),abs(mxy)) < canvas/2-1

# connectivity test
def _is_connected(img):
    m=np.array(img.convert('L'))<255
    if not m.any(): return False
    h,w=m.shape; start=tuple(np.argwhere(m)[0]); dq=deque([start]); seen={start}
    while dq:
        x,y=dq.popleft()
        for nx,ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
            if 0<=nx<h and 0<=ny<w and m[nx,ny] and (nx,ny) not in seen:
                seen.add((nx,ny)); dq.append((nx,ny))
    return len(seen)==m.sum()

def composite_connected(shapes, offs, canvas):
    img=draw_composite(shapes,offs,canvas,(canvas//2,canvas//2))
    return _is_connected(img)

# ═════════════════════════════ Offset generation ═══════════════════════════════
def _generate_touching_offsets(shapes, canvas):
    offs=[(0.0,0.0)]
    for i in range(1,len(shapes)):
        r_i=compute_shape_radius(shapes[i])
        for _ in range(1000):
            j=random.randrange(i)
            r_j=compute_shape_radius(shapes[j])
            d=r_i+r_j-random.uniform(0,0.05*min(r_i,r_j))
            theta=random.uniform(0,2*math.pi)
            dx=offs[j][0]+d*math.cos(theta)
            dy=offs[j][1]+d*math.sin(theta)
            if all(math.hypot(dx-ox,dy-oy) > 0.5*(r_i+compute_shape_radius(shapes[k]))
                   for k,(ox,oy) in enumerate(offs)):
                offs.append((dx,dy)); break
        else:
            offs.append((random.uniform(-1,1),random.uniform(-1,1)))
    return offs

def _random_shape_list(canvas):
    n=random.randint(2,5); L=[]
    for z in range(n):
        typ=random.choice(SHAPES); col=random.choice(COLORS)
        info={'type':typ,'color':col,'angle':0,'z':z}
        if typ in ('square','rectangle','oval'):
            w=random.uniform(canvas*0.07,canvas*0.2)
            h=w if typ=='square' else w*random.uniform(0.6,1.4)
            info.update(width=w, height=h)
        elif typ=='polygon':
            r=random.uniform(canvas*0.07,canvas*0.2)
            info.update(size=r, n=random.randint(5,8))
        else:
            s=random.uniform(canvas*0.07,canvas*0.2)
            info.update(size=s)
        L.append(info)
    return L

def gen_shapes_offsets(canvas, *, require_conn):
    for _ in range(300):
        shapes=_random_shape_list(canvas)
        offs=_generate_touching_offsets(shapes,canvas)
        if composite_fits_canvas(shapes,offs,canvas) and \
           (not require_conn or composite_connected(shapes,offs,canvas)):
            return shapes,offs
    raise RuntimeError("Could not generate initial connected shapes")

# ═════════════════════════════ Affine transform ════════════════════════════════
def apply_affine(img, sc, ang, C1, C2, canvas):
    ca,sa=math.cos(ang),math.sin(ang)
    mask=img.convert("L").point(lambda p:255 if p<255 else 0)
    bb=mask.getbbox()
    if bb:
        w,h=bb[2]-bb[0],bb[3]-bb[1]
        ex,ey=abs(w*ca)+abs(h*sa), abs(w*sa)+abs(h*ca)
        sc=min(sc, canvas/ex-1, canvas/ey-1)
    a,b=ca/sc, sa/sc; d,e=-sa/sc, ca/sc
    c=C1[0]-(a*C2[0]+b*C2[1]); f=C1[1]-(d*C2[0]+e*C2[1])
    return img.transform((canvas,canvas),Image.AFFINE,
                         (a,b,c,d,e,f),resample=Image.BICUBIC,fillcolor="white")

# ═════════════════════════════ Jitter helper ═══════════════════════════════════
def jitter_shapes(shapes, offs, canvas):
    """One attempt: returns new_shapes, new_offs, transform_list, attrs."""
    ns=[copy.deepcopy(s) for s in shapes]
    no=[(dx,dy) for dx,dy in offs]
    tlist=[]; attrs=[]
    num=max(NO_MIN_TRANSFORMS,random.randint(NO_MIN_TRANSFORMS,NO_MAX_TRANSFORMS))
    idxs=random.sample(range(len(ns)),num)
    for i in idxs:
        info=ns[i]; attrs.append({"type":info["type"],"color":info["color"]})
        mandatory=random.choice(['translate','rotate','scale'])
        if mandatory=='rotate' and info['type']=='circle':
            mandatory='translate'
        chosen={mandatory}
        for cat,p in (('translate',PROB_TRANSLATE),
                      ('rotate',PROB_ROTATE),
                      ('scale',PROB_SCALE),
                      ('reshape',PROB_RESHAPE)):
            if cat=='rotate' and info['type']=='circle': continue
            if cat not in chosen and random.random()<p: chosen.add(cat)
        tlist.extend(chosen)
        if 'translate' in chosen:
            dist=random.uniform(NO_MIN_TRANSLATE_PX,NO_MAX_TRANSLATE_PX)
            theta=random.uniform(0,2*math.pi)
            dx,dy=no[i]
            no[i]=(dx+dist*math.cos(theta),dy+dist*math.sin(theta))
        if 'rotate' in chosen:
            mag=random.uniform(JITTER_MIN_ROT,abs(NO_MAX_ROT))
            info['angle']=mag*random.choice([-1,1])
        if 'scale' in chosen:
            δ=random.uniform(RESCALE_MIN,RESCALE_MAX)
            factor=1+δ if random.random()<0.5 else 1-δ
            if info['type'] in ('square','rectangle','oval'):
                info['width']*=factor; info['height']*=factor
            elif info['type']=='polygon':
                info['size']*=factor
            else:
                info['size']*=factor
        if 'reshape' in chosen:
            r0=compute_shape_radius(info)
            nt=random.choice([t for t in SHAPES if t!=info['type']])
            if nt in ('square','rectangle','oval'):
                w=h=2*r0; h=w if nt=='square' else w*random.uniform(0.6,1.4)
                ns[i]={'type':nt,'color':info['color'],
                       'width':w,'height':h,'angle':info['angle'],'z':info['z']}
            elif nt=='polygon':
                ns[i]={'type':'polygon','color':info['color'],
                       'size':r0,'n':random.randint(5,8),
                       'angle':info['angle'],'z':info['z']}
            else:
                ns[i]={'type':nt,'color':info['color'],
                       'size':2*r0,'angle':info['angle'],'z':info['z']}
    return ns,no,tlist,attrs

# ═════════════════════ Human / Model view helpers ═══════════════════════════════
def save_human_view(i1,i2,truth,rid,canvas,odir,shapes):
    img1,img2=Image.open(i1),Image.open(i2)
    w,h=canvas,canvas
    c=Image.new("RGB",(w*2+10,h+LABEL_HEIGHT+10),"white")
    c.paste(img1,(0,LABEL_HEIGHT)); c.paste(img2,(w+10,LABEL_HEIGHT))
    d=ImageDraw.Draw(c)
    d.text((w//2-30,20),"Image 1","black")
    d.text((w+10+w//2-30,20),"Image 2","black")
    objs=", ".join(f"{s['color']} {s['type']}" for s in sorted(shapes,key=lambda x:x['z']))
    d.text((5,5),f"Objects: {objs}","black")
    d.text((w//2-20,h+LABEL_HEIGHT-5),truth.upper(),"black")
    p=os.path.join(odir,f"human_view_{rid}.png"); c.save(p); return p

def save_model_view(i1,i2,rid,canvas,odir):
    img1,img2=Image.open(i1),Image.open(i2)
    w,h=canvas,canvas
    mv=Image.new("RGB",(w*2+BAR_WIDTH,h+LABEL_HEIGHT),"white")
    d=ImageDraw.Draw(mv)
    d.rectangle([w,0,w+BAR_WIDTH-1,h+LABEL_HEIGHT],fill="black")
    mv.paste(img1,(0,LABEL_HEIGHT)); mv.paste(img2,(w+BAR_WIDTH,LABEL_HEIGHT))
    d.text((w//2-30,5),"Image 1","black")
    d.text((w+BAR_WIDTH+w//2-30,5),"Image 2","black")
    p=os.path.join(odir,f"model_view_{rid}.png"); mv.save(p); return p

# ═════════════════════════ Trial driver (shared) ═══════════════════════════════
def _run_trial(rid, canvas, odir, client,
               *, conn1: bool, conn2: bool, concat: bool,
               no_affine: bool = False, prompt_in: str = None):
    """
    Runs one trial, saving either separate (img1,img2) or combined (one canvas)
    images to `odir`. If client is None, skips inference but still saves the
    same images.
    """
    # 0) Few-shot demos (only if client provided)
    if FEW_SHOT and client is not None:
        ys, ns = generate_two_shot_examples(
            canvas, odir,
            one_canvas=concat,
            with_distractors=ENABLE_DISTRACTORS,
            allow_distractor_overlap=DISTRACTOR_CAN_OVERLAP,
            conn1=conn1, conn2=conn2, no_affine=no_affine,
        )
        client.yes0, client.yes1 = encode(ys[0]), encode(ys[1])
        client.no0,  client.no1  = encode(ns[0]), encode(ns[1])

    # 1) Draw image1
    shapes, offs = gen_shapes_offsets(canvas, require_conn=conn1)
    cx1 = cy1 = canvas // 2
    img1 = draw_composite(shapes, offs, canvas, (cx1, cy1))
    p1   = os.path.join(odir, f"a-1_{rid}.png")
    img1.save(p1, dpi=(500,500))

    truth = "yes" if random.random() < YES_PROB else "no"

    # 2) Draw image2 (affine or not)
    if no_affine:
        ang, sc, offs2 = 0.0, 1.0, offs
        img2 = img1.copy()
        cx2, cy2 = cx1, cy1
    else:
        while True:
            ang = random.uniform(*GLOBAL_ROT_RANGE)
            sc  = random.uniform(*GLOBAL_SCALE_RANGE)
            offs2 = [
                (dx*sc*math.cos(ang)-dy*sc*math.sin(ang),
                 dx*sc*math.sin(ang)+dy*sc*math.cos(ang))
                for dx,dy in offs
            ]
            if composite_fits_canvas(shapes, offs2, canvas) and \
               (not conn2 or composite_connected(shapes, offs2, canvas)):
                break
        cx2, cy2 = cx1, cy1
        img2 = apply_affine(img1, sc, ang, (cx1, cy1), (cx2, cy2), canvas)

    global_aff = (ang, sc, cx1, cy1, cx2, cy2)

    # 3) Jitter for NO‐cases
    transforms, jit_attrs, n_jit = [], [], 0
    if truth == "no":
        while True:
            ns, no_, tr, ja = jitter_shapes(shapes, offs2, canvas)
            if composite_fits_canvas(ns, no_, canvas) and \
               (not conn2 or composite_connected(ns, no_, canvas)):
                shapes2, offs2 = ns, no_
                transforms.extend(tr)
                jit_attrs.extend(ja)
                n_jit = len(ja)
                break
        img2 = draw_composite(shapes2, offs2, canvas, (cx2, cy2))

    # 4) Distractors
    if ENABLE_DISTRACTORS:
        R = max(
            math.hypot(dx, dy) + compute_shape_radius(s)
            for (dx, dy), s in zip(offs2, shapes)
        )
        img2 = _add_distractors(
            img2, (cx2, cy2), R, canvas,
            allow_overlap=DISTRACTOR_CAN_OVERLAP
        )

    # 5) Save image2
    p2 = os.path.join(odir, f"a-2_{rid}.png")
    img2.save(p2, dpi=(500,500))

    # 6) Combined view for concat trials (always)
    mv = None
    if concat:
        mv = save_model_view(p1, p2, rid, canvas, odir)

    # 7) Prompt
    prompt = prompt_in or (
        "The first image shows an object made of geometric shapes, which together "
        "form an object. Does this SAME object appear in the second image up to a "
        "rigid translation, rotation, and scale of the ENTIRE object as a whole? "
        "Respond with {yes} or {no}."
    )
    if ENABLE_DISTRACTORS:
        prompt += " There may be extra shapes; ignore them."

    # 8) Inference (skip if client is None)
    if client is None:
        ans = None
    else:
        if not concat:
            out = client.ask_pair(
                prompt,
                base64.b64encode(open(p1,"rb").read()).decode(),
                base64.b64encode(open(p2,"rb").read()).decode(),
                FEW_SHOT
            )
        else:
            b64 = base64.b64encode(open(mv,"rb").read()).decode()
            out = client.ask_single(prompt, b64, FEW_SHOT)
        ans = _extract_answer(out)

    return truth, ans, global_aff, transforms, shapes, n_jit, jit_attrs# ══════════════════════════════════════════════════════════════════════════════




def _imagenet_ds():
    global _IMAGENET_DS, _IMAGENET_IDX_BY_LABEL
    if _IMAGENET_DS is None:
        _IMAGENET_DS = load_dataset(
            "timm/mini-imagenet",
            split="validation",
            trust_remote_code=True
        )
        # --- build a label→indices map once, in ~150 ms ---------------
        idx_by_lbl = defaultdict(list)
        for i, lbl in enumerate(_IMAGENET_DS["label"]):
            idx_by_lbl[lbl].append(i)
        _IMAGENET_IDX_BY_LABEL = {k: v for k, v in idx_by_lbl.items()}
    return _IMAGENET_DS


def _random_imagenet_example(ds, *, seed):
    return ds.shuffle(seed=seed).select([0])[0]

def _random_alt_index_same_label(lbl, *, rnd, exclude_idx):
    """
    Return an index of the same ImageNet class `lbl`, but ≠ exclude_idx.
    """
    cand = _IMAGENET_IDX_BY_LABEL[lbl]
    if len(cand) == 1:
        return exclude_idx                  # degenerate (very unlikely)
    while True:
        alt = rnd.choice(cand)
        if alt != exclude_idx:
            return alt

def run_trial_5(run_id: str, canvas_size: int, out_dir: str, client):
    """
    • Image 1: one ImageNet patch centred on a white canvas.
    • Image 2: same patch after a random global affine OR
               a *different* patch of the same class + the same affine.
    """
    ds   = _imagenet_ds()
    seed = int.from_bytes(run_id.encode(), "little") & 0xFFFFFFFF
    rnd  = random.Random(seed + 1)

    # ─── pick the two samples ────────────────────────────────────────────────
    sam1 = _random_imagenet_example(ds, seed=seed)
    truth = "yes" if rnd.random() < YES_PROB else "no"
    if truth == "yes":
        sam2 = sam1
    else:
        idx1 = rnd.choice(range(len(ds)))
        sam1 = ds[idx1]
        lbl1 = sam1["label"]

        truth = "yes" if rnd.random() < YES_PROB else "no"
        if truth == "yes":
            sam2 = sam1
        else:
            alt_idx = _random_alt_index_same_label(lbl1,
                                                rnd=rnd,
                                                exclude_idx=idx1)
            sam2 = ds[alt_idx]

    # ─── choose patch size and canvas positions ──────────────────────────────
    patch_side = rnd.uniform(canvas_size * 0.35, canvas_size * 0.55)
    patch_side = int(patch_side)

    cx1 = cy1 = canvas_size // 2    # centre for both images

    def _make_canvas(rec):
        pil = rec["image"] if hasattr(rec["image"], "convert") \
               else Image.open(rec["image"]["path"])
        pil = pil.resize((patch_side, patch_side))
        cnv = Image.new("RGB", (canvas_size, canvas_size), "white")
        cnv.paste(pil, (cx1 - patch_side // 2, cy1 - patch_side // 2))
        return cnv

    img1 = _make_canvas(sam1)
    p1   = os.path.join(out_dir, f"a-1_{run_id}.png"); img1.save(p1)

    # ─── global affine for image 2 (rotation + scale + centred) ──────────────
    while True:
        ang = rnd.uniform(*GLOBAL_ROT_RANGE)
        sc  = rnd.uniform(*GLOBAL_SCALE_RANGE)
        # apply_affine rescales further if needed to keep inside the canvas,
        # so we do not need an explicit fits‑canvas check.
        img2_base = _make_canvas(sam2)
        img2 = apply_affine(img2_base, sc, ang,
                            (cx1, cy1), (cx1, cy1), canvas_size)
        break

    cx2 = cx1
    cy2 = cy1
    global_aff = (ang, sc, cx1, cy1, cx2, cy2)

    # ─── optional distractors ────────────────────────────────────────────────
    if ENABLE_DISTRACTORS:
        R_patch = math.sqrt(2) * patch_side / 2
        img2 = _add_distractors(img2, (cx2, cy2), R_patch,
                                canvas_size,
                                allow_overlap=DISTRACTOR_CAN_OVERLAP)

    p2 = os.path.join(out_dir, f"a-2_{run_id}.png"); img2.save(p2)

    # ─── prompt & model call ─────────────────────────────────────────────────
    prompt = (
        "The first image shows a single cropped ImageNet photograph on a white "
        "background. Does **exactly the same** photograph appear in the second "
        "image, up to a rigid translation, rotation, and scale of the ENTIRE "
        "photograph? If it was replaced by a different photo—even one of the same "
        "category—respond {no}. Otherwise respond {yes}."
    )

    save_human_view(p1, p2, truth, run_id, canvas_size, out_dir,
                    [{"type": "imagenet", "color": "n/a", "z": 0}])

    out = client.ask_pair(prompt, encode(img1), encode(img2), FEW_SHOT)
    ans = _extract_answer(out)

    # ─── analytics fields (no per‑shape jitters here) ────────────────────────
    transforms     = []
    n_jitter       = 0
    jittered_attrs = []
    shapes         = [{"type": "imagenet", "color": "n/a", "z": 0}]

    return (truth, ans, global_aff, transforms,
            shapes, n_jitter, jittered_attrs)
# trial wrappers
run_trial_1=lambda r,c,o,cl:_run_trial(r,c,o,cl,conn1=False,conn2=False,concat=False)
run_trial_2=lambda r,c,o,cl:_run_trial(r,c,o,cl,conn1=False,conn2=False,concat=True)
run_trial_3 = lambda r, c, o, cl: _run_trial(
    r, c, o, cl,
    conn1=False,          # start with a connected object (like T7)
    conn2=False,         # we do not force connectivity after jitter
    concat=False,        # separate images, not the concatenated bar view
    no_affine=True,       # <‑‑‑ key change
    prompt_in= "The first image shows an object made of geometric shapes, which together form an object. Does this SAME object appear in the second image? For example, if a component shape were to be rotated or translated, it would be a different object. Respond with {yes} or {no} (inside the curly brackets)."
)
run_trial_7=lambda r,c,o,cl:_run_trial(r,c,o,cl,conn1=True ,conn2=False,concat=False)
run_trial_8=lambda r,c,o,cl:_run_trial(r,c,o,cl,conn1=True ,conn2=False,concat=True)
run_trial_9=lambda r,c,o,cl:_run_trial(r,c,o,cl,conn1=True ,conn2=True ,concat=False, prompt_in= "“The first image shows an object made of 4-connected geometric shapes, which together form an object. Does this SAME object appear in the second image? For example, if a component shape were to be rotated or translated separately from the entire composite-object, it would be a different object. Respond with {yes} or {no} (inside the curly brackets).")
#run_trial_10 = trial10.run_trial_10
# ═════════════════════ distractor helper ═══════════════════════════════════════
def _draw_single(d,info,x0,y0):
    typ,col,ang=info['type'],info['color'],info.get('angle',0)
    if typ=='circle':
        r=info['size']/2; d.ellipse([x0-r,y0-r,x0+r,y0+r],fill=col)
    elif typ=='oval':
        w,h=info['width'],info['height']
        d.ellipse([x0-w/2,y0-h/2,x0+w/2,y0+h/2],fill=col)
    elif typ in ('square','rectangle'):
        w,h=info['width'],info['height']
        pts=[(x0-w/2,y0-h/2),(x0+w/2,y0-h/2),
             (x0+w/2,y0+h/2),(x0-w/2,y0+h/2)]
        if ang: pts=[rotate_point(px,py,x0,y0,ang) for px,py in pts]
        d.polygon(pts,fill=col)
    elif typ=='triangle':
        s,ht=info['size'],info['size']*math.sqrt(3)/2
        verts=[(x0,      y0-2/3*ht),
               (x0-s/2,  y0+1/3*ht),
               (x0+s/2,  y0+1/3*ht)]
        if ang: verts=[rotate_point(px,py,x0,y0,ang) for px,py in verts]
        d.polygon(verts,fill=col)
    elif typ=='polygon':
        rad,n=info['size'],info['n']
        verts=[(x0+rad*math.cos(2*math.pi*i/n),
                y0+rad*math.sin(2*math.pi*i/n)) for i in range(n)]
        if ang: verts=[rotate_point(px,py,x0,y0,ang) for px,py in verts]
        d.polygon(verts,fill=col)
    elif typ=='line':
        L=info['size']; p1,p2=(x0-L/2,y0),(x0+L/2,y0)
        if ang:
            p1=rotate_point(*p1,x0,y0,ang); p2=rotate_point(*p2,x0,y0,ang)
        d.line([p1,p2],fill=col,width=max(1,int(info['size']/10)))

def _add_distractors(
    img,
    center,
    radius,
    canvas,
    *,
    gap=10,
    cnt_rng=(1, 3),
    allow_overlap=False,
    max_shape_attempts=10,
    max_pos_attempts=250,
):
    """
    Draw 1–cnt_rng random distractors, but ensure none overlap the main object
    (within 'radius') nor each other (tracked in 'existing'). If after
    max_shape_attempts*max_pos_attempts no valid spot is found for a distractor,
    skip it.
    """
    d = ImageDraw.Draw(img)
    cx, cy = center
    existing = []  # list of (x, y, r_i) for placed distractors

    num_to_place = random.randint(*cnt_rng)
    for _ in range(num_to_place):
        shape_attempt = 0
        placed = False

        while shape_attempt < max_shape_attempts and not placed:
            shape_attempt += 1
            # pick a random shape
            typ = random.choice(SHAPES)
            col = random.choice(COLORS)

            # base size
            if typ in ('square','rectangle','oval'):
                w = random.uniform(canvas*0.05, canvas*0.15)
                h = w if typ=='square' else w*random.uniform(0.6,1.4)
                info = {'type':typ, 'color':col, 'width':w, 'height':h, 'angle':random.uniform(0,2*math.pi)}
                r_i = math.hypot(w/2, h/2)
            elif typ=='polygon':
                s = random.uniform(canvas*0.05, canvas*0.15)
                info = {'type':'polygon', 'color':col, 'size':s, 'n':random.randint(5,8), 'angle':random.uniform(0,2*math.pi)}
                r_i = s
            else:
                s = random.uniform(canvas*0.05, canvas*0.15)
                info = {'type':typ, 'color':col, 'size':s, 'angle':random.uniform(0,2*math.pi)}
                r_i = s/2

            # try to find a non-overlapping position
            pos_attempt = 0
            while pos_attempt < max_pos_attempts:
                pos_attempt += 1
                x = random.uniform(r_i, canvas - r_i)
                y = random.uniform(r_i, canvas - r_i)

                # check against main object
                if not allow_overlap:
                    if math.hypot(x-cx, y-cy) < (radius + r_i + gap):
                        continue

                # check against existing distractors
                if any(math.hypot(x-ex, y-ey) < (r_i + er + gap) for ex,ey,er in existing):
                    continue

                # found a valid spot!
                existing.append((x, y, r_i))
                _draw_single(d, info, x, y)
                placed = True
                break

            # if pos_attempt exhausted, shrink or pick new shape next loop
            if not placed:
                # reduce size to try to fit in next shape_attempt
                if 'width' in info and 'height' in info:
                    info['width'] *= 0.5
                    info['height'] *= 0.5
                    r_i *= 0.5
                elif 'size' in info:
                    info['size'] *= 0.5
                    r_i *= 0.5
                # loop back to pick new position (or new shape if next shape_attempt)
        
        # if after all shape_attempts still not placed, skip this distractor

    return img
# ═════════════════════════════ Analysis utils ══════════════════════════════════
def _extract_answer(t):
    m = _CURLY_RE.search(t)
    ans = (m.group(1) if m else t).strip().lower()
    if ans in ("yes", "no"):
        return ans
    for txt in (ans, t.lower()):
        y, n = txt.rfind("yes"), txt.rfind("no")
        if max(y, n) >= 0:
            return "yes" if y > n else "no"
    return ans


def analyze_transform_effects(records, label):
    """
    Fits four logistic models:
      1. magnitude model        — |θ|, |s−1|, |dx|, |dy|
      2. direction model        — |θ|, sign(s−1), |dx|, |dy|
      3. translation magnitude  — |dx|, |dy|
      4. translation direction  — signed dx, dy
    Uses _fit_logit_safe to retry on non-convergence and skips if still failing.
    """
    if not records:
        print(f"{label}: no data")
        return

    # 1) Extract raw features
    angles    = np.array([abs(r['angle']) for r in records])
    scales    = np.array([r['scale']       for r in records])
    scale_mag = np.abs(scales - 1.0)
    scale_dir = np.where(scales > 1.0, 1.0, -1.0)
    dxs       = np.array([abs(r['dx'])     for r in records])
    dys       = np.array([abs(r['dy'])     for r in records])
    y         = np.array([1 if r['correct'] else 0 for r in records])

    X_raw      = np.column_stack([angles, scale_mag, scale_dir, dxs, dys])
    feat_names = ['|θ|', '|s−1|', 'sign(s−1)', '|dx|', '|dy|']
    X_std      = StandardScaler().fit_transform(X_raw)

    def _safe_fit(col_idx, title):
        Xi    = X_std[:, col_idx]
        Xi    = np.column_stack([np.ones_like(y), Xi])  # add intercept
        names = ['intercept'] + [feat_names[i] for i in col_idx]

        model = _fit_logit_safe(y, Xi)
        if model is None:
            acc = y.mean()
            print(f"\n╔═ Affine effects on log-odds [{label}] — {title} ═╗")
            print("║  failed to converge after retries; using intercept-only.  ║")
            print(f"║  accuracy = {acc:.3f}                                    ║")
            print("╚" + "═"*55 + "╝")
            return

        params  = model.params
        se      = model.bse
        zvals   = params / se
        pvals   = model.pvalues
        ci_lo, ci_hi = model.conf_int().T
        odds    = np.exp(params)

        hdr = ['feature', 'β', 'SE', '95 % CI', 'z', 'p', 'odds']
        rows = []
        for j, nm in enumerate(names):
            rows.append([
                nm,
                f"{params[j]:+8.3f}",
                f"{se[j]:.3f}",
                f"[{ci_lo[j]:+.3f},{ci_hi[j]:+.3f}]",
                f"{zvals[j]:+.2f}",
                f"{pvals[j]:.3g}",
                f"{odds[j]:.3f}"
            ])

        print(f"\n╔═ Affine effects on log-odds [{label}] — {title} ═╗")
        print(tabulate(rows, headers=hdr, tablefmt="github"))

    # 2) Existing magnitude & direction models
    _safe_fit([0, 1, 3, 4], "magnitude model")
    _safe_fit([0, 2, 3, 4], "direction model")

    # 3) New: translation magnitude
    _safe_fit([3, 4], "translation magnitude model")

    # 4) New: translation direction (signed dx, dy)
    trans_raw = np.column_stack([[r['dx'] for r in records],
                                 [r['dy'] for r in records]])
    trans_std = StandardScaler().fit_transform(trans_raw)
    Xi_dir    = np.column_stack([np.ones_like(y), trans_std])
    names_dir = ['intercept', 'dx', 'dy']

    dir_model = _fit_logit_safe(y, Xi_dir)
    if dir_model is None:
        print(f"\n╔═ Translation direction effects on log-odds [{label}] ═╗")
        print("║  failed to converge after retries; skipping.               ║")
        print("╚" + "═"*55 + "╝")
    else:
        params  = dir_model.params
        se      = dir_model.bse
        zvals   = params / se
        pvals   = dir_model.pvalues
        ci_lo, ci_hi = dir_model.conf_int().T
        odds    = np.exp(params)

        hdr = ['feature', 'β', 'SE', '95 % CI', 'z', 'p', 'odds']
        rows = []
        for j, nm in enumerate(names_dir):
            rows.append([
                nm,
                f"{params[j]:+8.3f}",
                f"{se[j]:.3f}",
                f"[{ci_lo[j]:+.3f},{ci_hi[j]:+.3f}]",
                f"{zvals[j]:+.2f}",
                f"{pvals[j]:.3g}",
                f"{odds[j]:.3f}"
            ])

        print(f"\n╔═ Translation direction effects on log-odds [{label}] ═╗")
        print(tabulate(rows, headers=hdr, tablefmt="github"))   

# ─── Safe logit to avoid infinities ────────────────────────────────────────────
def _safe_logit(p, eps=1e-6):
    p = max(min(p, 1 - eps), eps)
    return math.log(p / (1 - p))

def _fit_logit_safe(endog, exog, max_retries=1):
    """
    Attempt to fit a Logit model with stronger convergence criteria
    and L2 regularization fallback. Returns a fitted result or None.
    """
    import numpy as np
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    # 1) Standardize exog (except intercept)
    X = exog.copy()
    # assume first column is intercept
    if X.shape[1] > 1:
        mean = X[:,1:].mean(axis=0)
        std  = X[:,1:].std(axis=0)
        std[std == 0] = 1.0
        X[:,1:] = (X[:,1:] - mean) / std

    # 2) Try unpenalized Logit with tightened tolerance
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = sm.Logit(endog, X).fit(
                method='newton',
                maxiter=500,
                tol=1e-6,
                disp=False
            )
        if hasattr(result, 'mle_retvals') and result.mle_retvals.get('converged', False):
            return result
    except Exception:
        pass

    # 3) Fall back to L2‐penalized (ridge) fit_regularized
    try:
        # alpha controls strength of L2; you can tune this if needed
        result = sm.Logit(endog, X).fit_regularized(
            method='l1',       # l1 here actually does elastic-net; with alpha small it approximates ridge
            alpha=0.1,         # smaller alpha → weaker penalty
            maxiter=500,
            tol=1e-6,
            disp=False
        )
        return result
    except Exception:
        return None

def _acc_ci(tp, fp, tn, fnc, alpha=0.05):
    """Wald 1-α CI for a proportion; returns (p, lo, hi)."""
    tot = tp + fp + tn + fnc
    if tot == 0:
        # no trials → treat accuracy as 0 with zero-width interval
        return 0.0, 0.0, 0.0

    p   = (tp + tn) / tot
    se  = math.sqrt(p * (1 - p) / tot)
    z   = norm.ppf(1 - alpha / 2)
    return p, p - z * se, p + z * se
# ─── Presence‐effects analyzer ────────────────────────────────────────────────
def analyze_presence_effects(records, categories, key, label):
    """
    Prints Δ‑log‑odds and Wald statistics; never crashes.
    Any undefined quantity (SE, CI, p) is reported as 'n/a'.
    """
    rows = []
    for cat in categories:
        n1 = acc1 = n0 = acc0 = 0
        for r in records:
            present = (
                any(attr.get('color') == cat or attr.get('type') == cat
                    for attr in r['jit_attrs'])
                if key == 'jit_attrs' else
                cat in r[key]
            )
            if present:
                n1  += 1
                acc1 += r['correct']
            else:
                n0  += 1
                acc0 += r['correct']

        # not enough data
        if n1 == 0 or n0 == 0:
            rows.append([cat, "n/a", "n/a", "n/a", "n/a", "n/a", n1, n0])
            continue

        p1, p0 = acc1 / n1, acc0 / n0
        β       = _safe_logit(p1) - _safe_logit(p0)
        odds    = math.exp(β)

        # SE / CI / p only if each denominator strictly > 0
        denom1 = n1 * p1 * (1 - p1)
        denom0 = n0 * p0 * (1 - p0)
        if denom1 > 0 and denom0 > 0:
            var    = 1/denom1 + 1/denom0
            se     = math.sqrt(var)
            z      = β / se
            p      = 2 * (1 - norm.cdf(abs(z)))
            ci_low = β - 1.96 * se
            ci_high= β + 1.96 * se
            rows.append([
                cat,
                f"{β:+.3f}",
                f"[{ci_low:+.3f},{ci_high:+.3f}]",
                f"{se:.3f}",
                f"{p:.3g}",
                f"{odds:.3f}",
                n1,
                n0
            ])
        else:
            # variance undefined – report point estimate only
            rows.append([
                cat,
                f"{β:+.3f}",
                "n/a",
                "n/a",
                "n/a",
                f"{odds:.3f}",
                n1,
                n0
            ])

    # order rows
    rows.sort(key=lambda r: (r[1] != "n/a",
                             abs(float(r[1]) if r[1] != "n/a" else 0)),
              reverse=True)

    hdr = ["category", "Δ log‑odds", "95 % CI", "SE",
           "p‑value", "odds", "# present", "# absent"]
    print(f"\n╔═ Effect of {key.replace('_',' ')} on accuracy  [{label}] ═╗")
    print(tabulate(rows, headers=hdr, tablefmt="github"))
# ═════════════════════════════ Main loop (updated) ═══════════════════════════════

def main():
    import argparse, os, tempfile, json, glob, base64, re, sys
    from collections import defaultdict
    from tabulate import tabulate
    from PIL import Image

    global ENABLE_DISTRACTORS, DISTRACTOR_CAN_OVERLAP, FEW_SHOT

    ap = argparse.ArgumentParser()
    #ap.add_argument("--models", nargs="+", default=["local:OpenGVLab/InternVL3-14B"])
    ap.add_argument("--models", nargs="+", default=[
                #    "openrouter:anthropic/claude-3.7-sonnet:thinking",
                #    "openrouter:google/gemini-2.5-pro-preview",
                #    "openai:o4-mini-2025-04-16",
                    #"openai:o3-2025-04-16",
                   # "local:google/gemma-3-27b-it",
                    # "local:allenai/Molmo-7B-D-0924",
                    #  "local:mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                    #"local:qwen",
                    #"local:llama",
                    #"local:microsoft/Phi-4-multimodal-instruct",
                    "local:OpenGVLab/InternVL3-14B",
                    "local:qwen2.5-vl-32b",

                                                    ])
    ap.add_argument("--num", type=int, default=5, help="Examples per trial")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--model_path", type=str, default=None)
    ap.add_argument("--no-distractors", action="store_true")
    ap.add_argument("--allow-distractor-overlap", action="store_true")
    ap.add_argument("--few-shot", action="store_true")
    ap.add_argument("--all-settings", action="store_true")
    ap.add_argument("--make-dataset", type=str, help="Directory to save generated dataset")
    ap.add_argument("--load-dataset", nargs="+", type=str, help="Directories of saved datasets to load")
    args = ap.parse_args()

    # parse model specs
    models = []
    for m in args.models:
        if ":" in m:
            backend, spec = m.split(":", 1)
        else:
            backend, spec = m, None
        models.append((m, backend.lower(), spec))

    # which trials to run
    TRIALS = (3,)

    # few-shot / distractor settings
    if args.all_settings:
        settings = [(False, False), (True, False), (False, True)]
    else:
        settings = [(args.few_shot, args.no_distractors)]

    # Prompt helper (unchanged)
    def make_prompt(t):
        if t == 3:
            prompt_in = (
                "“The first image shows an object made of geometric shapes, which together form an object. "
                "Does this SAME object appear in the second image? For example, if a component shape were to be "
                "rotated or translated, it would be a different object. Respond with {yes} or {no} "
                "(inside the curly brackets)."
            )
        elif t == 9:
            prompt_in = (
                "“The first image shows an object made of connected geometric shapes, which together form an object. "
                "Does this SAME object appear in the second image? For example, if a component shape were to be rotated "
                "or translated separately from the entire composite-object, it would be a different object. Respond with {yes} or {no} "
                "(inside the curly brackets)."
            )
        else:
            prompt_in = None

        default = (
            "The first image shows an object made of geometric shapes, which together form an object. "
            "Does this SAME object appear in the second image up to a rigid translation, rotation, and scale "
            "of the ENTIRE object as a whole? Respond with {yes} or {no}."
        )
        p = prompt_in or default
        if ENABLE_DISTRACTORS:
            p += " There may be extra shapes in Image 2 that are not part of the original object; as long as the object from Image 1 is present, the answer is yes even if there are other shapes present."
        return p


    # ─── make-dataset mode ─────────────────────────────────────────────────────
    if args.make_dataset:
        for t in TRIALS:
            for few_flag, no_dist in settings:
                FEW_SHOT = few_flag
                ENABLE_DISTRACTORS = not no_dist
                DISTRACTOR_CAN_OVERLAP = args.allow_distractor_overlap

                # base dir for this trial+setting
                base_dir = os.path.join(
                    args.make_dataset,
                    f"T{t}",
                    f"fs{int(FEW_SHOT)}_nd{int(not ENABLE_DISTRACTORS)}"
                )
                os.makedirs(base_dir, exist_ok=True)

                # 1) demos subfolder
                demo_dir = os.path.join(base_dir, "demo")
                os.makedirs(demo_dir, exist_ok=True)
                yes_pair, no_pair = generate_two_shot_examples(
                    CANVAS_SIZE, demo_dir,
                    one_canvas=(t in (2, 8)),
                    with_distractors=ENABLE_DISTRACTORS,
                    allow_distractor_overlap=DISTRACTOR_CAN_OVERLAP,
                    conn1=(t in (7, 8, 9)),
                    conn2=(t == 9),
                    no_affine=(t == 3),
                )
                yes_pair[0].save(os.path.join(demo_dir, "yes0.png"))
                yes_pair[1].save(os.path.join(demo_dir, "yes1.png"))
                no_pair[0].save(os.path.join(demo_dir, "no0.png"))
                no_pair[1].save(os.path.join(demo_dir, "no1.png"))

                # 2) examples subfolders
                for k in range(args.num):
                    rid = f"{t}_{k}"
                    ex_dir = os.path.join(base_dir, f"example_{k}")
                    os.makedirs(ex_dir, exist_ok=True)

                    # generate images (client=None skips inference)
                    truth, _, aff, transforms, shapes, n_jit, jit_attrs = globals()[f"run_trial_{t}"](
                        rid, CANVAS_SIZE, ex_dir, None
                    )

                    # rename/move outputs into ex_dir
                    img1_src = os.path.join(ex_dir, f"a-1_{rid}.png")
                    img2_src = os.path.join(ex_dir, f"a-2_{rid}.png")
                    if t in (2, 8):
                        # concatenated view
                        combo = os.path.join(ex_dir, f"model_view_{rid}.png")
                        os.replace(combo, os.path.join(ex_dir, "img1.png"))
                    else:
                        os.replace(img1_src, os.path.join(ex_dir, "img1.png"))
                        os.replace(img2_src, os.path.join(ex_dir, "img2.png"))

                    # write metadata
                    meta = {
                        "trial": t,
                        "few_shot": FEW_SHOT,
                        "enable_distractors": ENABLE_DISTRACTORS,
                        "allow_distractor_overlap": DISTRACTOR_CAN_OVERLAP,
                        "truth": truth,
                        "global_affine": aff,
                        "transforms": transforms,
                        "jit_attrs": jit_attrs,
                        "shapes": shapes,
                        "concat": (t in (2, 8)),
                        "prompt": make_prompt(t),
                    }
                    with open(os.path.join(ex_dir, "meta.json"), "w") as f:
                        json.dump(meta, f, indent=2)

        print("Finished making dataset.", flush=True)
        return
    # ─── load-dataset mode ─────────────────────────────────────────────────────
    if args.load_dataset:
        examples = defaultdict(lambda: defaultdict(list))

        for ds_dir in args.load_dataset:
            # iterate trials
            for trial_dir in glob.glob(os.path.join(ds_dir, "T*")):
                if not os.path.isdir(trial_dir):
                    continue
                t = int(os.path.basename(trial_dir)[1:])  # "T9" -> 9

                # iterate settings
                for setting_dir in glob.glob(os.path.join(trial_dir, "fs*_nd*")):
                    setting = os.path.basename(setting_dir)
                    m = re.match(r"fs([01])_nd([01])", setting)
                    if not m:
                        continue
                    fs_flag = bool(int(m.group(1)))
                    nd_flag = not bool(int(m.group(2)))

                    # load demos
                    demo_dir = os.path.join(setting_dir, "demo")
                    yes_pair = (
                        Image.open(os.path.join(demo_dir, "yes0.png")),
                        Image.open(os.path.join(demo_dir, "yes1.png"))
                    )
                    no_pair = (
                        Image.open(os.path.join(demo_dir, "no0.png")),
                        Image.open(os.path.join(demo_dir, "no1.png"))
                    )

                    # load each example
                    for ex_dir in glob.glob(os.path.join(setting_dir, "example_*")):
                        meta = json.load(open(os.path.join(ex_dir, "meta.json")))
                        img1_path = os.path.join(ex_dir, "img1.png")
                        img2_path = None if meta["concat"] else os.path.join(ex_dir, "img2.png")

                        b1 = base64.b64encode(open(img1_path, "rb").read()).decode()
                        b2 = None if img2_path is None else base64.b64encode(open(img2_path, "rb").read()).decode()

                        examples[t][(fs_flag, not meta["enable_distractors"])].append({
                            **meta,
                            "b1": b1,
                            "b2": b2,
                            "yes_pair": yes_pair,
                            "no_pair": no_pair,
                            "img1_path": img1_path,
                            "img2_path": img2_path
                        })
        # 3) run: MODEL -> TRIAL -> SETTING
        for mk, backend, spec in models:
            print(f"\n=== MODEL: {mk} ===")
            for t in TRIALS:
                print(f"\n--- TRIAL {t} ---")
                for few_flag, no_dist in settings:
                    group = examples[t].get((few_flag, no_dist), [])
                    if not group:
                        continue
                    print(f"\nSetting: few_shot={few_flag}, no_distractors={no_dist}")
                    # init metrics
                    confusion = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
                    recs = []
                    # one client per model+setting
                    yes_pair, no_pair = group[0]["yes_pair"], group[0]["no_pair"]
                    if 'client' in locals():
                        try:
                            # If the client has a torch model, move it to CPU
                            if hasattr(client, 'client') and isinstance(client.client, torch.nn.Module):
                                client.client.to('cpu')
                        except Exception:
                            pass
                        # Delete the client object and clear CUDA cache
                        del client
                        gc.collect()
                        torch.cuda.empty_cache()
                    if backend == "local":
                        client = InferenceClient("local", yes_pair, no_pair,
                                                 use_local=True, model_name=spec,
                                                 model_path=args.model_path)
                    elif backend == "openai":
                        client = InferenceClient("openai", yes_pair, no_pair,
                                                 model_name=spec)
                    else:
                        client = InferenceClient("openrouter", yes_pair, no_pair,
                                                 openrouter_model=spec)

                    # process each example
                    for ex in group:
                        if ex["concat"]:
                            raw_out = client.ask_single(ex["prompt"], ex["b1"], few_flag)
                        else:
                            raw_out = client.ask_pair(ex["prompt"], ex["b1"], ex["b2"], few_flag)
                        ans = _extract_answer(raw_out)
                        # verbose structured output
                        if args.verbose:
                            record = {
                                "model": mk,
                                "trial": t,
                                "few_shot": few_flag,
                                "no_distractors": no_dist,
                                "img1": ex["img1_path"],
                                "img2": ex["img2_path"],
                                "raw_out": raw_out,
                                "answer": ans,
                                "truth": ex["truth"]
                            }
                            print("VERBOSE_RESPONSE:", json.dumps(record),flush=True)

                        # update confusion
                        cm = confusion
                        if ans == "yes" and ex["truth"] == "yes":
                            cm["TP"] += 1
                        elif ans == "yes":
                            cm["FP"] += 1
                        elif ex["truth"] == "no":
                            cm["TN"] += 1
                        else:
                            cm["FN"] += 1

                        # record for effects
                        ang, sc, x1, y1, x2, y2 = ex["global_affine"]
                        recs.append({
                            "angle": ang,
                            "scale": sc,
                            "dx": x2 - x1,
                            "dy": y2 - y1,
                            "transforms": ex["transforms"],
                            "jit_attrs": ex["jit_attrs"],
                            "correct": (ans == ex["truth"])
                        })

                    # report metrics
                    p, lo, hi = _acc_ci(cm["TP"], cm["FP"], cm["TN"], cm["FN"])
                    label = f"Trial {t} ({'few-shot' if few_flag else 'zero-shot'},{'no-distractors' if no_dist else 'with-distractors'})"
                    print(f"\n>>> {label}: {p:.2%} (95% CI {lo:.2%}–{hi:.2%})")
                    print(tabulate([[k, v] for k, v in cm.items()],
                                   headers=["pred/true", "cnt"], tablefmt="grid"))
                    analyze_transform_effects(recs, f"{mk} T{t}")
                    analyze_presence_effects(recs, COLORS, 'jit_attrs', f"{mk} T{t} Colors")
                    analyze_presence_effects(recs, ["translate","rotate","scale"], 'transforms', f"{mk} T{t} Jitter Effects")
                    analyze_presence_effects(recs, SHAPES, 'jit_attrs', f"{mk} T{t} Shapes")
        torch.cuda.empty_cache()
        return

    # ─── normal random + inference ─────────────────────────────────────────────
    base_out = "outputs"
    os.makedirs(base_out, exist_ok=True)
    out = tempfile.mkdtemp(prefix="permute-connected-", dir=base_out)
    if args.verbose:
        print(f"[run] writing debug images to {out!r}")

    for mk, backend, spec in models:
        print(f"\n=== MODEL: {mk} ===")
        for t in TRIALS:
            print(f"\n--- TRIAL {t} ---")
            for few_flag, no_dist in settings:
                print(f"\nSetting: few_shot={few_flag}, no_distractors={no_dist}")
                FEW_SHOT = few_flag
                ENABLE_DISTRACTORS = not no_dist
                DISTRACTOR_CAN_OVERLAP = args.allow_distractor_overlap

                # one client per model+setting
                yes_imgs, no_imgs = generate_two_shot_examples(CANVAS_SIZE, out, one_canvas=FEW_SHOT)
                if backend == "local":
                    client = InferenceClient("local", yes_imgs, no_imgs,
                                             use_local=True, model_name=spec,
                                             model_path=args.model_path)
                elif backend == "openai":
                    client = InferenceClient("openai", yes_imgs, no_imgs,
                                             model_name=spec)
                else:
                    client = InferenceClient("openrouter", yes_imgs, no_imgs,
                                             openrouter_model=spec)

                confusion = {"TP":0,"FP":0,"TN":0,"FN":0}
                recs = []

                for k in range(args.num):
                    rid = f"{t}_{k}"
                    truth, ans, aff, transforms, shapes, n_jit, jit_attrs = \
                        globals()[f"run_trial_{t}"](rid, CANVAS_SIZE, out, client)
                    if ans == "yes" and truth == "yes":
                        confusion["TP"] += 1
                    elif ans == "yes":
                        confusion["FP"] += 1
                    elif truth == "no":
                        confusion["TN"] += 1
                    else:
                        confusion["FN"] += 1
                    ang, sc, x1, y1, x2, y2 = aff
                    recs.append({
                        "angle": ang, "scale": sc,
                        "dx": x2 - x1, "dy": y2 - y1,
                        "transforms": transforms,
                        "jit_attrs": jit_attrs,
                        "correct": (ans == truth)
                    })

                p, lo, hi = _acc_ci(confusion["TP"], confusion["FP"],
                                    confusion["TN"], confusion["FN"])
                label = f"Trial {t} ({'few-shot' if few_flag else 'zero-shot'},{'no-distractors' if no_dist else 'with-distractors'})"
                print(f"\n>>> {label}: {p:.2%} (95% CI {lo:.2%}–{hi:.2%})")
                print(tabulate([[k, v] for k, v in confusion.items()],
                               headers=["pred/true", "cnt"], tablefmt="grid"))
                analyze_transform_effects(recs, f"{mk} T{t}")
                analyze_presence_effects(recs, COLORS, 'jit_attrs', f"{mk} T{t} Colors")
                analyze_presence_effects(recs, ["translate","rotate","scale"], 'transforms', f"{mk} T{t} Jitter Effects")
                analyze_presence_effects(recs, SHAPES, 'jit_attrs', f"{mk} T{t} Shapes")

if __name__ == "__main__":
    main()





