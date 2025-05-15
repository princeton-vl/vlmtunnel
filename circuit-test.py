#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_circuitboard.py
────────────────────

Synthesises circuit‑board‑style images:

  • A central breadboard (10 numbered ports).
  • Between 4 – 10 peripheral components (each 1–3 ports) randomly
    placed around the canvas edges.
  • Randomly selected breadboard ports are connected to a random port
    on a random component by coloured wires that run in neat 90°
    segments (horizontal, then vertical).  Wires may cross.

Task (single question type)
---------------------------
  “Where does the wire from port N on the breadboard go?”
   – Answer with the *component label* in {curly braces}, e.g. {C3}.

Supports three back‑ends via the unchanged InferenceClient:

  --models openai openrouter local
  --use-local --local-model-name  <HF‑ID or nickname>
             --local-model-path   <FSDP checkpoint>

Reports:

  • Overall accuracy
  • Per‑component accuracy
  • Δ‑log‑odds of correctness when the target component is each label

Usage example
-------------
    export OPENAI_API_KEY=...
    export OPENROUTER_LAB_TOK=...
    python test_circuitboard.py \
        --models openai openrouter \
        --num 20 \
        --few-shot

The script will also save a “few_shot_examples/” folder with two
demonstration images for in‑context learning.
"""

import os, sys, argparse, random, math, re, base64, itertools, requests
from io import BytesIO
from collections import Counter, defaultdict
from typing import Tuple, List, Dict, Optional, Sequence
import warnings
from statsmodels.tools.sm_exceptions import PerfectSeparationError, ConvergenceWarning
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import gc

from transformers import (
    GenerationConfig,            # for Molmo
    AutoModelForCausalLM         # only used when model_name=="molmo"
)


from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    Pix2StructForConditionalGeneration,
    Pix2StructProcessor,
    GenerationConfig,
)

from scipy.stats import fisher_exact, chi2_contingency
import statsmodels.api as sm
import pandas as pd
import json
import glob
from infclient import run_inference, LocalFetcher


# ─────────── Constants ───────────
CANVAS_W, CANVAS_H  = 800, 600
BREAD_W,  BREAD_H   = 200, 320
BREAD_LEFT          = (CANVAS_W - BREAD_W)//2
BREAD_TOP           = (CANVAS_H - BREAD_H)//2
PORT_RADIUS         = 10
COMP_MIN_PORTS      = 1
COMP_MAX_PORTS      = 3
COMP_MIN            = 9
COMP_MAX            = 10
WIRE_WIDTH          = 4
WIRE_COLOURS        = ['red','orange','yellow','green','blue','purple']

MODEL_CODE          = "o3-2025-04-16"#"o4-mini-2025-04-16"#
FONT                = ImageFont.load_default()

def analyze_length_effect(records: List[dict]) -> None:
    """Logistic regression of correctness ~ wire length, with 95% CI & p-value."""
    lengths = np.array([r["length"] for r in records], dtype=float)
    success = np.array([r["correct"] for r in records], dtype=int)

    # design matrix: intercept + length
    X = sm.add_constant(lengths)

    # 1) Fit the model
    try:
        model = sm.Logit(success, X).fit(disp=False)
    except Exception as e:
        print(f"\n[Length regression skipped: model fit failed ({e})]")
        return

    # 2) Extract coefficient and p-value
    try:
        coef = float(model.params[1])
        pval = float(model.pvalues[1])
    except Exception:
        print("\n[Length regression: could not extract coefficient or p-value]")
        return

    # 3) Extract 95% confidence interval
    try:
        ci = model.conf_int()
        if hasattr(ci, "iloc"):
            lo, hi = float(ci.iloc[1, 0]), float(ci.iloc[1, 1])
        else:
            lo, hi = float(ci[1, 0]), float(ci[1, 1])
    except Exception:
        lo = hi = float("nan")

    # 4) Print summary
    print("\n=== Length effect (logistic regression) ===")
    print(f"log-odds coef: {coef:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   p={pval:.3g}")

def analyze_color_effect(records: List[dict]) -> None:
    """Δ log‑odds and Fisher p‑value for each wire colour."""
    colours = sorted({r["color"] for r in records})
    print("\n=== Colour effect on accuracy ===")
    print(f"{'color':<10}{'Δ log‑odds':>12}{'p‑value':>10}{'n':>6}")

    for col in colours:
        n1  = sum(1 for r in records if r["color"] == col)
        k1  = sum(r["correct"] for r in records if r["color"] == col)
        n0  = sum(1 for r in records if r["color"] != col)
        k0  = sum(r["correct"] for r in records if r["color"] != col)

        # Δ log‑odds
        def _safe(p, eps=1e-6):
            p = max(min(p, 1 - eps), eps)
            return math.log(p / (1 - p))
        delta = _safe(k1 / n1) - _safe(k0 / n0) if n1 and n0 else float("nan")

        # Fisher exact p‑value
        table = [[k1, n1 - k1], [k0, n0 - k0]]
        p_val = fisher_exact(table)[1] if all(min(row) >= 0 for row in table) else float("nan")

        print(f"{col:<10}{delta:+12.3f}{p_val:>10.3g}{n1:>6}")

# ─────────── Extended analyses ───────────
def analyze_distance_effect(records: List[dict]) -> None:
    """Logistic regression of correctness ~ euclidean distance, skipping on singularity."""
    distances = np.array([r["euclid_dist"] for r in records], dtype=float)
    success   = np.array([r["correct"]    for r in records], dtype=int)
    X = sm.add_constant(distances)
    try:
        model = sm.Logit(success, X).fit(disp=False)
    except PerfectSeparationError as e:
        print(f"\n[Distance regression skipped: perfect separation ({e})]")
        return
    except Exception as e:
        print(f"\n[Distance regression skipped: model fit failed ({e})]")
        return

    ci = model.conf_int()
    if isinstance(ci, np.ndarray):
        lo, hi = ci[1,0], ci[1,1]
    else:
        lo, hi = float(ci.iloc[1,0]), float(ci.iloc[1,1])
    coef = float(model.params[1])
    pval = float(model.pvalues[1])

    print("\n=== Distance effect (logistic regression) ===")
    print(f"log-odds coef: {coef:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   p={pval:.3g}")

def analyze_crossings_effect(records: List[dict]) -> None:
    """
    Logistic regression of correctness ~ number of crossings, with robustness
    to perfect separation, singular matrices, or zero variation in crossings.
    """
    import numpy as np
    from statsmodels.tools.sm_exceptions import PerfectSeparationError

    crosses = np.array([r["crossings"] for r in records], dtype=float)
    success = np.array([r["correct"]   for r in records], dtype=int)
    X = sm.add_constant(crosses)

    # 1) Skip if no variation in the predictor
    if crosses.size == 0 or np.all(crosses == crosses[0]):
        print("\n[Crossings regression skipped: no variation in crossings]")
        return

    # 2) Fit the model with exception handling
    try:
        model = sm.Logit(success, X).fit(disp=False)
    except PerfectSeparationError as e:
        print(f"\n[Crossings regression skipped: perfect separation ({e})]")
        return
    except np.linalg.LinAlgError as e:
        print(f"\n[Crossings regression skipped: singular matrix ({e})]")
        return
    except Exception as e:
        print(f"\n[Crossings regression skipped: model fit failed ({e})]")
        return

    # 3) Extract 95% CI, coefficient, and p‐value
    ci = model.conf_int()
    if isinstance(ci, np.ndarray):
        lo, hi = ci[1, 0], ci[1, 1]
    else:
        lo, hi = float(ci.iloc[1, 0]), float(ci.iloc[1, 1])
    coef = float(model.params[1])
    pval = float(model.pvalues[1])

    # 4) Print summary
    print("\n=== Crossings effect (logistic regression) ===")
    print(f"log-odds coef: {coef:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   p={pval:.3g}")
def analyze_color_bias(records: List[dict]) -> None:
    """
    Chi-squared test comparing the model’s predicted-wire-color distribution
    against the actual distribution of wire colors in each graph.
    """
    actual_counts = Counter(r["color"] for r in records)
    predicted_counts = Counter()

    for r in records:
        mapping = r["mapping"]
        pred    = r["pred"]
        # find a port that has that predicted component
        ports = [port for port, lab in mapping.items() if lab == pred]
        if ports:
            predicted_counts[r["winfo"][ports[0]]["color"]] += 1

    colors = sorted(actual_counts)
    obs = [predicted_counts[c] for c in colors]
    exp = [actual_counts[c] for c in colors]
    chi2, p, _, _ = chi2_contingency([obs, exp])

    print("\n=== Color bias (predicted vs actual) ===")
    print(f"Chi² = {chi2:.3f}, p = {p:.3g}")
    print(f"{'color':<10}{'actual':>10}{'predicted':>12}")
    for c in colors:
        print(f"{c:<10}{actual_counts[c]:>10}{predicted_counts[c]:>12}")


def analyze_multivariate_effect(records: List[dict]) -> None:
    """Logistic regression of correctness ~ length + euclid_dist + crossings."""
    import numpy as np

    # build a DataFrame
    df = pd.DataFrame({
        "length":      [r["length"]      for r in records],
        "euclid_dist": [r["euclid_dist"] for r in records],
        "crossings":   [r["crossings"]   for r in records],
        "success":     [r["correct"]     for r in records],
    })

    # design matrix
    X = sm.add_constant(df[["length", "euclid_dist", "crossings"]])

    # fit with error handling
    try:
        model = sm.Logit(df["success"], X).fit(disp=False)
    except Exception as e:
        print(f"\n[Multivariate regression skipped: {e}]")
        return

    ci = model.conf_int()
    print("\n=== Multivariate effect (logistic regression) ===")
    for i, var in enumerate(["length", "euclid_dist", "crossings"], start=1):
        if isinstance(ci, np.ndarray):
            lo, hi = ci[i, 0], ci[i, 1]
        else:
            lo, hi = float(ci.loc[var, 0]), float(ci.loc[var, 1])
        coef = float(model.params[var])
        pval = float(model.pvalues[var])
        print(f"{var:>12} coef {coef:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   p={pval:.3g}")

def analyze_color_frequency_bias(records: List[dict]) -> None:
    """
    For each color c:
      bias(c) = (times model picked c)/N  -  (avg prevalence of c across trials).
    Also computes overall bias toward the most-frequent color in each trial.
    """
    # 1) collect all colors seen
    colors = sorted({ r["winfo"][p]["color"]
                      for r in records
                      for p in r["mapping"].keys() })

    N = len(records)
    obs_counts       = Counter()                  # times model picked each color
    sum_prevalence   = {c: 0.0 for c in colors}   # sum of f_{i,c} over trials
    sum_delta_max    = 0.0                        # for overall most-freq bias

    for r in records:
        mapping = r["mapping"]
        winfo   = r["winfo"]

        # a) compute per-color prevalence in this trial
        total_wires = len(mapping)
        freq = Counter(winfo[port]["color"] for port in mapping)
        prevalence = { c: freq[c] / total_wires for c in colors }
        for c in colors:
            sum_prevalence[c] += prevalence[c]

        # b) record the model's pick
        pred = r["pred"]
        # find any port whose label == pred
        pred_ports = [port for port, lab in mapping.items() if lab == pred]
        if pred_ports:
            pick_color = winfo[pred_ports[0]]["color"]
            obs_counts[pick_color] += 1
        else:
            # if the model predicted an invalid component, skip
            continue

        # c) most-frequent color(s) this trial
        max_p = max(prevalence.values())
        max_colors = [c for c,v in prevalence.items() if v == max_p]
        indicator = 1 if pick_color in max_colors else 0
        sum_delta_max += (indicator - max_p)

    # 2) print per-color bias table
    print("\n=== Color-frequency bias ===")
    print(f"{'color':<10}{'bias':>10}{'obs_prob':>12}{'avg_prev':>12}")
    for c in colors:
        obs_prob = obs_counts[c] / N
        avg_prev = sum_prevalence[c] / N
        bias     = obs_prob - avg_prev
        print(f"{c:<10}{bias:>10.3f}{obs_prob:>12.3f}{avg_prev:>12.3f}")

    # 3) overall bias toward most-frequent color
    overall_bias = sum_delta_max / N
    print(f"\nOverall bias toward most-frequent color: {overall_bias:+.3f}")

def segments_intersect(seg1: Tuple[Tuple[int,int],Tuple[int,int]],
                       seg2: Tuple[Tuple[int,int],Tuple[int,int]]) -> bool:
    """
    Check proper intersection (excluding shared endpoints) between two line segments.
    Each segment is ((x1,y1),(x2,y2)).
    """
    (x1, y1), (x2, y2) = seg1
    (x3, y3), (x4, y4) = seg2

    def orient(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

    p, q, r, s = (x1,y1), (x2,y2), (x3,y3), (x4,y4)
    o1, o2 = orient(p,q,r), orient(p,q,s)
    o3, o4 = orient(r,s,p), orient(r,s,q)

    # proper intersection if orientations straddle each other
    return (o1 * o2 < 0) and (o3 * o4 < 0)

# ─────────── Utility: encode / decode ───────────
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
    m = _CURLY_RE.search(text)
    return m.group(1).strip() if m else text.strip()

# ─────────── Image Generation ───────────
def _draw_breadboard(draw: ImageDraw.Draw) -> List[Tuple[int,int]]:
    """Draw breadboard and return list of port-centre coords (len==10),
       with port numbers labelled just outside each circle."""
    # Breadboard rectangle
    draw.rectangle(
        [BREAD_LEFT, BREAD_TOP, BREAD_LEFT+BREAD_W, BREAD_TOP+BREAD_H],
        fill="#e0e0e0", outline="black", width=2
    )

    ports = []
    col_x = [BREAD_LEFT + BREAD_W*0.25, BREAD_LEFT + BREAD_W*0.75]
    spacing = BREAD_H / 6
    for c, x in enumerate(col_x):
        for i in range(5):
            y = BREAD_TOP + spacing*(i+1)
            ports.append((int(x), int(y)))

    # draw port circles and numbers just to the right
    for idx, (x, y) in enumerate(ports, start=1):
        # circle
        draw.ellipse(
            [x-PORT_RADIUS, y-PORT_RADIUS, x+PORT_RADIUS, y+PORT_RADIUS],
            fill="white", outline="black", width=2
        )
        # label text
        txt = str(idx)
        tw, th = draw.textbbox((0,0), txt, font=FONT)[2:]
        # position to the right, with small gap
        draw.text((x + PORT_RADIUS + 4, y - th/2), txt, fill="black", font=FONT)

    return ports

def _random_component_bbox(side:str, comp_w:int, comp_h:int)->Tuple[int,int,int,int]:
    """Return (left,top,right,bottom) for a component placed on given side."""
    margin = 20
    if side=='left':
        left  = margin
        top   = random.randint(margin, CANVAS_H-comp_h-margin)
    elif side=='right':
        left  = CANVAS_W - comp_w - margin
        top   = random.randint(margin, CANVAS_H-comp_h-margin)
    elif side=='top':
        left  = random.randint(margin, CANVAS_W-comp_w-margin)
        top   = margin
    else: # bottom
        left  = random.randint(margin, CANVAS_W-comp_w-margin)
        top   = CANVAS_H - comp_h - margin
    return (left, top, left+comp_w, top+comp_h)

def _draw_component(
    draw: ImageDraw.Draw,
    label: str,
    n_ports: int,
    existing_bbxs: List[Tuple[int, int, int, int]],
    *,
    fill_color: str = "#d0d0ff"
) -> Tuple[Dict, str]:
    """Draw one rectangular component, return its metadata + side chosen.

       fill_color  – hex string used for the rectangle fill.
    """
    comp_w, comp_h = 100, 60

    # pick a non‑overlapping side / position
    for _ in range(100):
        side = random.choice(["left", "right", "top", "bottom"])
        bbox = _random_component_bbox(side, comp_w, comp_h)
        l, t, r, b = bbox
        if all(r < L or l > R or b < T or t > B for (L, T, R, B) in existing_bbxs):
            break
    else:
        raise RuntimeError("Could not place component after 100 tries")

    # body + label
    draw.rectangle(bbox, fill=fill_color, outline="black", width=2)
    tw, th = draw.textbbox((0, 0), label, font=FONT)[2:]
    draw.text(((l + r - tw) / 2, (t + b - th) / 2), label, font=FONT, fill="black")

    # side that faces breadboard ⇒ hosts the terminals
    term_side = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}[side]

    # evenly spaced port circles
    ports = []
    if term_side in ("left", "right"):
        x = l if term_side == "left" else r
        ys = np.linspace(t + 10, b - 10, n_ports)
        for y in ys:
            ports.append((int(x), int(y)))
            draw.ellipse(
                [x - PORT_RADIUS, y - PORT_RADIUS, x + PORT_RADIUS, y + PORT_RADIUS],
                fill="white",
                outline="black",
                width=2,
            )
    else:
        y = t if term_side == "top" else b
        xs = np.linspace(l + 10, r - 10, n_ports)
        for x in xs:
            ports.append((int(x), int(y)))
            draw.ellipse(
                [x - PORT_RADIUS, y - PORT_RADIUS, x + PORT_RADIUS, y + PORT_RADIUS],
                fill="white",
                outline="black",
                width=2,
            )

    return {"label": label, "bbox": bbox, "ports": ports}, side

def do_segments_collide_strict(seg1: Tuple[Tuple[int,int],Tuple[int,int]],
                               seg2: Tuple[Tuple[int,int],Tuple[int,int]],
                               wire_width: int) -> bool:
    """
    Checks if two H/V wire segments visually collide, considering wire width.
    Assumes seg1 and seg2 are primarily horizontal or vertical (from orthogonal paths).
    Collision means:
    1. Their centerlines properly intersect (standard geometric intersection).
    2. They are parallel (both H or both V) and their bounding boxes (inflated by
       wire_width) overlap.
    3. One is H, one is V, and their bounding boxes (inflated by wire_width)
       intersect (catches corner cases where widths cause collision).
    """
    # Check for proper centerline intersection first
    if segments_intersect(seg1, seg2): # segments_intersect is your existing function
        return True

    # Define bounding boxes for the actual wire areas (not just centerlines)
    (x1_s1, y1_s1), (x2_s1, y2_s1) = seg1
    (x1_s2, y1_s2), (x2_s2, y2_s2) = seg2

    # Determine orientation and bounds for seg1's rectangle
    s1_is_h = (y1_s1 == y2_s1)
    s1_is_v = (x1_s1 == x2_s1)
    
    # Integer arithmetic for wire width halves
    # (width-1)//2 for the "lower" part from centerline, width//2 for "upper"
    # e.g., width 4: center c, pixels c-1, c, c+1, c+2. Range is [c-( (4-1)//2 ), c + (4//2) ] = [c-1, c+2]
    # e.g., width 3: center c, pixels c-1, c, c+1. Range is [c-( (3-1)//2 ), c + (3//2) ] = [c-1, c+1]
    ww_half_low = (wire_width - 1) // 2
    ww_half_high = wire_width // 2

    rect1_l, rect1_r, rect1_b, rect1_t = 0,0,0,0
    if s1_is_h: # seg1 is Horizontal
        rect1_l = min(x1_s1, x2_s1)
        rect1_r = max(x1_s1, x2_s1)
        rect1_b = y1_s1 - ww_half_low
        rect1_t = y1_s1 + ww_half_high
    elif s1_is_v: # seg1 is Vertical
        rect1_b = min(y1_s1, y2_s1)
        rect1_t = max(y1_s1, y2_s1)
        rect1_l = x1_s1 - ww_half_low
        rect1_r = x1_s1 + ww_half_high
    else: # seg1 is diagonal - this strict check is mainly for H/V paths.
          # If segments_intersect didn't catch it, assume no collision for this specialized check.
        return False 

    # Determine orientation and bounds for seg2's rectangle
    s2_is_h = (y1_s2 == y2_s2)
    s2_is_v = (x1_s2 == x2_s2)
    rect2_l, rect2_r, rect2_b, rect2_t = 0,0,0,0
    if s2_is_h: # seg2 is Horizontal
        rect2_l = min(x1_s2, x2_s2)
        rect2_r = max(x1_s2, x2_s2)
        rect2_b = y1_s2 - ww_half_low
        rect2_t = y1_s2 + ww_half_high
    elif s2_is_v: # seg2 is Vertical
        rect2_b = min(y1_s2, y2_s2)
        rect2_t = max(y1_s2, y2_s2)
        rect2_l = x1_s2 - ww_half_low
        rect2_r = x1_s2 + ww_half_high
    else: # seg2 is diagonal
        return False

    # Standard AABB (Axis-Aligned Bounding Box) intersection test
    # True if rectangles overlap
    no_horizontal_overlap = rect1_r <= rect2_l or rect1_l >= rect2_r
    no_vertical_overlap = rect1_t <= rect2_b or rect1_b >= rect2_t
    
    if no_horizontal_overlap or no_vertical_overlap:
        return False # No collision
    else:
        return True  # Collision detected

def _orthogonal_polyline(p0: Tuple[int,int], p1: Tuple[int,int], force_direction: Optional[str] = None) -> List[Tuple[int,int]]:
    """
    Return a two-segment (L-shaped) polyline between p0→p1.
    force_direction: "h_first" (horizontal then vertical) or 
                     "v_first" (vertical then horizontal) to override random choice.
    """
    x0, y0 = p0
    x1, y1 = p1

    horizontal_first_chosen: bool
    if force_direction == "h_first":
        horizontal_first_chosen = True
    elif force_direction == "v_first":
        horizontal_first_chosen = False
    else: # Default random choice
        horizontal_first_chosen = random.random() < 0.5
    
    if horizontal_first_chosen:
        # horizontal first, then vertical
        mid = (x1, y0)
    else:
        # vertical first, then horizontal
        mid = (x0, y1)

    return [p0, mid, p1]

def _diagonal_polyline(
        p0: Tuple[int,int],
        p1: Tuple[int,int],
        existing: List[Tuple[Tuple[int,int],Tuple[int,int]]]
    ) -> List[Tuple[int,int]]:
    """
    Two-segment diagonal path p0→mid→p1 (45° then 135°). If either leg would
    overlap an existing segment at the same slope, insert a third point mid2
    so the path becomes p0→mid→mid2→p1, all with smooth right-angle turns.
    """
    # margins so nothing hits the very edge
    margin = PORT_RADIUS + 2
    x0, y0 = p0
    x1, y1 = p1

    # decide which diagonal slope to use first
    slope1 = 1 if abs(x1 - x0) >= abs(y1 - y0) else -1
    slope2 = -1 / slope1

    # intersection of line1 (slope1 through p0) and line2 (slope2 through p1)
    b1 = y0 - slope1 * x0
    b2 = y1 - slope2 * x1
    xi = (b2 - b1) / (slope1 - slope2)
    yi = slope1 * xi + b1

    # clamp mid inside canvas
    xi = min(max(xi, margin), CANVAS_W - margin)
    yi = min(max(yi, margin), CANVAS_H - margin)
    mid = (int(xi), int(yi))

    # build the basic two-leg path
    path = [p0, mid, p1]
    def seg_slope(seg):
        (ax, ay), (bx, by) = seg
        dx = bx - ax
        dy = by - ay
        if dx == 0:
            return None
        return dy / dx


    eps = 1e-6
    for new_seg in zip(path, path[1:]):
            s_new = seg_slope(new_seg)
            for old_seg in existing:
                s_old = seg_slope(old_seg)

                # detect parallel: both vertical (None) or both numeric & equal
                parallel = (s_new is None and s_old is None) or (
                    s_new is not None and s_old is not None and abs(s_new - s_old) < eps
                )
                if parallel:
                    # need a detour: compute second kink mid2 as intersection of
                    #   line3 (slope2 thru mid)  &  line4 (slope1 thru p1)
                    b3 = mid[1] - slope2 * mid[0]
                    b4 = y1   - slope1 * x1
                    xi2 = (b4 - b3) / (slope2 - slope1)
                    yi2 = slope2 * xi2 + b3

                    # clamp again
                    xi2 = min(max(xi2, margin), CANVAS_W - margin)
                    yi2 = min(max(yi2, margin), CANVAS_H - margin)
                    mid2 = (int(xi2), int(yi2))

                    # return a 3-kink path: slope1 → slope2 → slope1
                    return [p0, mid, mid2, p1]

    # if no parallel overlap, just return the simple two-leg
    return path
def generate_circuit_image(
    *,
    min_components: int,
    max_components: int,
    min_ports: int,
    max_ports: int,
    min_wires: int,
    max_wires: int,
    min_cc_wires: int,
    max_cc_wires: int,
    wire_color_mode: str = "default",
    no_wire_crossing: bool = False,
    ) -> Tuple[Image.Image, int, str, Dict[int, str], Dict[int, Dict[str, object]]]:
    """
    Build a synthetic circuit and return:

    img               : PIL.Image
    query_port        : int  (breadboard port 1–10)
    correct_component : str  (e.g. "C3")
    mapping           : {breadboard_port → component_label}
    wire_info         : {
        breadboard_port → {
            'length': float,      # polyline length in pixels
            'euclid_dist': float, # straight-line distance in pixels
            'crossings': int,     # number of times this wire crosses any existing wire
            'color': str
        }
    }
    """
    COMPONENT_COLOURS = [
        "#ffd0d0", "#d0ffd0", "#d0d0ff", "#fff0d0",
        "#d0f0ff", "#f0d0ff", "#e0ffe0", "#ffe0ff",
    ]
    # Expanded wire colors for 'unique' mode, ensure enough for typical max wires
    EXTENDED_WIRE_COLOURS = WIRE_COLOURS + [
        '#FF1493', '#00FFFF', '#FFD700', '#ADFF2F', '#FF00FF', '#1E90FF', # HotPink, Aqua, Gold, GreenYellow, Fuchsia, DodgerBlue
        '#D2691E', '#8A2BE2', '#00FA9A', '#DC143C', '#7FFF00', '#BDB76B', # Chocolate, BlueViolet, MediumSpringGreen, Crimson, Chartreuse, DarkKhaki
        '#FF8C00', '#48D1CC', '#C71585', '#7CFC00', '#BA55D3', '#20B2AA'  # DarkOrange, MediumTurquoise, MediumVioletRed, LawnGreen, MediumOrchid, LightSeaGreen
    ]
    generated_single_color = None
    if wire_color_mode == "single":
        generated_single_color = random.choice(WIRE_COLOURS)

    # 1) create blank canvas
    img  = Image.new("RGB", (CANVAS_W, CANVAS_H), "white")
    draw = ImageDraw.Draw(img)

    # 2) draw breadboard and get its 10 port coordinates
    bread_ports = _draw_breadboard(draw)

    # 3) place components around edges
    n_components = random.randint(min_components, max_components)
    components, occupied = [], []
    for i in range(n_components):
        n_ports_i = random.randint(min_ports, max_ports)
        comp, _ = _draw_component(
            draw,
            f"C{i+1}",
            n_ports_i,
            occupied,
            fill_color=random.choice(COMPONENT_COLOURS),
        )
        occupied.append(comp["bbox"])
        components.append(comp)

    all_segments: List[Tuple[Tuple[int,int], Tuple[int,int]]] = []
    mapping: Dict[int,str] = {}
    wire_info: Dict[int,Dict[str,object]] = {}
    used_comp_ports = set()

    avail_bread = list(range(1, 11))
    random.shuffle(avail_bread)
    # Ensure min_wires is respected if possible, up to max_wires or available ports
    target_bb_wires = random.randint(min_wires, min(max_wires, len(avail_bread), len(components) * max_ports))
    
    bb_wires_drawn_count = 0

    # 4) connect breadboard → components
    for bb_port_idx in avail_bread:
        if bb_wires_drawn_count >= target_bb_wires:
            break

        p0_bb = bread_ports[bb_port_idx - 1]
        
        # Try to find a component with a free port
        random.shuffle(components) # Try components in random order
        connection_made_for_bb_port = False
        for comp_obj in components:
            free_comp_ports = [
                pt for pt in comp_obj["ports"]
                if (comp_obj["label"], pt) not in used_comp_ports
            ]
            if not free_comp_ports:
                continue

            p1_comp = random.choice(free_comp_ports)
            
            poly_path = None
            path_found_for_pair = False

            if no_wire_crossing:
                # Try horizontal-first orthogonal path
                poly_h_first = _orthogonal_polyline(p0_bb, p1_comp, force_direction="h_first")
                # USE do_segments_collide HERE
                collides_h_first = any(do_segments_collide_strict(seg_h, old_seg, WIRE_WIDTH) for seg_h in zip(poly_h_first, poly_h_first[1:]) for old_seg in all_segments)

                if not collides_h_first:
                    poly_path = poly_h_first
                    path_found_for_pair = True
                else:
                    # Try vertical-first orthogonal path
                    poly_v_first = _orthogonal_polyline(p0_bb, p1_comp, force_direction="v_first")
                    # USE do_segments_collide HERE
                    # USE do_segments_collide_strict HERE
                    collides_v_first = any(do_segments_collide_strict(seg_v, old_seg, WIRE_WIDTH) for seg_v in zip(poly_v_first, poly_v_first[1:]) for old_seg in all_segments)
                    if not collides_v_first:
                        poly_path = poly_v_first
                        path_found_for_pair = True
            else: # Original behavior (allow crossing, use diagonal)
                poly_path = _diagonal_polyline(p0_bb, p1_comp, all_segments)
                path_found_for_pair = True # Assume diagonal path is always "found"

            if path_found_for_pair and poly_path:
                used_comp_ports.add((comp_obj["label"], p1_comp)) # Occupy port only if path is found

                wire_col_bb: str
                if wire_color_mode == "single":
                    wire_col_bb = generated_single_color
                elif wire_color_mode == "unique":
                    color_idx = len(mapping) # Based on successfully mapped BB wires
                    wire_col_bb = EXTENDED_WIRE_COLOURS[color_idx % len(EXTENDED_WIRE_COLOURS)]
                else: # "default"
                    wire_col_bb = random.choice(WIRE_COLOURS)
                
                draw.line(poly_path, fill=wire_col_bb, width=WIRE_WIDTH)

                length = sum(math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(poly_path, poly_path[1:]))
                euclid_dist = math.hypot(p1_comp[0] - p0_bb[0], p1_comp[1] - p0_bb[1])
                
                current_crossings = 0
                for new_seg in zip(poly_path, poly_path[1:]):
                    for old_s in all_segments:
                        if segments_intersect(new_seg, old_s):
                            current_crossings +=1
                
                mapping[bb_port_idx] = comp_obj["label"]
                wire_info[bb_port_idx] = {
                    "length": length, "euclid_dist": euclid_dist,
                    "crossings": current_crossings, "color": wire_col_bb,
                }
                for a, b in zip(poly_path, poly_path[1:]):
                    all_segments.append((a, b))
                
                bb_wires_drawn_count += 1
                connection_made_for_bb_port = True
                break # Connected this breadboard port, move to the next one
        # If loop finishes without connection_made_for_bb_port, this bb_port_idx is skipped.

    # 5) connect component ↔ component wires (visual only)
    target_cc_wires = random.randint(min_cc_wires, max_cc_wires)
    cc_wires_drawn_count = 0
    
    # Create a list of all component ports with their component labels
    all_component_ports_with_labels = []
    for comp_obj in components:
        for p_idx, port_coord in enumerate(comp_obj["ports"]):
             all_component_ports_with_labels.append({'label': comp_obj["label"], 'coord': port_coord, 'id': (comp_obj["label"], port_coord)})

    for _ in range(target_cc_wires * 5): # Try more times to find valid pairs
        if cc_wires_drawn_count >= target_cc_wires:
            break

        available_for_cc = [p for p in all_component_ports_with_labels if p['id'] not in used_comp_ports]
        if len(available_for_cc) < 2:
            break

        comp1_port_info = random.choice(available_for_cc)
        # Find a port from a *different* component
        comp2_options = [p for p in available_for_cc if p['label'] != comp1_port_info['label']]
        if not comp2_options:
            continue
        comp2_port_info = random.choice(comp2_options)

        p1_cc, p2_cc = comp1_port_info['coord'], comp2_port_info['coord']
        
        poly_path_cc = None
        path_found_cc = False
        if no_wire_crossing:
            poly_h_cc = _orthogonal_polyline(p1_cc, p2_cc, force_direction="h_first")
            # USE do_segments_collide HERE
            if not any(do_segments_collide_strict(s, o, WIRE_WIDTH) for s in zip(poly_h_cc, poly_h_cc[1:]) for o in all_segments):
                poly_path_cc = poly_h_cc
                path_found_cc = True
            else:
                poly_v_cc = _orthogonal_polyline(p1_cc, p2_cc, force_direction="v_first")
                # USE do_segments_collide HERE
                # USE do_segments_collide_strict HERE
                if not any(do_segments_collide_strict(s, o, WIRE_WIDTH) for s in zip(poly_v_cc, poly_v_cc[1:]) for o in all_segments):
                    poly_path_cc = poly_v_cc
                    path_found_cc = True
        else:
            poly_path_cc = _diagonal_polyline(p1_cc, p2_cc, all_segments)
            path_found_cc = True

        if path_found_cc and poly_path_cc:
            used_comp_ports.add(comp1_port_info['id'])
            used_comp_ports.add(comp2_port_info['id'])

            cc_wire_color: str
            if wire_color_mode == "single":
                cc_wire_color = generated_single_color
            elif wire_color_mode == "unique":
                # Total wires so far: bb_wires + current_cc_wires
                color_idx = len(mapping) + cc_wires_drawn_count 
                cc_wire_color = EXTENDED_WIRE_COLOURS[color_idx % len(EXTENDED_WIRE_COLOURS)]
            else: # "default"
                cc_wire_color = random.choice(WIRE_COLOURS)

            draw.line(poly_path_cc, fill=cc_wire_color, width=WIRE_WIDTH)
            for a, b in zip(poly_path_cc, poly_path_cc[1:]):
                all_segments.append((a, b))
            cc_wires_drawn_count += 1
            
    # 6) pick a query port that is definitely wired
    if not mapping:
        # This can happen if min_wires is 0, or if no_wire_crossing prevented any wires.
        # Create a dummy response or handle as an edge case.
        # For robustness in testing, let's return a default if no wires are mapped.
        # The calling code might need to handle this (e.g. skip analysis for such an image).
        print("Warning: No breadboard wires were successfully placed for query generation.", file=sys.stderr)
        # Return a default query, image might be "empty" of connections.
        # This ensures the function doesn't crash but the trial might be invalid.
        return img, 1, "C1", {}, {} # Dummy query_port, correct_comp, empty mappings

    query_port = random.choice(list(mapping.keys()))
    correct_comp_label = mapping[query_port]

    return img, query_port, correct_comp_label, mapping, wire_info
# ─────────── Few‑shot example builder ───────────
def generate_two_shot_examples(
    odir: str = "few_shot_examples",
    **gen_kwargs,
) -> List[Tuple[Image.Image, str, str]]:
    """
    Produce two canonical examples (saved under `odir`) and return:
      [(image, prompt_text, correct_component), …]
    """
    os.makedirs(odir, exist_ok=True)
    examples = []
    for i in range(2):
        img, qport, comp_label, _, _ = generate_circuit_image(**gen_kwargs)
        img.save(f"{odir}/two_shot_{i}.png")
        prompt = (
            f"Which component does the wire from port {qport} on the breadboard, which is the gray rectangle with numbered ports, connect to? "
            "A wire is a series of connected, same colored lines that go from the center of a port, represented on the screen as a white circle, to another port. Each wire only connects two ports, one at either end. "
            "A wire will NEVER turn at the same spot that it intersects another wire, and wires do not change colors. "
            "Answer with the component label in curly braces, e.g {C0}."
        )
        examples.append((img, prompt, f"After following the wire, it connects to {comp_label}. {{{comp_label}}}"))
    return examples




class InferenceClient:
    """
    Supported back‑ends
      • name == "openai"   → OpenAI Completion (vision‑capable)
      • otherwise          → OpenRouter Completion (vision‑capable)
      • --use-local        → any local HF model, including:
            ─ "molmo‑7B‑D‑0924"    (chat/vision)
            ─ "pix2struct‑ai2d‑base" (seq2seq/vision)

    Environment variables
        OPENAI_LAB_KEY        – for the OpenAI back‑end
        OPENROUTER_LAB_TOK    – for the OpenRouter back‑end
    """

    # ────────────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        name: str,
        demos: List[Tuple[Image.Image, str, str]],
        *,
        api_key: Optional[str] = None,
        openrouter_model: str = "grok-1",
        use_local: bool = False,
        model_name: str = "",
        model_path: str | None = None,
    ):
        self.name        = name.lower()
        self.or_model    = name
        self.use_local   = use_local
        self.model_name  = model_name 
        self.model_path  = model_path
        self.demos      = demos

        # ───────── Local HF model ─────────
        if use_local:
            self.client, self.processor, self.device = LocalFetcher(
                model_path=self.model_path,
                model_name=self.model_name
            ).get_model_params()

        # ───────── OpenAI remote ─────────
        elif self.name == "openai":
            import openai

            key = api_key or os.getenv("OPENAI_LAB_KEY")
            if key is None:
                raise RuntimeError("OPENAI_LAB_KEY is not set")
            self.client = openai.OpenAI(api_key=key, timeout=1000)

        # ───────── OpenRouter remote ─────
        else:
            key = api_key or os.getenv("OPENROUTER_LAB_TOK")
            if key is None:
                raise RuntimeError("OPENROUTER_LAB_TOK is not set")
            self._or_endpoint = "https://openrouter.ai/api/v1/chat/completions"
            self._or_headers  = {
                "Authorization": f"Bearer {key}",
                "Content-Type":  "application/json",
            }

    # ────────────────────────────────── core multimodal generation helper ───────
    # ---------------------------------------------------------------------------
    @staticmethod
    def _data_url(b64: str) -> str:
        # CORRECT: returns the raw data‐URL string
        return f"data:image/png;base64,{b64}"

    def _or_post(self, messages: list[dict]) -> str:
        """Call OpenRouter and return the assistant’s reply text."""
        payload = {
            "model":     self.model_name,
            "messages":  messages,
            "reasoning": {"effort": "high"},
        }
        try:
            r = requests.post(
                self._or_endpoint,
                headers=self._or_headers,
                json=payload,
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[OpenRouter error] {e}")
            return random.choice(["{{C0}}", "{{C1}}"])

    # ─────────────────────────── ask_pair (unused in visual‑search) ─────────────
    def ask_pair(self, prompt: str, b1: str, b2: str, few_shot: bool) -> str:
        """Two‑image queries (not used in this benchmark)."""
        # Only needed if you extend the benchmark; logic analogous to ask_single.
        raise NotImplementedError("ask_pair is not used in visual‑search trials.")

    # ─────────────────────────────── ask_single (main path) ────────────────────
    def ask_single(self, prompt: str, b64_image: str, few_shot: bool) -> str:
        """
        Ask a single‐image query, routing to local, OpenAI, or OpenRouter back-ends.
        Fixed OpenRouter branch to send only JSON-serializable image_url payloads.
        """
        # ─── Local HF model ───────────────────────────────────────────────────
        if self.use_local:
            img = decode(b64_image)
            if not few_shot:
                return run_inference(
                    self.client, self.processor, self.model_name,
                    images=img, query=prompt, temperature=0.0001
                )
            # Build few-shot convo for local
            msgs = []
            for img_ex, ex_prompt, ex_answer in self.demos:
                msgs.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_ex},
                        {"type": "text",  "text": ex_prompt},
                    ],
                })
                msgs.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"{ex_answer}"}],
                })
            msgs.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text",  "text": prompt},
                ],
            })
            return run_inference(
                self.client, self.processor, self.model_name,
                messages=msgs, temperature=0.0001
            )

        # ─── OpenAI remote ────────────────────────────────────────────────────
        if self.name == "openai":
            try:
                input_payload = [{
                    "role": "user",
                    "content": [
                        {"type": "input_text",  "text": prompt},
                        {"type": "input_image", "image_url": f"data:image/png;base64,{b64_image}"}
                    ]
                }]
                if few_shot:
                    demos_payload = []
                    for img_ex, ex_prompt, ex_answer in self.demos:
                        demos_payload.append({
                            "role": "user",
                            "content": [
                                {"type": "input_text",  "text": ex_prompt},
                                {"type": "input_image", "image_url": f"data:image/png;base64,{encode(img_ex)}"}
                            ]
                        })
                        demos_payload.append({
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": f"{ex_answer}"}]
                        })
                    input_payload = demos_payload + input_payload
                resp = self.client.responses.create(
                    model=self.model_name,
                    reasoning={"effort": "high"},
                    input=input_payload,
                )
                return resp.output_text
            except Exception as e:
                print(f"[OpenAI error] {e}", file=sys.stderr)
                return random.choice(["{{C0}}", "{{C1{}}"])

        # ─── OpenRouter remote ────────────────────────────────────────────────
        # Only send text + image_url entries (no PIL.Image)
        
        messages = []
        if not few_shot:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text",      "text": prompt},
                    {"type": "image_url", "image_url": self._data_url(b64_image)},
                ],
            }]
        else:
            for img_ex, ex_prompt, ex_answer in self.demos:
                ex_b64 = encode(img_ex)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text",      "text": ex_prompt},
                        {"type": "image_url", "image_url": self._data_url(ex_b64)},
                    ],
                })
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"{ex_answer}"}],
                })
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text",      "text": prompt},
                    {"type": "image_url", "image_url": self._data_url(b64_image)},
                ],
            })

        return self._or_post(messages)
# ─────────── Analysis helpers ───────────
def _safe_logit(p,eps=1e-6): p=max(min(p,1-eps),eps); return math.log(p/(1-p))

def per_component_accuracy(recs:List[dict]):
    comps=sorted({r['label'] for r in recs})
    print("\n=== Per‑component accuracy ===")
    print(f"{'comp':<6}{'acc':>8}{'n':>6}")
    for c in comps:
        subset=[r for r in recs if r['label']==c]
        n=len(subset); acc=sum(r['correct'] for r in subset)/n if n else float('nan')
        print(f"{c:<6}{acc:>8.2%}{n:>6}")

def delta_log_odds(recs:List[dict]):
    comps=sorted({r['label'] for r in recs})
    print("\n=== Δ‑log‑odds of correctness by component ===")
    print(f"{'comp':<6}{'Δ log‑odds':>12}{'#':>6}")
    for c in comps:
        n1 = sum(1 for r in recs if r['label']==c);  k1=sum(r['correct'] for r in recs if r['label']==c)
        n0 = sum(1 for r in recs if r['label']!=c);  k0=sum(r['correct'] for r in recs if r['label']!=c)
        if n1==0 or n0==0: d=float('nan')
        else:
            d=_safe_logit(k1/n1) - _safe_logit(k0/n0)
        print(f"{c:<6}{d:>+12.3f}{n1:>6}")

# ─────────── Main ───────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=
                    [
                    # "openai:o3-2025-04-16",
                    "openrouter:anthropic/claude-3.7-sonnet:thinking",
                    "openrouter:google/gemini-2.5-pro-preview",
                    "openai:o4-mini-2025-04-16",
                    # "local:google/gemma-3-27b-it",
                    # "local:allenai/Molmo-7B-D-0924",
                    # "local:mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                    # "local:qwen",
                    # "local:qwen2.5-vl-32b",
                    # "local:OpenGVLab/InternVL3-14B",
                    # "local:microsoft/Phi-4-multimodal-instruct",
                    # "local:llama",
                    ])
    ap.add_argument("--num",          type=int, default=5)
    ap.add_argument("--few-shot",     action="store_true")
    ap.add_argument("--verbose",      action="store_true")
    ap.add_argument("--use-local",    action="store_true")
    ap.add_argument("--local-model-name", type=str, default="")
    ap.add_argument("--local-model-path", type=str, default=None)

    # generation controls
    ap.add_argument("--min-components",  type=int, default=5)
    ap.add_argument("--max-components",  type=int, default=10)
    ap.add_argument("--min-ports",       type=int, default=1)
    ap.add_argument("--max-ports",       type=int, default=3)
    ap.add_argument("--min-wires",       type=int, default=7)
    ap.add_argument("--max-wires",       type=int, default=15)
    ap.add_argument("--min-cc-wires",    type=int, default=0)
    ap.add_argument("--max-cc-wires",    type=int, default=6)

    # wire drawing options
    ap.add_argument("--wire-color-mode", type=str, default="default",
                    choices=["default", "single", "unique"],
                    help="Wire color strategy: 'default' (random from list), 'single' (all one color), 'unique' (each wire different color if possible)")
    ap.add_argument("--no-wire-crossing", action="store_true",
                    help="Attempt to draw wires that do not cross each other (may result in fewer wires and use orthogonal paths).")

    # new dataset flags
    ap.add_argument("--make-dataset", type=str,
                    help="Directory to save generated dataset (no inference)")
    ap.add_argument("--load-dataset", nargs="+", type=str,
                    help="Directories of saved datasets to load & evaluate")
    ap.add_argument("--all-settings", action="store_true",
                    help="Run both no-few-shot and few-shot settings")

    ap.add_argument("--test-image", action="store_true")
    args = ap.parse_args()
    settings = [(False,), (True,)] if args.all_settings else [(args.few_shot,)]

    gen_kwargs = dict(
        min_components=args.min_components,
        max_components=args.max_components,
        min_ports=args.min_ports,
        max_ports=args.max_ports,
        min_wires=args.min_wires,
        max_wires=args.max_wires,
        min_cc_wires=args.min_cc_wires,
        max_cc_wires=args.max_cc_wires,
        wire_color_mode=args.wire_color_mode,
        no_wire_crossing=args.no_wire_crossing,
    )

    # 1) Build two few-shot demos once
    demos = generate_two_shot_examples(**gen_kwargs)

    # 2) Parse model specs into (spec_string, backend, model_name)
    models = []
    for spec in args.models:
        if ":" in spec:
            backend, mdl = spec.split(":", 1)
        else:
            backend, mdl = spec, ""
        models.append((spec, backend.lower(), mdl))

    # 3) Handle test-image -- just generate one and exit
    if args.test_image:
        img, q, c, _, _ = generate_circuit_image(**gen_kwargs)
        img.save("test_circuit.png")
        print(f"Saved test_circuit.png (query={q}, answer={c})")
        sys.exit(0)

    # 4) MAKE-DATASET
    if args.make_dataset:
        ds_dir = args.make_dataset
        os.makedirs(ds_dir, exist_ok=True)

        for (few_flag,) in settings:
            sub = "with_fs" if few_flag else "no_fs"
            subdir = os.path.join(ds_dir, sub)
            os.makedirs(subdir, exist_ok=True)

            # save demos only for few-shot
            if few_flag:
                demo_dir = os.path.join(subdir, "few_shot_examples")
                os.makedirs(demo_dir, exist_ok=True)
                for idx, (img, prompt, ans) in enumerate(demos):
                    img.save(f"{demo_dir}/fs_{idx}.png")
                    with open(f"{demo_dir}/fs_{idx}.json", "w") as f:
                        json.dump({"prompt": prompt, "answer": ans}, f, indent=2)

            # save trials under subdir
            for i in range(args.num):
                img, q, comp, mapping, winfo = generate_circuit_image(**gen_kwargs)
                prompt = (
                    f"Which component does the wire from port {q} on the breadboard, "
                    "which is the gray rectangle with numbered ports, connect to? "
                    "A wire is a series of connected, same colored lines that go from the center of a port, represented on the screen as a white circle, to another port. Each wire only connects two ports, one at either end. "
                    "A wire will NEVER turn at the same spot that it intersects another wire, and wires do not change colors. "
                    "Answer with the component label in curly braces, e.g {C3}."
                )
                meta = {
                    "query_port":   q,
                    "correct_comp": comp,
                    "mapping":      mapping,
                    "winfo":        winfo,
                    "prompt":       prompt,
                    "few_shot":     few_flag
                }
                img.save(os.path.join(subdir, f"img_{i}.png"))
                with open(os.path.join(subdir, f"meta_{i}.json"), "w") as f:
                    json.dump(meta, f, indent=2)

        print(f"[make-dataset] saved to {ds_dir}")
        return
    
    if args.load_dataset:
        for spec, backend, mdl in models:
            client = InferenceClient(
                name=backend,
                demos=demos,
                api_key=None,
                openrouter_model=mdl,
                use_local=(backend == "local") or args.use_local,
                model_name=mdl,
                model_path=args.local_model_path
            )
            print(f"\n=== Loaded-DS Model: {spec} ===")

            for (few_flag,) in settings:
                sub = "with_fs" if few_flag else "no_fs"
                print(f"\n--- Setting few_shot={few_flag} ---")
                records: List[dict] = []

                # load every meta_*.json in each given dataset dir
                for ds_dir in args.load_dataset:
                    dir_ = os.path.join(ds_dir, sub)
                    for mf in glob.glob(os.path.join(dir_, "meta_*.json")):
                        info = json.load(open(mf))
                        mapping = {int(k): v for k, v in info["mapping"].items()}
                        winfo   = {int(k): v for k, v in info["winfo"].items()}
                        img_path = mf.replace("meta_", "img_").replace(".json", ".png")
                        b64 = base64.b64encode(open(img_path, "rb").read()).decode()

                        raw  = client.ask_single(info["prompt"], b64, few_flag)
                        pred = extract_braced(raw).upper()
                        qp   = info["query_port"]

                        rec = {
                            "label":       info["correct_comp"].upper(),
                            "pred":        pred,
                            "mapping":     mapping,
                            "winfo":       winfo,
                            "correct":     (pred == info["correct_comp"].upper()),
                            "length":      winfo[qp]["length"],
                            "euclid_dist": winfo[qp]["euclid_dist"],
                            "crossings":   winfo[qp]["crossings"],
                            "color":       winfo[qp]["color"],
                        }
                        records.append(rec)

                        if args.verbose:
                            vr = {
                            "model":       spec,
                            "meta_path":   mf,
                            "query_port":  qp,
                            "raw":         raw,
                            "label":       rec["label"],
                            "pred":        rec["pred"],
                            "correct":     rec["correct"],
                        }
                            print("VERBOSE_RESPONSE:", json.dumps(vr),flush=True)

                # print overall accuracy even if zero records
                n = len(records)
                correct = sum(r["correct"] for r in records)
                acc = correct / max(n, 1)
                print(f"Overall accuracy {correct}/{n} = {acc:.2%}")

                # only run the four analyses when we actually have data
                if n > 0:
                    analyze_distance_effect(records)
                    analyze_crossings_effect(records)
                    analyze_multivariate_effect(records)
                    analyze_color_frequency_bias(records)
                else:
                    print(f"[load_dataset] no records for few_shot={few_flag}, skipping detailed analyses.")

            # free GPU memory after this model
            try:
                if hasattr(client, "client") and isinstance(client.client, torch.nn.Module):
                    client.client.to("cpu")
            except:
                pass
            del client
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return
    
    # 6) LIVE-EVAL (unchanged)
    for spec, backend, mdl in models:
        client = InferenceClient(
            name=backend,
            demos=demos,
            api_key=None,
            openrouter_model=mdl,
            use_local=(backend == "local") or args.use_local,
            model_name=mdl,
            model_path=args.local_model_path
        )
        print(f"\n=== Live Model: {spec} ===")

        for (few_flag,) in settings:
            print(f"\n--- Setting few_shot={few_flag} ---")
            recs = []
            for _ in range(args.num):
                img, q, comp, mapping, winfo = generate_circuit_image(**gen_kwargs)
                prompt = (
                    f"Which component does the wire from port {q} on the breadboard, "
                    "which is the gray rectangle with numbered ports, connect to? "
                    "A wire is a series of connected, same colored lines that go from the center of a port, represented on the screen as a white circle, to another port. Each wire only connects two ports, one at either end. "
                    "A wire will NEVER turn at the same spot that it intersects another wire, and wires do not change colors. "
                    "Answer with the component label in curly braces, e.g {C3}."
                )
                raw  = client.ask_single(prompt, encode(img), few_flag)
                pred = extract_braced(raw).upper()
                recs.append({
                    "label":       comp.upper(),
                    "pred":        pred,
                    "mapping":     mapping,
                    "winfo":       winfo,
                    "correct":     (pred == comp.upper()),
                    "length":      winfo[q]["length"],
                    "euclid_dist": winfo[q]["euclid_dist"],
                    "crossings":   winfo[q]["crossings"],
                    "color":       winfo[q]["color"],
                })
                if args.verbose:
                    print(f"GT {comp} | pred {pred} | {pred==comp.upper()}", flush=True)

            # run analyses verbatim
            n   = len(recs)
            acc = sum(r["correct"] for r in recs) / max(n,1)
            print(f"Overall accuracy {sum(r['correct'] for r in recs)}/{n} = {acc:.2%}")
            analyze_distance_effect(recs)
            analyze_crossings_effect(recs)
            analyze_multivariate_effect(recs)
            analyze_color_frequency_bias(recs)

        # free GPU memory after both settings
        try:
            if hasattr(client, "client") and isinstance(client.client, torch.nn.Module):
                client.client.to("cpu")
        except:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()