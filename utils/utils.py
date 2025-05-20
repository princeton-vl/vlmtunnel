#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
common_utils.py
---------------
Common utility functions for VQA test scripts.
"""

import base64
import math
import re
from io import BytesIO
from typing import List, Tuple
from PIL import Image
from collections import Counter

from scipy.stats import norm 
import numpy as np 
from tabulate import tabulate

# regex for curly extract
_CURLY_RE = re.compile(r"\{([^{}]+)\}")

def extract_braced(text: str) -> str:
    """
    Extracts the first substring inside curly braces {}.
    If none found, returns the original text, stripped.
    """
    if not isinstance(text, str): # Handle cases where text might not be a string
        return str(text).strip()
    m = _CURLY_RE.search(text)
    return m.group(1).strip() if m else text.strip()



def _extract_yes_no_answer(text: str) -> str:
    """
    Extracts 'yes' or 'no' from text, case-insensitive.
    Returns 'error' if neither is found.
    """
    if not isinstance(text, str):
        return "error"
    text_lower = text.lower()
    if "{{yes}}" in text_lower or "yes" in re.findall(r'\b(yes)\b', text_lower):
        return "yes"
    if "{{no}}" in text_lower or "no" in re.findall(r'\b(no)\b', text_lower):
        return "no"
    if "yes" in text_lower:
        return "yes"
    if "no" in text_lower:
        return "no"
    return text 

# Image Encoding/Decoding
def encode(img: Image.Image) -> str:
    """Encodes a PIL Image to a base64 string."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def decode(b64_png: str) -> Image.Image:
    """Decodes a base64 string to a PIL Image."""
    byte_data = base64.b64decode(b64_png)
    img = Image.open(BytesIO(byte_data))
    img.load()  # Force actual pixel data into memory
    return img

# Mathematical Utilities
def _safe_logit(p: float, eps: float = 1e-7) -> float:
    """Safely computes logit, clamping p to [eps, 1-eps]."""
    p = max(min(p, 1.0 - eps), eps)
    return math.log(p / (1.0 - p))

def _norm_cdf(z: float) -> float:
    """Computes the CDF of a standard normal distribution."""
    return norm.cdf(z)

# Analysis Utilities
def analyze_presence_effects(
    records: List[dict],
    categories: List[str],
    key: str, # The key in each record dict that holds the category value for that record
    label: str # A label for the analysis, used in the printed output title
):
    """
    Computes Δ-log-odds, 95% CI, and p-value (Wald z-test)
    of accuracy when the feature specified by `key` == cat vs != cat.

    Args:
        records (List[dict]): A list of record dictionaries. Each record must contain
                               a 'correct' boolean/int field and the `key` field.
        categories (List[str]): A list of category values to analyze.
        key (str): The dictionary key in each record that indicates the category.
        label (str): A descriptive label for the print output.
    """
    eps = 1e-9

    rows = []
    for cat_value in categories:
        present_records = [r for r in records if r.get(key) == cat_value]
        # Records where the specific category is absent for the given key
        absent_records = [r for r in records if r.get(key) != cat_value]

        n1 = len(present_records)
        acc1 = sum(r['correct'] for r in present_records)
        n0 = len(absent_records)
        acc0 = sum(r['correct'] for r in absent_records)

        if n1 == 0 or n0 == 0: # Not enough data for comparison
            rows.append((str(cat_value), 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', n1, n0))
            continue

        p1_raw = acc1 / n1 if n1 > 0 else 0.0
        p0_raw = acc0 / n0 if n0 > 0 else 0.0


        p1 = np.clip(p1_raw, eps, 1 - eps)
        p0 = np.clip(p0_raw, eps, 1 - eps)

        # Δ-log-odds (coefficient)
        coeff = _safe_logit(p1, eps) - _safe_logit(p0, eps)
        odds_ratio = math.exp(coeff)


        var1_inv = n1 * p1 * (1 - p1)
        var0_inv = n0 * p0 * (1 - p0)

        if var1_inv <= 0 or var0_inv <= 0: # Avoid division by zero or log of non-positive
            se, z, pval, ci_lo, ci_hi = (float('nan'), float('nan'), float('nan'), float('nan'), float('nan'))
        else:
            var = (1 / var1_inv) + (1 / var0_inv)
            se = math.sqrt(var)
            z = coeff / se if se > 0 else float('nan') # Avoid division by zero if se is somehow zero
            pval = 2 * (1 - _norm_cdf(abs(z))) if not math.isnan(z) else float('nan')
            ci_lo = coeff - 1.96 * se
            ci_hi = coeff + 1.96 * se

        rows.append((
            str(cat_value),
            f"{coeff:+.3f}" if not math.isnan(coeff) else 'n/a',
            f"[{ci_lo:+.3f}, {ci_hi:+.3f}]" if not math.isnan(ci_lo) else 'n/a',
            f"{se:.3f}" if not math.isnan(se) else 'n/a',
            f"{pval:.3g}" if not math.isnan(pval) else 'n/a',
            f"{odds_ratio:.3f}" if not math.isnan(odds_ratio) else 'n/a',
            n1,
            n0
        ))

    # Sort rows by the absolute value of Δ log-odds, descending. Handle 'n/a'.
    def sort_key(row_data):
        try:
            return abs(float(row_data[1]))
        except (ValueError, TypeError):
            return -1 # Place 'n/a' or non-float values at the end

    rows.sort(key=sort_key, reverse=True)


    header = ["category", "Δ log‑odds", "95% CI", "SE", "p‑value", "Odds Ratio", "# present", "# absent"]
    print(f"\n=== Effect of {key} on accuracy [{label}] ===")
    print(tabulate(rows, headers=header, tablefmt="github"))