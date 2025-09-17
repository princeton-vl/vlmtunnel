#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyzer.py
───────────
Contains functions for analyzing model inference results for various VQA tasks.
This module expects 'utils.py' (with _safe_logit, _norm_cdf, etc.) to be in the PYTHONPATH.
"""
import sys # For stderr and exit
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError, ConvergenceWarning
from scipy.stats import fisher_exact, chi2_contingency
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple
import warnings
import math

from utils.utils import _safe_logit, _norm_cdf 
from sklearn.preprocessing import StandardScaler


# --- Helper Functions ---

def _perform_logistic_regression(
    records: List[Dict[str, Any]],
    feature_keys: List[str],
    dependent_var_key: str,
    analysis_prefix: str,
    analysis_suffix: str
) -> None:
    """
    Performs logistic regression and prints results.
    Models: dependent_var_key ~ feature_keys.
    Math: Fits a logistic regression model. Reports log-odds coefficients, 95% CIs, and p-values for each feature.
    The log-odds (logit) is ln(p/(1-p)).
    """
    if not records:
        print(f"\n[{analysis_prefix} regression skipped: no records]")
        return

    data_dict = {key: [r.get(key, float('nan')) for r in records] for key in feature_keys}
    data_dict[dependent_var_key] = [r.get(dependent_var_key, 0) for r in records] # Default to 0 for missing dependent var
    
    df = pd.DataFrame(data_dict).dropna()

    if len(df) < 2 or df[dependent_var_key].nunique() < 2:
        print(f"\n[{analysis_prefix} regression skipped: insufficient data after NaN removal (rows: {len(df)}, unique outcomes: {df[dependent_var_key].nunique()})]")
        return
    for key in feature_keys:
        if df[key].nunique() < 2:
            print(f"\n[{analysis_prefix} regression skipped: feature '{key}' has no variation]")
            return
            
    X = sm.add_constant(df[feature_keys])
    y = df[dependent_var_key]

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=UserWarning) # From statsmodels with pandas
            model = sm.Logit(y, X).fit(disp=False)
    except PerfectSeparationError as e:
        print(f"\n[{analysis_prefix} regression skipped: PerfectSeparationError ({e})]")
        return
    except np.linalg.LinAlgError as e:
        print(f"\n[{analysis_prefix} regression skipped: LinAlgError/Singular matrix ({e})]")
        return
    except Exception as e:
        print(f"\n[{analysis_prefix} regression skipped: model fit failed ({e})]")
        return

    print(f"\n=== {analysis_prefix}: {analysis_suffix} (logistic regression) ===")
    try:
        conf_int = model.conf_int()
        for param_name in model.params.index:
            if param_name == 'const':
                continue
            coef = float(model.params[param_name])
            pval = float(model.pvalues[param_name])
            ci_low, ci_high = float(conf_int.loc[param_name, 0]), float(conf_int.loc[param_name, 1])
            
            if len(feature_keys) == 1: # Single predictor
                print(f"log-odds coef: {coef:+.4f}    95% CI [{ci_low:+.4f}, {ci_high:+.4f}]    p={pval:.3g}")
            else: # Multiple predictors
                print(f"{param_name:>12} coef {coef:+.4f}    95% CI [{ci_low:+.4f}, {ci_high:+.4f}]    p={pval:.3g}")
    except Exception as e:
        print(f"\n[{analysis_prefix} regression: error extracting results ({e})]")

# --- Functions for test_circuitboard.py (Visual Circuit Tracing) ---

def analyze_length_effect(records: List[Dict[str, Any]]) -> None:
    """CB: Logistic regression of correctness ~ wire length."""
    _perform_logistic_regression(records, ["length"], "correct", "CB: Length", "effect")

def analyze_color_effect(records: List[Dict[str, Any]]) -> None:
    """CB: Δ log‑odds and Fisher p‑value for each wire colour vs others.
    Math: Δ log-odds = logit(P(correct|color)) - logit(P(correct|other colors)). Fisher's Exact Test for p-value.
    """
    if not records: print("\n[CB Color effect skipped: no records]"); return
    valid_records = [r for r in records if r.get("color") is not None and r.get("correct") is not None]
    if not valid_records: print("\n[CB Color effect skipped: no valid color/correctness data]"); return
    
    colors = sorted(list(set(r["color"] for r in valid_records)))
    if not colors: print("\n[CB Color effect skipped: no colors found]"); return

    print("\n=== CB: Colour effect on accuracy ===")
    print(f"{'color':<10}{'Δ log‑odds':>12}{'p‑value':>10}{'n (color)':>10}")
    for col in colors:
        n1 = sum(1 for r in valid_records if r["color"] == col)
        k1 = sum(1 for r in valid_records if r["color"] == col and r["correct"])
        n0 = sum(1 for r in valid_records if r["color"] != col)
        k0 = sum(1 for r in valid_records if r["color"] != col and r["correct"])

        delta_lo, p_fisher = float("nan"), float("nan")
        if n1 > 0 and n0 > 0:
            delta_lo = _safe_logit(k1 / n1) - _safe_logit(k0 / n0)
            try: _, p_fisher = fisher_exact([[k1, n1 - k1], [k0, n0 - k0]])
            except ValueError: pass # If counts are too small/problematic
        print(f"{col:<10}{delta_lo:+12.3f}{p_fisher:>10.3g}{n1:>10}")

def analyze_distance_effect(records: List[Dict[str, Any]]) -> None:
    """CB: Logistic regression of correctness ~ euclidean distance."""
    _perform_logistic_regression(records, ["euclid_dist"], "correct", "CB: Distance", "effect")

def analyze_crossings_effect(records: List[Dict[str, Any]]) -> None:
    """CB: Logistic regression of correctness ~ number of crossings."""
    _perform_logistic_regression(records, ["crossings"], "correct", "CB: Crossings", "effect")

def analyze_color_bias(records: List[Dict[str, Any]]) -> None:
    """CB: Chi-squared test for model's predicted-wire-color distribution vs actual.
    Math: Pearson's Chi-squared test for goodness of fit between observed predicted color frequencies and actual color frequencies.
    """
    if not records: print("\n[CB Color bias skipped: no records]"); return
    actual_counts = Counter(r["color"] for r in records if "color" in r)
    if not actual_counts: print("\n[CB Color bias skipped: no actual colors]"); return

    predicted_counts = Counter()
    for r in records:
        mapping, pred_label, winfo = r.get("mapping"), r.get("pred"), r.get("winfo")
        if not all([mapping, pred_label, winfo]): continue
        ports = [p for p, actual in mapping.items() if actual == pred_label]
        if ports and ports[0] in winfo and "color" in winfo[ports[0]]:
            predicted_counts[winfo[ports[0]]["color"]] += 1
    
    if not predicted_counts: print("\n[CB Color bias skipped: no predicted colors]"); return

    colors_test = sorted(list(actual_counts.keys()))
    obs_freq = [predicted_counts.get(c, 0) for c in colors_test]
    exp_freq = [actual_counts.get(c, 0) for c in colors_test]

    if sum(exp_freq) == 0 or sum(obs_freq) == 0: print("\n[CB Color bias skipped: zero total frequencies]"); return
    try:
        chi2, p, _, _ = chi2_contingency([obs_freq, exp_freq])
        print("\n=== CB: Color bias (Predicted via Comp-type vs. Queried Wire Color) ===")
        print(f"Chi² = {chi2:.3f}, p = {p:.3g}")
        print(f"{'color':<10}{'Actual (Queried)':>18}{'Predicted (via Comp)':>22}")
        for c in colors_test: print(f"{c:<10}{actual_counts.get(c,0):>18}{predicted_counts.get(c, 0):>22}")
    except ValueError as e: print(f"\n[CB Color bias Chi² test skipped: {e}]")

def analyze_multivariate_effect(records: List[Dict[str, Any]]) -> None:
    """CB: Logistic regression of correctness ~ length + euclid_dist + crossings."""
    _perform_logistic_regression(records, ["length", "euclid_dist", "crossings"], "correct", "CB: Multivariate", "effect")

def analyze_color_frequency_bias(records: List[Dict[str, Any]]) -> None:
    """CB: Compares model's color pick rate vs. average color prevalence in images."""
    if not records: print("\n[CB Color-frequency bias skipped: no records]"); return
    
    all_graph_colors = set()
    for r in records:
        for port_info in r.get("winfo", {}).values():
            if "color" in port_info: all_graph_colors.add(port_info["color"])
    colors_domain = sorted(list(all_graph_colors))
    if not colors_domain and len(records) > 0: print("[CB Color-frequency bias: no colors in graphs, but records exist.]")
    if not records: print("\n[CB Color-frequency bias skipped: N is zero]"); return

    obs_counts, sum_prevalence = Counter(), {c: 0.0 for c in colors_domain}
    sum_delta_max, n_eff_prev, n_eff_obs = 0.0, 0, 0

    for r in records:
        mapping, winfo, pred_label = r.get("mapping"), r.get("winfo"), r.get("pred")
        if not mapping or not winfo: continue
        n_eff_prev +=1
        
        trial_colors = [winfo[p]["color"] for p in mapping if p in winfo and "color" in winfo[p]]
        prevalence_trial = {c: 0.0 for c in colors_domain}
        if trial_colors:
            counts_trial = Counter(trial_colors)
            prevalence_trial = {c: counts_trial.get(c,0)/len(trial_colors) for c in colors_domain}
        for c in colors_domain: sum_prevalence[c] += prevalence_trial.get(c, 0.0)

        if pred_label is None: continue
        ports_to_pred = [p for p, actual in mapping.items() if actual == pred_label and p in winfo and "color" in winfo[p]]
        if ports_to_pred:
            color_of_pred_wire = winfo[ports_to_pred[0]]["color"]
            obs_counts[color_of_pred_wire] += 1
            n_eff_obs +=1
            if trial_colors:
                max_p = max(prevalence_trial.values()) if prevalence_trial else 0.0
                is_max_freq = 1 if prevalence_trial.get(color_of_pred_wire, -1.0) == max_p and max_p > 0 else 0
                sum_delta_max += (is_max_freq - max_p)
    
    print("\n=== CB: Color-frequency bias (Predicted Component's Wire Color vs. Avg Prevalence) ===")
    print(f"{'color':<10}{'bias':>10}{'obs_prob':>12}{'avg_prev':>12} (N_obs={n_eff_obs}, N_prev={n_eff_prev})")
    for c in colors_domain:
        obs_p = obs_counts[c]/n_eff_obs if n_eff_obs > 0 else 0.0
        avg_p = sum_prevalence[c]/n_eff_prev if n_eff_prev > 0 else 0.0
        print(f"{c:<10}{obs_p-avg_p:>10.3f}{obs_p:>12.3f}{avg_p:>12.3f}")
    print(f"\nOverall bias toward most-frequent color: {sum_delta_max/n_eff_prev if n_eff_prev > 0 else 0.0:+.3f}")

def per_component_accuracy(records: List[Dict[str, Any]]) -> None:
    """CB: Calculates accuracy for each ground truth component label."""
    if not records: print("\n[CB Per-component accuracy skipped: no records]"); return
    valid_records = [r for r in records if r.get("label") is not None and r.get("correct") is not None]
    if not valid_records: print("\n[CB Per-component accuracy skipped: no valid records]"); return
    
    component_labels = sorted(list(set(r['label'] for r in valid_records)))
    if not component_labels: print("\n[CB Per-component accuracy skipped: no component labels]"); return

    print("\n=== CB: Per‑component accuracy (ground truth component) ===")
    print(f"{'Component':<10}{'Accuracy':>10}{'N':>6}")
    for comp_label in component_labels:
        subset = [r for r in valid_records if r['label'] == comp_label]
        n_total = len(subset)
        acc = sum(r['correct'] for r in subset)/n_total if n_total > 0 else float('nan')
        print(f"{comp_label:<10}{acc:>9.2%}{n_total:>6}")

def delta_log_odds(records: List[Dict[str, Any]]) -> None:
    """CB: Δ log‑odds of correctness by ground truth component, calls analyze_presence_effects."""
    if not records: print("\n[CB Δ-log-odds by component skipped: no records]"); return
    valid_records = [r for r in records if r.get("label") is not None and r.get("correct") is not None]
    if not valid_records: print("\n[CB Δ-log-odds by component skipped: no valid records for analysis]"); return
    
    component_labels = sorted(list(set(r['label'] for r in valid_records)))
    if not component_labels: print("\n[CB Δ-log-odds by component skipped: no component labels found]"); return
    
    # This function is now a wrapper. analyze_presence_effects provides more stats.
    # The original only printed Δ log-odds and N. We use the more comprehensive version.
    analyze_presence_effects(valid_records, component_labels, 'label', "CB: Correctness by Component (Δ log-odds)")


# --- Functions for test_visual_attention.py ---

def analyze_presence_effects(
    records: List[Dict[str, Any]],
    categories_for_key: List[str], # Values the 'key_to_check' can take
    key_to_check: str,
    analysis_label: str
):
    """
    VA: Computes Δ-log-odds of accuracy when record[key_to_check] == category vs. != category.
    Math: For each category, calculates Δlog-odds = logit(P(correct|category_present)) - logit(P(correct|category_absent)).
          SE of Δlog-odds = sqrt(1/(k1*p1*(1-p1)) + 1/(k0*p0*(1-p0))), or using counts: sqrt(N1/(S1*(N1-S1)) + N0/(S0*(N0-S0))).
          Uses z-test (coeff/SE) for p-value via _norm_cdf. CI is coeff ± 1.96*SE.
    """
    if not records: print(f"\n[VA {analysis_label} ({key_to_check}) skipped: no records]"); return
    
    results = []
    for cat_value in categories_for_key:
        present_recs = [r for r in records if r.get(key_to_check) == cat_value and 'correct' in r]
        absent_recs = [r for r in records if r.get(key_to_check) != cat_value and 'correct' in r]
        n1, k1 = len(present_recs), sum(r['correct'] for r in present_recs)
        n0, k0 = len(absent_recs), sum(r['correct'] for r in absent_recs)

        coeff, se, p_val, ci_l, ci_h = [float('nan')] * 5
        if n1 > 0 and n0 > 0:
            p1, p0 = k1/n1, k0/n0
            coeff = _safe_logit(p1) - _safe_logit(p0)
            
            var_den1, var_den0 = k1 * (n1 - k1), k0 * (n0 - k0)
            if var_den1 > 0 and var_den0 > 0: # Avoid division by zero if k=0 or k=n
                variance = (n1 / var_den1) + (n0 / var_den0)
                if variance > 0:
                    se = math.sqrt(variance)
                    if se > 1e-9: # Avoid division by zero SE
                        z = coeff / se
                        p_val = 2 * (1 - _norm_cdf(abs(z)))
                        ci_l, ci_h = coeff - 1.96 * se, coeff + 1.96 * se
        results.append({'cat': cat_value, 'coeff': coeff, 'se': se, 'pval': p_val, 'ci_l': ci_l, 'ci_h': ci_h, 'n1': n1, 'n0': n0})

    results.sort(key=lambda x: (not math.isnan(x['coeff']), abs(x['coeff']) if not math.isnan(x['coeff']) else -1), reverse=True)
    
    print(f"\n=== VA: Δ-log-odds by {key_to_check} for {analysis_label} ===")
    print(f"{'Category':<12} {'Δ log-odds':>12} {'SE':>8} {'95% CI':>22} {'p-value':>10} {'N (pres)':>10} {'N (abs)':>10}")
    for r_item in results:
        ci_str = f"[{r_item['ci_l']:+.3f}, {r_item['ci_h']:+.3f}]" if not math.isnan(r_item['ci_l']) else "—"
        se_str = f"{r_item['se']:.3f}" if not math.isnan(r_item['se']) else "—"
        pval_str = f"{r_item['pval']:.4f}" if not math.isnan(r_item['pval']) else "—"
        coeff_str = f"{r_item['coeff']:+12.3f}" if not math.isnan(r_item['coeff']) else "—"
        print(f"{str(r_item['cat']):<12} {coeff_str} {se_str:>8} {ci_str:>22} {pval_str:>10} {r_item['n1']:>10} {r_item['n0']:>10}")

def cooccurrence_counts(records: list[dict], row_key: str, col_key: str, title: str):
    """VA: Prints a contingency table of counts for records[row_key] × records[col_key]."""
    if not records: print(f"\n[VA Co-occurrence ({title}) skipped: no records]"); return
    row_vals = sorted(list(set(r[row_key] for r in records if row_key in r)))
    col_vals = sorted(list(set(r[col_key] for r in records if col_key in r)))
    if not row_vals or not col_vals: print(f"\n[VA Co-occurrence ({title}) skipped: no row/col values]"); return

    matrix = defaultdict(lambda: defaultdict(int))
    for r in records:
        if row_key in r and col_key in r: matrix[r[row_key]][r[col_key]] += 1

    print(f"\n=== VA: {title} (Co-occurrence Table) ===")
    header = f"{row_key[:10]:<12}" + "".join(f"{str(c)[:7]:>8}" for c in col_vals)
    print(header + "\n" + "-" * len(header))
    for r_val in row_vals:
        print(f"{str(r_val)[:10]:<12}" + "".join(f"{matrix[r_val].get(c_val, 0):>8}" for c_val in col_vals))

def analyze_chain_presence_effects(
    records: List[Dict[str, Any]], 
    shape_types: List[str], 
    colors_list: List[str], 
    analysis_title_prefix: str
):
    """VA Chain: Δ-log-odds for start_color/start_shape presence by calling analyze_presence_effects."""
    chain_records = [r for r in records if r.get('trial') == 'chain']
    if not chain_records:
        print(f"\n[{analysis_title_prefix} (chain start features) skipped: no chain records]")
        return
    analyze_presence_effects(chain_records, colors_list, 'start_color', f"{analysis_title_prefix} (by start color)")
    analyze_presence_effects(chain_records, shape_types, 'start_shape', f"{analysis_title_prefix} (by start shape)")

def analyze_prediction_distribution(
    records: List[Dict[str, Any]],
    expected_categories: List[str],
    prediction_field: str,
    analysis_title: str
):
    """VA: Frequency & bias of categorical predictions vs. uniform prior.
    Math: Bias = logit(P_empirical(prediction)) - logit(P_uniform_prior).
          P_uniform_prior = 1 / len(expected_categories).
    """
    if not records: print(f"\n[VA {analysis_title} prediction dist skipped: no records]"); return
    if not expected_categories: print(f"\n[VA {analysis_title} prediction dist skipped: no categories]"); return

    counts = Counter(r.get(prediction_field) for r in records if prediction_field in r)
    total = len(records)
    uniform_prob = 1.0 / len(expected_categories)

    print(f"\n=== VA: {analysis_title} (Prediction Distribution, N={total}) ===")
    print(f"{'Category':<15} {'Count':>7} {'Empirical P(pred)':>18} {'Δ log‑odds (vs Unif)':>22}")
    for cat in expected_categories:
        obs_count = counts.get(cat, 0)
        emp_prob = obs_count / total if total else 0.0
        d_lo = _safe_logit(emp_prob) - _safe_logit(uniform_prob)
        print(f"{str(cat):<15} {obs_count:>7} {emp_prob:>18.3f} {d_lo:>+22.3f}")

def accuracy_by_category(
    records: List[Dict[str, Any]],
    category_values: List[str],
    ground_truth_key: str,
    analysis_title: str
):
    """VA/Generic: Per-category accuracy based on ground truth labels."""
    if not records: print(f"\n[{analysis_title} accuracy by category skipped: no records]"); return
    
    print(f"\n=== VA/Generic: {analysis_title} (Accuracy by Ground Truth '{ground_truth_key}') ===")
    print(f"{'Category':<15} {'Accuracy':>10} {'N':>6}")
    for cat_val in category_values:
        relevant = [r for r in records if r.get(ground_truth_key) == cat_val and 'correct' in r]
        n = len(relevant)
        acc = sum(r['correct'] for r in relevant)/n if n > 0 else float('nan')
        print(f"{str(cat_val):<15} {acc:>9.2%}{n:>6}")

def grid_prediction_heatmap(records: List[Dict[str, Any]], grid_size: int, analysis_title: str):
    """VA: Heatmap of model's predicted grid cell probabilities."""
    if not records: print(f"\n[VA {analysis_title} heatmap skipped: no records]"); return
    if grid_size <=0: print(f"\n[VA {analysis_title} heatmap skipped: invalid grid_size]"); return

    pred_matrix = np.zeros((grid_size, grid_size), dtype=float)
    valid_preds = 0
    for r in records:
        pred = r.get("pred")
        try:
            if isinstance(pred, str) and pred.isdigit(): pred = int(pred)
            elif not isinstance(pred, int): continue
            if 1 <= pred <= grid_size * grid_size:
                row, col = divmod(pred - 1, grid_size)
                pred_matrix[row, col] += 1
                valid_preds += 1
        except (ValueError, TypeError): continue
    
    if valid_preds > 0: pred_matrix /= valid_preds

    print(f"\n=== VA: {analysis_title} (Prediction Probability Heatmap, N_valid_preds={valid_preds}) ===")
    if valid_preds == 0: print("  No valid integer grid cell predictions found.")
    else:
        with np.printoptions(precision=3, suppress=True, linewidth=grid_size*10): print(pred_matrix)

def chain_color_difficulty(
    records: List[Dict[str, Any]], 
    colors_list: List[str], 
    analysis_title_prefix: str
):
    """VA Chain: Δ-log-odds for final color correctness. Uses analyze_presence_effects."""
    chain_records = [r for r in records if r.get("trial") == "chain"]
    if not chain_records:
        print(f"\n[{analysis_title_prefix} (chain final color) skipped: no chain records]")
        return
    analyze_presence_effects(chain_records, colors_list, key_to_check="label",
                             analysis_label=f"{analysis_title_prefix} (by final color)")

def analyze_chain_any_presence(
    records: List[Dict[str, Any]], 
    shape_types_list: List[str], 
    colors_list: List[str], 
    analysis_title_prefix: str
):
    """VA Chain: Δ-log-odds if color/shape appears anywhere in the chain."""
    chain_records = [r for r in records if r.get('trial') == 'chain']
    if not chain_records:
        print(f"\n[{analysis_title_prefix} (chain any-presence) skipped: no chain records]")
        return

    for feature_type, item_list, item_key_in_record, type_label in [
        (colors_list, 'chain_colors', "Color", "Any Color Presence"),
        (shape_types_list, 'chain_shapes', "Shape", "Any Shape Presence")
    ]:
        print(f"\n--- {analysis_title_prefix}: {type_label} in Chain ---")
        for item_val in feature_type:
            temp_recs = []
            for r_orig in chain_records:
                r_new = r_orig.copy()
                # Create a binary presence key for analyze_presence_effects
                r_new['temp_feature_is_present'] = item_val if item_val in r_new.get(item_key_in_record, []) else f"not_{item_val}"
                temp_recs.append(r_new)
            # Categories for analyze_presence_effects are the item itself and its negation
            analyze_presence_effects(temp_recs, [item_val, f"not_{item_val}"], 'temp_feature_is_present', 
                                     f"Chain Any-Presence of '{item_val}' ({type_label.split()[1]})")


def analyze_excess_popular_color(records: List[Dict[str, Any]], analysis_title: str):
    """
    VA: Mean excess probability of predicting the majority color.
    Requires: 'majority_freq', 'predicted_majority' in records.
    Math: Calculates mean of (I(predicted_majority) - majority_freq). Tests if mean is 0 using z-test.
          SE_mean = sqrt(Var(diffs) / N). CI = mean_diff ± 1.96*SE_mean.
    """
    relevant = [r for r in records if r.get('trial') in ["color_of_number", "chain"] and
                all(k in r for k in ['majority_freq', 'predicted_majority']) and
                isinstance(r['majority_freq'], (int, float)) and isinstance(r['predicted_majority'], bool)]
    if not relevant:
        print(f"\n=== VA: {analysis_title} ===\n(No relevant records for excess popular color analysis)"); return

    diffs = [(1.0 if r['predicted_majority'] else 0.0) - r['majority_freq'] for r in relevant]
    N = len(diffs)
    if N == 0: print(f"\n=== VA: {analysis_title} ===\n(N=0 after diff computation)"); return

    mean_d = sum(diffs)/N
    var_d = sum((d - mean_d)**2 for d in diffs)/(N-1) if N > 1 else 0.0
    se_mean_d = math.sqrt(var_d/N) if N > 0 and var_d >=0 else 0.0
    
    ci_l, ci_h, p_val = float('nan'), float('nan'), float('nan')
    if se_mean_d > 1e-9:
        z = mean_d / se_mean_d
        p_val = 2 * (1 - _norm_cdf(abs(z)))
        ci_l, ci_h = mean_d - 1.96*se_mean_d, mean_d + 1.96*se_mean_d
    
    print(f"\n=== VA: {analysis_title} (Excess Popular Color Prediction) ===")
    print(f"N trials: {N}\nMean excess prob (Δ): {mean_d:+.4f}\nSE of Δ: {se_mean_d:+.4f}")
    print(f"95% CI for Δ: [{ci_l:+.4f}, {ci_h:+.4f}]\np‑value (H₀: Δ=0): {p_val:.4f}")

# --- Functions for Object Re-identification (OBJREID) / Generic ---

def _fit_logit_safe(endog: np.ndarray, exog: np.ndarray) -> Optional[Any]:
    """Helper for OBJREID: Fits Logit, fallback to L1 regularization if needed."""
    if StandardScaler is None and exog.shape[1] > 1: # exog includes constant
        print("Warning (_fit_logit_safe): StandardScaler unavailable, features not scaled.", file=sys.stderr)
    
    try: # Standard Logit (Newton for robustness)
        model = sm.Logit(endog, exog).fit(method='newton', disp=False, maxiter=100)
        if model.mle_retvals.get('converged', False): return model
    except (PerfectSeparationError, np.linalg.LinAlgError): pass # Try L1
    except Exception: pass # Try L1

    try: # Fallback to L1 regularized
        return sm.Logit(endog, exog).fit_regularized(method='l1', alpha=0.01, disp=False, maxiter=200, trim_mode='auto')
    except Exception as e_reg:
        print(f"Warning: Regularized Logit fit also failed: {e_reg}", file=sys.stderr)
        return None

def analyze_transform_effects(records: List[Dict[str, Any]], analysis_label: str):
    """
    OBJREID: Logistic models for global affine transformation effects (correctness ~ transform_params).
    Math: Fits logistic regression. Predictors are scaled transformation parameters. Reports β, SE, CI, z, p, Odds Ratio.
    """
    if not records: print(f"\n[OBJREID {analysis_label} skipped: no records]"); return
    if StandardScaler is None:
        print(f"\n[OBJREID {analysis_label} skipped: StandardScaler not available.]"); return
    try: from tabulate import tabulate
    except ImportError: tabulate = None; print(f"Warning ({analysis_label}): 'tabulate' not found.", file=sys.stderr)

    feats = {
        '|θ|': np.array([abs(r.get('angle', 0.0)) for r in records]),
        '|s−1|': np.abs(np.array([r.get('scale', 1.0) for r in records]) - 1.0),
        'sign(s−1)': np.sign(np.array([r.get('scale', 1.0) for r in records]) - 1.0),
        '|dx|': np.abs(np.array([r.get('dx', 0.0) for r in records])),
        '|dy|': np.abs(np.array([r.get('dy', 0.0) for r in records])),
        'dx': np.array([r.get('dx', 0.0) for r in records]),
        'dy': np.array([r.get('dy', 0.0) for r in records])
    }
    feats['sign(s−1)'][feats['sign(s−1)'] == 0] = 1 # Avoid zero sign, default to positive effect
    y = np.array([1 if r.get('correct', False) else 0 for r in records])

    def _run_model(sel_feat_names: List[str], desc: str):
        X_raw = np.column_stack([feats[name] for name in sel_feat_names])
        X_scaled = StandardScaler().fit_transform(X_raw)
        X_const = sm.add_constant(X_scaled, prepend=True)
        
        model = _fit_logit_safe(y, X_const)
        title = f"OBJREID Affine Effects: {analysis_label} — {desc}"
        print(f"\n╔═ {title} ═╗")
        if model is None:
            print(f"║ Model failed. Accuracy: {y.mean() if len(y)>0 else float('nan'):.3f}".ljust(len(title)+3) + "║")
            print("╚" + "═"*(len(title)+4) + "╝"); return

        res = []
        param_names = ['Intercept'] + sel_feat_names
        for i, name in enumerate(param_names):
            beta, se_ = model.params[i], model.bse[i] if i < len(model.bse) else float('nan')
            ci = model.conf_int()[i] if i < model.conf_int().shape[0] else [float('nan')]*2
            p_val = getattr(model, 'pvalues', np.full_like(model.params,np.nan))[i]
            z_ = beta/se_ if se_ > 1e-9 else float('nan')
            odds = math.exp(beta) if not math.isnan(beta) else float('nan')
            res.append([name, f"{beta:+8.3f}", f"{se_:.3f}", f"[{ci[0]:+.2f},{ci[1]:+.2f}]", f"{z_:+.2f}", f"{p_val:.3g}", f"{odds:.3f}"])
        
        hdrs = ['Feature', 'β', 'Std.Err.', '95% CI', 'z', 'p-value', 'Odds Ratio']
        if tabulate: print(tabulate(res, headers=hdrs, tablefmt="github"))
        else: print("║ "+" | ".join(hdrs)+"\n║ "+"\n║ ".join([" | ".join(row) for row in res]))
        print("╚" + "═"*(len(title)+4) + "╝")

    _run_model(['|θ|', '|s−1|', '|dx|', '|dy|'], "Magnitude Model")
    _run_model(['|θ|', 'sign(s−1)', '|dx|', '|dy|'], "Scale Direction Model")
    _run_model(['|dx|', '|dy|'], "Translation Magnitude Model")
    _run_model(['dx', 'dy'], "Translation Direction Model")

def acc_ci_wald(tp: int, fp: int, tn: int, fn: int, alpha: float = 0.05) -> Tuple[float, float, float]:
    """OBJREID/Generic: Wald 1-α CI for accuracy. (accuracy, ci_low, ci_high).
    Math: Accuracy p = (TP+TN)/N. Wald CI: p ± z_{α/2} * sqrt(p(1-p)/N).
    """
    N = tp + fp + tn + fn
    if N == 0: return 0.0, 0.0, 0.0
    acc = (tp + tn) / N
    se = math.sqrt(acc * (1 - acc) / N) if N > 0 and 0 <= acc <= 1 and acc*(1-acc)>=0 else 0.0
    
    z_crit = 1.96 # Default for alpha=0.05 (95% CI)
    try:
        from scipy.stats import norm as scipy_norm
        z_crit = scipy_norm.ppf(1 - alpha / 2.0)
    except ImportError:
        if alpha != 0.05: print("Warning (acc_ci_wald): scipy missing, z for non-95% CI is approx.", file=sys.stderr) # type: ignore
    
    return acc, max(0.0, acc - z_crit * se), min(1.0, acc + z_crit * se)