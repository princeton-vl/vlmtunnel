import os
import sys
import argparse
import random
import json
import glob
import tempfile
import shutil
import math
from collections import Counter, defaultdict
from typing import Tuple, List, Dict, Optional, Any
from PIL import Image

# ── Core utilities ────────────────────────────────────────────────────────────
from utils.utils import (
    encode  as encode_pil_to_b64_str,
    decode  as decode_b64_str_to_pil,
    extract_braced,
    _extract_yes_no_answer,
    _safe_logit,
    _norm_cdf,
)

# ── Analysis helpers ──────────────────────────────────────────────────────────
from utils.analyzer import (
    analyze_presence_effects as util_analyze_presence_effects,
    analyze_length_effect,
    analyze_color_effect,
    analyze_distance_effect,
    analyze_crossings_effect,
    analyze_color_bias,
    analyze_multivariate_effect,
    analyze_color_frequency_bias,
    per_component_accuracy,
    delta_log_odds,
    analyze_presence_effects,
    cooccurrence_counts,
    analyze_chain_presence_effects,
    analyze_prediction_distribution,
    accuracy_by_category,
    grid_prediction_heatmap,
    chain_color_difficulty,
    analyze_chain_any_presence,
    analyze_excess_popular_color,
    analyze_transform_effects,
    acc_ci_wald,
)

# ── Inference client ──────────────────────────────────────────────────────────
from inference.inference_client import InferenceClient

# ── Task-specific image generators ────────────────────────────────────────────
from image_generators.image_generator_circuits import (
    generate_circuit_image,
    generate_two_shot_examples as gen_fs_circuits,
)
GEN_CIRCUITS_AVAILABLE = True

from image_generators.image_generator_vs import (
    generate_chain_image             as gen_va_chain_image,
    generate_two_shot_examples       as gen_fs_va,
    generate_chain_description_string as gen_va_chain_desc,
    SHAPE_TYPES as VA_SHAPE_TYPES_CONST,
    COLORS      as VA_COLORS_CONST,
)
GEN_VA_AVAILABLE = True

from image_generators.image_generator_objreid import (
    generate_objreid_trial_data,
    generate_objreid_two_shot_examples as gen_fs_objreid,
    CANVAS_SIZE_DEFAULT as OR_CANVAS_SIZE_DEFAULT_CONST,
    COLORS            as OR_COLORS_CONST,
    SHAPES            as OR_SHAPES_CONST,
)
GEN_OBJREID_AVAILABLE = True

BASE_OUTPUT_DIR = "./outputs" # For live run images and canonical demos for live runs
CIRCUITS_VA_DEMO_SUBDIR_NAME = "few_shot_examples"
OBJREID_DEMO_SUBDIR_NAME = "demo"

# -----------------------------------------------------------------------------
# SECTION 1: TASK HANDLER HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def _prepare_circuits_va_few_shot_turns(raw_demos: List[Tuple[Image.Image, str, str]]) -> List[Dict[str, Any]]:
    turns = []
    if not raw_demos: return turns
    for fs_img, fs_prompt, fs_ans in raw_demos:
        turns.append({"role": "user", "text": fs_prompt, "images_pil": [fs_img]})
        turns.append({"role": "assistant", "answer_text": fs_ans})
    return turns

def _prepare_objreid_few_shot_turns(raw_yes_pair: Tuple[Image.Image, Image.Image], raw_no_pair: Tuple[Image.Image, Image.Image], base_prompt: str) -> List[Dict[str, Any]]:
    turns = []
    turns.append({"role": "user", "text": base_prompt, "images_pil": [raw_yes_pair[0], raw_yes_pair[1]]})
    turns.append({"role": "assistant", "answer_text": "{{yes}}"})
    turns.append({"role": "user", "text": f"That is correct. {base_prompt}", "images_pil": [raw_no_pair[0], raw_no_pair[1]]})
    turns.append({"role": "assistant", "answer_text": "{{no}}"})
    return turns

def _load_dataset_specific_va_circ_demos(demo_path_root: str) -> Optional[List[Tuple[Image.Image, str, str]]]:
    loaded_demos = []
    if not os.path.isdir(demo_path_root): return None
    try:
        for i in range(2): # Max 2 demos for circuits
            img_path = os.path.join(demo_path_root, f"fs_{i}.png")
            json_path = os.path.join(demo_path_root, f"fs_{i}.json")
            if os.path.exists(img_path) and os.path.exists(json_path):
                img = Image.open(img_path)
                with open(json_path, 'r') as f: meta = json.load(f)
                loaded_demos.append((img, meta["prompt"], meta["answer"]))
            elif i == 0: return None # First demo must exist if any
            else: break # Second demo optional
        return loaded_demos if loaded_demos else None
    except Exception as e: print(f"  Warning: Failed to load VA/Circuit demos from {demo_path_root}. Error: {e}", file=sys.stderr); return None

def _load_dataset_specific_objreid_demos(demo_path_root: str) -> Optional[Tuple[Tuple[Image.Image, Image.Image], Tuple[Image.Image, Image.Image]]]:
    if not os.path.isdir(demo_path_root): return None
    try: # Original names: yes0.png, yes1.png, no0.png, no1.png
        yes_img1 = Image.open(os.path.join(demo_path_root, "yes0.png"))
        yes_img2 = Image.open(os.path.join(demo_path_root, "yes1.png"))
        no_img1 = Image.open(os.path.join(demo_path_root, "no0.png"))
        no_img2 = Image.open(os.path.join(demo_path_root, "no1.png"))
        return ((yes_img1, yes_img2), (no_img1, no_img2))
    except Exception as e: print(f"  Warning: Failed to load ObjReID demos from {demo_path_root}. Error: {e}", file=sys.stderr); return None

# -----------------------------------------------------------------------------
# SECTION 2: TASK HANDLER FUNCTIONS
# -----------------------------------------------------------------------------

def handle_circuits_task(args: argparse.Namespace, parsed_models: List[Tuple[str,str,str]],
                         run_settings_fs_bools: List[bool],
                         canonical_fs_demos_raw: Optional[List[Tuple[Image.Image, str, str]]]):
    print_prefix = "[Circuits Task]"
    if not GEN_CIRCUITS_AVAILABLE: print(f"{print_prefix} Generator disabled. Skipping.", file=sys.stderr); return
    
    task_live_output_dir = os.path.join(BASE_OUTPUT_DIR, "circuits_live_runs")
    gen_kwargs = {k: getattr(args, k) for k in ["min_components", "max_components", "min_ports", "max_ports", "min_wires", "max_wires", "min_cc_wires", "max_cc_wires", "wire_color_mode", "no_wire_crossing"]}

    if args.test_image:
        img, q, c, _, _ = generate_circuit_image(**gen_kwargs)
        test_img_path = os.path.join(BASE_OUTPUT_DIR, "circuits_test_images", "test_circuit_board.png")
        os.makedirs(os.path.dirname(test_img_path), exist_ok=True)
        img.save(test_img_path); print(f"{print_prefix} Saved: {test_img_path} (QueryPort={q}, CorrectComp={c})")
        return

    if args.make_dataset:
        dataset_root_dir = args.make_dataset
        os.makedirs(dataset_root_dir, exist_ok=True)
        print(f"{print_prefix} Generating dataset in {dataset_root_dir}...")
        for is_fs_setting in run_settings_fs_bools:
            setting_subdir_name = "with_fs" if is_fs_setting else "no_fs"
            final_ds_setting_path = os.path.join(dataset_root_dir, setting_subdir_name)
            os.makedirs(final_ds_setting_path, exist_ok=True)
            
            if is_fs_setting and gen_fs_circuits:
                demos_for_this_slice = gen_fs_circuits(odir=None, num_examples=2, **gen_kwargs)
                if demos_for_this_slice and len(demos_for_this_slice) == 2:
                    demo_save_path = os.path.join(final_ds_setting_path, CIRCUITS_VA_DEMO_SUBDIR_NAME)
                    os.makedirs(demo_save_path, exist_ok=True)
                    for idx, (pil_img, p_str, a_str) in enumerate(demos_for_this_slice):
                        pil_img.save(os.path.join(demo_save_path, f"fs_{idx}.png"))
                        with open(os.path.join(demo_save_path, f"fs_{idx}.json"), "w") as f: json.dump({"prompt": p_str, "answer": a_str}, f, indent=2)
                else: print(f"  Warning: Failed to generate 2 demos for Circuits dataset slice {setting_subdir_name}.", file=sys.stderr)
            
            for i in range(args.num_examples):
                img, q_p, cc_l, map_d, winfo_d = generate_circuit_image(**gen_kwargs)
                prompt_txt = (f"Which component does the wire from port {q_p} on the breadboard, which is the gray rectangle with "
                              f"numbered ports, connect to? A wire is a series of connected, same colored lines that go from the "
                              f"center of a port, represented on the screen as a white circle, to another port. Each wire only "
                              f"connects two ports, one at either end. A wire will NEVER turn at the same spot that it intersects "
                              f"another wire, and wires do not change colors. Answer with the component label in curly braces, e.g {{C3}}.")
                meta = {"query_port": q_p, "correct_comp": cc_l, "mapping": {str(k):v for k,v in map_d.items()},
                        "winfo": {str(k):v for k,v in winfo_d.items()}, "prompt": prompt_txt,
                        "few_shot": is_fs_setting}
                img.save(os.path.join(final_ds_setting_path, f"img_{i}.png"))
                with open(os.path.join(final_ds_setting_path, f"meta_{i}.json"), "w") as f: json.dump(meta, f, indent=2)
        print(f"{print_prefix} Dataset generation complete."); return

    client: Optional[InferenceClient] = None; last_model_spec = None
    for model_spec, backend, model_id_path in parsed_models:
        if model_spec != last_model_spec:
            if client: client.cleanup_local_model()
            client_canonical_fs_turns = _prepare_circuits_va_few_shot_turns(canonical_fs_demos_raw) if canonical_fs_demos_raw else []
            client = InferenceClient(backend, model_id_path, (backend=="local"), args.local_model_path, client_canonical_fs_turns)
            last_model_spec = model_spec
        
        print(f"\n{print_prefix} Model: {model_spec}")
        for fs_strategy_for_run in run_settings_fs_bools:
            print(f"  Overall Strategy: targeting {'FS-tagged' if fs_strategy_for_run else 'non-FS-tagged'} datasets / live with {'FS' if fs_strategy_for_run else 'no FS'}")
            records: List[Dict[str,Any]] = []

            if args.load_dataset:
                for ds_root in args.load_dataset:
                    setting_subdir_name = "with_fs" if fs_strategy_for_run else "no_fs"
                    dataset_setting_path = os.path.join(ds_root, setting_subdir_name)
                    if not os.path.isdir(dataset_setting_path):
                        print(f"    Warning: Dataset path not found {dataset_setting_path}", file=sys.stderr); continue
                    
                    item_specific_fs_turns = []
                    if fs_strategy_for_run:
                        loaded_demos_raw = _load_dataset_specific_va_circ_demos(os.path.join(dataset_setting_path, CIRCUITS_VA_DEMO_SUBDIR_NAME))
                        if loaded_demos_raw: item_specific_fs_turns = _prepare_circuits_va_few_shot_turns(loaded_demos_raw)
                    
                    meta_files = sorted(glob.glob(os.path.join(dataset_setting_path, "meta_*.json")))
                    num_to_run = min(args.num_examples, len(meta_files)) if args.num_examples > 0 else len(meta_files)
                    
                    for mfp_idx, mfp in enumerate(meta_files[:num_to_run]):
                        with open(mfp, 'r') as f: info = json.load(f)
                        img_p = mfp.replace("meta_", "img_").replace(".json", ".png")
                        if not os.path.exists(img_p): continue
                        pil_image = Image.open(img_p)
                        
                        use_fs_for_this_item = info.get("few_shot", False) and fs_strategy_for_run and bool(item_specific_fs_turns)
                        
                        raw_resp = client.ask(info["prompt"], [pil_image], use_fs_for_this_item, max_tokens=150,
                                              current_item_fs_turns=item_specific_fs_turns if use_fs_for_this_item else None)
                        pred = extract_braced(raw_resp).upper()
                        q_p_rec = info["query_port"]; winfo_rec = {int(k): v for k,v in info.get("winfo",{}).items()}
                        rec = {"label": info["correct_comp"].upper(), "pred": pred, "correct": (pred == info["correct_comp"].upper()),
                               "mapping": {int(k):v for k,v in info.get("mapping",{}).items()}, "winfo": winfo_rec,
                               "length": winfo_rec.get(q_p_rec,{}).get("length", float('nan')),
                               "euclid_dist": winfo_rec.get(q_p_rec,{}).get("euclid_dist", float('nan')),
                               "crossings": winfo_rec.get(q_p_rec,{}).get("crossings", float('nan')),
                               "color": winfo_rec.get(q_p_rec,{}).get("color", "unknown")}
                        records.append(rec)
                        if args.verbose:
                            print(f"      V: {os.path.basename(mfp)} GT:{rec['label']}|P:{pred}|C:{rec['correct']}", flush=True)
                            print(f"        Raw Model Output: {raw_resp}", flush=True)
            else: # Live inference
                if fs_strategy_for_run and not client.few_shot_turns and canonical_fs_demos_raw: client.few_shot_turns = _prepare_circuits_va_few_shot_turns(canonical_fs_demos_raw)
                elif not fs_strategy_for_run: client.few_shot_turns = []
                os.makedirs(task_live_output_dir, exist_ok=True)
                live_img_subdir = os.path.join(task_live_output_dir, model_spec.replace(":", "-"), f"fs_{int(fs_strategy_for_run)}")
                if args.verbose: os.makedirs(live_img_subdir, exist_ok=True)
                print(f"    Running Live Inference (using canonical FS if strategy is FS, images to {live_img_subdir if args.verbose else 'memory only'})...")
                for i in range(args.num_examples):
                    img, q_p, cc_l, map_d, winfo_d = generate_circuit_image(**gen_kwargs)
                    if args.verbose: img.save(os.path.join(live_img_subdir, f"live_img_ex{i}.png"))
                    prompt_live = (f"Which component does the wire from port {q_p} on the breadboard, which is the gray rectangle with "
                              f"numbered ports, connect to? A wire is a series of connected, same colored lines that go from the "
                              f"center of a port, represented on the screen as a white circle, to another port. Each wire only "
                              f"connects two ports, one at either end. A wire will NEVER turn at the same spot that it intersects "
                              f"another wire, and wires do not change colors. Answer with the component label in curly braces, e.g {{C3}}.")
                    raw_resp = client.ask(prompt_live, [img], fs_strategy_for_run, max_tokens=150)
                    pred_live = extract_braced(raw_resp).upper()
                    rec_live = {"label": cc_l.upper(), "pred": pred_live, "correct": (pred_live == cc_l.upper()),
                                "mapping": map_d, "winfo": winfo_d, "length": winfo_d.get(q_p,{}).get("length", float('nan')),
                                "euclid_dist": winfo_d.get(q_p,{}).get("euclid_dist", float('nan')),
                                "crossings": winfo_d.get(q_p,{}).get("crossings", float('nan')),
                                "color": winfo_d.get(q_p,{}).get("color", "unknown")}
                    records.append(rec_live)
                    if args.verbose:
                        print(f"      V Live ex{i}: GT:{rec_live['label']}|P:{pred_live}|C:{rec_live['correct']}", flush=True)
                        print(f"        Raw Model Output: {raw_resp}", flush=True)

            if records:
                n_ok = sum(r['correct'] for r in records); n_tot = len(records)
                print(f"    Overall Accuracy: {n_ok}/{n_tot} = {n_ok/n_tot:.2%}" if n_tot > 0 else "N/A")
                per_component_accuracy(records); delta_log_odds(records); analyze_length_effect(records)
                analyze_color_effect(records); analyze_distance_effect(records); analyze_crossings_effect(records)
                analyze_multivariate_effect(records); analyze_color_bias(records); analyze_color_frequency_bias(records)
    if client: client.cleanup_local_model()

def handle_visual_attention_task(args: argparse.Namespace, parsed_models: List[Tuple[str,str,str]],
                                 run_settings_fs_bools: List[bool],
                                 canonical_fs_demos_raw: Optional[List[Tuple[Image.Image, str, str]]]):
    print_prefix = "[VisualAttention Task (Chain Trial)]"
    if not GEN_VA_AVAILABLE: print(f"{print_prefix} Generator disabled. Skipping.", file=sys.stderr); return

    task_live_output_dir = os.path.join(BASE_OUTPUT_DIR, "visual_attention_live_runs")

    if args.test_image:
        img, sp, fc, _, _ = gen_va_chain_image(args.va_grid_size, args.va_cell_size, args.va_chain_length)
        test_img_path = os.path.join(BASE_OUTPUT_DIR, "visual_attention_test_images", "test_va_chain.png")
        os.makedirs(os.path.dirname(test_img_path), exist_ok=True)
        img.save(test_img_path); print(f"{print_prefix} Saved: {test_img_path} (Start: {sp}, Final Color: {fc})")
        return

    if args.make_dataset:
        dataset_root_dir = args.make_dataset
        os.makedirs(dataset_root_dir, exist_ok=True)
        print(f"{print_prefix} Generating dataset in {dataset_root_dir} (chain trial only)...")
        for is_fs_setting in run_settings_fs_bools:
            setting_subdir_name = f"chain_fs{int(is_fs_setting)}"
            final_ds_setting_path = os.path.join(dataset_root_dir, setting_subdir_name)
            os.makedirs(final_ds_setting_path, exist_ok=True)

            if is_fs_setting and gen_fs_va:
                demos_for_this_slice = gen_fs_va(args.va_grid_size, args.va_cell_size, args.va_chain_length, odir=None, num_examples=1)
                if demos_for_this_slice and len(demos_for_this_slice) == 1:
                    demo_save_path = os.path.join(final_ds_setting_path, CIRCUITS_VA_DEMO_SUBDIR_NAME)
                    os.makedirs(demo_save_path, exist_ok=True)
                    pil_img, p_str, a_str = demos_for_this_slice[0]
                    pil_img.save(os.path.join(demo_save_path, "fs_0.png"))
                    with open(os.path.join(demo_save_path, "fs_0.json"), "w") as f: json.dump({"prompt": p_str, "answer": a_str}, f, indent=2)
                else: print(f"  Warning: Failed to generate 1 demo for VA dataset slice {setting_subdir_name}.", file=sys.stderr)
            
            for i in range(args.num_examples):
                img, sp_obj, fc_obj, cp_obj, cc_obj = gen_va_chain_image(args.va_grid_size, args.va_cell_size, args.va_chain_length)
                prompt_txt = (f"Starting at the {sp_obj[1]} {sp_obj[0]}, follow the labels for {len(cp_obj)-1} steps. "
                              f"(For instance, in a different example of {len(cp_obj)-1} steps, you might start at a blue triangle, "
                              f"then go to a red square, then a blue circle. The answer would be blue.) "
                              f"After those steps, what color are you on? Answer with the color in curly braces, e.g. {{red}}.")
                meta = {"trial_type": "chain", "few_shot": is_fs_setting,
                        "start_pair": sp_obj, "final_color": fc_obj, "chain": cp_obj,
                        "color_counts": dict(cc_obj), "prompt": prompt_txt,
                        "gen_args": {"grid": args.va_grid_size, "cell": args.va_cell_size, "chain_len": args.va_chain_length}}
                img.save(os.path.join(final_ds_setting_path, f"img_{i}.png"))
                with open(os.path.join(final_ds_setting_path, f"meta_{i}.json"), "w") as f: json.dump(meta, f, indent=2)
        print(f"{print_prefix} Dataset generation complete."); return
    
    client: Optional[InferenceClient] = None; last_model_spec = None
    for model_spec, backend, model_id_path in parsed_models:
        if model_spec != last_model_spec:
            if client: client.cleanup_local_model()
            client_canonical_fs_turns = _prepare_circuits_va_few_shot_turns(canonical_fs_demos_raw) if canonical_fs_demos_raw else []
            client = InferenceClient(backend, model_id_path, (backend=="local"), args.local_model_path, client_canonical_fs_turns)
            last_model_spec = model_spec
        
        print(f"\n{print_prefix} Model: {model_spec}")
        for fs_strategy_for_run in run_settings_fs_bools:
            print(f"  Overall Strategy: targeting {'FS-tagged' if fs_strategy_for_run else 'non-FS-tagged'} datasets / live with {'FS' if fs_strategy_for_run else 'no FS'}")
            records_va: List[Dict[str,Any]] = []

            if args.load_dataset:
                for ds_root in args.load_dataset:
                    setting_subdir_name = f"chain_fs{int(fs_strategy_for_run)}"
                    dataset_setting_path = os.path.join(ds_root, setting_subdir_name)
                    if not os.path.isdir(dataset_setting_path):
                        print(f"    Warning: Dataset path not found {dataset_setting_path}", file=sys.stderr); continue
                    
                    item_specific_fs_turns = []
                    if fs_strategy_for_run:
                        loaded_demos_raw = _load_dataset_specific_va_circ_demos(os.path.join(dataset_setting_path, CIRCUITS_VA_DEMO_SUBDIR_NAME))
                        if loaded_demos_raw: item_specific_fs_turns = _prepare_circuits_va_few_shot_turns(loaded_demos_raw)

                    meta_files = sorted(glob.glob(os.path.join(dataset_setting_path, "meta_*.json")))
                    num_to_run = min(args.num_examples, len(meta_files)) if args.num_examples > 0 else len(meta_files)

                    for mfp_idx, mfp in enumerate(meta_files[:num_to_run]):
                        with open(mfp, 'r') as f: info = json.load(f)
                        img_idx_str = os.path.basename(mfp).replace("meta_","").replace(".json","")
                        img_p = os.path.join(dataset_setting_path, f"img_{img_idx_str}.png") # img_0.png, img_1.png etc.
                        if not os.path.exists(img_p):
                             img_p = os.path.join(dataset_setting_path, f"img_chain_{img_idx_str}.png") # Fallback for older naming
                             if not os.path.exists(img_p): continue
                        
                        pil_image = Image.open(img_p)
                        use_fs_for_this_item = info.get("few_shot", False) and fs_strategy_for_run and bool(item_specific_fs_turns)
                        
                        raw_resp = client.ask(info["prompt"], [pil_image], use_fs_for_this_item, max_tokens=100,
                                              current_item_fs_turns=item_specific_fs_turns if use_fs_for_this_item else None)
                        pred = extract_braced(raw_resp).lower()
                        rec = {"trial": "chain", "label": info["final_color"].lower(), "pred": pred, "correct": (pred == info["final_color"].lower()),
                               "start_color": info["start_pair"][1].lower(), "start_shape": info["start_pair"][0].lower(),
                               "chain_colors": [c.lower() for _,c in info.get("chain",[])],
                               "chain_shapes": [s.lower() for s,_ in info.get("chain",[])],
                               "color_counts": info.get("color_counts",{}), "source_file": mfp}
                        if rec["color_counts"] and sum(rec["color_counts"].values()) > 0:
                            rec["majority_freq"] = max(rec["color_counts"].values()) / sum(rec["color_counts"].values())
                            rec["predicted_majority"] = (pred == max(rec["color_counts"], key=rec["color_counts"].get).lower() if rec["color_counts"] else False)
                        records_va.append(rec)
                        if args.verbose:
                            print(f"      V: {os.path.basename(mfp)} GT:{rec['label']}|P:{pred}|C:{rec['correct']}", flush=True)
                            print(f"        Raw Model Output: {raw_resp}", flush=True)
            else: # Live VA
                if fs_strategy_for_run and not client.few_shot_turns and canonical_fs_demos_raw: client.few_shot_turns = _prepare_circuits_va_few_shot_turns(canonical_fs_demos_raw)
                elif not fs_strategy_for_run: client.few_shot_turns = []
                os.makedirs(task_live_output_dir, exist_ok=True)
                live_img_subdir = os.path.join(task_live_output_dir, model_spec.replace(":", "-"), f"fs_{int(fs_strategy_for_run)}")
                if args.verbose: os.makedirs(live_img_subdir, exist_ok=True)
                print(f"    Running Live Inference (using canonical FS if strategy is FS, images to {live_img_subdir if args.verbose else 'memory only'})...")
                for i in range(args.num_examples):
                    img, sp_obj, fc_obj, cp_obj, cc_obj = gen_va_chain_image(args.va_grid_size, args.va_cell_size, args.va_chain_length)
                    if args.verbose: img.save(os.path.join(live_img_subdir, f"live_va_img_ex{i}.png"))
                    prompt_live = (f"Starting at the {sp_obj[1]} {sp_obj[0]}, follow the labels for {len(cp_obj)-1} steps. "
                                   f"(For instance, in a different example of {len(cp_obj)-1} steps, you might start at a blue triangle, "
                                   f"then go to a red square, then a blue circle. The answer would be blue.) "
                                   f"After those steps, what color are you on? Answer with the color in curly braces, e.g. {{red}}.")
                    raw_resp = client.ask(prompt_live, [img], fs_strategy_for_run, max_tokens=100)
                    pred_live = extract_braced(raw_resp).lower()
                    rec_live = {"trial": "chain", "label": fc_obj.lower(), "pred": pred_live, "correct": (pred_live == fc_obj.lower()),
                                "start_color": sp_obj[1].lower(), "start_shape": sp_obj[0].lower(),
                                "chain_colors": [c.lower() for _,c in cp_obj], "chain_shapes": [s.lower() for s,_ in cp_obj],
                                "color_counts": dict(cc_obj) }
                    if rec_live["color_counts"] and sum(rec_live["color_counts"].values()) > 0:
                        rec_live["majority_freq"] = max(rec_live["color_counts"].values()) / sum(rec_live["color_counts"].values())
                        rec_live["predicted_majority"] = (pred_live == max(rec_live["color_counts"], key=rec_live["color_counts"].get).lower() if rec_live["color_counts"] else False)
                    records_va.append(rec_live)
                    if args.verbose:
                        print(f"      V Live ex{i}: GT:{rec_live['label']}|P:{pred_live}|C:{rec_live['correct']}", flush=True)
                        print(f"        Raw Model Output: {raw_resp}", flush=True)

            if records_va:
                n_ok_va = sum(r['correct'] for r in records_va); n_tot_va = len(records_va)
                print(f"    Overall Accuracy (Chain Trial): {n_ok_va}/{n_tot_va} = {n_ok_va/n_tot_va:.2%}" if n_tot_va > 0 else "N/A")
                title_va = f"VA Chain - Model {model_spec}, FS_Strategy={fs_strategy_for_run}"
                analyze_chain_presence_effects(records_va, VA_SHAPE_TYPES_CONST, VA_COLORS_CONST, title_va)
                chain_color_difficulty(records_va, VA_COLORS_CONST, title_va)
                analyze_chain_any_presence(records_va, VA_SHAPE_TYPES_CONST, VA_COLORS_CONST, title_va)
                analyze_prediction_distribution(records_va, VA_COLORS_CONST, "pred", f"{title_va} Final Color Pred Dist")
                accuracy_by_category(records_va, VA_COLORS_CONST, "label", f"{title_va} Acc by Final GT Color")
                analyze_excess_popular_color(records_va, title_va)
    if client: client.cleanup_local_model()


def handle_objreid_task(args: argparse.Namespace, parsed_models: List[Tuple[str,str,str]],
                        run_settings_objreid: List[Tuple[bool,bool]], 
                        canonical_fs_demos_raw: Optional[Tuple[Tuple[Image.Image, Image.Image], Tuple[Image.Image, Image.Image]]]):
    print_prefix = "[ObjReID Task (Trials 1,3,9)]"
    if not GEN_OBJREID_AVAILABLE: print(f"{print_prefix} Generator disabled. Skipping.", file=sys.stderr); return

    task_live_output_dir = os.path.join(BASE_OUTPUT_DIR, "objreid_live_runs")
    user_example_objreid_prompt = ("The first image shows an object made of connected geometric shapes, which together form an object. "
                                   "Does this SAME object appear in the second image? For example, if a component shape were to be "
                                   "rotated or translated separately from the entire composite-object, it would be a different object. "
                                   "Respond with {yes} or {no} (inside the curly brackets). There may be extra shapes in Image 2 "
                                   "that are not part of the original object; as long as the object from Image 1 is present, "
                                   "the answer is yes even if there are other shapes present.")

    if args.test_image:
        for trial_id_test in args.objreid_trials:
            c1,c2,na = (False,False,False); trial_name_suffix = ""
            if trial_id_test == 3: na = True; trial_name_suffix = "_pixel_perfect"
            elif trial_id_test == 9: c1,c2 = True,True; trial_name_suffix = "_standard_connected"
            else: trial_name_suffix = "_unconnected"
            
            test_img_path_root = os.path.join(BASE_OUTPUT_DIR, "objreid_test_images", f"T{trial_id_test}{trial_name_suffix}")
            os.makedirs(test_img_path_root, exist_ok=True)
            generate_objreid_trial_data(args.objreid_canvas_size, c1, c2, na, True, 
                                        not args.objreid_no_distractors, args.objreid_allow_distractor_overlap, 
                                        test_img_path_root, "")
            print(f"{print_prefix} Saved test images for T{trial_id_test} in {test_img_path_root}")
        return

    if args.make_dataset:
        dataset_root_dir = args.make_dataset
        os.makedirs(dataset_root_dir, exist_ok=True)
        print(f"{print_prefix} Generating dataset in {dataset_root_dir}...")
        for trial_id in args.objreid_trials:
            trial_path = os.path.join(dataset_root_dir, f"T{trial_id}")
            os.makedirs(trial_path, exist_ok=True)
            for is_fs_setting, has_distractors_setting in run_settings_objreid:
                setting_subdir_name = f"fs{int(is_fs_setting)}_nd{int(not has_distractors_setting)}"
                final_ds_setting_path = os.path.join(trial_path, setting_subdir_name)
                os.makedirs(final_ds_setting_path, exist_ok=True)
                
                c1_gen, c2_gen, no_aff_gen = (False,False,False)
                if trial_id == 3: no_aff_gen = True
                elif trial_id == 9: c1_gen, c2_gen = True, True

                if is_fs_setting and gen_fs_objreid:
                    demos_for_this_slice = gen_fs_objreid(args.objreid_canvas_size, None,
                                                          False, False, False,
                                                          add_distractors=has_distractors_setting,
                                                          allow_distractor_overlap=args.objreid_allow_distractor_overlap)
                    if demos_for_this_slice:
                        demo_save_path = os.path.join(final_ds_setting_path, OBJREID_DEMO_SUBDIR_NAME)
                        os.makedirs(demo_save_path, exist_ok=True)
                        demos_for_this_slice[0][0].save(os.path.join(demo_save_path, "yes0.png"))
                        demos_for_this_slice[0][1].save(os.path.join(demo_save_path, "yes1.png"))
                        demos_for_this_slice[1][0].save(os.path.join(demo_save_path, "no0.png"))
                        demos_for_this_slice[1][1].save(os.path.join(demo_save_path, "no1.png"))
                    else: print(f"  Warning: Failed to generate demos for ObjReID slice {setting_subdir_name}.", file=sys.stderr)

                for i in range(args.num_examples):
                    ex_dir = os.path.join(final_ds_setting_path, f"example_{i}")
                    truth_val = (random.random() < 0.5)
                    # Assuming generate_objreid_trial_data saves img1.png, img2.png in ex_dir if prefix is ""
                    _, _, shapes_data, affine_params, jitter_transforms, _, jitter_attrs = \
                        generate_objreid_trial_data(args.objreid_canvas_size, c1_gen, c2_gen, no_aff_gen, truth_val,
                                                    has_distractors_setting, args.objreid_allow_distractor_overlap, 
                                                    ex_dir, "") 
                    
                    current_prompt_text = f"ObjReID T{trial_id}. {user_example_objreid_prompt}"
                    if has_distractors_setting: current_prompt_text += " Please ignore any distractor objects."

                    meta = {"trial": trial_id, "few_shot": is_fs_setting,
                            "enable_distractors": has_distractors_setting,
                            "allow_distractor_overlap": args.objreid_allow_distractor_overlap,
                            "truth": "yes" if truth_val else "no",
                            "global_affine": affine_params, "transforms": jitter_transforms, "jit_attrs": jitter_attrs,
                            "shapes": shapes_data,
                            "concat": False, "prompt": current_prompt_text}
                    with open(os.path.join(ex_dir, "meta.json"), "w") as f: json.dump(meta, f, indent=2)
        print(f"{print_prefix} Dataset generation complete."); return

    client: Optional[InferenceClient] = None; last_model_spec = None
    model_records_all_settings_agg: List[Dict[str,Any]] = []

    for model_spec, backend, model_id_path in parsed_models:
        model_records_all_settings_agg.clear()
        if client and model_spec != last_model_spec: client.cleanup_local_model(); client = None
        last_model_spec = model_spec
        
        print(f"\n{print_prefix} Model: {model_spec}")
        
        for fs_strategy_for_run, img_dist_setting_for_run in run_settings_objreid:
            current_setting_desc = f"FS_Strategy={fs_strategy_for_run}, Img_Dist_Setting={img_dist_setting_for_run}, Img_Ovlp={args.objreid_allow_distractor_overlap}"
            print(f"  Setting: {current_setting_desc}")

            client_canonical_fs_turns = []
            if fs_strategy_for_run and canonical_fs_demos_raw:
                 client_canonical_fs_turns = _prepare_objreid_few_shot_turns(canonical_fs_demos_raw[0], canonical_fs_demos_raw[1], user_example_objreid_prompt)
            
            if client is None: client = InferenceClient(backend, model_id_path, (backend=="local"), args.local_model_path, client_canonical_fs_turns if fs_strategy_for_run else [])
            else: client.few_shot_turns = client_canonical_fs_turns if fs_strategy_for_run else []
            
            records_reid: List[Dict[str,Any]] = []

            if args.load_dataset:
                for ds_root in args.load_dataset:
                    for trial_id_load in args.objreid_trials:
                        setting_subdir_name = f"fs{int(fs_strategy_for_run)}_nd{int(not img_dist_setting_for_run)}"
                        dataset_trial_setting_path = os.path.join(ds_root, f"T{trial_id_load}", setting_subdir_name)
                        if not os.path.isdir(dataset_trial_setting_path):
                            print(f"    Warning: Dataset path not found {dataset_trial_setting_path}", file=sys.stderr); continue
                        
                        item_specific_fs_turns = []
                        if fs_strategy_for_run:
                            loaded_demos_raw = _load_dataset_specific_objreid_demos(os.path.join(dataset_trial_setting_path, OBJREID_DEMO_SUBDIR_NAME))
                            if loaded_demos_raw: item_specific_fs_turns = _prepare_objreid_few_shot_turns(loaded_demos_raw[0], loaded_demos_raw[1], user_example_objreid_prompt)
                        
                        ex_dirs = sorted([d for d in glob.glob(os.path.join(dataset_trial_setting_path, "example_*")) if os.path.isdir(d)])
                        num_to_run = min(args.num_examples, len(ex_dirs)) if args.num_examples > 0 else len(ex_dirs)

                        for mfp_idx, mfp_reid_dir in enumerate(ex_dirs[:num_to_run]):
                            meta_p = os.path.join(mfp_reid_dir, "meta.json")
                            img1_p_reid = os.path.join(mfp_reid_dir, "img1.png")
                            img2_p_reid = os.path.join(mfp_reid_dir, "img2.png")
                            if not(os.path.exists(meta_p) and os.path.exists(img1_p_reid) and os.path.exists(img2_p_reid)): continue
                            
                            with open(meta_p,'r') as f: info_reid = json.load(f)
                            pil1, pil2 = Image.open(img1_p_reid), Image.open(img2_p_reid)
                            
                            item_expects_fs_from_meta = info_reid.get("few_shot", False)
                            use_fs_for_this_item = item_expects_fs_from_meta and fs_strategy_for_run and bool(item_specific_fs_turns)
                            
                            prompt_for_call = info_reid.get("prompt", user_example_objreid_prompt)
                            if use_fs_for_this_item and "yes, that is correct" not in prompt_for_call.lower():
                                prompt_for_call = f"Yes, that is correct. {prompt_for_call}"
                            
                            raw_resp = client.ask(prompt_for_call, [pil1, pil2], use_fs_for_this_item, max_tokens=30,
                                                  current_item_fs_turns=item_specific_fs_turns if use_fs_for_this_item else None)
                            pred_reid = _extract_yes_no_answer(raw_resp)
                            truth_reid = info_reid.get("truth","error").lower()
                            
                            aff_p_reid = info_reid.get("global_affine", [0,1,0,0,0,0])
                            rec_reid = {"trial": trial_id_load, "correct": (pred_reid == truth_reid), "pred": pred_reid, "label": truth_reid,
                                        "angle": aff_p_reid[0], "scale": aff_p_reid[1], 
                                        "dx": aff_p_reid[2] if len(aff_p_reid) > 2 else 0, 
                                        "dy": aff_p_reid[3] if len(aff_p_reid) > 3 else 0,
                                        "transforms": info_reid.get("transforms", []), 
                                        "jit_attrs": info_reid.get("jit_attrs", []),
                                        "raw": raw_resp, "source_file": meta_p}
                            records_reid.append(rec_reid)
                            if args.verbose:
                                print(f"      V: {os.path.basename(mfp_reid_dir)} T:{truth_reid}|P:{pred_reid}|C:{rec_reid['correct']}", flush=True)
                                print(f"        Raw Model Output: {raw_resp}", flush=True)
            else: # Live ObjReID
                temp_img_base_dir = tempfile.mkdtemp(prefix="objreid_live_")
                print(f"    Running Live Inference (using canonical FS if strategy is FS, temp images in {temp_img_base_dir})")
                for trial_id_live in args.objreid_trials:
                    c1_lr, c2_lr, na_lr = (False,False,False)
                    if trial_id_live == 3: na_lr=True
                    elif trial_id_live == 9: c1_lr,c2_lr=True,True
                    
                    final_live_prompt = f"ObjReID T{trial_id_live}. {user_example_objreid_prompt}"
                    if img_dist_setting_for_run: final_live_prompt += " Please ignore any distractor objects."
                    if fs_strategy_for_run: final_live_prompt = f"Yes, that is correct. {final_live_prompt}"

                    for i_lr in range(args.num_examples):
                        live_ex_img_dir = os.path.join(temp_img_base_dir, f"T{trial_id_live}_ex{i_lr}") if args.verbose else None
                        if live_ex_img_dir: os.makedirs(live_ex_img_dir, exist_ok=True)
                        truth_lr_bool = (random.random() < 0.5)
                        img1lr,img2lr,shapeslr,afflr,jit_tlr,n_jlr,jit_alr = generate_objreid_trial_data(
                            args.objreid_canvas_size, c1_lr,c2_lr,na_lr, truth_lr_bool,
                            img_dist_setting_for_run, args.objreid_allow_distractor_overlap, live_ex_img_dir, ""
                        )
                        raw_resp = client.ask(final_live_prompt, [img1lr, img2lr], fs_strategy_for_run, max_tokens=30)
                        pred_olr = _extract_yes_no_answer(raw_resp)
                        rec_lr = {"trial": trial_id_live, "correct": (pred_olr == ("yes" if truth_lr_bool else "no")),
                                  "pred": pred_olr, "label": ("yes" if truth_lr_bool else "no"),
                                  "angle":afflr[0],"scale":afflr[1],"dx":afflr[2],"dy":afflr[3],
                                  "transforms":jit_tlr,"jit_attrs":jit_alr,"raw":raw_resp, "id":f"live_T{trial_id_live}_ex{i_lr}"}
                        records_reid.append(rec_lr)
                        if args.verbose:
                            print(f"        V Live T{trial_id_live} ex{i_lr}: T:{rec_lr['label']}|P:{pred_olr}|C:{rec_lr['correct']}", flush=True)
                            print(f"          Raw Model Output: {raw_resp}", flush=True)
                try: shutil.rmtree(temp_img_base_dir)
                except OSError as e: print(f"  Warning: could not remove temp dir {temp_img_base_dir}: {e}", file=sys.stderr)
            
            if records_reid:
                tp_r,fp_r,tn_r,fn_r =sum(r['pred']=='yes'and r['label']=='yes'for r in records_reid), sum(r['pred']=='yes'and r['label']=='no'for r in records_reid),sum(r['pred']=='no'and r['label']=='no'for r in records_reid), sum(r['pred']=='no'and r['label']=='yes'for r in records_reid)
                if (tp_r + fp_r + tn_r + fn_r) > 0 :
                    acc_r, cil_r, cih_r = acc_ci_wald(tp_r,fp_r,tn_r,fn_r)
                    print(f"    Overall Acc for Setting ({current_setting_desc}): {acc_r:.2%} (CI {cil_r:.2%}-{cih_r:.2%}), N={len(records_reid)}")
                else: print(f"    Overall Acc for Setting ({current_setting_desc}): N/A (0 records)")
                analyze_transform_effects(records_reid, f"ObjReID {model_spec} Setting ({current_setting_desc})")
                for color_cat in OR_COLORS_CONST:
                    temp_key_color = f"jit_color_{color_cat}_present"
                    for r_item in records_reid: r_item[temp_key_color] = any(attr.get('color') == color_cat for attr in r_item.get('jit_attrs', []))
                    util_analyze_presence_effects(records_reid, [True, False], temp_key_color, f"ObjReID {model_spec} Jittered Color '{color_cat}' Present ({current_setting_desc})")
                for shape_cat in OR_SHAPES_CONST:
                    temp_key_shape = f"jit_shape_{shape_cat}_present"
                    for r_item in records_reid: r_item[temp_key_shape] = any(attr.get('type') == shape_cat for attr in r_item.get('jit_attrs', []))
                    util_analyze_presence_effects(records_reid, [True, False], temp_key_shape, f"ObjReID {model_spec} Jittered Shape '{shape_cat}' Present ({current_setting_desc})")
                for transform_cat in ["translate","rotate","scale","reshape"]:
                    temp_key_transform = f"jit_transform_{transform_cat}_present"
                    for r_item in records_reid: r_item[temp_key_transform] = (transform_cat in r_item.get('transforms',[]))
                    util_analyze_presence_effects(records_reid, [True,False], temp_key_transform, f"ObjReID {model_spec} Jitter Type '{transform_cat}' Applied ({current_setting_desc})")
                model_records_all_settings_agg.extend(records_reid)
        
        if model_records_all_settings_agg:
            tp_mo,fp_mo,tn_mo,fn_mo =sum(r['pred']=='yes'and r['label']=='yes'for r in model_records_all_settings_agg), sum(r['pred']=='yes'and r['label']=='no'for r in model_records_all_settings_agg),sum(r['pred']=='no'and r['label']=='no'for r in model_records_all_settings_agg), sum(r['pred']=='no'and r['label']=='yes'for r in model_records_all_settings_agg)
            if (tp_mo + fp_mo + tn_mo + fn_mo) > 0:
                acc_mo,cil_mo,cih_mo = acc_ci_wald(tp_mo,fp_mo,tn_mo,fn_mo)
                print(f"\n  GRAND TOTAL for Model {model_spec} (ObjReID): Acc {acc_mo:.2%} (CI {cil_mo:.2%}-{cih_mo:.2%}), N={len(model_records_all_settings_agg)}")
            else: print(f"\n  GRAND TOTAL for Model {model_spec} (ObjReID): N/A (0 records)")
    if client: client.cleanup_local_model()


def run_main():
    parser = argparse.ArgumentParser(description="Unified VQA and Object Re-ID Test Suite.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["openai:gpt-5-2025-08-07"], help="Model specs (backend:model_id_or_path).")
    parser.add_argument("--num-examples", type=int, default=10, help="Number of examples per trial/setting.")
    parser.add_argument("--few-shot", action="store_true", help="General strategy: use few-shot for live inference and target FS-tagged datasets.")
    parser.add_argument("--all-settings", action="store_true", help="Run all FS/non-FS and other applicable setting combinations.")
    parser.add_argument("--verbose", action="store_true", help="Verbose output.")
    parser.add_argument("--local-model-path", type=str, default=None, help="Path to local HuggingFace model files.")
    parser.add_argument("--make-dataset", type=str, help="Exact root directory name to save generated datasets (e.g., './my-circuits-data').")
    parser.add_argument("--load-dataset", nargs="+", type=str, help="Exact root directories of saved datasets to load.")
    parser.add_argument("--test-image", action="store_true", help="Generate test images and exit.")

    subparsers = parser.add_subparsers(title="tasks", dest="task_name", required=True)
    parser_circuits = subparsers.add_parser("circuits", help="Circuit board tracing task.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_circuits.add_argument("--min-components", type=int, default=5); parser_circuits.add_argument("--max-components", type=int, default=10)
    parser_circuits.add_argument("--min-ports", type=int, default=1); parser_circuits.add_argument("--max-ports", type=int, default=3)
    parser_circuits.add_argument("--min-wires", type=int, default=7); parser_circuits.add_argument("--max-wires", type=int, default=15)
    parser_circuits.add_argument("--min-cc-wires", type=int, default=0); parser_circuits.add_argument("--max-cc-wires", type=int, default=6)
    parser_circuits.add_argument("--wire-color-mode", type=str, default="default", choices=["default", "single", "unique"])
    parser_circuits.add_argument("--no-wire-crossing", action="store_true")
    parser_circuits.set_defaults(func_to_call=handle_circuits_task)

    parser_va = subparsers.add_parser("visual_attention", help="Visual attention tasks (chain trial only).", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_va.add_argument("--va-grid-size", type=int, default=3); parser_va.add_argument("--va-cell-size", type=int, default=100)
    parser_va.add_argument("--va-chain-length", type=int, default=2)
    parser_va.set_defaults(func_to_call=handle_visual_attention_task)

    parser_objreid = subparsers.add_parser("objreid", help="Object re-identification task (Trials 1,3,9).", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_objreid.add_argument("--objreid-canvas-size", type=int, default=OR_CANVAS_SIZE_DEFAULT_CONST)
    parser_objreid.add_argument("--objreid-trials", nargs="+", type=int, choices=[1,3,9], default=[1,3,9], help="Specific ObjReID trials.")
    parser_objreid.add_argument("--objreid-no-distractors", action="store_true", help="CONTROL FOR IMAGE GENERATION: Disable distractors.")
    parser_objreid.add_argument("--objreid-allow-distractor-overlap", action="store_true", help="CONTROL FOR IMAGE GENERATION: Allow distractors to overlap.")
    parser_objreid.set_defaults(func_to_call=handle_objreid_task)
    
    args = parser.parse_args()
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    raw_canonical_demos_for_task: Any = None
    if args.few_shot or args.all_settings:
        demo_live_base_dir = os.path.join(BASE_OUTPUT_DIR, "canonical_demos_for_live_runs")
        os.makedirs(demo_live_base_dir, exist_ok=True)
        if args.task_name == "circuits" and GEN_CIRCUITS_AVAILABLE and gen_fs_circuits:
            circuits_demo_kwargs = {k: getattr(args, k) for k in ["min_components", "max_components", "min_ports", "max_ports","min_wires", "max_wires", "min_cc_wires", "max_cc_wires","wire_color_mode", "no_wire_crossing"]}
            raw_canonical_demos_for_task = gen_fs_circuits(odir=os.path.join(demo_live_base_dir, "circuits"), num_examples=2, **circuits_demo_kwargs)
        elif args.task_name == "visual_attention" and GEN_VA_AVAILABLE and gen_fs_va:
            raw_canonical_demos_for_task = gen_fs_va(args.va_grid_size, args.va_cell_size, args.va_chain_length, odir=os.path.join(demo_live_base_dir, "visual_attention_chain"))
        elif args.task_name == "objreid" and GEN_OBJREID_AVAILABLE and gen_fs_objreid:
            objreid_demo_c1, objreid_demo_c2, objreid_demo_na = False, False, False
            raw_canonical_demos_for_task = gen_fs_objreid(
                args.objreid_canvas_size, os.path.join(demo_live_base_dir, "objreid"),
                trial_conn1=objreid_demo_c1, trial_conn2=objreid_demo_c2, trial_no_affine=objreid_demo_na,
                add_distractors=not args.objreid_no_distractors,
                allow_distractor_overlap=args.objreid_allow_distractor_overlap
            )
    
    parsed_models_list: List[Tuple[str, str, str]] = []
    for m_spec in args.models:
        try: backend, model_id_path = m_spec.split(":", 1)
        except ValueError: print(f"Warning: Model spec '{m_spec}' malformed. Skipping.", file=sys.stderr); continue
        parsed_models_list.append((m_spec, backend.lower(), model_id_path))
    if not parsed_models_list and not args.make_dataset and not args.test_image:
        print("No valid models. Exiting.", file=sys.stderr); sys.exit(1)

    task_specific_run_settings: Any
    if args.task_name == "objreid":
        task_specific_run_settings = []
        possible_fs_settings = [False, True] if args.all_settings else [args.few_shot]
        possible_dist_settings = [True, False] if args.all_settings else [not args.objreid_no_distractors] # True if distractors ON
        for fs_s in possible_fs_settings:
            for dist_s in possible_dist_settings: task_specific_run_settings.append((fs_s, dist_s))
        if not task_specific_run_settings: task_specific_run_settings.append((args.few_shot, not args.objreid_no_distractors))
    else: # Circuits, VA
        task_specific_run_settings = [False, True] if args.all_settings else [args.few_shot]
            
    if hasattr(args, 'func_to_call'):
        args.func_to_call(args, parsed_models_list, task_specific_run_settings, raw_canonical_demos_for_task)
    else:
        parser.print_help()

if __name__ == "__main__":
    run_main()