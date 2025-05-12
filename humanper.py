#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
human_evaluate.py

Loads a dataset produced by permute-connected.py in make-dataset mode,
then presents each example to a human: outputs a composite image file path,
prompts for a yes/no answer, and scores the human at the end of each trial.
Designed for headless SSH environments—no GUI required.
"""
import argparse
import json
import os
import tempfile
from collections import defaultdict
from PIL import Image


def load_examples(dataset_dir):
    """
    Walk through dataset_dir, collect examples grouped by trial number.
    Each example directory must contain img1.png, optional img2.png, and meta.json.
    """
    trials = defaultdict(list)
    for root, dirs, files in os.walk(dataset_dir):
        if 'meta.json' in files and 'img1.png' in files:
            meta_path = os.path.join(root, 'meta.json')
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            trial = meta.get('trial')
            img1_path = os.path.join(root, 'img1.png')
            img2_path = os.path.join(root, 'img2.png') if 'img2.png' in files else None
            truth = meta.get('truth')
            trials[trial].append({
                'dir': root,
                'img1': img1_path,
                'img2': img2_path,
                'truth': truth
            })
    return trials


def save_composite(img1_path, img2_path=None):
    """
    Create a composite image side by side and save to a temp file.
    Returns the path to the saved composite image.
    """
    img1 = Image.open(img1_path)
    if img2_path:
        img2 = Image.open(img2_path)
        w1, h1 = img1.size
        w2, h2 = img2.size
        h = max(h1, h2)
        composite = Image.new('RGB', (w1 + w2, h), 'white')
        composite.paste(img1, (0, 0))
        composite.paste(img2, (w1, 0))
    else:
        composite = img1
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    composite.save(tmp.name)
    return tmp.name


def prompt_yes_no():
    """
    Prompt the user for a yes/no answer until valid input is given.
    Returns 'yes' or 'no'.
    """
    while True:
        ans = input("Answer (y/n): ").strip().lower()
        if ans in ('y', 'yes'):
            return 'yes'
        if ans in ('n', 'no'):
            return 'no'
        print("Please enter 'y' or 'n'.")


def evaluate_human(trials):
    """
    Iterate through each trial, present examples by file path, collect answers, and score.
    """
    for trial in sorted(trials.keys()):
        examples = trials[trial]
        print(f"\n=== Trial {trial}: {len(examples)} examples ===")
        correct = 0
        for idx, ex in enumerate(examples, 1):
            print(f"Example {idx}/{len(examples)}: {ex['dir']}")
            comp_path = save_composite(ex['img1'], ex['img2'])
            print(f"Open this file to view: {comp_path}")
            human_ans = prompt_yes_no()
            if human_ans == ex['truth']:
                print("Correct!\n")
                correct += 1
            else:
                print(f"Incorrect, truth is '{ex['truth']}'.\n")
        percent = correct / len(examples) * 100
        print(f"Trial {trial} Score: {correct}/{len(examples)} correct ({percent:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate human performance on primitivision dataset"
    )
    parser.add_argument(
        "dataset_dir", metavar="DATASET_DIR",
        help="Path to the dataset directory produced by permute-connected.py --make-dataset"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_dir):
        parser.error(f"Dataset directory not found: {args.dataset_dir}")

    trials = load_examples(args.dataset_dir)
    if not trials:
        print("No examples found in dataset directory.")
        return

    evaluate_human(trials)


if __name__ == '__main__':
    main()
