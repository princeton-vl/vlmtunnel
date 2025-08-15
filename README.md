# VLMs Have Tunnel Vision: Nonlocal Visual Reasoning Evaluation Suite
This suite evaluates Vision-Language Models (VLMs) on their capacity for nonlocal visual reasoning – tasks requiring the integration of evidence from multiple image regions. It's designed to test three core visual reasoning skills: comparative perception, saccadic search, and smooth visual search.


### Check out the paper here: https://arxiv.org/abs/2507.13361
-----------------------------------------
## Overview

The suite includes three main task categories:
1.  **Object Re-Identification**: Tests comparative perception by asking if a transformed object reappears in a second image.
2.  **Visual Scavenger Hunt (Chain Trial)**: Tests saccadic search by requiring models to follow a chain of visual cues across a grid.
3.  **Circuit Connections**: Tests smooth visual search by asking models to trace wires on a circuit board.

This code allows you to verify our results, generate new datasets, run models against those datasets, and analyze the results.

-----------------------------------------

##  Getting Started

### Prerequisites

1.  **Python 3.8+**
2.  **Required Libraries**:  Pillow numpy pandas statsmodels scipy scikit-learn openai requests torch transformers tabulate
    (Ensure `torch` and `transformers` are installed if you plan to use local models).
3.  **API Keys**: If using OpenAI or OpenRouter, set your API keys as environment variables:
    * `OPENAI_API_KEY="your_openai_key"`
    * `OPENROUTER_API_KEY="your_openrouter_key"`


##  Running the Script

All functionality is exposed through **`main.py`**.

**Note:** If you're using a conda environment, make sure to activate it first:
```bash
conda activate svlm
```

### 1. Basic pattern

```bash
python main.py [COMMON_OPTIONS] <TASK_NAME> [TASK_SPECIFIC_OPTIONS]
```

*Example*

```bash
python main.py --num-examples 20 --verbose objreid
```

### 2. Common options (work for every task)

* `--models BACKEND:MODEL_ID …`  
  Evaluate one or more models.  
  Examples: `openai:gpt-4o-2024-08-06`, `openrouter:google/gemini-2.5-pro-preview`, `local:Qwen/Qwen2.5-VL-32B-Instruct`
* `--num-examples INT` – number of examples to run
* `--few-shot` – enable few-shot evaluation (uses canonical demos made during make-dataset)
* `--all-settings` – sweep over FS/non-FS and other settings (e.g. distractors)
* `--verbose` – print per-example details and raw model outputs
* `--local-model-path PATH` – path to a local model (if not given inline in `--models`)
* `--make-dataset DIR` – generate a new dataset in *DIR*
* `--load-dataset DIR [DIR …]` – load existing dataset(s) and evaluate
* `--test-image` – generate **one** sample image for the chosen task and exit

#### NOTE
Only certain HF models are supported; check inference/infclient.py.

-----------------------------------------
## Task-specific details & examples

Below are the extra flags and typical commands for each task family.

---

### 1. Object Re-Identification (`objreid`)

**Generation-only flags**

| Flag | Meaning | Default |
|------|---------|---------|
| `--objreid-canvas-size INT` | Output image resolution | 512 |
| `--objreid-trials 1 3 9 …` | Which trial types to generate/evaluate | `1 3 9` |
| `--objreid-no-distractors` | Disable distractor objects | *off* |
| `--objreid-allow-distractor-overlap` | Allow distractors to overlap the main object | *off* |


Trial 9 corresponds to the 'Standard' variant from the paper; 1 corresponds to the 'Unconnected' variant, and 3 corresponds to the 'Pixel Perfect' variant.
**Example commands**

Create a *single* test image for trial 9:

```bash
python main.py --test-image objreid --objreid-trials 9
```

Make a 50-example dataset for trial 1, **few-shot**, **no distractors**:

```bash
python main.py \
  --make-dataset ./my_objreid_T1_fs_nodist \
  --num-examples 50 \
  --few-shot \
  objreid \
  --objreid-trials 1 \
  --objreid-no-distractors
```

Evaluate GPT-4o on the *fs1_nd0* split of trial 9 in an existing dataset:

```bash
python main.py \
  --load-dataset ./objreid-data \
  --models openai:gpt-4o-2024-08-06 \
  --few-shot \
  --verbose \
  objreid \
  --objreid-trials 9
```

Live inference with a **local** model (trial 3, 5 examples):

```bash
python main.py \
  --models local:/hfpath/to/your/local_model \
  --num-examples 5 \
  --verbose \
  objreid \
  --objreid-trials 3
```

---

### 2. Visual Scavenger Hunt (`visual_attention`)

**Generation-only flags**

| Flag | Meaning | Default |
|------|---------|---------|
| `--va-grid-size INT` | Grid dimension (e.g. `3` → 3×3) | 3 |
| `--va-cell-size INT` | Cell height/width in pixels | 100 |
| `--va-chain-length INT` | Number of clue hops | 2 |

**Example commands**

One test image with a 3-step chain:

```bash
python main.py --test-image visual_attention --va-chain-length 3
```

50-example few-shot dataset:

```bash
python main.py \
  --make-dataset ./my_va_dataset_fs1 \
  --num-examples 50 \
  --few-shot \
  visual_attention
```

Evaluate Gemini Flash on the non-few-shot split:

```bash
python main.py \
  --load-dataset ./my_va_dataset \
  --models openrouter:google/gemini-2.5-pro-preview \
  --num-examples 30 \
  --verbose \
  visual_attention
```

---

### 3. Circuit Connections (`circuits`)

**Generation-only flags**

| Flag | Description |
|------|-------------|
| `--min-components / --max-components INT` | Limits on number of components |
| `--min-ports / --max-ports INT` | Ports per component |
| `--min-wires / --max-wires INT` | Total wires |
| `--min-cc-wires / --max-cc-wires INT` | Wires within a *single* component |
| `--wire-color-mode [default|single|unique]` | Wire color strategy |
| `--no-wire-crossing` | Forbid wire crossings |

**Example commands**

Single test image, unique wire colors, no crossings:

```bash
python main.py \
  --test-image \
  circuits \
  --wire-color-mode unique \
  --no-wire-crossing
```

50-example dataset, **single-color** wires:

```bash
python main.py \
  --make-dataset ./my_circuits_singlecolor \
  --num-examples 50 \
  circuits \
  --wire-color-mode single
```

Evaluate GPT-4o on a few-shot, parameter-controlled live run:

```bash
python main.py \
  --models openai:gpt-4o-2024-08-06 \
  --num-examples 5 \
  --few-shot \
  --verbose \
  circuits \
  --min-components 3 --max-components 5 \
  --min-wires 4 --max-wires 8
```
