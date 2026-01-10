# Evaluation Guide

This script generates **one image per prompt** from a CSV file using a diffusion model and saves the results to disk.

---

## Input Data Format

The script expects a CSV file with the following columns.

### Required Columns
- `prompt` — Text prompt used for image generation

### Optional Columns
- `case_number` — Output image filename (defaults to row index)
- `evaluation_seed` or `sd_seed` — Random seed (default: `42`)
- `evaluation_guidance` — CFG scale (default: `7.5`)

### Example CSV
```csv
case_number,prompt,evaluation_seed,evaluation_guidance
0,"A cat sitting on a sofa",1234,7.5
1,"A futuristic city at sunset",5678,8.0
```

---

## Output Structure
```
<save_dir>/all/<category>/<case_number>.png
```

---

## Example Command
```bash
python generate_images.py   --data ./data/prompts.csv   --model_type sd   --model_id_or_path CompVis/stable-diffusion-v1-4   --save_dir ./results   --device cuda:0
```

---

# Nudity Evaluation Script

A lightweight, filesystem-friendly nudity evaluation utility built on top of **NudeNet**.  
Designed for batch image folders and experiment result directories.

---

## Supported Directory Layout

```
images/
├── 0.jpg
├── 1.png
└── ...
```

---

## CLI Usage

### Root Directory Mode
```bash
python nudity_eval.py   --mode root   --root /path/to/experiment   --threshold 0.6
```

### Image Path Mode
```bash
python nudity_eval.py   --mode path   --img_path /path/to/images   --threshold 0.6
```

---

# CSD Similarity Evaluation Scripts

This section documents **two evaluation scripts** used to compute block-wise similarity scores using a **CSD-CLIP** model.  
Each script produces a single JSON score per evaluated subdirectory.

---

## 1. Standard Image Similarity Evaluation

### Expected Directory Structure
```
--dir
├── model_A/
│   ├── 0.png
│   ├── 1.png
│   └── ...
├── model_B/
│   ├── 0.png
│   ├── 1.png
│   └── ...
```

**Notes**
- Each subfolder corresponds to a single model or experiment
- Images must be **index-aligned** with the reference embeddings

---

### CSD Checkpoint

To run this script, you need to provide a CSD model checkpoint.  
You can download it from [here](https://drive.google.com/file/d/1FX0xs8p-C7Ob-h5Y4cUhTeOepHzXv_46/view).

---

### CLI Usage
```bash
python csd_eval.py   --dir /path/to/results   --ckpt_path csd_checkpoint.pt   --refer_path reference_embeddings.pt   --device cuda:0
```

---

# UDATK Artistic Image Evaluation

This variant supports **multiple images per index**, commonly used for artistic or style-based datasets.

## Expected Directory Structure
```
--dir
├── model_A/
│   ├── sample_0/
│   │   ├── img1.png
│   │   └── img2.png
│   ├── sample_1/
│   │   ├── img1.png
│   │   └── ...
├── model_B/
│   └── ...
```

---

## CLI Usage
```bash
python csd_eval_udatk.py   --dir /path/to/results   --ckpt_path csd_checkpoint.pt   --refer_path reference_embeddings.pt   --device cuda:0
```

---

# Fine-Grained VQA Evaluation (Gecko-Style)

This script performs **yes/no VQA evaluation** per image using the OpenAI API and aggregates:
- Overall accuracy
- Concept score
- Non-concept score

---

## Requirements

- Python 3.9+
- Packages: `openai`, `pandas`, `numpy`, `tqdm`

Install dependencies:
```bash
pip install openai pandas numpy tqdm
```

---

## Authentication

Set your OpenAI API key:
```bash
export OPENAI_KEY="YOUR_KEY"
```

---

## Inputs

### 1) CSV File (`--file`)

The CSV must contain at least the following columns:

| Column     | Type   | Description                            |
|------------|--------|----------------------------------------|
| `prompt`   | string | Prompt used to generate the image      |
| `img_path` | string | Local path to the image file           |

**Example**
```csv
prompt,img_path
"a photo of a dog","/abs/path/to/0.png"
"a photo of a cat","/abs/path/to/1.png"
```

---

### 2) Questions JSON (`--questions_file`)

A JSON list (or dict indexed by row index) where each entry corresponds to a CSV row index.  
Each entry contains a list of QA objects.

Each QA object must include:
- `question` (string)
- `groundtruth` (string, e.g., `"Yes."` or `"No."`)

**Example**
```json
[
  [
    {"question": "Is there a dog in the image?", "groundtruth": "Yes."},
    {"question": "Is the dog wearing a hat?", "groundtruth": "No."}
  ],
  [
    {"question": "Is there a cat in the image?", "groundtruth": "Yes."}
  ]
]
```

**Important:**  
`questions[idx]` is accessed directly, so the JSON order **must align exactly** with the CSV row ordering.

---

## CLI Usage
```bash
python gecko_eval.py   --file /path/to/data.csv   --questions_file /path/to/questions.json
```

The output directory is automatically set to the same folder as the input CSV.
