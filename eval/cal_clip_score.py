import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse

import torch
import torch.nn.functional as F
from transformers import (
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    AutoTokenizer,
    AutoProcessor,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute CLIP scores for prompts/images in batched JSON logs."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Root directory containing subfolders of log_*.json files",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Number of examples to process per batch",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device to run inference on",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    input_path = args.input_path
    batch_size = args.batch_size
    device = args.device

    files = list(Path(input_path).rglob("**/log_*.json"))
    model_to_log = defaultdict(list)
    for f in files:
        model = str(f).split('/')[-4]
        model_to_log[model].append(str(f))
    print('models: ', list(model_to_log.keys()))
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    text_model = CLIPTextModelWithProjection.from_pretrained(
        "openai/clip-vit-large-patch14"
    ).to(device)
    vision_model = CLIPVisionModelWithProjection.from_pretrained(
        "openai/clip-vit-large-patch14"
    ).to(device)

    scores = {}
    with torch.no_grad():
        for model, imgtext_paths in tqdm(model_to_log.items(), desc="Models"):
            combined = []
            for path in imgtext_paths:
                with open(path, "r") as f:
                    combined.extend(json.load(f))

            model_scores = []
            for i in tqdm(range(0, len(combined), batch_size), leave=False, desc=model):
                batch = combined[i : i + batch_size]
                prompts = [item["prompt"] for item in batch]
                images = [Image.open(item["img_path"]) for item in batch]

                # Text
                text_inputs = tokenizer(
                    prompts, return_tensors="pt", padding="max_length", truncation=True
                ).to(device)
                text_embeds = text_model(**text_inputs).text_embeds
                text_embeds = F.normalize(text_embeds, dim=-1)

                # Images
                vision_inputs = processor(
                    images=images, return_tensors="pt"
                ).to(device)
                vision_embeds = vision_model(**vision_inputs).image_embeds
                vision_embeds = F.normalize(vision_embeds, dim=-1)

                # CLIP score = cosine similarity along the diagonal
                scores_mat = vision_embeds @ text_embeds.t()
                model_scores.extend(scores_mat.diag().cpu().tolist())

            scores[model] = float(np.mean(model_scores))

    # Save one JSON per model (or combine as needed)
    for model, score in scores.items():
        out_dir = Path(input_path) / model
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / "clip_scores.json"
        with open(save_path, "w") as f:
            json.dump({model: score}, f)

if __name__ == "__main__":
    main()
