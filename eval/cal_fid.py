from cleanfid import fid
from pathlib import Path
import json
import argparse

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
        "--real_folder",
        type=str,
        default='./coco10k'
    )
    return parser.parse_args()

args = parse_args()

dirs = [*Path(args.input_path).glob('*')]
fid_scores = {}

for path in dirs:
    if ".json" in str(path):
        continue
    model = str(path).split('/')[-1]
    score = fid.compute_fid(args.real_folder, path, mode="clean")
    fid_scores[model] = score

    print(model, score)

with open(args.input_path, 'w') as f:
    json.dump(fid_scores, f)
