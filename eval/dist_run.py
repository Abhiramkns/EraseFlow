import os
import time
import json
import random
import subprocess
import logging
import csv
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--file",        help="Path to the CSV file containing prompts + other fields", required=True)
    parser.add_argument("--concept", type=str, default=None)
    parser.add_argument("--gpus",        help="Comma-separated list of GPU IDs",                required=True)
    parser.add_argument("--ckpts_path", type=str, default=None)
    parser.add_argument("--save_dir",    help="Where to write split prompt CSVs",               required=True)
    parser.add_argument("--num_samples", help="Number of samples per run",                     type=int, default=1)
    # assume ckpts and concept are defined elsewhere or add args for them
    return parser.parse_args()

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    args    = parse_args()
    gpu_ids = args.gpus.split(',')

    # 1) READ ALL ROWS
    with open(args.file, newline='', encoding='utf-8') as csvfile:
        reader     = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames            # save header
        rows       = list(reader)                 # each row is a dict of all columns

    # 2) SHUFFLE
    random.seed(42)
    random.shuffle(rows)
    
    with open(args.ckpts_path, "r") as f:
        ckpts = json.load(f)

    for model, ckpt in ckpts.items(): 

        # 3) SPLIT into chunks
        n_gpus    = len(gpu_ids)
        print('n_gpus:', n_gpus)
        n_rows      = len(rows)
        base_size, remainder = divmod(n_rows, n_gpus)

        commands = {}
        concept = args.concept
        save_dir = f"{args.save_dir}/{concept}/{model}"
        for i in range(n_gpus):
            size = base_size + (1 if i < remainder else 0)
            start = sum(base_size + (1 if j < remainder else 0) for j in range(i))
            chunk = rows[start : start + size]
        
            # ensure output directory exists
            os.makedirs(save_dir, exist_ok=True)
            out_csv = os.path.join(save_dir, f"{model}_prompts_{i}.csv")

            # 4) WRITE OUT with full header
            with open(out_csv, 'w', newline='', encoding='utf-8') as outf:
                writer = csv.DictWriter(outf, fieldnames=fieldnames)
                writer.writeheader()
                cleaned_chunk = [
                    {k: v for k, v in row.items() if k in fieldnames and k is not None}
                    for row in chunk
                ]
                writer.writerows(cleaned_chunk)

            # build your per-model command exactly as before,
            # just pointing to `out_csv` instead of a .txt file
            if "SAFREE" in model:
                cmd = (
                    f"python src/gen_images.py "
                    f"--data '{out_csv}' "
                    f"--save_dir '{save_dir}' "
                    f"--num_samples {args.num_samples} "
                    f"--model_id_or_path '{ckpt}' "
                    f"--category '{concept}' "
                    f"--id '{i}' "
                    f"--safree -svf -lra "
                )
            elif "MACE" in model:
                cmd = (
                    f"python gen_images_dist.py "
                    f"--data '{out_csv}' "
                    f"--model_type sd "
                    f"--save_dir '{save_dir}' "
                    f"--num_samples {args.num_samples} "
                    f"--model_id_or_path {ckpt} "
                    f"--category '{concept}' "
                    f"--id '{i}'"
                )
            else:
                cmd = (
                    f"python gen_images_dist.py "
                    f"--data '{out_csv}' "
                    f"--model_type sd "
                    f"--save_dir '{save_dir}' "
                    f"--num_samples {args.num_samples} "
                    f"--target_ckpt {ckpt} "
                    f"--category '{concept}' "
                    f"--id '{i}'"
                )

            commands[f"{model}_{i}"] = cmd

        # dump all commands to JSON
        with open("all_commands.json", 'w', encoding='utf-8') as jf:
            json.dump(commands, jf, indent=2)

        # GPU‐launch loop (unchanged)…
        queue  = list(commands.items())
        active = []

        while queue or active:
            while queue and len(active) < len(gpu_ids):
                name, cmd = queue.pop(0)
                used      = {a['gpu'] for a in active}
                free_gpu  = next(g for g in gpu_ids if g not in used)

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = free_gpu
                logging.info(f"Launching {name} on GPU {free_gpu}: {cmd}")
                proc = subprocess.Popen(cmd, shell=True, env=env)
                active.append({'name': name, 'proc': proc, 'gpu': free_gpu})

            time.sleep(1)

            for job in active[:]:
                if job['proc'].poll() is not None:
                    job['proc'].communicate()
                    logging.info(f"{job['name']} on GPU {job['gpu']} finished")
                    active.remove(job)

    print("All tasks completed.")

if __name__ == "__main__":
    main()
