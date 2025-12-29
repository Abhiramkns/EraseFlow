import os
import argparse
import pandas as pd
import torch

from load_model import load_model


def load_dataset(csv_path: str, category: str) -> pd.DataFrame:
    dataset = pd.read_csv(csv_path)
    
    print(f"{category} dataset size: {len(dataset)}")
    return dataset


def resolve_prompt_and_case(row, fallback_idx):
    if "prompt" in row:
        return row["prompt"], row.get("case_number", fallback_idx)
    return None, None


def main(args):
    output_dir = os.path.join(args.save_dir, "all", args.category)
    os.makedirs(output_dir, exist_ok=True)

    print("Configuration:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    model, _, _ = load_model(
        args.model_type,
        args.target_ckpt,
        args.device,
        args.model_id_or_path,
        args.attack_type == "trajDiff",
        args.attack_type,
        target_ckpt_2=args.target_ckpt_2,
    )

    dataset = load_dataset(args.data, args.category)

    for idx, row in dataset.iterrows():
        target_prompt, case_num = resolve_prompt_and_case(row, idx)
        if not isinstance(target_prompt, str):
            continue

        seed = int(row.get("evaluation_seed", row.get("sd_seed", 42)))
        guidance = float(row.get("evaluation_guidance", 7.5))

        print(
            f"[{idx}] Case {case_num} | Seed {seed} | "
            f"Guidance {guidance} | Prompt: {target_prompt}"
        )

        generator = torch.Generator(device=args.device).manual_seed(seed)

        images = model(
            prompt=target_prompt,
            negative_prompt="",
            height=512,
            width=512,
            num_inference_steps=(
                28 if args.model_type == "sd3" else args.num_inference_steps
            ),
            guidance_scale=guidance,
            generator=generator,
        ).images

        save_path = os.path.join(output_dir, f"{case_num}.png")
        images[0].save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default="./data/tmp_prompt.csv")
    parser.add_argument("--model_id_or_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="./results/tmp")
    parser.add_argument("--model_type", type=str, default="sd")
    parser.add_argument("--target_ckpt", type=str, default=None)
    parser.add_argument("--target_ckpt_2", type=str, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=50)

    parser.add_argument(
        "--category", type=str, default="nudity", choices=["nudity"]
    )
    parser.add_argument("--attack_type", type=str, default="i2p")
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()
    main(args)
