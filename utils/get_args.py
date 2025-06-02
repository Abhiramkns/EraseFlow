# args.py

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="EraseFlow‐style LoRA training")

    # Core model & LoRA arguments
    parser.add_argument(
        "--model_id",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="HuggingFace repo or local path for Stable Diffusion."
    )
    parser.add_argument(
        "--target_prompt",
        type=str,
        required=True,
        help="Prompt whose concept we want to erase."
    )
    parser.add_argument(
        "--anchor_prompt",
        type=str,
        required=True,
        help="Safe prompt used for sampling."
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name for this run (subfolder under save_dir)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Whether to attach LoRA to UNet."
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="Rank for LoRA adapters."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["fp16", "bf16", "fp32"],
        default="bf16",
        help="Mixed precision for non‐trainable parts."
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Use 8‐bit AdamW (requires bitsandbytes)."
    )
    
    # Switch epoch: after this epoch, skip sampling
    parser.add_argument(
        "--switch_epoch",
        type=int,
        default=20,
        help="Last epoch (inclusive) to run the sampling phase. If not set, sampling runs every epoch."
    )

    # Optimizer / training hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size per training step."
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of DDIM sampling steps."
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        required=True,
        help="Total number of training epochs."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate for LoRA parameters."
    )
    parser.add_argument(
        "--flow_learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate for z_model."
    )
    parser.add_argument(
        "--trainadam_beta1",
        type=float,
        default=0.9,
        help="β1 for AdamW."
    )
    parser.add_argument(
        "--trainadam_beta2",
        type=float,
        default=0.999,
        help="β2 for AdamW."
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="ε for AdamW."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Text‐to‐image guidance scale."
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="η parameter for DDIM sampling."
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=2.5,
        help="Flow constant β used in z_loss (encourages z_model ≈ β)."
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max norm for gradient clipping."
    )
    parser.add_argument(
        "--cfg",
        action="store_true",
        help="Enable explicit classifier‐free guidance in the UNet forward pass."
    )

    # Checkpointing / output
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="Directory under which subdir `--name` will be created."
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=1,
        help="Save a LoRA checkpoint every N epochs."
    )

    return parser.parse_args()
