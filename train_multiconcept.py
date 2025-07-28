# train.py

import os
import logging
import time
import torch
import contextlib
import json
import random
import numpy as np
import wandb

from utils.util import set_seed
from utils.model_setup import (
    load_pipeline,
    configure_unet_lora,
    setup_optimizer_and_scaler,
    save_lora_checkpoint,
)
from utils.sampling import generate_negative_prompt_embeddings, sample_multiconcept_epoch
from utils.train_step import train_multiconcept_eraseflow_step
from utils.get_args import get_args  # ← your arg parser

def main(args):
    # ─── LOGGING ─────────────────────────────────────────────
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # ─── W&B SETUP ────────────────────────────────────────────
    wandb.login()  # Assumes you've set WANDB_API_KEY
    wandb.init(
        project=(args.wandb_project if hasattr(args, "wandb_project") else "EraseFlow"),
        name=args.name,
        config=vars(args),
    )
    # ────────────────────────────────────────────────────────────

    # reproducibility
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 1) Load pipeline & scheduler, cast models
    pipeline, weight_dtype = load_pipeline(args, device)
    # 2) Attach LoRA to UNet and freeze base weights
    unet = configure_unet_lora(pipeline, args, device, weight_dtype)

    # 3) Build optimizer (LoRA + z_model) and potential GradScaler
    optimizer, scaler, z_model = setup_optimizer_and_scaler(unet, args, device)

    # 4) Decide which autocast to use
    if args.use_lora:
        args.autocast = contextlib.nullcontext
    else:
        def func_autocast():
            return torch.cuda.amp.autocast(dtype=weight_dtype)
        args.autocast = func_autocast

    # 5) Precompute negative prompt embedding (for classifier‐free guidance)
    neg_prompt = generate_negative_prompt_embeddings(pipeline, device)
    train_neg_prompt_embeds = neg_prompt.repeat(1, 1, 1)

    # 6) Output directory for saving LoRA weights
    output_dir = os.path.join(args.save_dir, args.name)
    os.makedirs(output_dir, exist_ok=True)

    global_step = 0
    start_time = time.time()

    # load your concept definitions
    with open(args.multiconcept_list, 'r') as f:
        multiconcept_list = json.load(f)

    # split into forget vs. retain
    forget_list = {}
    retain_list = {}
    for concept, anchor in multiconcept_list.items():
        if concept == anchor:
            retain_list[concept] = anchor
        else:
            forget_list[concept] = anchor

    # Number of *concepts* (not images) to draw each epoch
    NUM_CONCEPTS_PER_EPOCH = 5
    SWITCH_EPOCH = args.switch_epoch
    forgot_samples_record = {}
    forget_samples_images = {}
    for epoch in range(args.num_epochs):
        logger.info(f"Starting epoch {epoch}/{args.num_epochs - 1}")

        # ─── One big gradient accumulation ──────────────────────
        unet.zero_grad()

        # Pick with replacement
        selected_forget = random.choices(
            list(forget_list.items()), k=1 # NUM_CONCEPTS_PER_EPOCH
        )
        selected_retain = random.choices(
            list(retain_list.items()), k=NUM_CONCEPTS_PER_EPOCH
        )

        # We’ll keep track of how many “forget” vs “retain” samples we actually accumulate
        forget_sample_count = 0
        retain_sample_count = 0

        # ─── FORGET LOSS (weight = 1.0) ────────────────────────
        forget_loss = 0
        for concept, anchor_concept in selected_forget:
            # generate samples + images for logging
            if epoch == 0 or epoch < SWITCH_EPOCH:
                unet.eval()
                with args.autocast():
                    tmp_forget_samples, images = sample_multiconcept_epoch(
                        pipeline, args, concept, anchor_concept, device, neg_prompt
                    )
                forgot_samples_record[concept] = tmp_forget_samples
                forget_samples_images[concept] = images
            forget_samples = forgot_samples_record[concept]
            images = forget_samples_images[concept]
            # log up to 4 example images
            if forget_samples:
                wandb.log({
                    f"forget/{concept}": [
                        wandb.Image(
                            (img.to(dtype=torch.float32).detach().cpu().permute(1,2,0).numpy() if isinstance(img, torch.Tensor) else img),
                            caption=concept
                        )
                        for img in images[:4]
                    ]
                }, step=global_step)

            # accumulate loss→backward
            unet.train()
            for sample in forget_samples:
                forget_loss += train_multiconcept_eraseflow_step(
                    sample,
                    unet,
                    pipeline,
                    optimizer,
                    scaler,
                    z_model,
                    args,
                    train_neg_prompt_embeds,
                    loss_w=1.0
                )
                forget_sample_count += 1

        # ─── RETAIN LOSS (weight = 0.1) ───────────────────────
        retain_loss = 0
        for concept, anchor_concept in selected_retain:
            unet.eval()
            with args.autocast():
                pipeline.disable_lora()
                samples, images = sample_multiconcept_epoch(
                    pipeline, args, concept, anchor_concept, device, neg_prompt
                )
                pipeline.enable_lora()
            if samples:
                wandb.log({
                    f"retain/{concept}": [
                        wandb.Image(
                            (img.to(dtype=torch.float32).detach().cpu().permute(1,2,0).numpy() if isinstance(img, torch.Tensor) else img),
                            caption=concept
                        )
                        for img in images[:4]
                    ]
                }, step=global_step)

            unet.train()
            for sample in samples:
                retain_loss += train_multiconcept_eraseflow_step(
                    sample,
                    unet,
                    pipeline,
                    optimizer,
                    scaler,
                    z_model,
                    args,
                    train_neg_prompt_embeds,
                    loss_w=args.preserve_weight
                )
                retain_sample_count += 1

        # ─── OPTIMIZER STEP & CLIPPING ─────────────────────────
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(z_model, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(z_model, args.max_grad_norm)
            optimizer.step()

        global_step += 1

        loss_val = forget_loss + retain_loss
        # You can log the *average* per‑sample loss if your train_step returns it.
        # Assuming the last call to train_step stored the last loss in `loss_val`:
        logger.info(f"Epoch {epoch} | step {global_step} | last_loss {loss_val:.4f} | forget_loss {forget_loss:.4f} | retain_loss {retain_loss:.4f}")
        wandb.log({"train/last_loss": loss_val, "train/forget_loss": forget_loss, "train/retain_loss": retain_loss}, step=global_step)

        # ─── SAVE CHECKPOINT ───────────────────────────────────
        if epoch % args.save_freq == 0 or epoch == args.num_epochs - 1:
            save_lora_checkpoint(unet, output_dir, epoch)

    # ────────── CLEANUP ───────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info("Training complete.")
    logger.info(f"Total training time: {elapsed/3600:.2f} hours ({elapsed:.1f} seconds).")
    wandb.finish()


if __name__ == "__main__":
    args = get_args()
    main(args)
