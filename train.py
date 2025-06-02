# train.py

import os
import logging
import time
import torch
import contextlib

from utils.util import set_seed
from utils.model_setup import (
    load_pipeline,
    configure_unet_lora,
    setup_optimizer_and_scaler,
    save_lora_checkpoint,
)
from utils.sampling import generate_negative_prompt_embeddings, sample_epoch
from utils.train_step import train_eraseflow_step
from utils.get_args import get_args  # ← our updated argument‐parsing module

def main(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

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
    last_samples = None

    # ──────────── Start timing ────────────
    start_time = time.time()

    for epoch in range(args.num_epochs):
        logger.info(f"Starting epoch {epoch}/{args.num_epochs - 1}")

        # Determine whether to sample fresh latents or reuse previous
        if args.switch_epoch is None or epoch <= args.switch_epoch:
            # SAMPLE PHASE (no gradients)
            unet.zero_grad()
            unet.eval()
            last_samples = sample_epoch(pipeline, args, device, neg_prompt)
        else:
            # Use samples from the last sampling epoch
            logger.info(f"Epoch {epoch} > switch_epoch ({args.switch_epoch}): reusing previous samples")
            # last_samples remains unchanged

        # TRAIN PHASE
        if last_samples is not None:
            unet.train()
            for sample in last_samples:
                loss_val = train_eraseflow_step(
                    sample,
                    unet,
                    pipeline,
                    optimizer,
                    scaler,
                    z_model,
                    args,
                    train_neg_prompt_embeds,
                )
                global_step += 1
                if global_step % 10 == 0:
                    logger.info(f"Epoch {epoch} | step {global_step} | loss {loss_val:.4f}")

        # SAVE CHECKPOINT
        if epoch % args.save_freq == 0 or epoch == args.num_epochs - 1:
            save_lora_checkpoint(unet, output_dir, epoch)

    # ────────── End timing ──────────
    elapsed = time.time() - start_time
    hours = elapsed / 3600
    seconds = elapsed
    logger.info("Training complete.")
    logger.info(f"Total training time: {hours:.2f} hours ({seconds:.1f} seconds).")

if __name__ == "__main__":
    args = get_args()
    main(args)
