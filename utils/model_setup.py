# model_setup.py

import os
import logging
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from utils.util import unwrap_model

def load_pipeline(args, device):
    """
    Load StableDiffusionPipeline (without safety checker), switch to DDIMScheduler,
    and cast VAE + text_encoder to desired precision.
    """
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=weight_dtype
    )
    scheduler_config = {}
    scheduler_config.update(pipeline.scheduler.config)
    pipeline.scheduler = DDIMScheduler.from_config(scheduler_config)

    # freeze and cast VAE & text_encoder
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.vae.to(device, dtype=weight_dtype)
    pipeline.text_encoder.to(device, dtype=weight_dtype)

    # disable safety checker
    pipeline.safety_checker = None

    # progress-bar config
    pipeline.set_progress_bar_config(
        position=1,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    return pipeline, weight_dtype

def configure_unet_lora(pipeline, args, device, weight_dtype):
    """
    Freeze original UNet, attach a LoRA adapter, and cast LoRA weights if needed.
    """
    unet = pipeline.unet
    # freeze all existing weights
    unet.requires_grad_(False)
    for p in unet.parameters():
        p.requires_grad_(False)

    assert args.use_lora, "LoRA must be enabled (args.use_lora=True)."
    unet.to(device, dtype=weight_dtype)

    unet_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    # during mixed‐precision, LoRA weights remain fp32
    if args.mixed_precision in ["fp16", "bf16"]:
        cast_training_params(unet, dtype=torch.float32)

    return unet

def setup_optimizer_and_scaler(unet, args, device):
    """
    Create optimizer (8‐bit AdamW if requested) over
    LoRA parameters + z_model parameter, plus optional GradScaler.
    """
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("Please install bitsandbytes (pip install bitsandbytes) for 8‐bit Adam.")
    else:
        optimizer_cls = torch.optim.AdamW

    # z_model is a single scalar parameter to learn the flow constant
    z_model = torch.nn.Parameter(
        torch.tensor(-0.1953, device=device, dtype=torch.float32, requires_grad=True)
    )

    # collect LoRA layers
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
    param_groups = [
        {"params": lora_layers, "lr": args.learning_rate},
        {"params": z_model, "lr": args.flow_learning_rate},
    ]

    optimizer = optimizer_cls(
        param_groups,
        betas=(args.trainadam_beta1, args.trainadam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    scaler = None
    if args.mixed_precision in ["fp16", "bf16"]:
        scaler = torch.amp.GradScaler()

    return optimizer, scaler, z_model

def save_lora_checkpoint(unet, output_dir, epoch):
    """
    Unwrap compiled UNet, extract LoRA weights, and save them to disk.
    """
    save_path = os.path.join(output_dir, f"checkpoint_epoch{epoch}")
    os.makedirs(save_path, exist_ok=True)

    unwrapped = unwrap_model(unet)
    unet_lora_state_dict = convert_state_dict_to_diffusers(
        get_peft_model_state_dict(unwrapped)
    )

    StableDiffusionPipeline.save_lora_weights(
        save_directory=save_path,
        unet_lora_layers=unet_lora_state_dict,
        safe_serialization=True,
    )
    logging.info(f"Saved LoRA checkpoint to {save_path}")
