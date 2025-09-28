# train_flux.py
from collections import defaultdict
import contextlib
import os
import datetime
import time
import json
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import FluxPipeline
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import flow_grpo.prompts
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.flux_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_flux import encode_prompt
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper

import copy

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    """
    Compute token-wise embeddings, pooled embeddings, and text_ids via Flux encode_prompt.
    Returns:
        prompt_embeds: (B, seq_len, H)
        pooled_prompt_embeds: (B, H)
        text_ids: (seq_len, 3)
    """
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
            text_encoders,
            tokenizers,
            prompt,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
        text_ids = text_ids.to(device)
    return prompt_embeds, pooled_prompt_embeds, text_ids


def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, return_inverse=True, return_counts=True
    )
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()


def get_sigmas(noise_scheduler, timesteps, accelerator, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
    timesteps = timesteps.to(accelerator.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def compute_log_prob(
    transformer, 
    pipeline, 
    sample, 
    j, 
    embeds, 
    pooled_embeds, 
    text_ids, 
    latent_image_ids,
    config, 
    calculate_pb=False
):
    guidance_scale = config.sample.guidance_scale
    if pipeline.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=pooled_embeds.device, dtype=torch.float32)              
        guidance = guidance.expand(sample['latents'][:, j].shape[0])
    else:
        guidance = None

    noise_pred = transformer(
        hidden_states=sample['latents'][:, j],
        timestep=sample['timesteps'][:, j] / 1000,
        guidance=guidance,
        pooled_projections=pooled_embeds,
        encoder_hidden_states=embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        return_dict=False,
    )[0]
    return sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        sample['timesteps'][:, j],
        sample['latents'][:, j].float(),
        prev_sample=sample['next_latents'][:, j].float(),
        noise_level=config.sample.noise_level,
        calculate_pb=calculate_pb,
    )


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, 'checkpoints', f'checkpoint-{global_step}')
    save_root_lora = os.path.join(save_root, 'lora')
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)

def prepare_latent_image_ids(pipeline, batch_size, height, width, device, dtype):
    latent_image_ids = pipeline._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
    return latent_image_ids

def main(_):
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime('%Y.%m.%d_%H.%M.%S')
    config.run_name = (config.run_name or unique_id) + '_' + unique_id

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    if accelerator.is_main_process:
        wandb.init(project='flow_grpo')

    logger.info(f"\n{config}")
    set_seed(config.seed, device_specific=True)

    # load Flux pipeline
    pipeline = FluxPipeline.from_pretrained(
        config.pretrained.model,
        cache_dir='.'
    )
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)
    
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2]
    tokenizers    = [pipeline.tokenizer, pipeline.tokenizer_2]

    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc='Timestep',
        dynamic_ncols=True,
    )

    inference_dtype = torch.float32
    if accelerator.mixed_precision == 'fp16': inference_dtype = torch.float16
    if accelerator.mixed_precision == 'bf16': inference_dtype = torch.bfloat16

    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.transformer.to(accelerator.device)

    if config.use_lora:
        target_modules = [
            'attn.add_k_proj','attn.add_q_proj','attn.add_v_proj',
            'attn.to_add_out','attn.to_k','attn.to_out.0','attn.to_q','attn.to_v',
        ]
        transformer_lora_config = LoraConfig(
            r=32, lora_alpha=64, init_lora_weights='gaussian', target_modules=target_modules
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
            pipeline.transformer.set_adapter('default')
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
        pipeline.transformer.enable_gradient_checkpointing()
    
    transformer = pipeline.transformer
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)

    if config.allow_tf32: torch.backends.cuda.matmul.allow_tf32 = True

    optimizer_cls = torch.optim.AdamW
    if config.train.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit

    z_model = torch.nn.Parameter(torch.tensor(-0.1953, device=accelerator.device, dtype=torch.float32, requires_grad=True))
    param_groups = [
        {'params': transformer_trainable_parameters, 'lr': config.train.learning_rate},
        {'params': z_model, 'lr': config.train.flow_learning_rate},
    ]
    optimizer = optimizer_cls(
        param_groups,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # prepare negative prompts
    neg_prompt_embed, neg_pooled_prompt_embed, neg_text_ids = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)
    train_neg_text_ids = neg_text_ids.repeat(config.train.batch_size, 1, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    transformer, optimizer = accelerator.prepare(transformer, optimizer)

    # ----- TRAINING LOOP -----
    samples_per_epoch = config.sample.train_batch_size * accelerator.num_processes * config.sample.num_batches_per_epoch
    total_train_batch_size = config.train.batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f" Sample batch size/device = {config.sample.train_batch_size}")
    logger.info(f" Train batch size/device  = {config.train.batch_size}")
    logger.info(f" Grad acc steps           = {config.train.gradient_accumulation_steps}")
    logger.info(f" Samples/epoch            = {samples_per_epoch}")
    logger.info(f" Total train batch size   = {total_train_batch_size}")
    logger.info(f" Inner epochs             = {config.train.num_inner_epochs}")

    epoch = global_step = 0
    switch_epoch = config.switch_epoch
    for epoch in range(config.num_epochs):
        if epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
            save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)

        # ------- SAMPLING -------
        pipeline.transformer.eval()

        if epoch == 0 or epoch < switch_epoch:
            perm_samples, prompts = [], []
            for _ in tqdm(range(config.sample.num_batches_per_epoch), desc=f"Epoch {epoch}: sampling", disable=not accelerator.is_local_main_process):
                prompts = config.anchor_prompt
                pe, ppe, txt_ids = compute_text_embeddings(prompts, text_encoders, tokenizers, 128, accelerator.device)
                
                # run sampling with logprob
                _, lat_list, lp_list = pipeline_with_logprob(
                    pipeline,
                    prompt=None,
                    prompt_2=None,
                    text_ids=txt_ids.squeeze(0),
                    height=config.resolution,
                    width=config.resolution,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    negative_prompt=None,
                    num_images_per_prompt=config.sample.train_batch_size,
                    prompt_embeds=pe,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    pooled_prompt_embeds=ppe,
                    negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                    output_type='pt',
                    noise_level=config.sample.noise_level,
                )

                latents = torch.stack(lat_list, dim=1)
                log_probs= torch.stack(lp_list, dim=1)
                
                timesteps = pipeline.scheduler.timesteps.repeat(config.sample.train_batch_size, 1)
                prompts = config.prompt
                pe, ppe, txt_ids = compute_text_embeddings(prompts, text_encoders, tokenizers, 128, accelerator.device)

                perm_samples.append({
                    'prompt_embeds': pe,
                    'pooled_prompt_embeds': ppe,
                    'text_ids': txt_ids,
                    'timesteps': timesteps,
                    'latents': latents[:, :-1],
                    'next_latents': latents[:, 1:],
                    'log_probs': log_probs,
                })

        # collate samples
        samples = {}
        for k in perm_samples[0]:
            tensors = []
            for s in perm_samples:
                tensors.append(s[k])
            samples[k] = torch.cat(tensors, dim=0)

        # optional wandb logging of images
        
        if accelerator.is_main_process:
            with tempfile.TemporaryDirectory() as tmpdir:
                images, _, _ = pipeline_with_logprob(
                    pipeline, 
                    prompt=None, 
                    prompt_2=None, 
                    text_ids=txt_ids.squeeze(0),
                    height=config.resolution, width=config.resolution,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    negative_prompt=None,
                    num_images_per_prompt=config.sample.train_batch_size,
                    prompt_embeds=pe,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    pooled_prompt_embeds=ppe,
                    negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                    output_type='pil', 
                    noise_level=config.sample.noise_level,
                )
                imgs = images[:min(15, len(images))]
                for idx, pil in enumerate(imgs):
                    pil.save(os.path.join(tmpdir, f'{idx}.jpg'))
                wandb.log({'images':[wandb.Image(os.path.join(tmpdir,f'{i}.jpg')) for i in range(len(imgs))]}, step=global_step)

        total_batch_size, num_timesteps = samples['timesteps'].shape
        assert num_timesteps == config.sample.num_steps
        
        torch.cuda.empty_cache()
        
        perm_height = config.resolution or pipeline.default_sample_size * pipeline.vae_scale_factor
        perm_width = config.resolution or pipeline.default_sample_size * pipeline.vae_scale_factor
        height = 2 * (int(perm_height) // (pipeline.vae_scale_factor * 2))
        width = 2 * (int(perm_width) // (pipeline.vae_scale_factor * 2))
        cached_latent_ids = pipeline._prepare_latent_image_ids(
            config.train.batch_size, 
            height // 2, 
            width // 2, 
            accelerator.device,
            inference_dtype
        )

        # ------- TRAINING -------
        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}
            batched = {k: v.view(config.sample.num_batches_per_epoch, -1, *v.shape[1:]) for k, v in samples.items()}
            samples_batched = [dict(zip(batched, x)) for x in zip(*batched.values())]

            pipeline.transformer.train()
            for i, sample in enumerate(tqdm(samples_batched, desc=f"Epoch {epoch}.{inner_epoch}: training", disable=not accelerator.is_local_main_process)):
                embeds = sample["prompt_embeds"]
                pooled_embeds = sample["pooled_prompt_embeds"]
                text_ids = sample['text_ids']
                total_loss=0.0
                all_logp, all_logpb=[], []
                train_timesteps=list(range(num_train_timesteps))
                for j in train_timesteps:
                    with accelerator.accumulate(transformer):
                        with autocast():
                            _, logp, logpb, _, _ = compute_log_prob(
                                transformer, 
                                pipeline, 
                                sample, 
                                j, 
                                embeds, 
                                pooled_embeds, 
                                text_ids.squeeze(0), 
                                cached_latent_ids.squeeze(0),
                                config, 
                                calculate_pb=True
                            )

                            torch.cuda.empty_cache()

                        all_logp.append(logp)
                        all_logpb.append(logpb)
                        loss_flow=logp
                        total_loss+=loss_flow
                total_loss = (total_loss + (z_model - config.beta)).pow(2).mean()
                accelerator.backward(total_loss)
                optimizer.step()
                optimizer.zero_grad()
                if accelerator.is_main_process:
                    wandb.log({'loss': total_loss.item(),'flow': z_model.item(), 'avg_logp': torch.stack(all_logp).mean().item(),'avg_logpb': torch.stack(all_logpb).mean().item()}, step=global_step)
                global_step+=1
            if config.train.ema: ema.step(transformer_trainable_parameters, global_step)

if __name__ == '__main__':
    app.run(main)