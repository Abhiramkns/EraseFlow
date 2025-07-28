# sampling.py

import torch
from utils.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob

def generate_negative_prompt_embeddings(pipeline, device):
    """
    Create a single‐token negative prompt embedding (just an empty string)
    to use for classifier‐free guidance.
    """
    neg = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(device)
    )[0]
    return neg

def sample_epoch(pipeline, args, device, neg_prompt_embed):
    """
    For each epoch: 
      - encode anchor_prompt
      - run DDIM sampling w/ log‐prob until latent sequence
      - stack latents, prepare train prompt embeddings + timesteps
      - return a list of dicts containing latents, next_latents, embeddings, etc.
    """
    samples = []

    # Anchor prompts → their embeddings
    prompts = [args.anchor_prompt] * args.batch_size
    prompt_ids = pipeline.tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
    ).input_ids.to(device)
    anchor_prompt_embeds = pipeline.text_encoder(prompt_ids)[0]

    # Expand negative‐prompt embeddings to match batch size
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(len(anchor_prompt_embeds), 1, 1)

    # Inform scheduler how many diffusion steps we take this epoch
    pipeline.scheduler.set_timesteps(args.num_steps, device=device)

    # Perform one pass of inference‐mode sampling (no gradients)
    with torch.inference_mode():
        ret_tuple = pipeline_with_logprob(
            pipeline,
            prompt_embeds=anchor_prompt_embeds,
            negative_prompt_embeds=sample_neg_prompt_embeds,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            eta=args.eta,
            output_type="pt",
            return_unetoutput=False,
            num_images_per_prompt=1,
        )
        _, _, latents, _ = ret_tuple

    # latents has shape (batch_size, num_steps+1, 4, 64, 64)
    latents = torch.stack(latents, dim=1)

    # Now encode train (target) prompts
    train_prompts = [args.target_prompt] * args.batch_size
    train_prompt_ids = pipeline.tokenizer(
        train_prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
    ).input_ids.to(device)
    train_prompt_embeds = pipeline.text_encoder(train_prompt_ids)[0]

    # Build a timesteps tensor: shape (batch_size, num_steps)
    timesteps = pipeline.scheduler.timesteps.repeat(len(train_prompt_embeds), 1)

    samples.append({
        "prompt_embeds": train_prompt_embeds,
        "anchor_prompt_embeds": anchor_prompt_embeds,
        "timesteps": timesteps,
        # we drop the last timestep for “latents” and shift 1 for next_latents
        "latents": latents[:, :-1],
        "next_latents": latents[:, 1:],
    })

    return samples

def sample_multiconcept_epoch(pipeline, args, concept, anchor_concept, device, neg_prompt_embed, forget_set=False):
    """
    For each epoch: 
      - encode anchor_prompt
      - run DDIM sampling w/ log‐prob until latent sequence
      - stack latents, prepare train prompt embeddings + timesteps
      - return a list of dicts containing latents, next_latents, embeddings, etc.
    """
    samples = []

    # Anchor prompts → their embeddings
    prompts = [anchor_concept] * args.batch_size
    prompt_ids = pipeline.tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
    ).input_ids.to(device)
    anchor_prompt_embeds = pipeline.text_encoder(prompt_ids)[0]

    # Expand negative‐prompt embeddings to match batch size
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(len(anchor_prompt_embeds), 1, 1)

    # Inform scheduler how many diffusion steps we take this epoch
    pipeline.scheduler.set_timesteps(args.num_steps, device=device)

    # Perform one pass of inference‐mode sampling (no gradients)
    with torch.inference_mode():
        ret_tuple = pipeline_with_logprob(
            pipeline,
            prompt_embeds=anchor_prompt_embeds,
            negative_prompt_embeds=sample_neg_prompt_embeds,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            eta=args.eta,
            output_type="pt",
            return_unetoutput=False,
            num_images_per_prompt=1,
        )
        _, _, latents, _ = ret_tuple

    # latents has shape (batch_size, num_steps+1, 4, 64, 64)
    latents = torch.stack(latents, dim=1)

    # Now encode train (target) prompts
    train_prompts = [concept] * args.batch_size
    train_prompt_ids = pipeline.tokenizer(
        train_prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
    ).input_ids.to(device)
    train_prompt_embeds = pipeline.text_encoder(train_prompt_ids)[0]
    with torch.inference_mode():
        ret_tuple = pipeline_with_logprob(
            pipeline,
            prompt_embeds=train_prompt_embeds,
            negative_prompt_embeds=sample_neg_prompt_embeds,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            eta=args.eta,
            output_type="pt",
            return_unetoutput=False,
            num_images_per_prompt=1,
        )
        images, _, _, _ = ret_tuple

    # Build a timesteps tensor: shape (batch_size, num_steps)
    timesteps = pipeline.scheduler.timesteps.repeat(len(train_prompt_embeds), 1)

    samples.append({
        "prompt_embeds": train_prompt_embeds,
        "anchor_prompt_embeds": anchor_prompt_embeds,
        "timesteps": timesteps,
        # we drop the last timestep for “latents” and shift 1 for next_latents
        "latents": latents[:, :-1],
        "next_latents": latents[:, 1:],
    })

    return samples, images
