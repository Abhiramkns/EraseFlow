# train_step.py
import math
import torch
from utils.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob

def train_eraseflow_step(sample, unet, pipeline, optimizer, scaler, z_model, args, train_neg_prompt_embeds):
    """
    Given a single `sample` dict (with latents, next_latents, embeddings, timesteps):
      1. Optionally concatenate unconditional + conditional embeddings if args.cfg
      2. Loop over a subset of timesteps to compute log‐forward & log‐backward
      3. Compute flow‐loss and z_loss, backpropagate, and step optimizer (with GradScaler if needed)
    Returns the scalar loss value (float).
    """
    # build “embeds” for UNet input
    if args.cfg:
        # classifier-free: concat neg + pos prompt embeddings
        embeds = torch.cat([train_neg_prompt_embeds, sample["prompt_embeds"]])
    else:
        embeds = sample["prompt_embeds"]

    total_loss = 0.0

    # choose timesteps indices (same as original: 10 random in [0..39], plus [40..49])
    indices = torch.cat([torch.randperm(40)[:10], torch.arange(40, 50)]).to(sample["latents"].device)

    for j in indices:
        j = int(j.item())
        j_latent = sample["latents"][:, j]
        next_j_latent = sample["next_latents"][:, j]

        with args.autocast():
            if args.cfg:
                # forward twice: uncond + text
                noise_pred = unet(
                    torch.cat([j_latent] * 2),
                    torch.cat([sample["timesteps"][:, j]] * 2),
                    embeds,
                ).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                noise_pred = unet(
                    j_latent,
                    sample["timesteps"][:, j],
                    embeds,
                ).sample

            # compute log‐forward (log_pf) and log‐backward (log_pb) under the DDIM step
            _, log_pf, log_pb = ddim_step_with_logprob(
                pipeline.scheduler,
                noise_pred,
                sample["timesteps"][:, j],
                j_latent,
                eta=args.eta,
                prev_sample=next_j_latent,
                calculate_pb=True,
            )

        # GFlowNet loss = (log_pf - log_pb)
        loss_flow = log_pf - log_pb  # shape (batch_size,)
        total_loss = total_loss + loss_flow

        # free up intermediate tensors
        torch.cuda.empty_cache()

    # compute z‐loss: encourages z_model ≈ log(beta). Here z_model models the log(Z).
    z_target = args.beta
    z_loss = z_model - z_target
    total_loss = total_loss + z_loss

    # mean‐squared: (sum over batch & timesteps + z_loss).pow(2).mean()
    total_loss = torch.mean(total_loss.pow(2))

    if scaler is not None:
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(z_model, args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(z_model, args.max_grad_norm)
        optimizer.step()

    optimizer.zero_grad()

    return total_loss.item()
