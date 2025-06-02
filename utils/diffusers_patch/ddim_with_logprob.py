# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

# Modified from https://github.com/huggingface/diffusers/blob/fc6acb6b97e93d58cb22b5fee52d884d77ce84d8/src/diffusers/schedulers/scheduling_ddim.py

from typing import Optional, Tuple, Union

import math
import torch

try:
    from diffusers.utils import randn_tensor
except ImportError:
    from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput, DDIMScheduler

import numpy as np

def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)


def _get_variance(self, timestep, prev_timestep):
    alpha_prod_t = torch.gather(self.alphas_cumprod, 0, timestep.cpu()).to(
        timestep.device
    )
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        self.alphas_cumprod.gather(0, prev_timestep.cpu()),
        self.final_alpha_cumprod,
    ).to(timestep.device)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance


def ddim_step_with_logprob(
    self: DDIMScheduler,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    eta: float = 1.0,
    use_clipped_model_output: bool = False,
    generator=None,
    prev_sample: Optional[torch.FloatTensor] = None,

    calculate_pb: bool = False, logp_mean=True,
    prev_timestep: int =None,
) -> Union[DDIMSchedulerOutput, Tuple]:
    """
    Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
    process from the learned model outputs (most often the predicted noise).

    Args:
        model_output (`torch.FloatTensor`): direct output from learned diffusion model.
        timestep (`int`): current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            current instance of sample being created by diffusion process.
        eta (`float`): weight of noise for added noise in diffusion step.
        use_clipped_model_output (`bool`): if `True`, compute "corrected" `model_output` from the clipped
            predicted original sample. Necessary because predicted original sample is clipped to [-1, 1] when
            `self.config.clip_sample` is `True`. If no clipping has happened, "corrected" `model_output` would
            coincide with the one provided as input and `use_clipped_model_output` will have not effect.
        generator: random number generator.
        variance_noise (`torch.FloatTensor`): instead of generating noise for the variance using `generator`, we
            can directly provide the noise for the variance itself. This is useful for methods such as
            CycleDiffusion. (https://arxiv.org/abs/2210.05559)
        return_dict (`bool`): option for returning tuple rather than DDIMSchedulerOutput class

        sample: x_t
        prev_sample: x_{t-1} (closer to clean image)

    Returns:
        [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
        [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
        returning a tuple, the first element is the sample tensor.

    """
    assert isinstance(self, DDIMScheduler)
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
    # Ideally, read DDIM paper in-detail understanding

    # Notation (<variable name> -> <name in paper>
    # - pred_noise_t -> e_theta(x_t, t)
    # - pred_original_sample -> f_theta(x_t, t) or x_0
    # - std_dev_t -> sigma_t
    # - eta -> η
    # - pred_sample_direction -> "direction pointing to x_t"
    # - pred_prev_sample -> "x_{t-1}"

    # 1. get previous step value (=t-1)
    if prev_timestep is None:
        prev_timestep = (
            timestep - self.config.num_train_timesteps // self.num_inference_steps
        )
    # to prevent OOB on gather
    prev_timestep = torch.clamp(prev_timestep, 0, self.config.num_train_timesteps - 1)

    # 2. compute alphas, betas
    # self.alphas_cumprod  torch.Size([1000])
    alpha_prod_t = self.alphas_cumprod.gather(0, timestep.cpu())  # torch scalar
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        self.alphas_cumprod.gather(0, prev_timestep.cpu()),
        self.final_alpha_cumprod,
    )
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape).to(
        sample.device
    )
    # alpha_prod_t = alpha_prod_t.to(sample.dtype)  # float32 -> bf16
    # alpha_prod_t_prev = alpha_prod_t_prev.to(sample.dtype)  # float32 -> bf16

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (
            sample - beta_prod_t ** (0.5) * model_output
        ) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (
            sample - alpha_prod_t ** (0.5) * pred_original_sample
        ) / beta_prod_t ** (0.5)
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (
            beta_prod_t**0.5
        ) * model_output
        pred_epsilon = (alpha_prod_t**0.5) * model_output + (
            beta_prod_t**0.5
        ) * sample
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`"
        )

    # 4. Clip or threshold "predicted x_0"
    # cifar ddpm: self.config.thresholding = False, self.config.clip_sample_range = 1.0
    # SD: self.config.thresholding = False, self.config.clip_sample = False
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = _get_variance(self, timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)  # eta is 1.0
    std_dev_t = _left_broadcast(std_dev_t, sample.shape).to(sample.device)

    if use_clipped_model_output: # not used?
        # the pred_epsilon is always re-derived from the clipped x_0 in Glide
        pred_epsilon = (
            sample - alpha_prod_t ** (0.5) * pred_original_sample
        ) / beta_prod_t ** (0.5)

    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
        0.5
    ) * pred_epsilon

    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample_mean = (
        alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    )

    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )

    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * variance_noise

    # log prob of prev_sample given prev_sample_mean and std_dev_t
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2))
        - torch.log(std_dev_t)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    if logp_mean:
        # mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    else:
        log_prob = log_prob.sum(dim=tuple(range(1, log_prob.ndim)))

    if calculate_pb:
        assert prev_sample is not None
        alpha_ddim = alpha_prod_t / alpha_prod_t_prev  # (bs, 4, 64, 64)
        pb_mean = alpha_ddim.sqrt() * prev_sample
        pb_std = (1 - alpha_ddim).sqrt()
        log_pb = (
                -((sample.detach() - pb_mean.detach()) ** 2) / (2 * (pb_std ** 2))
                - torch.log(pb_std)
                - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        if logp_mean:
            log_pb = log_pb.mean(dim=tuple(range(1, sample.ndim)))
        else:
            log_pb = log_pb.sum(dim=tuple(range(1, sample.ndim)))
        return prev_sample.type(sample.dtype), log_prob, log_pb

    else:
        return prev_sample.type(sample.dtype), log_prob
    # output is float32 as the self.alpha is float32


@torch.no_grad()
def pred_orig_latent(self: DDIMScheduler, model_output, sample: torch.FloatTensor, timestep: int):
    # 2. compute alphas, betas
    # self.alphas_cumprod  torch.Size([1000])
    alpha_prod_t = self.alphas_cumprod.gather(0, timestep.cpu())  # torch scalar
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    alpha_prod_t = alpha_prod_t.to(sample.dtype) # float32 -> bf16
    beta_prod_t = 1 - alpha_prod_t

    if self.config.prediction_type == "epsilon":
        pred_original_sample = (
            sample - beta_prod_t ** (0.5) * model_output
        ) / alpha_prod_t ** (0.5)
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (
            beta_prod_t**0.5
        ) * model_output
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`"
        )
    return pred_original_sample


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


# given x_{t-1} "prev_sample", compute x_t "sample"
def step_backward(self: DDIMScheduler,
    timestep: int,
    prev_sample: torch.FloatTensor,
    generator=None,):

    prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
    # to prevent OOB on gather
    prev_timestep = torch.clamp(prev_timestep, 0, self.config.num_train_timesteps - 1)

    # 2. compute alphas, betas
    # self.alphas_cumprod  torch.Size([1000])
    alpha_prod_t = self.alphas_cumprod.gather(0, timestep.cpu())  # torch scalar
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        self.alphas_cumprod.gather(0, prev_timestep.cpu()),
        self.final_alpha_cumprod,
    )
    alpha_prod_t = _left_broadcast(alpha_prod_t, prev_sample.shape).to(prev_sample.device)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, prev_sample.shape).to(prev_sample.device)
    # beta_prod_t = 1 - alpha_prod_t

    alpha_ddim = alpha_prod_t / alpha_prod_t_prev  # (bs, 4, 64, 64)
    pb_mean = alpha_ddim.sqrt() * prev_sample
    pb_std = (1 - alpha_ddim).sqrt()

    sample = pb_mean + pb_std * randn_tensor(
        prev_sample.shape,
        generator=generator,
        device=prev_sample.device,
        dtype=prev_sample.dtype,
    )
    return sample

import math
import numpy as np
import torch

# You might already have this dataclass defined elsewhere in your codebase.
from dataclasses import dataclass
from typing import Optional, Union, Tuple

@dataclass
class FlowMatchEulerDiscreteSchedulerOutput:
    prev_sample: torch.FloatTensor
    forward_logprob: torch.FloatTensor
    reverse_logprob: torch.FloatTensor


def sd3_step(
    self,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    prev_sample=None,
    s_churn: float = 0.0,
    s_tmin: float = 0.0,
    s_tmax: float = float("inf"),
    s_noise: float = 1.0,
    generator: Optional[torch.Generator] = None,
    return_dict: bool = True,
    use_manual_index=False,
) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
    r"""
    Predict the sample from the previous timestep by reversing the SDE,
    and compute the forward and reverse process log probabilities.
    
    In this modified step function we assume that the reverse process (going
    from sample \(x\) at time \(t\) to \(x_{\text{prev}}\) at time \(t-1\)) is given by:
    
    \[
    x_{\text{prev}} = x - \Delta\,\widehat{\epsilon} + \epsilon, \quad\text{with}\quad \epsilon\sim\mathcal{N}(0,(s_{\mathrm{noise}}\Delta)^2 I)
    \]
    
    where:
        - \(\Delta = \sigma - \sigma_{\mathrm{next}}\) (with \(\sigma\) obtained from the schedule),
        - \(\widehat{\epsilon}\) is the `model_output`,
        - \(s_{\mathrm{noise}}\) scales the variance of the added noise.
    
    The reverse process log probability is computed as the log probability of having
    sampled \(x_{\text{prev}}\) from \(\mathcal{N}\bigl(\mu_{\mathrm{rev}}, (s_{\mathrm{noise}}\Delta)^2 I\bigr)\),
    with \(\mu_{\mathrm{rev}} = x - \Delta\,\widehat{\epsilon}\).
    
    The forward process (or “reconstruction”) is obtained by inverting the above,
    i.e. by taking:
    
    \[
    \mu_{\mathrm{fwd}} = x_{\text{prev}} + \Delta\,\widehat{\epsilon},
    \]
    
    and then evaluating the log probability of recovering \(x\) from \(\mu_{\mathrm{fwd}}\)
    under \(\mathcal{N}\bigl(\mu_{\mathrm{fwd}}, (s_{\mathrm{noise}}\Delta)^2 I\bigr)\).
    
    Args:
        model_output (`torch.FloatTensor`):
            The model prediction (often an estimate of the noise).
        timestep (`float` or `torch.FloatTensor`):
            The current timestep; used for internal indexing.
        sample (`torch.FloatTensor`):
            The current sample (at time \(t\)).
        s_noise (`float`, defaults to 1.0):
            Scaling factor for the noise level in the reverse process (and so for the variance).
        generator (`torch.Generator`, *optional*):
            The random number generator for reproducible sampling.
        return_dict (`bool`, defaults to True):
            Whether to return a dataclass or a tuple.
    
    Returns:
        [`FlowMatchEulerDiscreteSchedulerOutput`] or `tuple`:
            A container with:
                - `prev_sample`: the predicted sample at the previous timestep,
                - `forward_logprob`: log probability of the forward “reconstruction” process,
                - `reverse_logprob`: log probability of the reverse (denoising) process.
    """
    # Ensure step index is initialized
    if self.step_index is None:
        self._init_step_index(timestep)
        
    # Ensure numerical precision.
    sample = sample.to(torch.float32)
    
    sigma = self.sigmas[self.step_index]
    sigma_next = self.sigmas[self.step_index + 1]
    
    # Δ (delta) represents the positive difference in noise levels.
    delta = sigma - sigma_next
    
    # --- Noising Process (Reverse) ---
    # Compute the deterministic mean for the noising process.
    mean_noising = sample - delta * model_output
    
    if prev_sample is None:
        # Generate noise and compute the noisy sample.
        if generator is not None:
            noise = torch.randn_like(sample, generator=generator)
        else:
            noise = torch.randn_like(sample)
        computed_prev_sample = mean_noising + s_noise * delta * noise
    else:
        computed_prev_sample = prev_sample
    
    # Compute log probability constants.
    # D is the product of dimensions (excluding the batch dimension).
    D = np.prod(sample.shape[1:])
    variance = (s_noise * delta) ** 2
    log_norm_const = torch.log(torch.tensor(2 * math.pi * variance, device=sample.device))
    
    # The noising log probability: likelihood of obtaining computed_prev_sample from the noising process.
    noising_error = torch.sum(((computed_prev_sample - mean_noising) ** 2), dim=tuple(range(1, sample.ndim)))
    noising_logprob = -0.5 * (D * log_norm_const + noising_error)
    
    # --- Denoising Process (Forward) ---
    # Invert the noising process to obtain the denoising mean.
    denoising_mean = computed_prev_sample + delta * model_output
    denoising_error = torch.sum(((sample - denoising_mean) ** 2), dim=tuple(range(1, sample.ndim)))
    denoising_logprob = -0.5 * (D * log_norm_const + denoising_error)
    
    # Advance the internal step counter.
    self._step_index += 1

    computed_prev_sample = computed_prev_sample.to(dtype=model_output.dtype)
    denoising_logprob = denoising_logprob.to(dtype=model_output.dtype)
    noising_logprob = noising_logprob.to(dtype=model_output.dtype)
    
    return computed_prev_sample, denoising_logprob, noising_logprob

@torch.no_grad()
def sd3_pred_orig_latent(self, model_output, sample, timestep):
    sample = sample.to(torch.float32)

    l_scheduler_timesteps = len(self.timesteps)
    scheduler_timesteps = self.timesteps.repeat(len(timestep), 1)
    timestep = timestep.repeat(1, l_scheduler_timesteps)

    indices = (scheduler_timesteps == timestep).nonzero()
    pos = 1 if len(indices) > 1 else 0
    curr_index = indices[pos][0].item()

    sigma = self.sigmas[curr_index]
    sigma_next = self.sigmas[curr_index+1]

    prev_sample = sample + (sigma_next - sigma)*model_output

    prev_sample = prev_sample.to(model_output.dtype)

    return prev_sample