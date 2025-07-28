from transformers.modeling_outputs import BaseModelOutputWithPooling
from typing import Callable, List, Optional, Union
import torch
import torch.nn.functional as F
# from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers import StableDiffusionPipeline

from diffusers.utils import logging
import torch.nn.functional as F
# from einops import rearrange

from transformers import CLIPModel

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
)

from torchvision import transforms
from tqdm import tqdm
from torchvision import transforms as tfms

from utils.models.flow.pipeline_rf import RectifiedFlowPipeline

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def f_beta(z, btype='sigmoid', upperbound_timestep=10, concept_type='nudity'):
    if "artists-" in concept_type:
        t = 5.5  # Midpoint between the two means
        k = 3.5     # Adjust the value of k as needed
    else:
        t = 5.333  # Midpoint between the two means
        k = 2.5     # Adjust the value of k as needed

    if btype=="tanh":
        _value = math.tanh(k * (10 * z - t))
        output = round(upperbound_timestep / 2. * (_value + 1))
    elif btype=="sigmoid":
        sigmoid_scale = 2.0
        _value = sigmoid(sigmoid_scale * k * (10 * z - t))
        output = round(upperbound_timestep * (_value))
    else:
        NotImplementedError('btype is incorrect')
    return output

def projection_matrix(E):
    """Calculate the projection matrix onto the subspace spanned by E."""   
    P = E @ torch.pinverse(E.T @ E) @ E.T
    return P


def projection_and_orthogonal(input_embeddings, masked_input_subspace_projection, concept_subspace_projection):
    ie = input_embeddings
    ms = masked_input_subspace_projection
    cs = concept_subspace_projection
    device = ie.device
    dim = ms.shape[0]
    
    uncond_e, text_e = ie.chunk(2)
    new_text_e = (torch.eye(dim).to(device) - cs) @ ms @ torch.squeeze(text_e).T
    new_text_e = new_text_e.T[None, :]
    new_embeddings = torch.concat([uncond_e, new_text_e])
    return new_embeddings

def safree_projection(input_embeddings, p_emb, masked_input_subspace_projection, concept_subspace_projection, 
                alpha=0., max_length=77, logger=None):
    ie = input_embeddings
    ms = masked_input_subspace_projection
    cs = concept_subspace_projection
    device = ie.device
    (n_t, dim) = p_emb.shape   

    I_m_cs = torch.eye(dim).to(device) - cs
    dist_vec = I_m_cs @ p_emb.T
    dist_p_emb = torch.norm(dist_vec, dim=0)
        
    means = []
    
    # Loop through each item in the tensor
    for i in range(n_t):
        # Remove the i-th item and calculate the mean of the remaining items
        mean_without_i = torch.mean(torch.cat((dist_p_emb[:i], dist_p_emb[i+1:])))
        # Append the mean to the list
        means.append(mean_without_i)

    # Convert the list of means to a tensor
    mean_dist = torch.tensor(means).to(device)
    rm_vector = (dist_p_emb < (1. + alpha) * mean_dist).float() # 1 for safe tokens 0 for trigger tokens
    n_removed = n_t - rm_vector.sum()
    if logger is not None:
        logger.log(f"Among {n_t} tokens, we remove {int(n_removed)}.")
    else:
        print(f"Among {n_t} tokens, we remove {int(n_removed)}.")
    
    # match this with the token size   
    ones_tensor = torch.ones(max_length).to(device)
    ones_tensor[1:n_t+1] = rm_vector
    ones_tensor = ones_tensor.unsqueeze(1)
        
    uncond_e, text_e = ie.chunk(2)
    text_e = text_e.squeeze()
    new_text_e = I_m_cs @ ms @ text_e.T
    new_text_e = new_text_e.T
    
    merged_text_e = torch.where(ones_tensor.bool(), text_e, new_text_e)
    new_embeddings = torch.concat([uncond_e, merged_text_e.unsqueeze(0)])
    return new_embeddings


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


class ModifiedRectifiedFlowPipeline(RectifiedFlowPipeline):
    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        requires_safety_checker: bool = True,
        clip_model: CLIPModel = CLIPModel.from_pretrained("openai/clip-vit-large-patch14"),
    ):
        
        super(ModifiedRectifiedFlowPipeline, self).__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            None,
            feature_extractor, 
        )
        self.clip_model = clip_model
        self.safety_checker = None
        self.cut_out_size = (
            feature_extractor.size
            if isinstance(feature_extractor.size, int)
            else feature_extractor.size["shortest_edge"]
        )
        self.normalize = transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
        
    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask

    def _encode_embeddings(self, prompt, prompt_embeddings, attention_mask=None):
        output_attentions = self.text_encoder.text_model.config.output_attentions
        output_hidden_states = (
            self.text_encoder.text_model.config.output_hidden_states
        )
        return_dict = self.text_encoder.text_model.config.use_return_dict
        hidden_states = self.text_encoder.text_model.embeddings(inputs_embeds=prompt_embeddings)
        
        bsz, seq_len = prompt.shape[0], prompt.shape[1]
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype)
                
        causal_attention_mask = causal_attention_mask.to(
            hidden_states.device
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = self.text_encoder.text_model._expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.text_encoder.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.text_encoder.text_model.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=prompt.device), prompt.to(torch.int).argmax(dim=-1)
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _new_encode_negative_prompt_space(self, negative_prompt_space, max_length, num_images_per_prompt, pooler_output=True):
        device = self._execution_device

        uncond_input = self.tokenizer(
            negative_prompt_space,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        uncond_embeddings = self.text_encoder(
            uncond_input.input_ids.to(device),
            attention_mask=uncond_input.attention_mask.to(device),
        )
        if not pooler_output:
            uncond_embeddings = uncond_embeddings[0]
            bs_embed, seq_len, _ = uncond_embeddings.shape
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
        else:
            uncond_embeddings = uncond_embeddings.pooler_output
        
        return uncond_embeddings

    def _masked_encode_prompt(self, prompt):
        device = self._execution_device
        
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        n_real_tokens = untruncated_ids.shape[1] -2

        if untruncated_ids.shape[1] > self.tokenizer.model_max_length:
            untruncated_ids = untruncated_ids[:, :self.tokenizer.model_max_length]
            n_real_tokens = self.tokenizer.model_max_length -2
        masked_ids = untruncated_ids.repeat(n_real_tokens, 1)

        for i in range(n_real_tokens):
            masked_ids[i, i+1] = 0

        masked_embeddings = self.text_encoder(
            masked_ids.to(device),
            attention_mask=None,
        )
        return masked_embeddings.pooler_output

    def _new_encode_prompt(self, prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, 
                            prompt_ids=None, prompt_embeddings=None, token_mask=None, debug=False):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        device = self._execution_device

        if prompt_embeddings is not None:
            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            text_embeddings = self._encode_embeddings(
                prompt_ids,
                prompt_embeddings,
                attention_mask=attention_mask,
            )
            text_input_ids = prompt_ids
        else:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            ################################################################################################
            if token_mask is not None:
                mask_iids = torch.where(token_mask == 0, torch.zeros_like(token_mask), text_input_ids[0].to(device)).int()
                mask_iids = mask_iids[mask_iids != 0]
                tmp_ones = torch.ones_like(token_mask) * 49407
                tmp_ones[:len(mask_iids)] = mask_iids
                text_input_ids = tmp_ones.int()
                text_input_ids = text_input_ids[None, :]                            
            ################################################################################################

            text_embeddings = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
        text_embeddings = text_embeddings[0]
        
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)
            
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings, text_input_ids, text_inputs.attention_mask

    @torch.enable_grad()
    def cond_fn(
        self,
        latents,
        timestep,
        index,
        text_embeddings,
        noise_pred_original,
        text_embeddings_clip,
        clip_guidance_scale,
        dt = 1.0,
    ):
        latents = latents.detach().requires_grad_()

        # latent_model_input = self.scheduler.scale_model_input(latents, timestep)
        latent_model_input = latents
        
        # predict the noise residual
        noise_pred = self.unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample

        # if isinstance(self.scheduler, (PNDMScheduler, DDIMScheduler, DPMSolverMultistepScheduler)):
        #     alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        #     beta_prod_t = 1 - alpha_prod_t
        #     # compute predicted original sample from predicted noise also called
        #     # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        #     pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

        #     fac = torch.sqrt(beta_prod_t)
        #     sample = pred_original_sample * (fac) + latents * (1 - fac)
        # elif isinstance(self.scheduler, LMSDiscreteScheduler):
        #     sigma = self.scheduler.sigmas[index]
        #     sample = latents - sigma * noise_pred
        # else:
        #     raise ValueError(f"scheduler type {type(self.scheduler)} not supported")
        
        sample = latents + (timestep/1000)*noise_pred

        sample = 1 / self.vae.config.scaling_factor * sample
        image = self.vae.decode(sample).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        
        image = transforms.Resize(self.cut_out_size)(image)
        image = self.normalize(image).to(latents.dtype)

        image_embeddings_clip = self.clip_model.get_image_features(image)
        image_embeddings_clip = image_embeddings_clip / image_embeddings_clip.norm(p=2, dim=-1, keepdim=True)
       
        loss = spherical_dist_loss(image_embeddings_clip, text_embeddings_clip).mean() * clip_guidance_scale

        grads = torch.autograd.grad(loss, latents)[0]

        # if isinstance(self.scheduler, LMSDiscreteScheduler):
        #     latents = latents.detach() + grads * (sigma**2)
        #     noise_pred = noise_pred_original
        # else:
        #     noise_pred = noise_pred_original - torch.sqrt(beta_prod_t) * grads
        
        # noise_pred = noise_pred_original - grads
        latents = latents - grads
        return noise_pred, latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_space: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        prompt_ids = None,
        prompt_embeddings = None,
        return_latents = False,
        safree_dict = {},
        clip_guidance_scale = 0,
        clip_prompt = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
        Examples:
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        sf = safree_dict
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps, prompt_embeds=prompt_embeddings)

        # 2. Define call parameters
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size = 1 
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # 3. Encode input prompt
        text_embeddings, text_input_ids, attention_mask = self._new_encode_prompt(
            prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, 
            prompt_ids, prompt_embeddings            
        )
        
        if clip_guidance_scale > 0:
            print("clip guidance")
            # import ipdb; ipdb.set_trace()
            if clip_prompt is not None:
                print("clip prompt: ", clip_prompt)
                clip_text_input = self.tokenizer(
                    clip_prompt,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(self.device)
            else:
                clip_text_input = text_input_ids.to(self.device)
            text_embeddings_clip = self.clip_model.get_text_features(clip_text_input)
            text_embeddings_clip = text_embeddings_clip / text_embeddings_clip.norm(p=2, dim=-1, keepdim=True)
            # duplicate text embeddings clip for each generation per prompt
            text_embeddings_clip = text_embeddings_clip.repeat_interleave(num_images_per_prompt, dim=0)
            
        if sf["safree"]:
            print("safree")
            negspace_text_embeddings = self._new_encode_negative_prompt_space(negative_prompt_space, 77, num_images_per_prompt)
            project_matrix = projection_matrix(negspace_text_embeddings.T)
            masked_embs = self._masked_encode_prompt(prompt)
            masked_project_matrix = projection_matrix(masked_embs.T)
            rescaled_text_embeddings = safree_projection(text_embeddings, masked_embs,
                                                                    masked_project_matrix, 
                                                                    project_matrix,
                                                                    alpha=sf["alpha"],
                                                                    logger=sf["logger"])                        
        else:
            negspace_text_embeddings = None
            project_matrix = None
        
        if sf["svf"]:
            print("svf")
            proj_ort = projection_and_orthogonal(text_embeddings, masked_project_matrix, project_matrix)
            _, text_e = text_embeddings.chunk(2)
            s_attn_mask = attention_mask.squeeze()
            
            text_e = text_e.squeeze()
            _, proj_ort_e = proj_ort.chunk(2)
            proj_ort_e = proj_ort_e.squeeze()                    
            proj_ort_e_act = proj_ort_e[s_attn_mask == 1]
            text_e_act = text_e[s_attn_mask == 1]
            sim_org_onp_act = F.cosine_similarity(proj_ort_e_act, text_e_act)
            beta = (1 - sim_org_onp_act.mean().item())
            
            beta_adjusted = f_beta(beta, upperbound_timestep=sf['up_t'], concept_type=sf['category'])
            sf["logger"].log(f"beta : {beta}, adjusted_beta: {beta_adjusted}")
            
        # 4. Prepare timesteps
        # self.scheduler.set_timesteps(num_inference_steps, device=device)
        # timesteps = self.scheduler.timesteps
        
        timesteps = [(1. - i/num_inference_steps) * 1000. for i in range(num_inference_steps)]

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        dt = 1.0 / num_inference_steps
        
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        all_latents = [latents]
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                # latents = [bs, 4, 64, 64]
                
                if sf['lra']:
                    latent_model_input = torch.cat([latents] * 3) if do_classifier_free_guidance else latents
                    # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                else:
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                if sf["svf"]:
                    _text_embeddings = rescaled_text_embeddings if (sf["safree"] \
                                    and (i <= beta_adjusted)) \
                                    else text_embeddings                                                                
                else:                    
                    _text_embeddings = rescaled_text_embeddings if (sf["safree"] \
                                            and (sf["re_attn_t"][0] <= i <= sf["re_attn_t"][1])) \
                                            else text_embeddings                    
                
                # predict the noise residual
                if sf['lra']:                    
                    _, text_e = text_embeddings.chunk(2)
                    combined_text_embeddings = torch.cat([_text_embeddings, text_e])                                        
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=combined_text_embeddings).sample                    
                else:
                    vec_t = torch.ones((latent_model_input.shape[0],), device=latents.device) * t 
                    noise_pred = self.unet(latent_model_input, vec_t, encoder_hidden_states=_text_embeddings).sample
                # perform guidance
                if do_classifier_free_guidance:
                    if sf["lra"]:
                        noise_pred_uncond, noise_pred_text, _ = noise_pred.chunk(3)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    else:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                if clip_guidance_scale > 0:
                    # import ipdb; ipdb.set_trace()
                    text_embeddings_for_guidance = (
                        text_embeddings.chunk(2)[1] if do_classifier_free_guidance else text_embeddings
                    )
                    noise_pred, latents = self.cond_fn(
                        latents,
                        t,
                        i,
                        text_embeddings_for_guidance,
                        noise_pred,
                        text_embeddings_clip,
                        clip_guidance_scale,
                        dt=dt,
                    )
                                 
                    if sf['lra']:
                        latent_model_input = torch.cat([latents] * 3) if do_classifier_free_guidance else latents
                        # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    else:
                        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                        # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    
                    if sf['lra']:                    
                        _, text_e = text_embeddings.chunk(2)
                        combined_text_embeddings = torch.cat([_text_embeddings, text_e])                                        
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=combined_text_embeddings).sample                    
                    else:
                        vec_t = torch.ones((latent_model_input.shape[0],), device=latents.device) * t 
                        noise_pred = self.unet(latent_model_input, vec_t, encoder_hidden_states=_text_embeddings).sample
                    # perform guidance
                    if do_classifier_free_guidance:
                        if sf["lra"]:
                            noise_pred_uncond, noise_pred_text, _ = noise_pred.chunk(3)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        else:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                latents = latents + dt*noise_pred # self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                
                all_latents.append(latents)
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # latents: [#, 4, 64, 64]
        if return_latents:
            return latents, all_latents

        # 8. Post-processing
        image = self.decode_latents(latents)
        
        # 9. Run safety checker
        # image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return image, all_latents
        # return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
        # return StableDiffusionPipelineOutput(images=image)    
        
    @torch.no_grad()
    def invert(
        self,
        target_image,
        prompt,
        guidance_scale=3.5,
        num_inference_steps=80,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt="",
        device="cuda",
    ):

        with torch.no_grad():
            # import ipdb; ipdb.set_trace()
            latent = self.vae.encode(tfms.functional.to_tensor(target_image).unsqueeze(0).to(self.device) * 2 - 1)
        start_latents = 0.18215 * latent.latent_dist.sample()

        # Encode prompt
        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # Latents are now the specified start latents
        latents = start_latents.clone()

        # We'll keep a list of the inverted latents as the process goes on
        intermediate_latents = []

        # Set num inference steps
        # self.unlearned_model.scheduler.set_timesteps(num_inference_steps, device=device)

        # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
        timesteps = list(reversed([(1. - i/num_inference_steps) * 1000. for i in range(num_inference_steps)]))
        dt = 1.0 / num_inference_steps
        for i, t in enumerate(tqdm(timesteps[1:], total=num_inference_steps - 1)):

            # We'll skip the final iteration
            if i >= num_inference_steps - 1:
                continue

            # t = timesteps[i]

            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            # latent_model_input = self.unlearned_model.scheduler.scale_model_input(latent_model_input, t)

            # Predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
            latents = latents - dt*noise_pred

            # Store
            intermediate_latents.append(latents)

        return torch.cat(intermediate_latents)