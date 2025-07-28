import copy
from tqdm import tqdm
import torch
import torch.nn.functional as F

from torchvision import transforms as tfms

class TrajDiff:
    def __init__(self, base_model, unlearned_model, device="cuda"):
        self.base_model = base_model
        self.unlearned_model = unlearned_model
        self.device = device
    
    # Calculates the trajectory difference between base_model and the unlearned model for a target_prompt.
    def compute(self, target_prompt, erase_id, **kwargs):
        kwargs_base_model = kwargs
        if "SAFREE" in erase_id:
            kwargs_base_model = copy.deepcopy(kwargs)
            kwargs_base_model["safree_dict"]["safree"] = False
            kwargs_base_model["safree_dict"]["svf"] = False
            kwargs_base_model["safree_dict"]["lra"] = False
            kwargs_base_model["negative_prompt"] = ""
        _, base_latents = self.base_model(target_prompt, **kwargs_base_model)
        _, unlearned_latents = self.unlearned_model(target_prompt, latents=base_latents[0], **kwargs)
       
        diff = 0
        for i in range(len(unlearned_latents)):
            diff += torch.norm(base_latents[i] - unlearned_latents[i], p=2, dim=(-1, -2, -3))
            # diff += F.mse_loss(base_latents[i], unlearned_latents[i])
        return diff
    
    # Calculates the trajectory difference between a random latent and the inverted latent for a target_prompt.
    def compute_v2(self, target_image, target_prompt, erase_id, num_inference_steps=50, **kwargs):
        kwargs_base_model = kwargs
        # if "SAFREE" in erase_id:
        #     kwargs_base_model = copy.deepcopy(kwargs)
        #     kwargs_base_model["safree_dict"]["safree"] = False
        #     kwargs_base_model["safree_dict"]["svf"] = False
        #     kwargs_base_model["safree_dict"]["lra"] = False
            
        inverted_latent = self.unlearned_model.invert(target_image, target_prompt, num_inference_steps=num_inference_steps, device=self.device)[-1]    
        
        _, base_latents = self.unlearned_model(target_prompt, **kwargs_base_model)
        _, unlearned_latents = self.unlearned_model(target_prompt, latents=inverted_latent.unsqueeze(0), **kwargs)

        diff = 0
        for i in range(len(unlearned_latents)):
            diff += torch.norm(base_latents[i] - unlearned_latents[i], p=2, dim=(-1, -2, -3))
        
        return diff  
    
    def compute_v3(self, target_image, target_prompt, erase_id, num_inference_steps=50, **kwargs):
        inverted_latent = self.base_model.invert(target_image, target_prompt, num_inference_steps=num_inference_steps, device=self.device)[-1]    
        
        _, base_latents = self.base_model(target_prompt, latents=inverted_latent.unsqueeze(0), **kwargs)
        _, unlearned_latents = self.unlearned_model(target_prompt, latents=inverted_latent.unsqueeze(0), **kwargs)

        diff = 0
        for i in range(len(unlearned_latents)):
            diff += torch.norm(base_latents[i] - unlearned_latents[i], p=2, dim=(-1, -2, -3))
        
        return diff 

    
    ## Inversion
    @torch.no_grad()
    def invert(
        self,
        start_latents,
        prompt,
        guidance_scale=3.5,
        num_inference_steps=80,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt="",
        device="cuda",
    ):

        # Encode prompt
        text_embeddings = self.unlearned_model._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # Latents are now the specified start latents
        latents = start_latents.clone()

        # We'll keep a list of the inverted latents as the process goes on
        intermediate_latents = []

        # Set num inference steps
        self.unlearned_model.scheduler.set_timesteps(num_inference_steps, device=device)

        # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
        timesteps = reversed(self.unlearned_model.scheduler.timesteps)

        for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

            # We'll skip the final iteration
            if i >= num_inference_steps - 1:
                continue

            t = timesteps[i]

            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.unlearned_model.scheduler.scale_model_input(latent_model_input, t)

            # Predict the noise residual
            noise_pred = self.unlearned_model.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
            next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
            alpha_t = self.unlearned_model.scheduler.alphas_cumprod[current_t]
            alpha_t_next = self.unlearned_model.scheduler.alphas_cumprod[next_t]

            # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
            latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
                1 - alpha_t_next
            ).sqrt() * noise_pred

            # Store
            intermediate_latents.append(latents)

        return torch.cat(intermediate_latents)
