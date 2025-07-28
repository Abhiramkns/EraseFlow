import copy
from tqdm import tqdm
import torch
import torch.nn.functional as F

from torchvision import transforms as tfms

class TDR:
    def __init__(self, base_model, unlearned_model, device="cuda"):
        self.base_model = base_model
        self.unlearned_model = unlearned_model
        self.device = device
    
    # Calculates the trajectory difference between base_model and the unlearned model for a target_prompt.
    def compute(self, target_prompt, close_prompt, erase_id, **kwargs):
        kwargs_base_model = kwargs
        if "SAFREE" in erase_id:
            kwargs_base_model = copy.deepcopy(kwargs)
            kwargs_base_model["safree_dict"]["safree"] = False
            kwargs_base_model["safree_dict"]["svf"] = False
            kwargs_base_model["safree_dict"]["lra"] = False
            kwargs_base_model["negative_prompt"] = ""
        
        base_target_img, base_latents = self.base_model(target_prompt, **kwargs_base_model)
        unlearned_target_img, unlearned_latents = self.unlearned_model(target_prompt, latents=base_latents[0], **kwargs)
        target_diff = 0
        for i in range(len(unlearned_latents)):
            target_diff += torch.norm(base_latents[i] - unlearned_latents[i], p=2, dim=(-1, -2, -3))
            # diff += F.mse_loss(base_latents[i], unlearned_latents[i])
        
        base_unrelated_img, base_latents = self.base_model(close_prompt, **kwargs_base_model)
        unlearned_unrelated_img, unlearned_latents = self.unlearned_model(close_prompt, latents=base_latents[0], **kwargs)
        closer_diff = 0
        for i in range(len(unlearned_latents)):
            closer_diff += torch.norm(base_latents[i] - unlearned_latents[i], p=2, dim=(-1, -2, -3))
        
        
        res = target_diff/(closer_diff + 1e-6)
        
        return res, target_diff, closer_diff, base_target_img, unlearned_target_img, base_unrelated_img, unlearned_unrelated_img