import os
import numpy as np
import torch.nn.functional as F
from transformers import (
    CLIPVisionModelWithProjection,
    CLIPTextModelWithProjection,
    AutoTokenizer,
    CLIPImageProcessor
)

import torch

def calculate_retain_score(model, gen, dataset, device, args, logger, all_imgdir, negative_prompt=None, negative_prompt_space=None):
    clip_score = []
    
    vmodel = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device)
    tmodel = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device)
    img_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    
    
    print('Gen: ', gen)
    
    for _iter, data in dataset.iterrows():
        # MMA-diffusion
        if "adv_prompt" in data:
            target_prompt = data['adv_prompt']
            case_num = _iter
        # Concept removal
        elif "sensitive prompt" in data:
            target_prompt = data["sensitive prompt"]
            case_num = _iter
        elif "prompt" in data:
            target_prompt = data["prompt"]
            try:
                case_num = data["case_number"]
            except:
                case_num = _iter
            
        clip_prompt = None
        try:
            clip_prompt = data["orig_prompt"]
        except:
            pass
        
        guidance = data.guidance if hasattr(data,'guidance') else 7.5
        
        # if args.model_type == "rf":
        #     guidance = 1.5
        
        # borrowed from RECE repo
        try:
            seed = data.evaluation_seed if hasattr(data,'evaluation_seed') else data.sd_seed
        except:
            seed = 42
            
        # if args.model_type == "rf":
        #     guidance = 1.5
        
        logger.log(f"Seed: {seed}, Iter: {_iter}, Case#: {case_num}: target prompt: {target_prompt}, clip prompt: {clip_prompt}")
        # check if data is broken
        if not isinstance(target_prompt, str) or not isinstance(seed, int) or not isinstance(guidance, (int, float)):
            continue

        # import ipdb; ipdb.set_trace()

        imgs, _ = model(
            target_prompt,
            clip_prompt=clip_prompt,
            num_images_per_prompt=args.num_samples,
            guidance_scale=guidance,
            num_inference_steps=args.num_inference_steps,
            negative_prompt=negative_prompt,
            negative_prompt_space=negative_prompt_space,
            height=512,
            width=512,
            generator=gen.manual_seed(seed),
            safree_dict={"re_attn_t": [int(tr) for tr in args.re_attn_t.split(",")],
                            "alpha": args.sf_alpha,
                            "logger": logger,
                            "safree": args.safree,
                            "svf": args.self_validation_filter,
                            "lra": args.latent_re_attention,
                            "up_t": args.up_t,
                            "category": args.category
                            },                
            clip_guidance_scale=args.clip_guidance_scale
        )
        
        
        tinputs = tokenizer([target_prompt], return_tensors="pt", padding=True, truncation=True).to(device)
        tembeds = tmodel(**tinputs).text_embeds
        tmebeds = F.normalize(tembeds, dim=-1)
        
        vinputs = img_processor(imgs[0], return_tensors="pt").to(device)
        vembeds = vmodel(**vinputs).image_embeds
        vembeds = F.normalize(vembeds, dim=-1)
        
        score = (tmebeds @ vembeds.T)[0].item()
        clip_score.append(score)
        
        
        _save_path = os.path.join(all_imgdir, f"{case_num}.png")
        imgs[0].save(_save_path)

    avg_score = np.mean(clip_score)
    return avg_score


def calculate_retain_score_sd3(model, gen, dataset, device, args, logger, all_imgdir, negative_prompt=None, negative_prompt_space=None):
    clip_score = []
    
    vmodel = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device)
    tmodel = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device)
    img_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    
    
    print('Gen: ', gen)
    
    for _iter, data in dataset.iterrows():
        # MMA-diffusion
        if "adv_prompt" in data:
            target_prompt = data['adv_prompt']
            case_num = _iter
        # Concept removal
        elif "sensitive prompt" in data:
            target_prompt = data["sensitive prompt"]
            case_num = _iter
        elif "prompt" in data:
            target_prompt = data["prompt"]
            try:
                case_num = data["case_number"]
            except:
                case_num = _iter
            
        clip_prompt = None
        try:
            clip_prompt = data["orig_prompt"]
        except:
            pass
        
        guidance = data.guidance if hasattr(data,'evaluation_guidance') else 7.5
        
        # if args.model_type == "rf":
        #     guidance = 1.5
        
        # borrowed from RECE repo
        try:
            seed = data.evaluation_seed if hasattr(data,'evaluation_seed') else data.sd_seed
        except:
            seed = 42
            
        # if args.model_type == "rf":
        #     guidance = 1.5
        
        logger.log(f"Seed: {seed}, Iter: {_iter}, Case#: {case_num}: target prompt: {target_prompt}, clip prompt: {clip_prompt}")
        # check if data is broken
        if not isinstance(target_prompt, str) or not isinstance(seed, int) or not isinstance(guidance, (int, float)):
            continue

        # import ipdb; ipdb.set_trace()

        imgs = model(
            target_prompt,
            negative_prompt="",
            num_inference_steps=28,
            guidance_scale=guidance,
            generator=torch.Generator().manual_seed(seed),
        ).images
        
        
        tinputs = tokenizer([target_prompt], return_tensors="pt", padding=True, truncation=True).to(device)
        tembeds = tmodel(**tinputs).text_embeds
        tmebeds = F.normalize(tembeds, dim=-1)
        
        vinputs = img_processor(imgs[0], return_tensors="pt").to(device)
        vembeds = vmodel(**vinputs).image_embeds
        vembeds = F.normalize(vembeds, dim=-1)
        
        score = (tmebeds @ vembeds.T)[0].item()
        clip_score.append(score)
        
        
        _save_path = os.path.join(all_imgdir, f"{case_num}.png")
        imgs[0].save(_save_path)

    avg_score = np.mean(clip_score)
    return avg_score
