import os
import argparse
import pandas as pd
import json 

from PIL import Image

from utils.main_utils import Logger, read_json, dummy
from load_model import load_model

import os
import numpy as np
import torch.nn.functional as F
from transformers import (
    CLIPVisionModelWithProjection,
    CLIPTextModelWithProjection,
    AutoTokenizer,
    CLIPImageProcessor
)

def merge_dicts_append(dict1, dict2):    
    for key in dict2:
        assert key in dict1.keys()        
        if isinstance(dict2[key], list):
            dict1[key].append(dict2[key])
        else:
            dict1[key] += [dict2[key]]        
    return dict1

def load_dataset(data_f, category, rm_cols, debug=False):
    # Read CSV file
    dataset = pd.read_csv(data_f)
    # delete index column
    
    if 'Unnamed: 0' in dataset.columns:
        dataset = dataset.drop(columns=['Unnamed: 0'])  
    dataset["nudity_toxicity"] = 0.0
    
    print(f"{category} dataset size: {dataset.shape[0]}")
    return dataset

def main(args):
    all_imgdir = os.path.join(args.save_dir, "all", args.category)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(all_imgdir, exist_ok=True)
    
    log_f = os.path.join(args.save_dir, "logs.txt")
    logger = Logger(log_f)
    

    logger.log("All configurations provided:")
    for arg in vars(args):
        logger.log(f"{arg}: {getattr(args, arg)}")
        
    rm_cols = ["inappropriate_percentage", "nudity_percentage", "q16_percentage", "sd_safety_percentage", "lexica_url"]
    
    model, gen, base_model = load_model(args.model_type, args.target_ckpt, args.device, args.model_id_or_path, True if args.attack_type == "trajDiff" else False, args.attack_type, target_ckpt_2=args.target_ckpt_2)
        
    ################################################
    negative_prompt_space = None
    negative_prompt = ""
    if args.safree :
        negative_prompt_space = [
            "Sexual Acts", "Content Meant to Arouse Sexual Excitement", "Nudity",
            "Pornography", "Erotic Art", "Lustful", "Seductive", "Orgasmic", "Libido",
            "Kinky", "Sexual Orientation", "Sexual Attraction", "Sexual Intercourse",
            "Sexual Pleasure", "Sexual Fantasy", "Carnal Desires", "Sexual Gratification"
        ]
        negative_prompt = ", ".join(negative_prompt_space)
        
        if args.latent_re_attention:
            from utils.free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d
            
            freeu_hyps = args.freeu_hyp.split('-')
            b1, b2, s1, s2 = float(freeu_hyps[0]), float(freeu_hyps[1]), float(freeu_hyps[2]), float(freeu_hyps[3])
            
            register_free_upblock2d(model, b1=b1, b2=b2, s1=s1, s2=s2)
            register_free_crossattn_upblock2d(model, b1=b1, b2=b2, s1=s1, s2=s2)
    ################################################
    # clip_score = []
    
    # vmodel = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(args.device)
    # tmodel = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(args.device)
    # img_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    # tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    with open(args.data_path, "r") as f:
        target_prompts = f.readlines()

    for i, prompt in enumerate(target_prompts):
        j = i*args.nsamples

        test_prompts = (
            [f"a portrait of {prompt.strip()}"] * (args.nsamples//5)+
            [f"a sketch of {prompt.strip()}"] * (args.nsamples//5)+
            [f"an oil painting of {prompt.strip()}"] * (args.nsamples//5) +
            [f"{prompt.strip()} in an official photo"] * (args.nsamples//5) +
            [f"an image capturing {prompt.strip()} at a public event"] * (args.nsamples//5)
        )

        for _iter in range(len(test_prompts)):
            case_num = j + _iter
            
            guidance = 7.5
            
            logger.log(f"Iter: {_iter}, Case#: {case_num}: target prompt: {test_prompts[_iter]}")
            imgs, _ = model(
                test_prompts[_iter],
                clip_prompt=None,
                num_images_per_prompt=args.num_samples,
                guidance_scale=guidance,
                num_inference_steps=args.num_inference_steps,
                negative_prompt=negative_prompt,
                negative_prompt_space=negative_prompt_space,
                height=512,
                width=512,
                generator=gen.manual_seed(_iter),
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
            
            
            _save_path = os.path.join(all_imgdir, f"{case_num}.png")
            imgs[0].save(_save_path)
            
        # ipdb.set_trace()
    #     tinputs = tokenizer([args.target_prompt], return_tensors="pt", padding=True, truncation=True).to(args.device)
    #     tembeds = tmodel(**tinputs).text_embeds
    #     tmebeds = F.normalize(tembeds, dim=-1)
        
    #     vinputs = img_processor(imgs[0], return_tensors="pt").to(args.device)
    #     vembeds = vmodel(**vinputs).image_embeds
    #     vembeds = F.normalize(vembeds, dim=-1)
        
    #     score = (tmebeds @ vembeds.T)[0].item()
    #     clip_score.append(score)

    # avg_score = np.mean(clip_score)
    # print(avg_score)
    # with open(f"{args.save_dir}/average_clip_score.txt", "w") as f:
    #     f.write(f"Average CLIP Score:  {avg_score}")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id_or_path', type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="./results/tmp")
    parser.add_argument('--nsamples', type=int, default=200)
    parser.add_argument("--model_type", type=str, default="sd")
    parser.add_argument("--num-samples", type=int, default=1, help="number of images to generate with SD")
    parser.add_argument("--target_ckpt", type=str, default=None, help="target checkpoint path")
    parser.add_argument("--target_ckpt_2", type=str, default=None, help="target checkpoint path 2")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    
    parser.add_argument("--category", type=str, default="nudity")
    parser.add_argument("--valid_case_numbers", default="0,100000", type=str)
    parser.add_argument("--erase-id", type=str, default="esd")
    parser.add_argument("--name", type=str, default="SD")
    
    # SLD config
    parser.add_argument("--safe_level", type=str, default="MAX")

    # Safe + Free ? --> SAFREE!
    parser.add_argument("--safree", action="store_true")
    parser.add_argument("--self_validation_filter", "-svf", action="store_true")
    parser.add_argument("--latent_re_attention", "-lra", action="store_true")
    parser.add_argument("--sf_alpha", default=0.01, type=float)
    parser.add_argument("--re_attn_t", default="-1,1001", type=str)
    parser.add_argument("--freeu_hyp", default="1.0-1.0-0.9-0.2", type=str)
    parser.add_argument("--up_t", default=10, type=int)
    
    # Attack type?
    parser.add_argument("--attack_type", type=str, default="i2p")
    
    parser.add_argument("--clip_guidance_scale", type=float, default=0.0)
    
    parser.add_argument("--device", default="cuda:0", type=str, help="first gpu device")

    parser.add_argument("--data_path", default="/data/data/matt/gfn/GFN_Unlearn/data/celebs_retain.txt", type=str)
    
    args = parser.parse_args()
    main(args)