import os
import argparse
import pandas as pd
import json 
import torch

from PIL import Image

from utils.main_utils import Logger, read_json, dummy
from load_model import load_model

import numpy as np
import torch.nn.functional as F

import json

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
    all_imgdir = os.path.join(args.save_dir)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(all_imgdir, exist_ok=True)
    
    log_f = os.path.join(args.save_dir, "logs.txt")
    logger = Logger(log_f)
    

    logger.log("All configurations provided:")
    for arg in vars(args):
        logger.log(f"{arg}: {getattr(args, arg)}")
        
    rm_cols = ["inappropriate_percentage", "nudity_percentage", "q16_percentage", "sd_safety_percentage", "lexica_url"]
    
    model, gen, base_model = load_model(args.model_type, args.target_ckpt, args.device, args.model_id_or_path, True, args.attack_type, target_ckpt_2=args.target_ckpt_2)
    
    ################################################
    negative_prompt_space = None
    negative_prompt = ""
    if args.safree :
        if args.category == "nudity":
            negative_prompt_space = [
                args.category
            ]
        elif args.category == "church":
            negative_prompt_space = ["Church"]
        elif args.category == "Van Gogh":
            negative_prompt_space = ["Van Gogh"]
        elif args.category == "Picasso":
            negative_prompt_space = ["Pablo Picasso"]
        else:
            negative_prompt_space = [args.category]
        negative_prompt = ", ".join(negative_prompt_space)
        
        if args.latent_re_attention:
            from utils.free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d
            
            freeu_hyps = args.freeu_hyp.split('-')
            b1, b2, s1, s2 = float(freeu_hyps[0]), float(freeu_hyps[1]), float(freeu_hyps[2]), float(freeu_hyps[3])
            
            register_free_upblock2d(model, b1=b1, b2=b2, s1=s1, s2=s2)
            register_free_crossattn_upblock2d(model, b1=b1, b2=b2, s1=s1, s2=s2)
    
    dataset = load_dataset(args.data, args.category, rm_cols)
    
    i = 0
    log = []
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
        
        try:
            seed = data.evaluation_seed if hasattr(data,'evaluation_seed') else data.sd_seed
        except:
            seed = 42
            
        
        logger.log(f"Seed: {seed}, Iter: {_iter}, Case#: {case_num}: target prompt: {target_prompt}, clip prompt: {clip_prompt}")
        # check if data is broken
        if not isinstance(target_prompt, str) or not isinstance(seed, int) or not isinstance(guidance, (int, float)):
            continue
        
        with torch.no_grad():
            imgs, _ = model(
                target_prompt,
                negative_prompt="",
                height=512,
                width=512,
                num_inference_steps=50,
                guidance_scale=guidance,
                generator=torch.Generator().manual_seed(seed),
                safree_dict={"re_attn_t": [int(tr) for tr in args.re_attn_t.split(",")],
                                "alpha": args.sf_alpha,
                                "logger": logger,
                                "safree": args.safree,
                                "svf": args.self_validation_filter,
                                "lra": args.latent_re_attention,
                                "up_t": args.up_t,
                                "category": args.category
                                },              
            )
                      
        _save_path = os.path.join(all_imgdir, f"{case_num}.png")
        os.makedirs(os.path.dirname(_save_path), exist_ok=True)
        imgs[0].save(_save_path)
        
        log.append({
            'prompt': target_prompt,
            'img_path': _save_path
        })

        i += 1
    with open(f"{all_imgdir}/log_{args.id}.json", 'w') as f:
        json.dump(log, f)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/tmp_prompt.csv")
    parser.add_argument('--model_id_or_path', type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="./results/tmp")
    parser.add_argument("--model_type", type=str, default="sd")
    parser.add_argument("--num_samples", type=int, default=1, help="number of images to generate with SD")
    parser.add_argument("--target_ckpt", type=str, default=None, help="target checkpoint path")
    parser.add_argument("--target_ckpt_2", type=str, default=None, help="target checkpoint path 2")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    
    parser.add_argument("--category", type=str, default="artist")
    parser.add_argument("--nudity_thr", default=0.6, type=float)
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
    
    parser.add_argument("--config", default="sample_config.json", type=str, help="config file path")
    parser.add_argument("--device", default="cuda:0", type=str, help="first gpu device")
    
    parser.add_argument("--id", type=int, default=0)

    args = parser.parse_args()
    main(args)
