from diffusers import DiffusionPipeline
import torch
from diffusers import DDPMScheduler, UNet2DModel, FlaxKarrasVeScheduler
from PIL import Image
import torch
import numpy as np 
from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDPMScheduler, HeunDiscreteScheduler, KarrasVeScheduler, DPMSolverMultistepScheduler, LMSDiscreteScheduler, DDIMScheduler #FlaxKarrasVeOutput
from tqdm.auto import tqdm
import os 
from torch import nn 
import torch.nn.functional as F 
import json
import glob
import argparse
from transformers import AutoTokenizer
import pandas as pd 
import open_clip 


def parse_args():
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="EraseFlow/NUDITY" # "stabilityai/stable-diffusion-2-1" #"/scratch/aj3281/DCR/DCR/sd-finetuned-org_parameters_instancelevel_blip_nodup_laion/checkpoint/", #"/scratch/aj3281/diffusers/examples/text_to_image/sd-pokemon-model/old_model/",
    )
    parser.add_argument(
        "--guidance",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--start_iter",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--end_iter",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="",
    )
    parser.add_argument(
        "--discriminator_guidance_scale",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--guidance_loss_scale",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--guidance_loss_scale_towards",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="DPM",
    )
    parser.add_argument(
        "--concept_erasure",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--concept_towards",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
    )
    return parser.parse_args()
    
args = parse_args()

guidance = args.guidance


vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", use_safetensors=True) #stabilityai/stable-diffusion-2-1
tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False) #CLIPTokenizer.from_pretrained(pretrained_path, subfolder="tokenizer")

text_encoder = CLIPTextModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="text_encoder", use_safetensors=True
)
unet = UNet2DConditionModel.from_pretrained(
   args.pretrained_model_name_or_path, subfolder="unet", use_safetensors=True
)
scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", use_safetensors=True)

scheduler_heun =  DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")  

torch_device = "cuda"
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)

vae.requires_grad = False 
unet.requires_grad = False 
text_encoder.requires_grad = False
tokenizer.requires_grad = False


for param in unet.parameters():
    param.requires_grad = False 

for param in vae.parameters():
    param.requires_grad = False 

for param in text_encoder.parameters():
    param.requires_grad = False 



device = torch_device

height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
num_inference_steps = 50  # Number of denoising steps
guidance_scale = args.guidance_scale  # Scale for classifier-free guidance
generator = torch.manual_seed(42)#.to(torch_device)  # Seed generator to create the initial latent noise
torch.cuda.manual_seed_all(42)
batch_size = 1
clip = False 
# torch.autograd.set_detect_anomaly(True)



if clip == True:
    
    ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-g-14", #"ViT-L-14",#
            pretrained="laion2b_s12b_b42k",#"laion2b_s32b_b82k",#
            device=torch_device,
        )
    ref_tokenizer = open_clip.get_tokenizer("ViT-g-14")#"ViT-L-14")
    ref_clip_preprocess.transforms.pop(2)
    ref_clip_preprocess.transforms.pop(2)
    ref_model.requires_grad = False 
    ref_tokenizer.requires_grad = False
    
    for param in ref_model.parameters():
        param.requires_grad = False 
    # for param in text_encoder.parameters():
    #     param.requires_grad = False 
    

outdir = args.outdir 

if not os.path.exists(outdir):
    os.makedirs(outdir)

if not os.path.exists(outdir + "/Images/"):
    os.makedirs(outdir + "/Images/")


filename = outdir + "/latents.npy"
n_samples = args.end_iter


def loss_fn(x):
    # x = discriminator(x.to(torch_device)) # model outputs
    
    x = F.softmax(x)[:,1] # to probs (10 classes)
    # x = x * torch.arange(2)[None, :].to(torch_device) # to a score, many ways to do this!
    return -x.mean()


#@title sampling function (with guidance and some extra tricks)
def sample(prompt, guidance_loss_scale, guidance_scale=7.5,
         negative_prompt = None, 
         num_inference_steps=50, start_latents = None,
         early_stop = None, discriminator_guidance_scale=2.0, cfg_norm=False, cfg_decay=False): #cfg_decay was originally True cfg_norm was also true, early stop=20

    
    # If no starting point is passed, create one
    if start_latents is None:
      # start_latents = torch.randn((len(prompt), 4, 64, 64), device=torch_device)
        latents = torch.randn((len(prompt), unet.config.in_channels, height // 8, width // 8),
            # generator=generator,
            device=torch_device,)
    
    scheduler.set_timesteps(num_inference_steps)
    scheduler_heun.set_timesteps(num_inference_steps)

    # Encode the prompt
    # text_embeddings = pipe._encode_prompt(prompt, device, 1, True, negative_prompt)
    # print("length of prompts", len(prompt))
    
    
    with torch.no_grad():
        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        # text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
        
        max_length = text_input.input_ids.shape[-1]
    
        if args.negative_prompt is not None or guidance_loss_scale != 0: 
            if args.negative_prompt is None:
                uncond_input_neg = tokenizer([args.concept_erasure] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
            else:
                uncond_input_neg = tokenizer([args.negative_prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
                
            uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
            uncond_embeddings_neg = text_encoder(uncond_input_neg.input_ids.to(torch_device))[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings, uncond_embeddings_neg])
                    
        else:
            uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
    # Create our random starting point
    latents = latents.clone()
    latents *= scheduler.init_noise_sigma

    # Prepare the scheduler
    scheduler.set_timesteps(num_inference_steps, device=torch_device)
    scheduler_heun.set_timesteps(num_inference_steps, device=torch_device)

    # Loop through the sampling timesteps 
    for i, (t, t_heun) in tqdm(enumerate(zip(scheduler.timesteps, scheduler_heun.timesteps))):

        # if i > early_stop: guidance_loss_scale = 0 # Early stop (optional)

        # sigma = scheduler_heun.sigmas[i]

        # Set requires grad
        if guidance_loss_scale != 0: latents = latents.detach().requires_grad_()

        if args.negative_prompt is not None or guidance_loss_scale != 0: 
            latent_model_input = torch.cat([latents] * 3)
        else:
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)

        # Apply any scaling required by the scheduler
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # text_embeddings.requires_grad = False 
        
        # predict the noise residual with the unet
        if guidance_loss_scale != 0:
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        else:
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform CFG
        cfg_scale = guidance_scale
        if cfg_decay: cfg_scale = 1 + guidance_scale * (1-i/num_inference_steps)

        if args.negative_prompt is not None or guidance_loss_scale != 0: 
            noise_pred_uncond, noise_pred_text, noise_pred_neg = noise_pred.chunk(3)
        else:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        
        if guidance_loss_scale != 0:
            loss = -1*guidance_loss_scale*torch.exp(-torch.norm(noise_pred_text - noise_pred_neg)/float(args.sigma))
            # loss = -1*guidance_loss_scale*(torch.norm(noise_pred_text - noise_pred_neg)) # WITHOUT EXP
            cond_grad = torch.autograd.grad(loss, latents, retain_graph=False, create_graph=False)[0]
            latents = latents.detach() - discriminator_guidance_scale * cond_grad #* (sigma**2 + 1)
            # noise_pred_text = noise_pred_text + discriminator_guidance_scale * cond_grad
            torch.cuda.empty_cache()              
            

        if args.negative_prompt is not None:
            # print(args.negative_prompt)    
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_neg) 
            # noise_pred = noise_pred_neg + cfg_scale * (noise_pred_text - noise_pred_neg) 
            # noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond) - cfg_scale * (noise_pred_neg - noise_pred_uncond)
        else:
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)

        # Normalize (see https://enzokro.dev/blog/posts/2022-11-15-guidance-expts-1/)
        if cfg_norm:
          noise_pred = noise_pred * (torch.linalg.norm(noise_pred_uncond) / torch.linalg.norm(noise_pred))          

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        del noise_pred 
        

    return latents.detach()
    

data = pd.read_csv(args.prompt).to_numpy() 

try: 
    prompts = pd.read_csv(args.prompt)['prompt'].to_numpy()
except:
    prompts = pd.read_csv(args.prompt)['adv_prompt'].to_numpy()
    

try:
    seeds = pd.read_csv(args.prompt)['evaluation_seed'].to_numpy()
except:
    try:
        seeds = pd.read_csv(args.prompt)['sd_seed'].to_numpy()
    except:
        seeds = [42 for i in range(len(prompts))]

try: 
    guidance_scales = pd.read_csv(args.prompt)['evaluation_guidance'].to_numpy()
except:
    try:
        guidance_scales = pd.read_csv(args.prompt)['sd_guidance_scale'].to_numpy()
    except:
        guidance_scales = [7.5 for i in range(len(prompts))]

import time

i = args.start_iter
n_samples = len(data)

avg_time = 0

while i < n_samples and i< args.end_iter:
    
    torch.cuda.empty_cache()
    try:
        
        seed = int(seeds[i])#int(data[i][3])
    except:
        seed = int(seeds[i][0])
    prompt = [prompts[i]] #[data[i][2]]
    guidance_scale = float(guidance_scales[i])#7.5 #data[i][3]
    print(prompt, seed, guidance_scale)

    generator = torch.manual_seed(seed)  # Seed generator to create the initial latent noise
    torch.cuda.manual_seed_all(seed)
    

    if i+ batch_size > n_samples:
        batch_size = n_samples - i

    start_time = time.time()
    
    latents2 = sample(prompt, guidance_loss_scale=args.guidance_loss_scale, guidance_scale = guidance_scale, discriminator_guidance_scale=args.discriminator_guidance_scale)

    with torch.no_grad():
        latents2 = 1./vae.config.scaling_factor * latents2 #0.18215
        
        images = vae.decode(latents2).sample

        del latents2 
        
        for j in range(batch_size):
            image = images[j]
            image = (image / 2 + 0.5).clamp(0, 1).squeeze()
            image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
            image = Image.fromarray(image)
            
            end_time = time.time()
            avg_time += end_time - start_time
            
            image.save(f"{outdir}/Images/{i+j}.png")            

    i += batch_size 

avg_time = avg_time/float(i)
print(avg_time)