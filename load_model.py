import copy 
import torch
from diffusers import DPMSolverMultistepScheduler, LMSDiscreteScheduler, DDIMScheduler, StableDiffusion3Pipeline
from diffusers import FluxPipeline
from peft import PeftModel
from safetensors.torch import load_file

from eval_utils.models.sd.pipeline_modified_sd import ModifiedStableDiffusionPipeline
from eval_utils.models.flow.pipeline_modified_rf import ModifiedRectifiedFlowPipeline

from diffusers import StableDiffusionXLPipeline

def load_model(model_type="sdxl", target_ckpt=None, device="cuda", model_id_or_path=None, return_base_model=False, attack_type="i2p_nudity", target_ckpt_2=None, ddim=False):
    base_model_id = None
    print('model_type: ', model_type)
    if model_type == "sd":
        model_id = "CompVis/stable-diffusion-v1-4" if model_id_or_path is None else model_id_or_path
        base_model_id = "CompVis/stable-diffusion-v1-4"
    elif model_type == "sdxl":
        model_id = "stabilityai/stable-diffusion-xl-base-1.0" if model_id_or_path is None else model_id_or_path
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    elif model_type == "sd3":
        model_id = "stabilityai/stable-diffusion-3-medium-diffusers" if model_id_or_path is None else model_id_or_path
        base_model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
    elif model_type == "flux":
        model_id = "black-forest-labs/FLUX.1-dev" if model_id_or_path is None else model_id_or_path
        base_model_id = "black-forest-labs/FLUX.1-dev"
    else:
        model_id = "XCLIU/2_rectified_flow_from_sd_1_5" if model_id_or_path is None else model_id_or_path
        base_model_id = "XCLIU/2_rectified_flow_from_sd_1_5"

    print('model_id: ', model_id)

    # print('model id: ', model_id)    
    if ddim:
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    else:
        scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    # if "unlearndiff" in attack_type and model_type == "sd":
    # scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    
    if model_type == "sd":
        model = ModifiedStableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler)
    elif model_type == "sdxl":
        model = StableDiffusionXLPipeline.from_pretrained(model_id, scheduler=scheduler)
    elif model_type == "sd3":
        model = StableDiffusion3Pipeline.from_pretrained(
            model_id, cache_dir=".", torch_dtype=torch.float16
        )
    elif model_type == "flux":
        print('Initializing Flux model')
        model = FluxPipeline.from_pretrained(model_id).to(device)
    else:
        model = ModifiedRectifiedFlowPipeline.from_pretrained(model_id, scheduler=scheduler)

    base_model = None
    if return_base_model:
        if "mace" in model_id:
            if model_type == "sd":
                base_model = ModifiedStableDiffusionPipeline.from_pretrained(base_model_id, scheduler=scheduler)
            else:
                base_model = ModifiedRectifiedFlowPipeline.from_pretrained(base_model_id, scheduler=scheduler)
        else:
            base_model = copy.deepcopy(model)
    
    if target_ckpt is not None:
        if "lora" in target_ckpt and model_type == "sd" :
            print('target ckpt: ', target_ckpt)
            model.load_lora_weights(target_ckpt)
            # model.unet.load_attn_procs(target_ckpt)
        elif "lora" in target_ckpt and ( model_type == "sd3" or model_type == "flux" ) :
            print('Loading Flow Matching models')
            model.transformer = PeftModel.from_pretrained(model.transformer, target_ckpt)
            model.transformer.set_adapter("default")
        elif "safetensors" in target_ckpt:
            ckpt = load_file(target_ckpt)
            if model_type == "sd":
                model.unet.load_state_dict(ckpt, strict=False)
            else:
                model.transformer.load_state_dict(ckpt, strict=False)
        else:
            ckpt = torch.load(target_ckpt, map_location="cpu", weights_only=True)
                
            weight_initial = list(ckpt.keys())[0]
            if "text_model" in weight_initial or "text_encoder" in weight_initial:
                model.text_encoder.load_state_dict(ckpt, strict=False)
            else:
                model.unet.load_state_dict(ckpt)
    if target_ckpt_2 is not None:
        ckpt = torch.load(target_ckpt_2, map_location="cpu")

        weight_initial = list(ckpt.keys())[0]
        if "text_model" in weight_initial or "text_encoder" in weight_initial:
            model.text_encoder.load_state_dict(ckpt, strict=False)
        else:
            model.unet.load_state_dict(ckpt)

    # import ipdb; ipdb.set_trace()
    
    if model_type == "sd":
        model.safety_checker = None
        model.feature_extractor = None
        model.requires_safety_checker = False

        model.vae.to(device)
        model.text_encoder.to(device)
        model.clip_model.to(device)
        model.unet.to(device)
        model.unet.train()
        
        if return_base_model:
            base_model.safety_checker = None
            base_model.feature_extractor = None
            base_model.requires_safety_checker
            
            base_model.vae.to(device)
            base_model.text_encoder.to(device)
            base_model.clip_model.to(device)
            base_model.unet.to(device)
            base_model.unet.train()
    elif model_type == "flux":
        model.safety_checker = None
        model.vae.requires_grad_(False)
        model.text_encoder.requires_grad_(False)
        model.text_encoder_2.requires_grad_(False)
        
        model.to(dtype=torch.bfloat16)
        model.transformer.eval()
    else:
        model.safety_checker = None
        model.vae.requires_grad_(False)
        model.text_encoder.requires_grad_(False)
        model.text_encoder_2.requires_grad_(False)
        model.text_encoder_3.requires_grad_(False)

        model.to(device)
        model.transformer.eval()
    
    generator = torch.Generator(device=device)
    return model, generator, base_model