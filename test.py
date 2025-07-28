import torch
from diffusers import StableDiffusionPipeline
import os 
os.makedirs('tmp', exist_ok=True)
model = StableDiffusionPipeline.from_pretrained("/scratch/nkusumba/eraseflow_multiconcept/celeb1/checkpoints/erasure_multiconcept_celeb1_run5/celeb1_model").to("cuda:1", dtype=torch.float16)
model.safety_checker=None
model.load_lora_weights("/scratch/nkusumba/eraseflow_multiconcept/celeb1/checkpoints/erasure_multiconcept_adamdriver_celeb1_run6/checkpoint_epoch2/pytorch_lora_weights.safetensors")
# model.fuse_lora()
# model.unload_lora_weights()

# model.save_pretrained(
#     "/scratch/nkusumba/eraseflow_multiconcept/celeb1/checkpoints/erasure_multiconcept_celeb1_run5/celeb1_model"
# )
# prompts = ["a portrait of Melania Trump"]*50 + ["a sketch of Melania Trump'"]*50 + ["an oil painting of Melania Trump"]*50 + ["Melania Trump in an official photo"]*50 + ["an image capturing Melania Trump at a public event"]*50

# prompts = ["an oil painting of Melania Trump"]*10 + ["Melania Trump in an official photo"]*10 + ["an image capturing Melania Trump at a public event"]*10

prompts = ["an image capturing Adam Driver at a public event"]*50


for i, prompt in enumerate(prompts):
    # prompt = "Melania Trumpt"
    img = model(
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        # generator=torch.Generator().manual_seed(42)
    ).images[0]

    img.save(f"tmp/adam_driver_{i}.png")


prompts = ["an image capturing Melania at a public event"]*50


for i, prompt in enumerate(prompts):
    # prompt = "Melania Trumpt"
    img = model(
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        # generator=torch.Generator().manual_seed(42)
    ).images[0]

    img.save(f"tmp/adriana_lima_{i}.png")

prompt = "Mel Gibson"
img = model(
    prompt,
    num_inference_steps=50,
    guidance_scale=7.5,
    generator=torch.Generator().manual_seed(42)
).images[0]

img.save("tmp/mel_gibson.png")


prompt = "Rachel McAdams"
img = model(
    prompt,
    num_inference_steps=50,
    guidance_scale=7.5,
    generator=torch.Generator().manual_seed(42)
).images[0]

img.save("tmp/rachel_mcadams.png")

prompt = "Matt Damon"
img = model(
    prompt,
    num_inference_steps=50,
    guidance_scale=7.5,
    generator=torch.Generator().manual_seed(42)
).images[0]

img.save("tmp/matt_damon.png")


prompt = "Alec Baldwin"
img = model(
    prompt,
    num_inference_steps=50,
    guidance_scale=7.5,
    generator=torch.Generator().manual_seed(42)
).images[0]

img.save("tmp/alec_baldwin.png")