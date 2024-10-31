#!/bin/env python

import torch
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL

# Load the SD1.5 VAE (ensure it is compatible with your SD1.5 version)
# IF you have a cached version, this will work.
# If you dont have a cached version... sorry, you'll have to find
# a current model name to use
vae_model_name = "runwayml/stable-diffusion-v1-5"
vae_model = AutoencoderKL.from_pretrained(vae_model_name, subfolder="vae")
vae_model.eval()

input_image = Image.open("input.png").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((512, 512)),   # Ensure your image matches the VAE's input size
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Scale to match VAE requirements
])
input_tensor = transform(input_image).unsqueeze(0)  # Add batch dimension

# Encode the image
with torch.no_grad():
    encoded = vae_model.encode(input_tensor).latent_dist.sample()

print("encoded tensor shape:" , encoded.shape)

def savelatent(latents,fname):
    with torch.no_grad():
        image = vae_model.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).astype("uint8")
    pil_image = Image.fromarray(image)

    pil_image.save(fname+".png")

savelatent(encoded,"output-sd")
