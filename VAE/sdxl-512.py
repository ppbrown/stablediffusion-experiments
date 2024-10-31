import torch
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the SDXL VAE directly and move it to the GPU
vae_model_name = "stabilityai/sdxl-vae"
vae_model = AutoencoderKL.from_pretrained(vae_model_name)
vae_model.to(device)
vae_model.eval()

# Load and preprocess the input image
input_image = Image.open("input.png").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to 512x512 for consistency
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Coordinate transformation for VAE input
])
input_tensor = transform(input_image).unsqueeze(0).to(device)  # Add batch dimension and move to device

# Encode the image
with torch.no_grad():
    encoded = vae_model.encode(input_tensor).latent_dist.sample()

def savelatent(latents, fname):
    with torch.no_grad():
        decoded_image = vae_model.decode(latents).sample

    decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)  # Undo normalization
    decoded_image = decoded_image.cpu().permute(0, 2, 3, 1).numpy()[0]  # Move to CPU and format for PIL
    decoded_image = (decoded_image * 255).astype("uint8")
    pil_image = Image.fromarray(decoded_image)

    pil_image.save(f"{fname}.png")

savelatent(encoded, "output-xl512")
