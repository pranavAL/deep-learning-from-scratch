import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Sinusoidal Time Embedding Function
class SinusoidalTimeEmbedding(nn.Module):
  def __init__(self, time_embed_dim):
    super(SinusoidalTimeEmbedding, self).__init__()
    self.time_embed_dim = time_embed_dim

  def forward(self, t):
    device = t.device
    half_dim = self.time_embed_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = t[:, None] * emb[None, :]
    time_emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return time_emb  #Shape: [B, time_emb_dim]

# Convolutional Block
class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, time_emb_dim):
    super(ConvBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(out_channels)

    # Inject time embedding
    self.time_mlp = nn.Linear(time_emb_dim, out_channels)

  def forward(self, x, t_emb):
    # Conv + BatchNorm + ReLU
    x = F.relu(self.bn1(self.conv1(x)))

    # Add time embedding
    t_emb = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1) # Shape : [B, out_channels, 1, 1]
    x = x + t_emb # Inject time embedding

    # Conv + BatchNorm + ReLU
    x = F.relu(self.bn2(self.conv2(x)))
    return x

# Define the UNet architecture for diffusion denoising with sinusoidal time embedding
class UNet(nn.Module):
  def __init__(self, in_channels=1, out_channels=1, time_emb_dim=256):
    super(UNet, self).__init__()

    # Time embedding module (sinusoidal embedding)
    self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)

    # Encoder: Downsampling path
    self.down1 = ConvBlock(in_channels, 64, time_emb_dim)
    self.down2 = ConvBlock(64, 128, time_emb_dim)
    self.down3 = ConvBlock(128, 256, time_emb_dim)
    self.down4 = ConvBlock(256, 512, time_emb_dim)

    # Bottleneck
    self.bottleneck = ConvBlock(512, 1024, time_emb_dim)

    # Decoder: Upsampling path
    self.up1 = ConvBlock(512 + 512, 512, time_emb_dim)
    self.up2 = ConvBlock(256 + 256, 256, time_emb_dim)
    self.up3 = ConvBlock(128 + 128, 128, time_emb_dim)
    self.up4 = ConvBlock(64 + 64, 64, time_emb_dim)

    # Output Layer
    self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    # Downsampling (for max pooling)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    # Upsampling (for transposed Convolution)
    self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
    self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
    self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

  def forward(self, x, t):
    # Get time embedding using sinusoidal function
    t_emb = self.time_embedding(t)

    # Encoder
    d1 = self.down1(x, t_emb)                          # Shape: [B, 64, H, W]
    d2 = self.down2(self.pool(d1), t_emb)              # Shape: [B, 128, H/2, W/2]
    d3 = self.down3(self.pool(d2), t_emb)              # Shape: [B, 256, H/4, W/4]
    d4 = self.down4(self.pool(d3), t_emb)              # Shape: [B, 512, H/8, W/8]

    # Bottleneck
    bottleneck = self.bottleneck(self.pool(d4), t_emb) # Shape: [B, 1024, H/16, W/16]

    # Decoder with Skip Connection
    u1 = self.upconv1(bottleneck)                      # Shape: [B, 512, H/8, W/8]
    u1 = torch.cat([u1, d4], dim=1)                    # Concatention along channel axis
    u1 = self.up1(u1, t_emb)

    u2 = self.upconv2(u1)                              # Shape: [B, 256, H/4, W/4]
    u2 = torch.cat([u2, d3], dim=1)
    u2 = self.up2(u2, t_emb)

    u3 = self.upconv3(u2)                              # Shape: [B, 128, H/2, W/2]
    u3 = torch.cat([u3, d2], dim=1)
    u3 = self.up3(u3, t_emb)

    u4 = self.upconv4(u3)                              # Shape: [B, 64, H, w]
    u4 = torch.cat([u4, d1], dim=1)
    u4 = self.up4(u4, t_emb)

    # Output layer (denoised image)
    out = self.final_conv(u4)                          # Shape: [B, out_channels, H, W]
    return out

def get_cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule for smoothing the noise addition
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2)**2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0.0001, 0.9999)
    return betas

def forward_diffusion(x_0, t, alphas_cumprod, device):
    """
    Forward Diffusion Process: Adds noise to the original data.

    Args:
        x_0: Original data (batch_size, channels, height, width).
        t: Time step tensor (batch_size,).
        alphas_cumprod: Cumulative product of alphas (timesteps,)
    """
    noise = torch.randn_like(x_0).to(device) # Gaussian noise
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1, 1)
    x_t = sqrt_alphas_cumprod * x_0 + sqrt_one_minus_alphas_cumprod * noise
    return x_t, noise

def reverse_process(model, x_T, betas, alphas_cumprod, device, t):
    """
    Reverse diffusion process: Generates data by denoising from x_T to x_0.

    Args:
        model: Trained denoising model.
        x_T: Initial noisy data (batch_size, channels, height, width)
        betas: Beta schedule (timesteps,)
        alphas_cumprod: Cumulative product of alphas (timesteps,)
        timesteps: Number of timesteps
    """
    x = x_T

    t_tensor = torch.full((x.size(0),), t, dtype=torch.long).to(device)
    predicted_noise = model(x, t_tensor)
    beta = betas[t]
    alpha = 1 - beta
    alpha_cumprod = alphas_cumprod[t]
    sqrt_recip_alpha = 1 / torch.sqrt(alpha)

    x = sqrt_recip_alpha.view(-1, 1, 1, 1) * (x - (beta / torch.sqrt(1 - alpha_cumprod)).view(-1, 1, 1, 1) * predicted_noise)

    if t > 0:
        noise = torch.randn_like(x).to(device)
        sigma = torch.sqrt(beta).view(-1, 1, 1, 1)
        x = x + sigma * noise

    return x

def train_diffusion_model(model, dataloader, optimizer, betas, alphas_cumprod, device, timesteps, num_epochs=20):
    model.train()
    mse_loss = nn.MSELoss()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (data, _) in enumerate(dataloader):

            data = data.to(device)
            optimizer.zero_grad()

            # Sample random timesteps for each example in the batch
            t = torch.randint(0, timesteps, (data.size(0),), device=device)

            # Add noise to the data at timestep t
            x_t, noise = forward_diffusion(data, t, alphas_cumprod, device)

            # Predict the noise using the model
            noise_pred = model(x_t, t)

            # Compute loss between true and predicted noise
            loss = mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()


        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")


def generate_and_save_samples(model, betas, alphas_cumprod, device, timesteps, num_samples=16):
    """
    Generates samples using the trained diffusion model and save them as images.
    """
    model.eval()
    with torch.no_grad():
        # Start from pure noise
        x = torch.randn(num_samples, 1, 128, 128).to(device)

        frames = []
        for t in reversed(range(timesteps)):
            # Reverse diffusion step
            x = reverse_process(model, x, betas, alphas_cumprod, device, t)

            if t % (timesteps // 10) == 0 or t == timesteps - 1:  # Save 10 frames + the final result
                # Clamp and rescale the image
                x_show = torch.clamp((x + 1) / 2, 0, 1)

                # Create a grid of images
                grid = make_grid(x_show, nrow=8)
                np_grid = grid.cpu().numpy()

                # Convert to PIL Image
                img = Image.fromarray((np.transpose(np_grid, (1, 2, 0)) * 255).astype(np.uint8))

                # Add timestep text to the image
                draw = ImageDraw.Draw(img)
                font = ImageFont.load_default()
                draw.text((10, 10), f"Timestep: {t}/{timesteps-1}", font=font, fill=(255, 255, 255))

                frames.append(img)

        # Save as animated GIF
        frames[0].save("diffusion_process_with_timesteps.gif", save_all=True, append_images=frames[1:],
                        duration=500, loop=0)
        print("Animated GIF saved as 'diffusion_process_with_timesteps.gif'")

def main():
  batch_size = 64
  learning_rate = 2e-4
  num_epochs = 20
  timesteps = 1000
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Define cosine noise schedule
  betas = get_cosine_beta_schedule(timesteps)
  alphas = 1.0 - betas
  alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

  transform = transforms.Compose([
      transforms.Resize((128, 128)),
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
  ])

  train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

  model = UNet().to(device)
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  # Train the diffusion model
  train_diffusion_model(model, train_loader, optimizer, betas, alphas_cumprod, device, timesteps, num_epochs)

  # Generate and save samples
  generate_and_save_samples(model, betas, alphas_cumprod, device, timesteps, num_samples=16)

if __name__ == "__main__":
  main()
