"""
Diffusion Model architecture for metal artifact reduction.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal positional embeddings for diffusion timesteps."""
    
    def __init__(self, dim: int):
        """
        Args:
            dim: Embedding dimension
        """
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Generate sinusoidal embeddings for timesteps.
        
        Args:
            time: Tensor of timestep indices
            
        Returns:
            Positional embeddings
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """Convolutional block for U-Net."""
    
    def __init__(self, in_ch: int, out_ch: int):
        """
        Args:
            in_ch: Number of input channels
            out_ch: Number of output channels
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through block."""
        return self.conv(x)


class ConditionalUnet(nn.Module):
    """
    Conditional U-Net for diffusion model.
    Takes noisy image and condition tensors (clean image, masks).
    """
    
    def __init__(self, input_channels: int = 4, output_channels: int = 1, 
                 time_emb_dim: int = 32):
        """
        Args:
            input_channels: Number of input channels (noisy img + conditions)
            output_channels: Number of output channels (noise prediction)
            time_emb_dim: Dimension of time embeddings
        """
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        self.time_proj = nn.Linear(time_emb_dim, 512)

        # Encoder (downsampling)
        self.down1 = Block(input_channels, 64)   # -> 64 channels
        self.down2 = Block(64, 128)              # -> 128 channels
        self.down3 = Block(128, 256)             # -> 256 channels

        # Bottleneck
        self.bot1 = Block(256, 512)
        self.bot2 = Block(512, 512)
        self.bot3 = Block(512, 256)

        # Decoder (upsampling) with skip connections
        self.up1 = Block(512, 128)   # Input: 256 (from bot) + 256 (skip x3) = 512
        self.up2 = Block(256, 64)    # Input: 128 (from up1) + 128 (skip x2) = 256
        self.out = nn.Conv2d(128, output_channels, 1)  # Input: 64 (from up2) + 64 (skip x1) = 128

        # Pooling and upsampling
        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            t: Timestep indices (batch_size,)
            
        Returns:
            Predicted noise (batch_size, output_channels, height, width)
        """
        # Time embedding
        t_emb = self.time_mlp(t)
        t_emb = self.time_proj(t_emb)
        # Reshape for broadcasting: (batch, channels) -> (batch, channels, 1, 1)
        t_emb = t_emb[(...,) + (None,) * 2]

        # Encoder
        x1 = self.down1(x)                      # (B, 64, H, W)
        x2 = self.down2(self.max_pool(x1))      # (B, 128, H/2, W/2)
        x3 = self.down3(self.max_pool(x2))      # (B, 256, H/4, W/4)

        # Bottleneck
        x4 = self.bot1(self.max_pool(x3))       # (B, 512, H/8, W/8)
        x4 = x4 + t_emb                         # Add time embedding
        x4 = self.bot2(x4)                      # (B, 512, H/8, W/8)
        x4 = self.bot3(x4)                      # (B, 256, H/8, W/8)

        # Decoder with skip connections
        x = self.upsample(x4)                   # (B, 256, H/4, W/4)
        x = torch.cat((x, x3), dim=1)           # (B, 512, H/4, W/4)
        x = self.up1(x)                         # (B, 128, H/4, W/4)

        x = self.upsample(x)                    # (B, 128, H/2, W/2)
        x = torch.cat((x, x2), dim=1)           # (B, 256, H/2, W/2)
        x = self.up2(x)                         # (B, 64, H/2, W/2)

        x = self.upsample(x)                    # (B, 64, H, W)
        x = torch.cat((x, x1), dim=1)           # (B, 128, H, W)
        
        # Final output
        output = self.out(x)                    # (B, 1, H, W)
        return output


class DiffusionModel(nn.Module):
    """
    Diffusion model wrapper managing noise schedules and diffusion process.
    """
    
    def __init__(self, model: nn.Module, time_steps: int = 1000, device: str = "cuda"):
        """
        Args:
            model: U-Net model for noise prediction
            time_steps: Number of diffusion steps
            device: Device for computation ("cuda" or "cpu")
        """
        super().__init__()
        self.model = model.to(device)
        self.time_steps = time_steps
        self.device = device

        # Noise schedule (linear)
        self.betas = torch.linspace(1e-4, 0.02, time_steps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def forward_diffusion_sample(self, x_0: torch.Tensor, 
                                 t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to clean image (forward diffusion process).
        
        Args:
            x_0: Clean image
            t: Timestep indices
            
        Returns:
            Tuple of (noisy_image, noise)
        """
        noise = torch.randn_like(x_0)
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
        return x_t, noise

    def compute_losses(self, real_art_img: torch.Tensor, 
                      condition_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute diffusion training loss.
        
        Args:
            real_art_img: Target artifact image
            condition_tensor: Condition tensors (clean image + masks)
            
        Returns:
            MSE loss between predicted and actual noise
        """
        # Random timesteps
        t = torch.randint(0, self.time_steps, 
                         (real_art_img.shape[0],)).to(self.device).long()
        
        # Forward diffusion
        x_t, noise = self.forward_diffusion_sample(real_art_img, t)

        # Model input: noisy image + conditions
        model_input = torch.cat((x_t, condition_tensor), dim=1)

        # Predict noise
        noise_pred = self.model(model_input, t)

        # MSE loss
        loss = F.mse_loss(noise_pred, noise)
        return loss
