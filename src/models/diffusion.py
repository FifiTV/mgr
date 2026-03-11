"""
Diffusion Model architecture for metal artifact reduction.
Supports both light and standard U-Net architectures.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from enum import Enum


class DiffusionArchitecture(Enum):
    """Available diffusion model architectures."""
    LIGHT = "light"       # Lightweight U-Net (test)
    STANDARD = "standard" # Advanced U-Net (Standard)


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


class ConditionalUnetLight(nn.Module):
    """
    Lightweight Conditional U-Net for diffusion model (original).
    Takes noisy image and condition tensors (clean image, masks).
    Good for limited GPU memory.
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
    Automatically selects appropriate U-Net architecture.
    """
    
    def __init__(self, architecture: str = "light", time_steps: int = 1000, 
                 device: str = "cuda", input_channels: int = 4, 
                 output_channels: int = 1):
        """
        Args:
            architecture: U-Net architecture ('light' or 'standard')
            time_steps: Number of diffusion steps
            device: Device for computation ("cuda" or "cpu")
            input_channels: Number of input channels
            output_channels: Number of output channels
        """
        super().__init__()
        
        # Build appropriate U-Net model
        arch_lower = architecture.lower()
        if arch_lower == "light":
            self.model = ConditionalUnetLight(input_channels, output_channels).to(device)
        elif arch_lower == "standard":
            self.model = ConditionalUnetStandard(input_channels, output_channels).to(device)
        else:
            raise ValueError(f"Unknown architecture: {architecture}. Use 'light' or 'standard'")
        
        self.architecture = arch_lower
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

        # Predict noise (architecture-specific forward pass)
        if self.architecture == "light":
            noise_pred = self.model(model_input, t)
        else:  # standard
            # Standard model has sinusoidal embeddings built-in
            noise_pred = self.model(model_input, t)

        # MSE loss
        loss = F.mse_loss(noise_pred, noise)
        return loss


# Standard
class ResnetBlock(nn.Module):
    """ResNet block for advanced diffusion model with time injection."""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.silu = nn.SiLU()

        # Residual connection (handles channel changes)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.silu(h)

        # Time injection
        time_emb = self.time_mlp(t_emb)
        time_emb = time_emb[(...,) + (None,) * 2]  # (B, C) -> (B, C, 1, 1)
        h = h + time_emb

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.silu(h)

        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-Attention block for capturing global context."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).view(B, 3, C, H * W)
        q, k, v = qkv.unbind(1)
        
        # Attention computation
        attn = (q.transpose(-2, -1) @ k) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = (v @ attn.transpose(-2, -1)).view(B, C, H, W)
        return x + self.proj(out)


class ConditionalUnetStandard(nn.Module):
    """
    Advanced Conditional U-Net with ResBlocks and Self-Attention (SOTA).
    Better quality than light version, requires more GPU memory.
    """
    
    def __init__(self, input_channels: int = 4, output_channels: int = 1, 
                 time_emb_dim: int = 256):
        super().__init__()

        # Time embedding with sinusoidal + MLP
        self.sinusoidal_emb = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4)
        )
        time_dim = time_emb_dim * 4

        # Encoder (ResNet blocks instead of simple convolutions)
        self.init_conv = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        
        self.down1 = ResnetBlock(64, 128, time_dim)
        self.down2 = ResnetBlock(128, 256, time_dim)
        self.down3 = ResnetBlock(256, 512, time_dim)

        # Bottleneck (with Self-Attention)
        self.bot1 = ResnetBlock(512, 512, time_dim)
        self.bot_attn = AttentionBlock(512)
        self.bot2 = ResnetBlock(512, 512, time_dim)

        # Decoder (with enlarged inputs for skip connections)
        self.up1 = ResnetBlock(512 + 256, 256, time_dim) 
        self.up2 = ResnetBlock(256 + 128, 128, time_dim)
        self.up3 = ResnetBlock(128 + 64, 64, time_dim)

        self.out = nn.Conv2d(64, output_channels, 1)

        # Pooling and upsampling
        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x: torch.Tensor, t: torch.Tensor, t_emb: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (noisy image + conditions)
            t: Timestep indices or embeddings
            t_emb: Optional pre-computed time embeddings
            
        Returns:
            Predicted noise
        """
        # Generate time embeddings if not provided
        if t_emb is None:
            if isinstance(t, torch.Tensor) and t.dtype in [torch.int32, torch.int64]:
                # If t is indices, generate embeddings
                t_emb = self.sinusoidal_emb(t)
            else:
                t_emb = t
        
        t_emb = self.time_mlp(t_emb)

        # Encoder
        x0 = self.init_conv(x)                       # (B, 64, H, W)
        x1 = self.down1(self.max_pool(x0), t_emb)    # (B, 128, H/2, W/2)
        x2 = self.down2(self.max_pool(x1), t_emb)    # (B, 256, H/4, W/4)
        x3 = self.down3(self.max_pool(x2), t_emb)    # (B, 512, H/8, W/8)

        # Bottleneck
        x4 = self.bot1(x3, t_emb)                    # (B, 512, H/8, W/8)
        x4 = self.bot_attn(x4)                       # (B, 512, H/8, W/8)
        x4 = self.bot2(x4, t_emb)                    # (B, 512, H/8, W/8)

        # Decoder with skip connections
        x = self.upsample(x4)                        
        x = torch.cat((x, x2), dim=1)                # Skip from x2
        x = self.up1(x, t_emb)                       # (B, 256, H/4, W/4)

        x = self.upsample(x)                         
        x = torch.cat((x, x1), dim=1)                # Skip from x1
        x = self.up2(x, t_emb)                       # (B, 128, H/2, W/2)

        x = self.upsample(x)                         
        x = torch.cat((x, x0), dim=1)                # Skip from x0
        x = self.up3(x, t_emb)                       # (B, 64, H, W)
        
        return self.out(x)