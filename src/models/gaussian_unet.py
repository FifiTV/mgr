"""
Gaussian Vicinal DDPM -- U-Net with AdaLN conditioning on feature vector y.

Architecture:
    Input : [B, 3, 512, 512]  -- concat(x_t, I_clean, M_metal)
    Output: [B, 1, 512, 512]  -- predicted noise eps

    Encoder: 4 levels, spatial 512->256->128->64->32, channels 64->128->256->512
    Self-attention at 64x64 and 32x32 resolutions (flash attention via
    F.scaled_dot_product_attention, requires PyTorch >= 2.0)
    AdaLN conditioning on y (6-dim feature vector) in every ResNet block
    Sinusoidal time embedding added in every ResNet block
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ─────────────────────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal position embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = math.log(10000) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=device) * -freqs)
        emb = t[:, None].float() * freqs[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResNetBlock(nn.Module):
    """
    ResNet block with:
        - sinusoidal time embedding (added to activations)
        - AdaLN conditioning on label vector y (scale + shift after norm)

    Forward signature: (x, t_emb, y) -> x'
    """

    def __init__(self, in_ch: int, out_ch: int, t_dim: int, y_dim: int = 6):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # Time: linear -> out_ch, added to activations between the two convs
        self.t_proj = nn.Linear(t_dim, out_ch)

        # AdaLN for y: MLP -> (scale, shift) pair applied to norm2 output
        self.y_proj = nn.Sequential(
            nn.Linear(y_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 2 * out_ch),
        )

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor,
                t_emb: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)

        # Add time embedding
        h = h + self.t_proj(t_emb)[:, :, None, None]

        # AdaLN: scale + shift from feature vector
        scale, shift = self.y_proj(y).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.skip(x)


class SelfAttention(nn.Module):
    """
    Multi-head self-attention with GroupNorm pre-norm and residual.
    Uses F.scaled_dot_product_attention (flash attention when available).
    """

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = channels // num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv  = nn.Conv1d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W)

        q, k, v = self.qkv(h).chunk(3, dim=1)  # each [B, C, N]

        # Reshape to (B, heads, N, head_dim) for scaled_dot_product_attention
        N = H * W
        q = q.view(B, self.num_heads, self.head_dim, N).transpose(2, 3)
        k = k.view(B, self.num_heads, self.head_dim, N).transpose(2, 3)
        v = v.view(B, self.num_heads, self.head_dim, N).transpose(2, 3)

        h = F.scaled_dot_product_attention(q, k, v)  # memory-efficient / flash attn
        h = h.transpose(2, 3).contiguous().view(B, C, N)
        h = self.proj(h).view(B, C, H, W)

        return x + h


# ── U-Net ────────────────────────────────────────────────────────────────────────

class GaussianUNet(nn.Module):
    """
    Conditional U-Net for Gaussian Vicinal DDPM.

    Input:  concat(x_t, I_clean, M_metal)  [B, 3, 512, 512]
    Output: predicted noise eps             [B, 1, 512, 512]

    Conditioning:
        t : diffusion timestep    -> sinusoidal embedding + MLP
        y : feature vector [0,1]^6 -> AdaLN (scale+shift) in every ResNet block
    """

    def __init__(self,
                 in_ch: int = 3,
                 out_ch: int = 1,
                 base_ch: int = 64,
                 t_emb_dim: int = 256,
                 y_dim: int = 6,
                 attn_heads: int = 8):
        super().__init__()
        ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]  # 64, 128, 256, 512
        t_dim = t_emb_dim * 4  # 1024

        # ── Time embedding ────────────────────────────────────────────────────────
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(t_emb_dim),
            nn.Linear(t_emb_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

        # ── Encoder ───────────────────────────────────────────────────────────────
        self.enc_proj = nn.Conv2d(in_ch, ch[0], 3, padding=1)

        # Level 1: 512x512, 64ch, no attention
        self.enc1 = nn.ModuleList([
            ResNetBlock(ch[0], ch[0], t_dim, y_dim),
            ResNetBlock(ch[0], ch[0], t_dim, y_dim),
        ])
        self.down1 = nn.Conv2d(ch[0], ch[0], 3, stride=2, padding=1)

        # Level 2: 256x256, 128ch, no attention
        self.enc2 = nn.ModuleList([
            ResNetBlock(ch[0], ch[1], t_dim, y_dim),
            ResNetBlock(ch[1], ch[1], t_dim, y_dim),
        ])
        self.down2 = nn.Conv2d(ch[1], ch[1], 3, stride=2, padding=1)

        # Level 3: 128x128, 256ch, no attention
        self.enc3 = nn.ModuleList([
            ResNetBlock(ch[1], ch[2], t_dim, y_dim),
            ResNetBlock(ch[2], ch[2], t_dim, y_dim),
        ])
        self.down3 = nn.Conv2d(ch[2], ch[2], 3, stride=2, padding=1)

        # Level 4: 64x64, 512ch, WITH attention
        self.enc4 = nn.ModuleList([
            ResNetBlock(ch[2], ch[3], t_dim, y_dim),
            SelfAttention(ch[3], attn_heads),
            ResNetBlock(ch[3], ch[3], t_dim, y_dim),
            SelfAttention(ch[3], attn_heads),
        ])
        self.down4 = nn.Conv2d(ch[3], ch[3], 3, stride=2, padding=1)

        # ── Bottleneck: 32x32, 512ch, WITH attention ──────────────────────────────
        self.bot = nn.ModuleList([
            ResNetBlock(ch[3], ch[3], t_dim, y_dim),
            SelfAttention(ch[3], attn_heads),
            ResNetBlock(ch[3], ch[3], t_dim, y_dim),
        ])

        # ── Decoder ───────────────────────────────────────────────────────────────
        # Level 4d: 32->64, WITH attention (symmetric to enc4)
        self.up4  = nn.ConvTranspose2d(ch[3], ch[3], 2, stride=2)
        self.dec4 = nn.ModuleList([
            ResNetBlock(ch[3] + ch[3], ch[2], t_dim, y_dim),
            SelfAttention(ch[2], attn_heads),
            ResNetBlock(ch[2], ch[2], t_dim, y_dim),
        ])

        # Level 3d: 64->128
        self.up3  = nn.ConvTranspose2d(ch[2], ch[2], 2, stride=2)
        self.dec3 = nn.ModuleList([
            ResNetBlock(ch[2] + ch[2], ch[1], t_dim, y_dim),
            ResNetBlock(ch[1], ch[1], t_dim, y_dim),
        ])

        # Level 2d: 128->256
        self.up2  = nn.ConvTranspose2d(ch[1], ch[1], 2, stride=2)
        self.dec2 = nn.ModuleList([
            ResNetBlock(ch[1] + ch[1], ch[0], t_dim, y_dim),
            ResNetBlock(ch[0], ch[0], t_dim, y_dim),
        ])

        # Level 1d: 256->512
        self.up1  = nn.ConvTranspose2d(ch[0], ch[0], 2, stride=2)
        self.dec1 = nn.ModuleList([
            ResNetBlock(ch[0] + ch[0], ch[0], t_dim, y_dim),
            ResNetBlock(ch[0], ch[0], t_dim, y_dim),
        ])

        # ── Output ────────────────────────────────────────────────────────────────
        self.out_norm = nn.GroupNorm(8, ch[0])
        self.out_conv = nn.Conv2d(ch[0], out_ch, 1)

    # ─────────────────────────────────────────────────────────────────────────────

    def _run_blocks(self, h: torch.Tensor, blocks: nn.ModuleList,
                    t_emb: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        for block in blocks:
            if isinstance(block, ResNetBlock):
                h = block(h, t_emb, y)
            else:
                h = block(h)
        return h

    def forward(self, x: torch.Tensor,
                t: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, 3, 512, 512]  concat(x_t, I_clean, M_metal)
            t : [B]                diffusion timesteps (integer)
            y : [B, 6]             normalized feature vector in [0, 1]^6

        Returns:
            [B, 1, 512, 512]  predicted noise
        """
        t_emb = self.time_mlp(t)                              # [B, t_dim]

        # ── Encoder ──────────────────────────────────────────────────────────────
        h  = self.enc_proj(x)                                 # [B, 64, 512, 512]

        h  = self._run_blocks(h, self.enc1, t_emb, y)
        s1 = h                                                 # skip [B, 64,  512, 512]
        h  = self.down1(h)                                    # [B, 64,  256, 256]

        h  = self._run_blocks(h, self.enc2, t_emb, y)
        s2 = h                                                 # skip [B, 128, 256, 256]
        h  = self.down2(h)                                    # [B, 128, 128, 128]

        h  = self._run_blocks(h, self.enc3, t_emb, y)
        s3 = h                                                 # skip [B, 256, 128, 128]
        h  = self.down3(h)                                    # [B, 256,  64,  64]

        h  = self._run_blocks(h, self.enc4, t_emb, y)
        s4 = h                                                 # skip [B, 512,  64,  64]
        h  = self.down4(h)                                    # [B, 512,  32,  32]

        # ── Bottleneck ────────────────────────────────────────────────────────────
        h  = self._run_blocks(h, self.bot, t_emb, y)         # [B, 512, 32, 32]

        # ── Decoder ───────────────────────────────────────────────────────────────
        h  = self.up4(h)                                      # [B, 512, 64, 64]
        h  = torch.cat([h, s4], dim=1)                       # [B, 1024, 64, 64]
        h  = self._run_blocks(h, self.dec4, t_emb, y)        # [B, 256,  64, 64]

        h  = self.up3(h)                                      # [B, 256, 128, 128]
        h  = torch.cat([h, s3], dim=1)                       # [B, 512, 128, 128]
        h  = self._run_blocks(h, self.dec3, t_emb, y)        # [B, 128, 128, 128]

        h  = self.up2(h)                                      # [B, 128, 256, 256]
        h  = torch.cat([h, s2], dim=1)                       # [B, 256, 256, 256]
        h  = self._run_blocks(h, self.dec2, t_emb, y)        # [B,  64, 256, 256]

        h  = self.up1(h)                                      # [B,  64, 512, 512]
        h  = torch.cat([h, s1], dim=1)                       # [B, 128, 512, 512]
        h  = self._run_blocks(h, self.dec1, t_emb, y)        # [B,  64, 512, 512]

        # ── Output ───────────────────────────────────────────────────────────────
        return self.out_conv(F.silu(self.out_norm(h)))        # [B, 1, 512, 512]


# ── DDPM wrapper ─────────────────────────────────────────────────────────────────

class GaussianDDPM(nn.Module):
    """
    DDPM wrapper for GaussianUNet.

    Manages noise schedule buffers and provides:
        q_sample  -- forward diffusion q(x_t | x_0)
        forward   -- sample t, add noise, predict noise; returns (eps, eps_pred)
    """

    def __init__(self,
                 unet: GaussianUNet,
                 T: int = 1000,
                 beta_schedule: str = 'linear',
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02):
        super().__init__()
        self.unet = unet
        self.T    = T

        if beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, T)
        elif beta_schedule == 'cosine':
            steps = T + 1
            s = 0.008
            x = torch.linspace(0, T, steps)
            f = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
            f = f / f[0]
            betas = (1 - f[1:] / f[:-1]).clamp(0.0, 0.999)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule!r}")

        alphas          = 1.0 - betas
        alphas_cumprod  = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas',                      betas)
        self.register_buffer('alphas',                     alphas)
        self.register_buffer('alphas_cumprod',             alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod',        alphas_cumprod.sqrt())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', (1 - alphas_cumprod).sqrt())

    def q_sample(self, x0: torch.Tensor,
                 t: torch.Tensor,
                 eps: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: q(x_t | x_0) = sqrt(ab_t)*x0 + sqrt(1-ab_t)*eps."""
        a  = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sa = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return a * x0 + sa * eps

    def forward(self, x0: torch.Tensor,
                condition: torch.Tensor,
                y: torch.Tensor):
        """
        Sample random t and eps, run forward diffusion + noise prediction.

        Args:
            x0        : [B, 1, H, W]  target I_error (normalized to [-1, 1])
            condition : [B, 2, H, W]  I_clean + M_metal (2 channels)
                        Internally concatenated with x_t -> [B, 3, H, W] for UNet.
            y         : [B, 6]        normalized feature vector

        Returns:
            eps      : [B, 1, H, W]  true noise
            eps_pred : [B, 1, H, W]  predicted noise
        """
        B      = x0.shape[0]
        t      = torch.randint(0, self.T, (B,), device=x0.device)
        eps    = torch.randn_like(x0)
        x_t    = self.q_sample(x0, t, eps)

        model_in = torch.cat([x_t, condition], dim=1)     # [B, 3, H, W] = x_t + I_clean + M_metal
        eps_pred = self.unet(model_in, t, y)

        return eps, eps_pred

    @torch.no_grad()
    def sample(self, condition: torch.Tensor,
               y_target: torch.Tensor) -> torch.Tensor:
        """
        DDPM reverse sampling: generate I_error from pure noise.

        Args:
            condition : [B, 2, H, W]  I_clean + M_metal
            y_target  : [B, 6]        target feature vector

        Returns:
            [B, 1, H, W]  generated I_error (normalized)
        """
        B, _, H, W = condition.shape
        x = torch.randn(B, 1, H, W, device=condition.device)

        for t_val in reversed(range(self.T)):
            t_batch = torch.full((B,), t_val, device=x.device, dtype=torch.long)
            model_in = torch.cat([x, condition], dim=1)
            eps_pred = self.unet(model_in, t_batch, y_target)

            beta_t  = self.betas[t_val]
            alpha_t = self.alphas[t_val]
            ab_t    = self.alphas_cumprod[t_val]

            mean = (x - beta_t / (1 - ab_t).sqrt() * eps_pred) / alpha_t.sqrt()

            if t_val > 0:
                x = mean + beta_t.sqrt() * torch.randn_like(x)
            else:
                x = mean

        return x
