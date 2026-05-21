"""infer_utils.py — shared inference utilities for Gaussian Vicinal DDPM.

Imported by infer_gaussian.py and src/experiments/*.py.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

import tomllib
from src.models.gaussian_unet import GaussianUNet, GaussianDDPM
from src.datasets.gaussian_dataset import FEATURE_COLS

SHAPE = (512, 512)


# ── Config / IO ────────────────────────────────────────────────────────────────

def load_config(path: str | Path = 'config.toml') -> dict:
    with open(path, 'rb') as f:
        return tomllib.load(f)


def load_raw(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(SHAPE)


# ── Model ──────────────────────────────────────────────────────────────────────

def load_model(model_path: Path, cfg: dict,
               device: torch.device) -> tuple[GaussianDDPM, list[str]]:
    """Load GaussianDDPM from checkpoint, preferring EMA weights.

    Returns (ddpm, feature_cols).
    """
    gaus = cfg.get('gaussian', {})
    feature_cols = gaus.get('feature_cols', None) or FEATURE_COLS
    y_dim = len(feature_cols)

    unet = GaussianUNet(
        in_ch=3, out_ch=1,
        base_ch=gaus.get('base_channels', 64),
        t_emb_dim=gaus.get('t_emb_dim', 256),
        y_dim=y_dim,
        attn_heads=gaus.get('attn_heads', 8),
    )
    ddpm = GaussianDDPM(
        unet=unet,
        T=gaus.get('T', 1000),
        beta_schedule=gaus.get('beta_schedule', 'linear'),
        beta_start=gaus.get('beta_start', 1e-4),
        beta_end=gaus.get('beta_end', 0.02),
    )

    # Prefer EMA weights even when raw-weight path is given
    ema_path = model_path.parent / (
        model_path.stem.replace('_ema', '').replace('gaussian_unet', 'gaussian_unet_ema')
        + model_path.suffix
    )
    if '_ema' not in model_path.stem and ema_path.exists():
        load_path = ema_path
        print(f'  Using EMA weights: {load_path}')
    else:
        load_path = model_path

    state = torch.load(load_path, map_location=device, weights_only=True)
    # EMA checkpoint stores tensors on CPU regardless of map_location
    state = {k: v.to(device) for k, v in state.items()}
    # Raw-weight checkpoint wraps keys under 'model_state'
    if 'model_state' in state:
        state = state['model_state']
    unet.load_state_dict(state)
    ddpm = ddpm.to(device)
    ddpm.eval()
    print(f'  Loaded: {load_path}  (y_dim={y_dim}, features={feature_cols})')
    return ddpm, feature_cols


# ── Sampling ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def ddpm_sample(ddpm: GaussianDDPM,
                condition: torch.Tensor,
                y_target: torch.Tensor,
                stride: int = 1,
                seed: int | None = None) -> np.ndarray:
    """DDPM reverse sampling.

    stride=1  → delegates to ddpm.sample() — identical to training visualisation,
                 guaranteed correct.
    stride>1  → custom loop skipping steps (faster, slightly lower quality).

    Args:
        condition: [B, 2, H, W]  (I_clean_norm cat M_metal)
        y_target:  [B, y_dim]    feature vector in [0, 1]
        stride:    1 = full 1000 steps; 10 ≈ 10× faster
        seed:      if given, sets CPU + CUDA RNG before sampling

    Returns:
        [H, W] float32 numpy array (normalised I_error)
    """
    if seed is not None:
        torch.manual_seed(seed)
        if condition.device.type == 'cuda':
            torch.cuda.manual_seed(seed)

    if stride == 1:
        # Exact same path as trainer's generate_samples → guaranteed correct
        x = ddpm.sample(condition, y_target)
        return x.squeeze(0).squeeze(0).cpu().numpy()

    # Strided loop for faster (debug) sampling
    B, _, H, W = condition.shape
    x = torch.randn(B, 1, H, W, device=condition.device)
    for t_val in reversed(range(0, ddpm.T, stride)):
        t_batch  = torch.full((B,), t_val, device=x.device, dtype=torch.long)
        model_in = torch.cat([x, condition], dim=1)
        eps_pred = ddpm.unet(model_in, t_batch, y_target)
        beta_t   = ddpm.betas[t_val]
        alpha_t  = ddpm.alphas[t_val]
        ab_t     = ddpm.alphas_cumprod[t_val]
        mean = (x - beta_t / (1 - ab_t).sqrt() * eps_pred) / alpha_t.sqrt()
        x = mean + beta_t.sqrt() * torch.randn_like(x) if t_val > 0 else mean

    return x.squeeze(0).squeeze(0).cpu().numpy()
