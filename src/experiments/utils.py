"""
Shared helpers for H/S experiment scripts.
"""

import re
import tomllib
from pathlib import Path

import numpy as np
import torch

from src.models.cyclegan import Generator
from src.models.diffusion import DiffusionModel


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_raw(path: Path, shape: tuple = (512, 512)) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(shape)


def match_pairs(target_dir: Path, baseline_dir: Path) -> list[tuple[Path, Path]]:
    """Match Target <-> Baseline files by img number."""
    def img_id(p: Path) -> str | None:
        m = re.search(r'img(\d+)', p.stem)
        return m.group(1) if m else None

    tmap = {img_id(p): p for p in sorted(target_dir.glob("*.raw")) if img_id(p)}
    pairs = []
    for p in sorted(baseline_dir.glob("*.raw")):
        i = img_id(p)
        if i and i in tmap:
            pairs.append((tmap[i], p))
    return pairs


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def normalize_hu(img: np.ndarray) -> np.ndarray:
    lo, hi = img.min(), img.max()
    return 2.0 * (img - lo) / (hi - lo) - 1.0 if hi - lo > 1e-5 else img


def to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float().to(device)


def compute_masks(clean: np.ndarray, art: np.ndarray, cfg: dict,
                  bloom_max: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Compute mask_M and mask_A — mirrors CTDataset / infer.py logic."""
    diff = np.abs(art - clean)
    metal = (diff > cfg["data"]["metal_threshold_hu"]).astype(np.float32)
    w = np.log1p(diff)
    denom = np.log1p(bloom_max) if bloom_max else w.max()
    artifact = np.clip(w / (denom + 1e-6), 0.0, 1.0).astype(np.float32)
    return metal, artifact


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_cyclegan_ab(path: Path, cfg: dict, device: torch.device) -> Generator:
    gen = Generator(
        input_nc=cfg["models"]["generator_input_nc"],
        output_nc=cfg["models"]["generator_output_nc"],
        n_residual_blocks=cfg["models"]["generator_n_residual"],
    ).to(device)
    gen.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    gen.eval()
    return gen


def load_diffusion_model(path: Path, cfg: dict, device: torch.device) -> DiffusionModel:
    """Load diffusion weights; prefers *_ema.pth if present next to path."""
    ema_path = path.parent / (path.stem + "_ema" + path.suffix)
    load_path = ema_path if ema_path.exists() else path
    diff = DiffusionModel(
        architecture=cfg["diffusion"].get("architecture", "standard"),
        time_steps=cfg["diffusion"].get("time_steps", 1000),
        device=device,
        input_channels=cfg["models"]["unet_input_channels"],
        output_channels=cfg["models"]["unet_output_channels"],
    )
    state = torch.load(load_path, map_location=device, weights_only=True)
    state = {k: v.to(device) for k, v in state.items()}
    diff.model.load_state_dict(state)
    diff.model.eval()
    return diff


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_cyclegan_ab(gen: Generator,
                          clean_t: torch.Tensor,
                          mask_m_t: torch.Tensor,
                          mask_a_t: torch.Tensor) -> np.ndarray:
    return gen(torch.cat([clean_t, mask_m_t, mask_a_t], dim=1)).squeeze().cpu().numpy()


@torch.no_grad()
def generate_diffusion(diff: DiffusionModel,
                        clean_t: torch.Tensor,
                        mask_m_t: torch.Tensor,
                        mask_a_t: torch.Tensor,
                        steps: int = 100) -> np.ndarray:
    condition = torch.cat([clean_t, mask_m_t, mask_a_t], dim=1)
    device = clean_t.device
    T = diff.time_steps
    stride = max(1, T // steps)
    step_seq = list(reversed(range(0, T, stride)))

    x = torch.randn(1, 1, clean_t.shape[2], clean_t.shape[3], device=device)
    for t_val in step_seq:
        t = torch.full((1,), t_val, device=device, dtype=torch.long)
        eps = diff.model(torch.cat([x, condition], dim=1), t)
        alpha_t     = diff.alphas[t_val]
        alpha_bar_t = diff.alphas_cumprod[t_val]
        beta_t      = diff.betas[t_val]
        mu = (x - beta_t / (1.0 - alpha_bar_t).sqrt() * eps) / alpha_t.sqrt()
        x  = mu + (beta_t.sqrt() * torch.randn_like(x) if t_val > 0 else 0.0)
    return x.clamp(-1.0, 1.0).squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def find_profile_row(img: np.ndarray) -> int:
    """Row with the highest horizontal gradient energy — best cross-section through streak."""
    return int(np.argmax(np.abs(np.diff(img, axis=1)).sum(axis=1)))


def gmsd(img1: np.ndarray, img2: np.ndarray, c: float = 170.0) -> float:
    """Gradient Magnitude Similarity Deviation. Lower = more perceptually similar."""
    def grad_mag(img):
        gx = np.abs(np.diff(img, axis=1, prepend=img[:, :1]))
        gy = np.abs(np.diff(img, axis=0, prepend=img[:1, :]))
        return np.sqrt(gx ** 2 + gy ** 2)

    gm1 = grad_mag(img1)
    gm2 = grad_mag(img2)
    gms_map = (2.0 * gm1 * gm2 + c) / (gm1 ** 2 + gm2 ** 2 + c)
    return float(gms_map.std())


def radial_power_spectrum(img: np.ndarray, n_bins: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged power spectrum. Returns (radii, mean_power)."""
    H, W = img.shape
    fft   = np.fft.fftshift(np.fft.fft2(img))
    power = np.abs(fft) ** 2

    cy, cx = H // 2, W // 2
    y, x   = np.ogrid[:H, :W]
    r      = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)

    r_max  = min(cy, cx)
    bins   = np.linspace(0, r_max, n_bins + 1)
    bidx   = np.clip(np.digitize(r.ravel(), bins) - 1, 0, n_bins - 1)

    radii   = 0.5 * (bins[:-1] + bins[1:])
    profile = np.zeros(n_bins)
    counts  = np.zeros(n_bins)
    np.add.at(profile, bidx, power.ravel())
    np.add.at(counts,  bidx, 1)
    profile /= np.maximum(counts, 1)
    return radii, profile


def hf_ratio(img: np.ndarray, low_radius_frac: float = 0.25) -> float:
    """Ratio of high-frequency power to total power (excluding DC)."""
    H, W  = img.shape
    fft   = np.fft.fftshift(np.fft.fft2(img))
    power = np.abs(fft) ** 2

    cy, cx = H // 2, W // 2
    r_low  = int(min(cy, cx) * low_radius_frac)
    y, x   = np.ogrid[:H, :W]
    r      = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

    total = power[r > 0].sum()
    high  = power[r > r_low].sum()
    return float(high / total) if total > 0 else 0.0
