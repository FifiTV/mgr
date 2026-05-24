"""eval_utils.py — wspólne funkcje dla skryptów ewaluacyjnych.

Import:
    from src.experiments.eval_utils import (
        load_generated_samples, find_aapm_pair, load_hospital_images,
        normalize_features, frechet_distance, frechet_per_feature,
        sinogram_stat_features, RADON_THETA, FEATURE_COLS,
    )
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from scipy.ndimage import binary_fill_holes
from skimage.transform import radon, resize

log = logging.getLogger(__name__)

SHAPE       = (512, 512)
RADON_THETA = np.linspace(0, 180, 90, endpoint=False)

# Reexport for convenience
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.datasets.gaussian_dataset import FEATURE_COLS                        # noqa: F401
from src.features.feature_extraction import extract_features                   # noqa: F401

ERROR_SCALE_DEFAULT = 642.8


# ── Ładowanie danych ────────────────────────────────────────────────────────────

def load_raw(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(SHAPE)


def load_generated_samples(gen_dir: Path, bodies: list[str] | None = None) -> list[dict]:
    """Załaduj próbki wygenerowane (*.json + _gen_error.raw).

    Każdy rekord: {json_path, raw_path, img_id, body, error_scale}
    Opcjonalnie filtruje po liście body.
    """
    samples = []
    for json_path in sorted(gen_dir.glob('*.json')):
        raw_path = json_path.with_name(json_path.stem + '_gen_error.raw')
        if not raw_path.exists():
            log.warning(f'Brak .raw dla {json_path.name}, pomijam')
            continue
        with open(json_path) as f:
            meta = json.load(f)
        body = meta['body']
        if bodies and body not in bodies:
            continue
        samples.append({
            'json_path':   json_path,
            'raw_path':    raw_path,
            'img_id':      meta['img_id'],
            'body':        body,
            'error_scale': meta.get('error_scale', ERROR_SCALE_DEFAULT),
        })
    log.info(f'Wygenerowane próbki: {len(samples)}'
             + (f' (body={bodies})' if bodies else ''))
    return samples


def find_aapm_pair(rpi_dir: Path, body: str, img_id: int) -> tuple[Path, Path] | None:
    """Zwróć (art_path, clean_path) lub None gdy brak plików."""
    body_dir   = rpi_dir / body
    art_path   = body_dir / 'Baseline' / f'training_body_metalart_img{img_id}_512x512x1.raw'
    clean_path = body_dir / 'Target'   / f'training_body_nometal_img{img_id}_512x512x1.raw'
    if art_path.exists() and clean_path.exists():
        return art_path, clean_path
    return None


def load_hospital_images(real_dir: Path) -> list[tuple[Path, np.ndarray]]:
    """Załaduj obrazy szpitalne — zwraca listę (path, array).

    Uwaga: obrazy używane są bezpośrednio (bez ekstrakcji streaków).
    """
    result = []
    for f in sorted(real_dir.glob('*.raw')):
        result.append((f, load_raw(f)))
    log.info(f'Obrazy szpitalne: {len(result)}')
    return result


# ── Normalizacja cech ────────────────────────────────────────────────────────────

def normalize_features(df: pd.DataFrame, clip_stats: dict) -> pd.DataFrame:
    """Normalizuj cechy raw tymi samymi statystykami co AAPM (clip_stats.json)."""
    df = df.copy()
    for col in FEATURE_COLS:
        if col not in clip_stats or col not in df.columns:
            continue
        s = clip_stats[col]
        vals = df[col].astype(float)
        if s.get('log', False):
            vals = np.log1p(vals)
        lo, hi = s['lo'], s['hi']
        df[col] = (vals.clip(lo, hi) - lo) / (hi - lo + 1e-9)
    return df


# ── Fréchet distance ────────────────────────────────────────────────────────────

def frechet_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Fréchet distance między dwoma zbiorami wektorów (N×D i M×D).

    FD = ||μ_X - μ_Y||² + Tr(Σ_X + Σ_Y - 2·√(Σ_X Σ_Y))
    """
    if X.shape[0] < 2 or Y.shape[0] < 2:
        log.warning('Za mało próbek dla frechet_distance (min 2)')
        return float('nan')

    mu_x, cov_x = X.mean(axis=0), np.cov(X, rowvar=False)
    mu_y, cov_y = Y.mean(axis=0), np.cov(Y, rowvar=False)
    cov_x, cov_y = np.atleast_2d(cov_x), np.atleast_2d(cov_y)

    diff     = mu_x - mu_y
    sqrt_cov = sqrtm(cov_x @ cov_y)
    if np.iscomplexobj(sqrt_cov):
        sqrt_cov = sqrt_cov.real

    return float(diff @ diff + np.trace(cov_x + cov_y - 2.0 * sqrt_cov))


def frechet_per_feature(df_a: pd.DataFrame, df_b: pd.DataFrame,
                         label_a: str = 'A', label_b: str = 'B') -> dict:
    """FD per cecha (1D) + łączne 6D."""
    feats_a = df_a[FEATURE_COLS].dropna().values
    feats_b = df_b[FEATURE_COLS].dropna().values

    results = {}
    for i, col in enumerate(FEATURE_COLS):
        results[col] = frechet_distance(feats_a[:, i:i+1], feats_b[:, i:i+1])

    results['combined_6D'] = frechet_distance(feats_a, feats_b)
    log.info(f'FD({label_a} vs {label_b}):  '
             + '  '.join(f'{c}={v:.3f}' for c, v in results.items()))
    return results


# ── Sinogram features ────────────────────────────────────────────────────────────

def sinogram_stat_features(img: np.ndarray) -> np.ndarray:
    """270D wektor cech sinogramu: [mean, std, P95] × 90 kątów.

    Wejście: dowolny obraz CT w HU (nie trzeba wyciągać artefaktu).
    """
    sg = radon(img.astype(np.float64), theta=RADON_THETA, circle=True)
    return np.concatenate([
        sg.mean(axis=0),
        sg.std(axis=0),
        np.percentile(sg, 95, axis=0),
    ]).astype(np.float32)


# ── RadImageNet FID (opcjonalne) ────────────────────────────────────────────────

def radimagenet_features(images: list[np.ndarray], weights_path: Path,
                          device_str: str = 'auto') -> np.ndarray:
    """Wyciągnij 2048D cechy z RadImageNet ResNet50.

    images: lista tablic HU (512×512) — mogą to być pełne CT lub sinogramy.
    Zwraca: (N, 2048) float32.
    """
    import torch
    import torchvision.models as tv_models

    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if device_str == 'auto' else torch.device(device_str))

    obj = torch.load(weights_path, map_location='cpu', weights_only=False)
    if isinstance(obj, torch.nn.Module):
        # Full model saved with torch.save(model, ...) — Lab-Rasool/RadImageNet format
        backbone = obj
    else:
        # State dict — load into a fresh ResNet50
        backbone = tv_models.resnet50()
        backbone.load_state_dict(obj, strict=False)
    if hasattr(backbone, 'fc'):
        backbone.fc = torch.nn.Identity()
    backbone.eval().to(device)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    feats = []
    for img in images:
        # Normalizuj do [0,1], resize 224×224, powiel do 3 kanałów
        lo, hi = img.min(), img.max()
        img_n = (img - lo) / (hi - lo + 1e-6)
        img_r = resize(img_n, (224, 224), anti_aliasing=True)
        t = torch.from_numpy(img_r).float().unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
        t = (t.to(device) - mean) / std
        with torch.no_grad():
            feats.append(backbone(t).squeeze().cpu().numpy())

    return np.array(feats, dtype=np.float32)


def imagenet_features(images: list[np.ndarray],
                      device_str: str = 'auto') -> np.ndarray:
    """Extract 2048D features from ResNet50 pretrained on ImageNet.

    Standard FID baseline — compare against Med-FID (RadImageNet).
    Normalization: HU clip [-1000, 3000] → [0, 1] (fixed window, not per-image).
    """
    import torch
    import torchvision.models as tv_models

    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if device_str == 'auto' else torch.device(device_str))

    backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
    backbone.fc = torch.nn.Identity()
    backbone.eval().to(device)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    feats = []
    for img in images:
        # Fixed HU window [-1000, 3000] → [0, 1] (same for all images, unlike per-image min/max)
        img_norm = (np.clip(img, -1000.0, 3000.0) + 1000.0) / 4000.0
        img_r = resize(img_norm, (224, 224), anti_aliasing=True)
        t = torch.from_numpy(img_r).float().unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
        t = (t.to(device) - mean) / std
        with torch.no_grad():
            feats.append(backbone(t).squeeze().cpu().numpy())

    return np.array(feats, dtype=np.float32)


def sinogram_images(images: list[np.ndarray]) -> list[np.ndarray]:
    """Przelicz listę obrazów CT na sinogramy (do podania do radimagenet_features)."""
    return [radon(img.astype(np.float64), theta=RADON_THETA, circle=True).astype(np.float32)
            for img in images]


# ── Zapis wyników ────────────────────────────────────────────────────────────────

def save_fid_csv(comparisons: dict[str, dict], out_path: Path):
    """Zapisz wyniki FID (słownik comparison→{feature→value}) do CSV."""
    rows = []
    for comparison, per_feat in comparisons.items():
        if not isinstance(per_feat, dict):
            continue
        for feat, val in per_feat.items():
            rows.append({'comparison': comparison, 'feature': feat, 'frechet_distance': val})
    if rows:
        pd.DataFrame(rows).to_csv(out_path, index=False)
        log.info(f'Zapisano: {out_path}')
