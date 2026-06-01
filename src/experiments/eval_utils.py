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
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from scipy.ndimage import binary_fill_holes, gaussian_filter
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


def load_magnet_images(magnet_dir: Path) -> list[tuple[Path, np.ndarray]]:
    """Load magnet/jitter images — files whose names start with 'jitter'.

    Shape is parsed from the filename (e.g. H512_W512); falls back to SHAPE.
    Each file is read with np.fromfile(path, dtype=np.float32) and reshaped.
    Returns list of (path, array) pairs, sorted by filename.
    """
    result = []
    for f in sorted(magnet_dir.glob('jitter*.raw')):
        m = re.search(r'H(\d+)_W(\d+)', f.stem)
        h, w = (int(m.group(1)), int(m.group(2))) if m else SHAPE
        img = np.fromfile(f, dtype=np.float32).reshape(h, w)
        result.append((f, img))
    log.info(f'Magnet/jitter images: {len(result)} from {magnet_dir}')
    return result


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
                      weights_path: Path | None = None,
                      device_str: str = 'auto') -> np.ndarray:
    """Extract 2048D features from ResNet50 pretrained on ImageNet.

    Standard FID baseline — compare against Med-FID (RadImageNet).
    Normalization: HU clip [-1000, 3000] → [0, 1] (fixed window, not per-image).

    weights_path: optional local .pth file. If None, torchvision downloads automatically
                  to ~/.cache/torch/hub/checkpoints/ (requires internet on first run).
    """
    import torch
    import torchvision.models as tv_models

    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if device_str == 'auto' else torch.device(device_str))

    backbone = tv_models.resnet50()
    if weights_path is not None and Path(weights_path).exists():
        state = torch.load(weights_path, map_location='cpu', weights_only=True)
        backbone.load_state_dict(state)
        log.info(f'ImageNet weights loaded from {weights_path}')
    else:
        # Automatic download via torchvision (cached in ~/.cache/torch/hub/checkpoints/)
        backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
        log.info('ImageNet weights loaded from torchvision cache')
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


# ── Pixel-level image quality metrics ─────────────────────────────────────────

DATA_RANGE = 2.0   # images normalized to [-1, 1]

# generated file suffix → which reference to compare against
MODEL_SUFFIXES = {
    "cyclegan_ab": "real_art",   # artifact generation → vs ground-truth artifact
    "diffusion":   "real_art",   # artifact generation → vs ground-truth artifact
    "cyclegan_ba": "clean",      # artifact removal    → vs clean CT
}

METRIC_NAMES_BASE  = ["ssim", "psnr", "mae", "gmsd"]
METRIC_NAMES_LPIPS = METRIC_NAMES_BASE + ["lpips"]


def normalize_hu(img: np.ndarray) -> np.ndarray:
    """Per-image min-max normalization to [-1, 1] — identical to infer.py."""
    lo, hi = img.min(), img.max()
    return 2.0 * (img - lo) / (hi - lo) - 1.0 if hi - lo > 1e-5 else img


def body_from_img_id(img_id: int) -> str:
    """
    Map img_id to RPI body folder name.
    Convention: body1 = img1-1000, body2 = img1001-2000, ...
    Formula: body_num = (img_id - 1) // 1000 + 1
    Examples: img7001 → body8, img8001 → body9
    """
    return f"body{(img_id - 1) // 1000 + 1}"


_RPI_TEMPLATES = {
    "real_art": "Baseline/training_body_metalart_img{id}_512x512x1.raw",
    "clean":    "Target/training_body_nometal_img{id}_512x512x1.raw",
}


def find_rpi_ref(rpi_dir: Path, img_id: int, ref_type: str) -> Path | None:
    """Return path to RPI reference file, or None if not found.

    Uses find_aapm_pair internally; ref_type is 'real_art' or 'clean'.
    """
    body = body_from_img_id(img_id)
    pair = find_aapm_pair(rpi_dir, body, img_id)
    if pair is None:
        return None
    art_path, clean_path = pair
    return art_path if ref_type == "real_art" else clean_path


# ── Generated-file discovery ──────────────────────────────────────────────────

def find_gen_dir(variant_dir: Path) -> Path:
    """
    Find directory that contains the generated .raw files.
    Layout A (infer.py): files are in variant/raw/
    Layout B (flat):     files are in variant/ directly
    """
    raw_dir = variant_dir / "raw"
    if raw_dir.exists():
        for p in raw_dir.glob("*.raw"):
            if any(p.stem.endswith(f"_{t}") for t in MODEL_SUFFIXES):
                return raw_dir
    for p in variant_dir.glob("*.raw"):
        if any(p.stem.endswith(f"_{t}") for t in MODEL_SUFFIXES):
            return variant_dir
    return raw_dir  # fallback


def find_ref_dir(variant_dir: Path) -> Path:
    """
    Find directory with reference .raw files when not using RPI.
    Layout A: refs in variant/raw/img/
    Layout B: refs in variant/raw/
    """
    img_dir = variant_dir / "raw" / "img"
    if img_dir.exists() and any(img_dir.glob("*_real_art.raw")):
        return img_dir
    raw_dir = variant_dir / "raw"
    if raw_dir.exists() and any(raw_dir.glob("*_real_art.raw")):
        return raw_dir
    return img_dir  # fallback


def find_eval_pairs(variant_dir: Path,
                    model_types: list[str],
                    rpi_dir: Path | None) -> list[dict]:
    """
    Returns list of dicts:
      variant, img_id, model_type, generated (Path), reference (Path),
      ref_is_rpi (bool — True means reference needs normalize_hu before use)
    """
    gen_dir = find_gen_dir(variant_dir)
    if not gen_dir.exists():
        log.warning(f"Generated dir not found: {gen_dir}")
        return []

    ref_dir = None if rpi_dir else find_ref_dir(variant_dir)

    pairs = []
    for gen_path in sorted(gen_dir.glob("*.raw")):
        stem = gen_path.stem

        matched_type = None
        for mtype in model_types:
            if stem.endswith(f"_{mtype}"):
                matched_type = mtype
                break
        if matched_type is None:
            continue

        m = re.search(r"img(\d+)", stem)
        if not m:
            log.warning(f"Cannot extract img_id from {gen_path.name}")
            continue
        img_id = int(m.group(1))

        ref_type   = MODEL_SUFFIXES[matched_type]
        ref_is_rpi = rpi_dir is not None

        if rpi_dir:
            ref_path = find_rpi_ref(rpi_dir, img_id, ref_type)
            if ref_path is None:
                body = body_from_img_id(img_id)
                log.warning(f"RPI reference not found: img{img_id} ({ref_type}) "
                            f"expected in {rpi_dir}/{body}/")
                continue
        else:
            prefix   = stem[: -(len(matched_type) + 1)]
            ref_path = ref_dir / f"{prefix}_{ref_type}.raw"
            if not ref_path.exists():
                log.warning(f"Reference not found: {ref_path.name}")
                continue

        pairs.append({
            "variant":    variant_dir.name,
            "img_id":     img_id,
            "model_type": matched_type,
            "generated":  gen_path,
            "reference":  ref_path,
            "ref_is_rpi": ref_is_rpi,
        })

    return pairs


# ── Pixel-level metrics ───────────────────────────────────────────────────────

def ssim(img1: np.ndarray, img2: np.ndarray,
         data_range: float = DATA_RANGE, sigma: float = 1.5) -> float:
    """SSIM with Gaussian window (sigma=1.5).

    Numerically equivalent to skimage.metrics.structural_similarity(
    gaussian_weights=True, sigma=1.5, data_range=data_range).
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    a, b = img1.astype(np.float64), img2.astype(np.float64)

    mu_a, mu_b   = gaussian_filter(a, sigma), gaussian_filter(b, sigma)
    mu_a2, mu_b2 = mu_a * mu_a, mu_b * mu_b
    mu_ab        = mu_a * mu_b

    sig_a2 = gaussian_filter(a * a, sigma) - mu_a2
    sig_b2 = gaussian_filter(b * b, sigma) - mu_b2
    sig_ab = gaussian_filter(a * b, sigma) - mu_ab

    num = (2.0 * mu_ab + C1) * (2.0 * sig_ab + C2)
    den = (mu_a2 + mu_b2 + C1) * (sig_a2 + sig_b2 + C2)
    return float(np.mean(num / den))


def psnr(img1: np.ndarray, img2: np.ndarray,
         data_range: float = DATA_RANGE) -> float:
    """Peak Signal-to-Noise Ratio [dB]. Higher = better."""
    mse = float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * np.log10(data_range ** 2 / mse)


def mae(img1: np.ndarray, img2: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(img1.astype(np.float64) - img2.astype(np.float64))))


def gmsd(img1: np.ndarray, img2: np.ndarray) -> float:
    """Gradient Magnitude Similarity Deviation. Lower = more similar.

    c = 0.0026 * data_range² scales the original 8-bit constant to [-1,1] images.
    """
    c = 0.0026 * (DATA_RANGE ** 2)

    def gm(img: np.ndarray) -> np.ndarray:
        gx = np.abs(np.diff(img, axis=1, prepend=img[:, :1]))
        gy = np.abs(np.diff(img, axis=0, prepend=img[:1, :]))
        return np.sqrt(gx ** 2 + gy ** 2)

    g1, g2  = gm(img1), gm(img2)
    gms_map = (2.0 * g1 * g2 + c) / (g1 ** 2 + g2 ** 2 + c)
    return float(gms_map.std())


def compute_metrics(gen: np.ndarray, ref: np.ndarray,
                    lpips_fn=None) -> dict:
    """Compute all pixel-level metrics for one (generated, reference) pair."""
    result = {
        "ssim": ssim(gen, ref),
        "psnr": psnr(gen, ref),
        "mae":  mae(gen, ref),
        "gmsd": gmsd(gen, ref),
    }
    if lpips_fn is not None:
        result["lpips"] = lpips_fn(gen, ref)
    return result


def aggregate_metrics(scores: list[dict],
                      metric_names: list[str]) -> dict:
    """Compute mean / std / median for each metric over a list of score dicts."""
    result = {}
    for k in metric_names:
        vals = np.array([s[k] for s in scores if k in s], dtype=np.float64)
        if not len(vals):
            continue
        result[f"{k}_mean"]   = float(vals.mean())
        result[f"{k}_std"]    = float(vals.std())
        result[f"{k}_median"] = float(np.median(vals))
    return result


# ── LPIPS (VGG16 via torchvision — no extra package) ─────────────────────────

_VGG16_URL = "https://download.pytorch.org/models/vgg16-397923af.pth"


class LPIPS_VGG:
    """Simplified LPIPS using pretrained VGG16 features (torchvision).

    Computes normalised L2 distance between multi-scale VGG16 activations.
    Input images: numpy [H, W] float32 in [-1, 1].

    weights_path: optional path to a local vgg16 .pth file.
      - If the file exists there: loaded directly (no internet needed).
      - If the path is given but missing: downloaded from PyTorch CDN to that path.
      - If None: torchvision downloads to ~/.cache/torch/hub/checkpoints/ as usual.
    """
    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD  = [0.229, 0.224, 0.225]
    _LAYER_ENDS    = [4, 9, 16, 23]  # relu1_2, relu2_2, relu3_3, relu4_3

    def __init__(self, device: str = "cpu",
                 weights_path: Path | None = None):
        import torch
        import torch.nn as nn
        import torchvision.models as tvm
        self.device = torch.device(device)

        if weights_path is not None:
            weights_path = Path(weights_path)
            if not weights_path.exists():
                log.info(f"VGG16 weights not found at {weights_path} — downloading...")
                weights_path.parent.mkdir(parents=True, exist_ok=True)
                torch.hub.download_url_to_file(_VGG16_URL, str(weights_path))
                log.info(f"Saved to {weights_path}")
            else:
                log.info(f"Loading VGG16 weights from {weights_path}")
            vgg = tvm.vgg16(weights=None)
            vgg.load_state_dict(
                torch.load(weights_path, map_location="cpu", weights_only=True)
            )
            vgg_feat = vgg.features.to(self.device).eval()
        else:
            vgg_feat = tvm.vgg16(
                weights=tvm.VGG16_Weights.IMAGENET1K_V1
            ).features.to(self.device).eval()

        for p in vgg_feat.parameters():
            p.requires_grad = False

        children = list(vgg_feat.children())
        prev = 0
        self.blocks = []
        for end in self._LAYER_ENDS:
            b = nn.Sequential(*children[prev:end]).to(self.device).eval()
            for p in b.parameters():
                p.requires_grad = False
            self.blocks.append(b)
            prev = end

        mean = torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1).to(self.device)
        std  = torch.tensor(self._IMAGENET_STD ).view(1, 3, 1, 1).to(self.device)
        self._mean, self._std = mean, std

    def _prep(self, img_np: np.ndarray):
        import torch
        t = torch.from_numpy(img_np).float().to(self.device)
        t = (t + 1.0) / 2.0
        t = t.unsqueeze(0).expand(3, -1, -1).unsqueeze(0)
        return (t - self._mean) / self._std

    def __call__(self, img1_np: np.ndarray, img2_np: np.ndarray) -> float:
        import torch
        x, y, total = self._prep(img1_np), self._prep(img2_np), 0.0
        with torch.no_grad():
            for block in self.blocks:
                x, y = block(x), block(y)
                norm = lambda f: f / f.pow(2).sum(1, keepdim=True).sqrt().clamp(1e-8)
                total += float((norm(x) - norm(y)).pow(2).mean())
        return total / len(self.blocks)


def build_lpips(device: str,
                weights_path: Path | None = None) -> "LPIPS_VGG | None":
    """Build LPIPS_VGG or return None on failure.

    weights_path: local .pth file; downloaded there if missing.
                  If None, torchvision uses its default cache.
    """
    try:
        inst = LPIPS_VGG(device=device, weights_path=weights_path)
        log.info(f"LPIPS: VGG16 ready on {device}")
        return inst
    except Exception as e:
        log.warning(f"LPIPS unavailable: {e}")
        return None
