"""
Feature Extraction Pipeline — Metal Artifact CT
================================================
All features computed on the difference image:
    Ierror = Imetal - Iclean  [units: HU, float32]

Data: AAPM Grand Challenge (.raw, float32, 512x512)
    - Iclean: valid HU range (-1100 to ~2500 HU)
    - Imetal: HU + metal saturation (up to ~65000) -> clipped to +-3000 HU

Features:
    1. peak_amplitude        -- P99 of bright pixels [HU]
    2. spatial_extent        -- fraction of body area covered by streak [0-1]
    3. bbox_ratio            -- streak bounding box diagonal / image diagonal [0-1]
    4. dark_to_bright_ratio  -- energy of dark streaks / bright blooms
    5. angular_concentration -- CV of angular FFT power profile (streak directionality)
    6. texture_roughness     -- Var(Laplacian) / Var(intensity) inside streak mask

Key design decisions:
    - HU_CLIP = 3000: physical CT limit; values above indicate hardware saturation
    - body_mask from Iclean + binary_fill_holes: eliminates metal influence on the mask
    - tau = P90 of bright pixels: Otsu unreliable for unimodal CT histograms
    - angular_concentration: high-pass (sigma=5) before FFT removes DC component
    - texture_roughness: relative eps + min 50 pixels -- robust to outliers
    - bbox_ratio: geometric alternative to spatial_extent (different variance)
"""

import numpy as np
from pathlib import Path
from scipy import ndimage
from scipy.ndimage import gaussian_filter, binary_fill_holes
import pandas as pd
import re
from tqdm.notebook import tqdm

# ── Global constants ────────────────────────────────────────────────────────────
SHAPE    = (512, 512)
N_ANGLES = 360
HU_CLIP  = 3000.0   # [HU] physical limit -- above this is detector saturation

# Polar grid -- built once, reused for every image
# Each FFT pixel gets an angle index theta in [0, N_ANGLES)
_cy, _cx     = SHAPE[0] // 2, SHAPE[1] // 2
_y, _x       = np.ogrid[-_cy:SHAPE[0]-_cy, -_cx:SHAPE[1]-_cx]
_angles_flat = np.arctan2(_y, _x).ravel()                          # [-pi, pi]
_angle_bins  = np.linspace(-np.pi, np.pi, N_ANGLES + 1)
_angle_idx   = np.clip(
    np.digitize(_angles_flat, _angle_bins) - 1, 0, N_ANGLES - 1
)


# ── Loading ─────────────────────────────────────────────────────────────────────
def load_raw(path: Path) -> np.ndarray:
    """Load a .raw file as float32, reshape to 512x512."""
    return np.fromfile(path, dtype=np.float32).reshape(SHAPE)


# ── Preprocessing ───────────────────────────────────────────────────────────────
def preprocess(Imetal: np.ndarray, Iclean: np.ndarray):
    """
    Build the difference image and body mask.

    Clipping to +-HU_CLIP:
        Imetal contains hardware saturation (>3000 HU) near metal.
        Values above HU_CLIP are a reconstruction artifact, not a physical
        signal -- removed before feature extraction.

    body_mask from Iclean (not Imetal):
        Imetal has distorted HU values near metal, which would corrupt
        the -500 HU threshold. Iclean is clean.
        binary_fill_holes closes air cavities (sinuses, ear canals)
        inside the body contour -- improves the denominator in
        spatial_extent and dark_to_bright_ratio.

    Returns:
        Ierror_clipped : np.ndarray [float64], range [-3000, 3000] HU
        body_mask      : np.ndarray [bool],    True = patient tissue
    """
    Ierror = (Imetal - Iclean).astype(np.float64)
    Ierror_clipped = np.clip(Ierror, -HU_CLIP, HU_CLIP)

    # Threshold -500 HU: air (<-500) vs tissue (>-500)
    body_mask = binary_fill_holes(Iclean > -500)

    return Ierror_clipped, body_mask


def compute_tau(Ierror_clipped: np.ndarray, body_mask: np.ndarray) -> float:
    """
    Threshold separating 'bright streak' from artifact background.

    Why not Otsu:
        The histogram of bright Ierror pixels in CT is unimodal
        (exponentially decreasing from 0 to max). Otsu does not find
        a meaningful threshold in a unimodal distribution -- it returns
        a value close to the median, not the streak boundary.

    Solution -- P90 of bright pixels inside the body:
        We define a streak as the 'top 10% brightest artifact pixels
        inside the body'. Deterministic, interpretable and robust to
        distribution shape.

    Returns:
        tau [HU] -- streak intensity threshold
    """
    pos = Ierror_clipped[(Ierror_clipped > 0) & body_mask]
    if len(pos) < 10:
        return 50.0   # fallback for very weak artifacts
    return float(np.percentile(pos, 90))


# ── Features ────────────────────────────────────────────────────────────────────
def peak_amplitude(Ierror_clipped: np.ndarray, body_mask: np.ndarray) -> float:
    """
    P99 of bright pixels inside the body [HU].

    P99 instead of max: rejects isolated noisy pixels (outliers).
    Inside body_mask: eliminates artifacts outside the patient body.
    Clipping to HU_CLIP in preprocess guarantees P99 <= 3000 HU.

    Expected range: 200-3000 HU
    """
    bright = Ierror_clipped[(Ierror_clipped > 0) & body_mask]
    return float(np.percentile(bright, 99)) if len(bright) > 0 else 0.0


def spatial_extent(
    Ierror_clipped: np.ndarray, body_mask: np.ndarray, tau: float
) -> float:
    """
    Fraction of body area covered by bright streak [0-1].

    Denominator = body_mask (with binary_fill_holes) instead of full image:
    eliminates FOV and patient positioning effects.
    Numerator = pixels > tau (P90) inside body = top 10% of streak.

    Expected range: 0.005-0.15
    """
    n_body = int(body_mask.sum())
    if n_body == 0:
        return 0.0
    n_streak = int(((Ierror_clipped > tau) & body_mask).sum())
    return n_streak / n_body


def dark_to_bright_ratio(
    Ierror_clipped: np.ndarray, body_mask: np.ndarray, tau: float, eps: float = 1.0
) -> float:
    """
    Ratio of dark streak energy to bright bloom energy [dimensionless].

    Symmetrically uses the same threshold tau for both sides:
        dark   = pixels < -tau  (photon starvation)
        bright = pixels >  tau  (beam hardening)
    Avoids threshold 0 which would capture background noise.

    eps = 1.0 [HU*px]: numerical stability when bright -> 0
    (weak artifacts where almost no bright pixels exist).

    Value > 1: dark streaks dominate (strong photon starvation)
    Value < 1: bright blooms dominate (beam hardening)
    Expected range: 0.1-2.0
    """
    region = body_mask
    dark   = float(np.abs(Ierror_clipped[(Ierror_clipped < -tau) & region]).sum())
    bright = float(Ierror_clipped[(Ierror_clipped > tau) & region].sum())
    return dark / (bright + eps)


def angular_concentration(
    Ierror_clipped: np.ndarray, body_mask: np.ndarray
) -> float:
    """
    Coefficient of variation (CV) of the angular FFT power profile.

    Measures artifact directionality:
        High value -> sharp, directional streaks (e.g. 2 main bands)
        Low value  -> isotropic bloom (uniform across all angles)

    Why high-pass (sigma=5):
        DC component and low frequencies (anatomy, global gradients)
        dominate the power spectrum and blur the angular profile ->
        CV would be underestimated. High-pass isolates structures
        smaller than ~5px = streaks.

    Polar grid _angle_idx built once globally for performance.

    Expected range: 0.5-10.0
    """
    Imasked  = Ierror_clipped * body_mask
    highpass = Imasked - gaussian_filter(Imasked, sigma=5)

    power   = np.abs(np.fft.fftshift(np.fft.fft2(highpass))) ** 2
    profile = np.bincount(
        _angle_idx, weights=power.ravel(), minlength=N_ANGLES
    ).astype(np.float64)

    mean_p = profile.mean()
    return float(profile.var() / mean_p**2) if mean_p > 1e-10 else 0.0


def texture_roughness(
    Ierror_clipped: np.ndarray, body_mask: np.ndarray, tau: float,
    min_mask_px: int = 50
) -> float:
    """
    Texture roughness inside the streak mask [dimensionless].

    Var(Laplacian) / Var(intensity) -- both computed ONLY inside the mask:
        Laplacian ~ second derivative -> measures edge sharpness and noise
        Normalization by Var(intensity) makes the result independent of:
            - artifact amplitude (strong vs weak bloom)
            - artifact size (small vs large extent)
        Measures pure texture 'roughness'.

    Low value  -> smooth gradient (beam hardening without scatter)
    High value -> jagged, noisy artifact (scatter + detector noise)

    Changes v2:
        - min_mask_px=50 instead of 4: with small sample var_int -> 0
          and result explodes (observed outliers: max=474 at n<10 px)
        - relative eps = max(1% * var_int, 1e-6): scales with data amplitude
          instead of fixed 1e-6 which is too small for HU data
    """
    mask = (Ierror_clipped > tau) & body_mask
    if mask.sum() < min_mask_px:
        return 0.0
    lap     = ndimage.laplace(Ierror_clipped)
    var_lap = float(lap[mask].var())
    var_int = float(Ierror_clipped[mask].var())
    eps     = max(0.01 * var_int, 1e-6)   # relative eps -- scales with data
    return var_lap / (var_int + eps)


def bbox_ratio(
    Ierror_clipped: np.ndarray, body_mask: np.ndarray, tau: float
) -> float:
    """
    Ratio of streak bounding box diagonal to image diagonal [0-1].

    Why alongside spatial_extent:
        spatial_extent counts pixels (area) -- sensitive to artifact density
        but not to its geometric span.
        bbox_ratio measures spatial reach in geometric terms --
        how far the streak extends from the metal center to the image edge.
        The two features are complementary and may have different variance
        depending on the dataset.

    Implementation:
        Bounding box = min/max row and column of pixels > tau inside body_mask.
        bbox diagonal / image diagonal -> normalized to [0, 1].

    Value near 0 -> artifact concentrated around metal (small implant)
    Value near 1 -> artifact spans the whole image (large implant, strong streak)

    Expected range: 0.1-1.0
    """
    streak_mask = (Ierror_clipped > tau) & body_mask
    if streak_mask.sum() < 10:
        return 0.0

    rows = np.where(streak_mask.any(axis=1))[0]
    cols = np.where(streak_mask.any(axis=0))[0]

    height    = float(rows[-1] - rows[0] + 1)
    width     = float(cols[-1] - cols[0] + 1)
    bbox_diag = np.sqrt(height**2 + width**2)
    img_diag  = np.sqrt(SHAPE[0]**2 + SHAPE[1]**2)   # ~724 px for 512x512

    return float(bbox_diag / img_diag)


def extract_features(Imetal: np.ndarray, Iclean: np.ndarray) -> dict:
    """
    Full pipeline for a single CT image pair.

    Args:
        Imetal : raw CT image with metal artifact [float32, HU]
        Iclean : reference image without metal    [float32, HU]

    Returns:
        Dictionary with 6 features + diagnostic (tau).
        All features are unnormalized -- Min-Max normalization
        is performed separately after collecting statistics from the full dataset.
    """
    Ierror_clipped, body_mask = preprocess(Imetal, Iclean)
    tau = compute_tau(Ierror_clipped, body_mask)

    return {
        'peak_amplitude':        peak_amplitude(Ierror_clipped, body_mask),
        'spatial_extent':        spatial_extent(Ierror_clipped, body_mask, tau),
        'bbox_ratio':            bbox_ratio(Ierror_clipped, body_mask, tau),
        'dark_to_bright_ratio':  dark_to_bright_ratio(Ierror_clipped, body_mask, tau),
        'angular_concentration': angular_concentration(Ierror_clipped, body_mask),
        'texture_roughness':     texture_roughness(Ierror_clipped, body_mask, tau),
        'tau':                   tau,
    }


# ── Dataset loop ────────────────────────────────────────────────────────────────
def run_dataset(base_dir: Path) -> pd.DataFrame:
    """
    Extract features for all pairs in a single body dataset.

    Args:
        base_dir : path to the body directory
                   (must contain Baseline/ and Target/ subdirectories)

    Returns:
        DataFrame with features indexed by img_id, sorted ascending.
        Saves results to ../results/features_body1.csv
    """
    def _img_id(path: Path) -> int:
        m = re.search(r'img(\d+)', path.stem)
        return int(m.group(1)) if m else -1

    baseline_files = sorted(
        [p for p in (base_dir / "Baseline").glob("*.raw")
         if re.search(r'img(\d+)', p.stem)],
        key=_img_id
    )

    records = []
    for img_art_path in tqdm(baseline_files, desc="Feature extraction"):
        img_id       = _img_id(img_art_path)
        i_clean_path = (base_dir / "Target" /
                        f"training_body_nometal_img{img_id}_512x512x1.raw")
        if not i_clean_path.exists():
            continue
        try:
            row = extract_features(load_raw(img_art_path), load_raw(i_clean_path))
            row['img_id'] = img_id
            records.append(row)
        except Exception as e:
            print(f"  ERROR img{img_id}: {e}")

    df = pd.DataFrame(records).set_index('img_id').sort_index()

    n_saturated = (df['tau'] >= HU_CLIP * 0.97).sum()
    if n_saturated:
        print(f"Warning: {n_saturated} images with tau >= {HU_CLIP*0.97:.0f} HU "
              f"(saturated, features less precise, but retained)")
    print(f"Total: {len(df)} images")

    out = Path("../results/features_body1.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"Saved: {out}  ({len(df)} rows)")

    return df
