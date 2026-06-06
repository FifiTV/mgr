"""eval_generated.py — evaluation of generated CT data quality.

Groups compared:
  A = I_clean_AAPM + I_error_generated   (synthesized CT from our model)
  B = I_clean_AAPM + I_error_AAPM        (AAPM CT reconstructed without raw metal)
  C = Hospital_raw                        (real hospital CT, used directly)
  D = Magnet/jitter                       (optional real CT, used directly)

Experiments
───────────
SSIM  Masked SSIM     — pixel-level quality inside artifact mask (A vs B)
Phys  Physical FID    — Frechet on 6 physical features (A vs B; hospital excluded)

FID — 2×2 matrix (input image × encoder backbone):

              CT image            CT sinogram
  ┌──────────────────────────────────────────────────┐
  │ ImageNet   [1] FID-CT-IN     [2] FID-Sino-IN     │
  │ RadImageNet[3] FID-CT-RIN    [4] FID-Sino-RIN    │
  └──────────────────────────────────────────────────┘

  [5] FID-Sino-Stats — 270D sinogram statistics (no neural network,
                       fully interpretable: mean/std/P95 per projection angle)

Rationale for the 2×2 structure:
  CT image   → encoder sees artifact appearance, anatomy, texture
  Sinogram   → encoder sees projection physics, acquisition character
  Two images can look visually similar but differ in sinogram (and vice versa).
  Agreement across [1]+[2] or [3]+[4] gives double confirmation from different
  perspectives — a stronger argument than any single metric.

  [5] is kept as an NN-free, physics-grounded reference point.

Usage:
  python eval_generated.py \\
    --generated data/raw/generated/body8/raw data/raw/generated/body9/raw \\
    --rpi       data/raw/RPI \\
    --real      data/raw/real \\
    --features  results/features_norm.csv \\
    --clip-stats results/feature_clip_stats.json \\
    --out       results/eval_generated \\
    [--bodies body8 body9] \\
    [--magnet  data/raw/magnet] \\
    [--rad-imagenet models/RadImageNet-ResNet50_notop.pt] \\
    [--mask-mode {unmasked,masked,both}]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes
from skimage.metrics import structural_similarity as ssim

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
log = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.experiments.eval_utils import (
    FEATURE_COLS, find_aapm_pair, frechet_distance, frechet_per_feature,
    imagenet_features, load_generated_samples, load_hospital_images,
    load_magnet_images, load_raw,
    normalize_features, radimagenet_features, save_fid_csv,
    sinogram_images, sinogram_stat_features,
)
from src.features.feature_extraction import compute_tau, extract_features, preprocess

HU_LO, HU_HI = -1000.0, 3000.0
ERR_CLIP      = 3000.0
_HU_RANGE     = HU_HI - HU_LO  # 4000.0


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _hu_norm(img: np.ndarray) -> np.ndarray:
    """Clip to [HU_LO, HU_HI] and map to [0, 1] with a fixed window.

    Applied to ALL groups (A, B, C, D) before feature extraction so that
    every encoder receives consistently scaled input regardless of origin.
    Fixed window avoids per-image normalization artefacts (e.g. an image that
    is mostly air would be stretched very differently under per-image min/max).
    """
    return ((np.clip(img, HU_LO, HU_HI) - HU_LO) / _HU_RANGE).astype(np.float32)


def _prepare_ct_images(
        group_a: list[dict], group_b: list[dict],
        hospital: list[tuple], magnet: list[tuple],
        use_masked: bool,
        max_c: int | None = None,
        max_d: int | None = None,
) -> tuple[list, list, list, list]:
    """Return (imgs_a, imgs_b, imgs_c, imgs_d) normalised to [0, 1].

    All groups are mapped through _hu_norm (fixed HU window) so that the
    feature extractors receive identically scaled inputs.
    A/B use 'I_sino_masked' when use_masked=True, otherwise 'I_sino'.
    C/D are reference groups — always unmasked.
    max_c / max_d limit C/D size (Radon transform is slow).
    """
    key      = 'I_sino_masked' if use_masked else 'I_sino'
    imgs_a   = [_hu_norm(a[key]) for a in group_a]
    imgs_b   = [_hu_norm(b[key]) for b in group_b]
    hosp_src = hospital if max_c is None else hospital[:max_c]
    magn_src = (magnet or []) if max_d is None else (magnet or [])[:max_d]
    imgs_c   = [_hu_norm(img) for _, img in hosp_src]
    imgs_d   = [_hu_norm(img) for _, img in magn_src]
    return imgs_a, imgs_b, imgs_c, imgs_d


def _fid_all_pairs(
        fa: np.ndarray, fb: np.ndarray,
        fc: np.ndarray, fd: np.ndarray,
        results: dict,
        prefix: str = '',
) -> dict:
    """Compute Frechet distance for all group pairs and store in results."""
    pairs = [
        (fa, fb, 'Generated(A)', 'AAPM(B)'),
        (fa, fc, 'Generated(A)', 'Hospital(C)'),
        (fb, fc, 'AAPM(B)',      'Hospital(C)'),
    ]
    if fd.shape[0] >= 2:
        pairs += [
            (fa, fd, 'Generated(A)', 'Magnet(D)'),
            (fb, fd, 'AAPM(B)',      'Magnet(D)'),
            (fc, fd, 'Hospital(C)',   'Magnet(D)'),
        ]
    for xa, xb, la, lb in pairs:
        key = f'{prefix}{la}_vs_{lb}'
        if xa.shape[0] >= 2 and xb.shape[0] >= 2:
            results[key] = frechet_distance(xa, xb)
            log.info(f'  FD({la} vs {lb}) = {results[key]:.4f}')
        else:
            results[key] = None
            log.warning(f'  Not enough samples: {key}  '
                        f'(n_{la.split("(")[1][0]}={xa.shape[0]}, '
                        f'n_{lb.split("(")[1][0]}={xb.shape[0]})')
    return results


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE GROUP CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_groups(samples: list[dict], rpi_dir: Path) -> tuple[list[dict], list[dict]]:
    """Match each generated sample to its AAPM pair and build image groups.

    Returns two equal-length lists (only samples with an existing AAPM pair):
      group_a — generated: I_clean + I_error_gen
      group_b — AAPM:      I_clean + I_error_aapm  (no raw metal point)

    Each record contains:
      I_error_*      — clipped I_error in HU  (for SSIM and Physical FID)
      I_synth/I_art  — full HU image          (for Physical FID feature extraction)
      I_sino         — clipped to [HU_LO, HU_HI]         (for FID experiments)
      I_sino_masked  — I_sino with artifact mask applied  (for masked FID variants)
      M_artifact     — binary artifact mask               (for diagnostics)
    """
    group_a, group_b = [], []

    for s in samples:
        pair = find_aapm_pair(rpi_dir, s['body'], s['img_id'])
        if pair is None:
            log.warning(f"  No AAPM pair for {s['body']}/img{s['img_id']}, skipping")
            continue

        I_art   = load_raw(pair[0])
        I_clean = load_raw(pair[1])

        gen_norm    = load_raw(s['raw_path'])
        I_error_gen = gen_norm * s['error_scale']

        I_error_gen_clip  = np.clip(I_error_gen,     -ERR_CLIP, ERR_CLIP)
        I_error_real_clip = np.clip(I_art - I_clean, -ERR_CLIP, ERR_CLIP)

        body_mask     = binary_fill_holes(I_clean > -500)
        I_err_for_tau, _ = preprocess(I_art, I_clean)
        tau           = compute_tau(I_err_for_tau, body_mask)
        M_artifact    = (np.abs(I_error_real_clip) > tau) & body_mask

        I_sino_a = np.clip(I_clean + I_error_gen_clip,  HU_LO, HU_HI).astype(np.float32)
        I_sino_b = np.clip(I_clean + I_error_real_clip, HU_LO, HU_HI).astype(np.float32)

        I_sino_a_masked = np.where(M_artifact, I_sino_a, HU_LO).astype(np.float32)
        I_sino_b_masked = np.where(M_artifact, I_sino_b, HU_LO).astype(np.float32)

        group_a.append({
            'img_id':        s['img_id'],
            'body':          s['body'],
            'I_synth':       (I_clean + I_error_gen_clip).astype(np.float32),
            'I_clean':       I_clean,
            'I_error_gen':   I_error_gen_clip,
            'I_sino':        I_sino_a,
            'I_sino_masked': I_sino_a_masked,
            'M_artifact':    M_artifact,
        })
        group_b.append({
            'img_id':        s['img_id'],
            'body':          s['body'],
            'I_art':         I_art,
            'I_clean':       I_clean,
            'I_error_aapm':  I_error_real_clip,
            'I_sino':        I_sino_b,
            'I_sino_masked': I_sino_b_masked,
            'M_artifact':    M_artifact,
        })

    log.info(f'A/B pairs: {len(group_a)} (from {len(samples)} generated samples)')
    return group_a, group_b


# ══════════════════════════════════════════════════════════════════════════════
# SSIM
# ══════════════════════════════════════════════════════════════════════════════

def run_ssim_exp(group_a: list[dict], group_b: list[dict]) -> dict:
    """Masked SSIM — I_error_gen vs I_error_aapm inside the artifact mask."""
    log.info('=== [SSIM] Masked SSIM ===')
    scores = []

    for a, b in zip(group_a, group_b):
        I_error_aapm = b['I_error_aapm']
        I_error_gen  = a['I_error_gen']
        I_clean      = b['I_clean']

        body_mask = binary_fill_holes(I_clean > -500)
        I_err_clip, _ = preprocess(b['I_art'], I_clean)
        tau = compute_tau(I_err_clip, body_mask)
        artifact_mask = (np.abs(I_error_aapm) > tau) & body_mask

        n_px = int(artifact_mask.sum())
        if n_px < 50:
            log.warning(f"  img{a['img_id']}: artifact mask too small ({n_px}px), skipping")
            continue

        data_range = float(I_error_aapm.max() - I_error_aapm.min()) + 1e-6
        _, ssim_map = ssim(I_error_aapm, I_error_gen,
                           data_range=data_range, win_size=11, full=True)
        score = float(ssim_map[artifact_mask].mean())

        scores.append({'img_id': a['img_id'], 'body': a['body'],
                       'masked_ssim': score, 'mask_pixels': n_px})
        log.info(f"  {a['body']}/img{a['img_id']:5d}  ssim={score:.4f}  mask={n_px}px")

    if not scores:
        return {'scores': [], 'mean': None, 'std': None, 'n': 0}

    vals   = [r['masked_ssim'] for r in scores]
    result = {'scores': scores, 'mean': float(np.mean(vals)),
              'std': float(np.std(vals)), 'n': len(scores)}
    log.info(f'  mean={result["mean"]:.4f} +/- {result["std"]:.4f}  n={result["n"]}')
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICAL FID  (A vs B only — hospital/magnet excluded, no clean reference)
# ══════════════════════════════════════════════════════════════════════════════

def run_physical_fid(group_a: list[dict], group_b: list[dict],
                     features_norm_path: Path, clip_stats: dict,
                     bodies: list[str] | None = None,
                     magnet_images: list[tuple] | None = None,
                     ) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Physical FID — Frechet distance on 6 physical features.

    Hospital excluded: no (I_art, I_clean) pairs available for feature extraction.
    Magnet/jitter: features extracted with I_clean=zeros (signal vs zero baseline).
    """
    log.info('=== [Phys] Physical FID — 6 physical features ===')

    records_gen = []
    for a in group_a:
        row = extract_features(a['I_synth'], a['I_clean'])
        row['img_id'] = a['img_id']
        row['body']   = a['body']
        records_gen.append(row)
    df_gen_raw = pd.DataFrame(records_gen)
    df_gen     = normalize_features(df_gen_raw, clip_stats)

    df_aapm = pd.read_csv(features_norm_path, index_col='img_id')
    if bodies and 'source' in df_aapm.columns:
        df_aapm = df_aapm[df_aapm['source'].isin(bodies)]
    log.info(f'  Generated: {len(df_gen)}  AAPM: {len(df_aapm)}')

    if len(df_gen) < 2:
        log.warning('  [Phys] Skipped — fewer than 2 generated samples with matched AAPM pairs')
        return {'Generated_vs_AAPM': None, 'n_gen': len(df_gen), 'n_aapm': len(df_aapm)}, df_gen, df_aapm

    result = {}
    result['Generated_vs_AAPM'] = frechet_per_feature(df_gen, df_aapm, 'Generated', 'AAPM')
    result['n_gen']  = len(df_gen)
    result['n_aapm'] = len(df_aapm)

    if magnet_images:
        records_mag = []
        for i, (path, img) in enumerate(magnet_images):
            row = extract_features(img, np.zeros_like(img))
            row['img_id'] = i
            records_mag.append(row)
        df_mag_raw = pd.DataFrame(records_mag)
        df_mag     = normalize_features(df_mag_raw, clip_stats)
        log.info(f'  Magnet/jitter: {len(df_mag)}')

        if len(df_mag) >= 2:
            result['Generated_vs_Magnet'] = frechet_per_feature(df_gen, df_mag, 'Generated', 'Magnet')
            result['AAPM_vs_Magnet']      = frechet_per_feature(df_aapm, df_mag, 'AAPM', 'Magnet')
            result['n_magnet'] = len(df_mag)
        else:
            log.warning('  Magnet group: fewer than 2 images, skipping FD')
    else:
        df_mag = pd.DataFrame()

    return result, df_gen, df_aapm


# ══════════════════════════════════════════════════════════════════════════════
# FID EXPERIMENTS — 2×2 matrix
#
#              CT image              CT sinogram
#  ImageNet    [1] run_fid_ct_in     [2] run_fid_sino_in
#  RadImageNet [3] run_fid_ct_rin    [4] run_fid_sino_rin
#
# + [5] run_fid_sino_stats — 270D statistics, no neural network
# ══════════════════════════════════════════════════════════════════════════════

def run_fid_ct_in(group_a: list[dict], group_b: list[dict],
                  hospital: list[tuple], magnet: list[tuple] | None,
                  use_masked: bool,
                  imagenet_path: Path | None = None) -> dict:
    """[1] FID — ImageNet ResNet50 on CT images.

    Encoder trained on natural images; captures global appearance and texture.
    """
    tag = 'masked' if use_masked else 'full'
    log.info(f'=== [1] FID-CT-ImageNet [{tag}] ===')

    imgs_a, imgs_b, imgs_c, imgs_d = _prepare_ct_images(
        group_a, group_b, hospital, magnet, use_masked)
    log.info(f'  A={len(imgs_a)}  B={len(imgs_b)}  C={len(imgs_c)}  D={len(imgs_d)}')

    fa = imagenet_features(imgs_a, imagenet_path) if imgs_a else np.empty((0, 2048))
    fb = imagenet_features(imgs_b, imagenet_path) if imgs_b else np.empty((0, 2048))
    fc = imagenet_features(imgs_c, imagenet_path) if imgs_c else np.empty((0, 2048))
    fd = imagenet_features(imgs_d, imagenet_path) if imgs_d else np.empty((0, 2048))
    log.info(f'  Features: A{fa.shape}  B{fb.shape}  C{fc.shape}  D{fd.shape}')

    results = {'method': f'ImageNet_ResNet50_2048D_CT{"_masked" if use_masked else ""}'}
    _fid_all_pairs(fa, fb, fc, fd, results)
    return results


def run_fid_sino_in(group_a: list[dict], group_b: list[dict],
                    hospital: list[tuple], magnet: list[tuple] | None,
                    use_masked: bool,
                    imagenet_path: Path | None = None,
                    max_c: int | None = None,
                    max_d: int | None = None) -> dict:
    """[2] FID — ImageNet ResNet50 on sinogram images.

    Encoder sees projection-domain structure: acquisition physics, angular patterns.
    ImageNet bias on natural textures — compare with [4] for RadImageNet perspective.
    """
    tag = 'masked' if use_masked else 'full'
    log.info(f'=== [2] FID-Sino-ImageNet [{tag}] ===')

    imgs_a, imgs_b, imgs_c, imgs_d = _prepare_ct_images(
        group_a, group_b, hospital, magnet, use_masked, max_c, max_d)
    log.info(f'  CT images: A={len(imgs_a)}  B={len(imgs_b)}  '
             f'C={len(imgs_c)}  D={len(imgs_d)}')

    log.info('  Computing sinograms...')
    sinos_a = sinogram_images(imgs_a)
    sinos_b = sinogram_images(imgs_b)
    sinos_c = sinogram_images(imgs_c) if imgs_c else []
    sinos_d = sinogram_images(imgs_d) if imgs_d else []

    fa = imagenet_features(sinos_a, imagenet_path) if sinos_a else np.empty((0, 2048))
    fb = imagenet_features(sinos_b, imagenet_path) if sinos_b else np.empty((0, 2048))
    fc = imagenet_features(sinos_c, imagenet_path) if sinos_c else np.empty((0, 2048))
    fd = imagenet_features(sinos_d, imagenet_path) if sinos_d else np.empty((0, 2048))
    log.info(f'  Features: A{fa.shape}  B{fb.shape}  C{fc.shape}  D{fd.shape}')

    results = {'method': f'ImageNet_ResNet50_2048D_Sino{"_masked" if use_masked else ""}'}
    _fid_all_pairs(fa, fb, fc, fd, results)
    return results


def run_fid_ct_rin(group_a: list[dict], group_b: list[dict],
                   hospital: list[tuple], magnet: list[tuple] | None,
                   use_masked: bool,
                   rad_path: Path) -> dict:
    """[3] FID — RadImageNet ResNet50 on CT images.

    Encoder pretrained on radiology images — domain-aware, less ImageNet bias.
    Compare with [1] to see if ImageNet vs medical encoder changes the ranking.
    """
    tag = 'masked' if use_masked else 'full'
    log.info(f'=== [3] FID-CT-RadImageNet [{tag}] ===')

    imgs_a, imgs_b, imgs_c, imgs_d = _prepare_ct_images(
        group_a, group_b, hospital, magnet, use_masked)
    log.info(f'  A={len(imgs_a)}  B={len(imgs_b)}  C={len(imgs_c)}  D={len(imgs_d)}')

    fa = radimagenet_features(imgs_a, rad_path) if imgs_a else np.empty((0, 2048))
    fb = radimagenet_features(imgs_b, rad_path) if imgs_b else np.empty((0, 2048))
    fc = radimagenet_features(imgs_c, rad_path) if imgs_c else np.empty((0, 2048))
    fd = radimagenet_features(imgs_d, rad_path) if imgs_d else np.empty((0, 2048))
    log.info(f'  Features: A{fa.shape}  B{fb.shape}  C{fc.shape}  D{fd.shape}')

    results = {'method': f'RadImageNet_ResNet50_2048D_CT{"_masked" if use_masked else ""}'}
    _fid_all_pairs(fa, fb, fc, fd, results)
    return results


def run_fid_sino_rin(group_a: list[dict], group_b: list[dict],
                     hospital: list[tuple], magnet: list[tuple] | None,
                     use_masked: bool,
                     rad_path: Path,
                     max_c: int | None = None,
                     max_d: int | None = None) -> dict:
    """[4] FID — RadImageNet ResNet50 on sinogram images.

    Best of both worlds: projection-domain input + radiology-aware encoder.
    Two images similar visually may still differ in sinogram — this catches that.
    """
    tag = 'masked' if use_masked else 'full'
    log.info(f'=== [4] FID-Sino-RadImageNet [{tag}] ===')

    imgs_a, imgs_b, imgs_c, imgs_d = _prepare_ct_images(
        group_a, group_b, hospital, magnet, use_masked, max_c, max_d)
    log.info(f'  CT images: A={len(imgs_a)}  B={len(imgs_b)}  '
             f'C={len(imgs_c)}  D={len(imgs_d)}')

    log.info('  Computing sinograms...')
    sinos_a = sinogram_images(imgs_a)
    sinos_b = sinogram_images(imgs_b)
    sinos_c = sinogram_images(imgs_c) if imgs_c else []
    sinos_d = sinogram_images(imgs_d) if imgs_d else []

    fa = radimagenet_features(sinos_a, rad_path) if sinos_a else np.empty((0, 2048))
    fb = radimagenet_features(sinos_b, rad_path) if sinos_b else np.empty((0, 2048))
    fc = radimagenet_features(sinos_c, rad_path) if sinos_c else np.empty((0, 2048))
    fd = radimagenet_features(sinos_d, rad_path) if sinos_d else np.empty((0, 2048))
    log.info(f'  Features: A{fa.shape}  B{fb.shape}  C{fc.shape}  D{fd.shape}')

    results = {'method': f'RadImageNet_ResNet50_2048D_Sino{"_masked" if use_masked else ""}'}
    _fid_all_pairs(fa, fb, fc, fd, results)
    return results


def run_fid_sino_stats(group_a: list[dict], group_b: list[dict],
                       hospital: list[tuple], magnet: list[tuple] | None,
                       use_masked: bool,
                       max_c: int | None = None,
                       max_d: int | None = None) -> dict:
    """[5] FID — 270D sinogram statistics (no neural network).

    Features: [mean, std, P95] × 90 projection angles = 270D vector per image.
    No NN required — fully interpretable: each dimension = one projection angle.
    Resistant to ImageNet/RadImageNet encoder bias.
    """
    tag = 'masked' if use_masked else 'full'
    log.info(f'=== [5] FID-Sino-Stats (270D) [{tag}] ===')

    imgs_a, imgs_b, imgs_c, imgs_d = _prepare_ct_images(
        group_a, group_b, hospital, magnet, use_masked, max_c, max_d)
    log.info(f'  A={len(imgs_a)}  B={len(imgs_b)}  C={len(imgs_c)}  D={len(imgs_d)}')

    log.info('  Computing sinogram statistics...')
    fa = np.array([sinogram_stat_features(img) for img in imgs_a]) if imgs_a else np.empty((0, 270))
    fb = np.array([sinogram_stat_features(img) for img in imgs_b]) if imgs_b else np.empty((0, 270))
    fc = (np.array([sinogram_stat_features(img) for img in imgs_c])
          if imgs_c else np.empty((0, 270)))
    fd = (np.array([sinogram_stat_features(img) for img in imgs_d])
          if imgs_d else np.empty((0, 270)))
    log.info(f'  Features: A{fa.shape}  B{fb.shape}  C{fc.shape}  D{fd.shape}')

    results = {'method': f'SinogramStats_270D{"_masked" if use_masked else ""}'}
    _fid_all_pairs(fa, fb, fc, fd, results)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def _plot_distributions(df_gen: pd.DataFrame, df_aapm: pd.DataFrame, out_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, col in zip(axes.ravel(), FEATURE_COLS):
        for df, label, color, ls in [
            (df_aapm, 'AAPM',      'steelblue',  '-'),
            (df_gen,  'Generated', 'darkorange', '--'),
        ]:
            if col not in df.columns:
                continue
            vals = df[col].dropna()
            lo, hi = vals.quantile(0.01), vals.quantile(0.99)
            vals = vals.clip(lo, hi)
            ax.hist(vals, bins=40, density=True, alpha=0.45,
                    color=color, histtype='stepfilled')
            ax.hist(vals, bins=40, density=True, alpha=1.0,
                    color=color, histtype='step', lw=1.5, ls=ls, label=label)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel('normalized value')
        ax.legend(fontsize=8)

    plt.suptitle('Physical features — AAPM vs Generated', fontsize=13)
    plt.tight_layout()
    path = out_dir / 'physical_features_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f'Saved: {path}')


def _plot_ssim_histogram(ssim_result: dict, out_dir: Path):
    if not ssim_result.get('scores'):
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    scores = [r['masked_ssim'] for r in ssim_result['scores']]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=20, color='steelblue', edgecolor='k', alpha=0.8)
    ax.axvline(np.mean(scores), color='red', lw=2, ls='--',
               label=f'mean={np.mean(scores):.3f}')
    ax.set_xlabel('Masked SSIM')
    ax.set_ylabel('Image count')
    ax.set_title('Masked SSIM — Generated vs AAPM (inside artifact mask)')
    ax.legend()
    path = out_dir / 'ssim_histogram.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f'Saved: {path}')


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHT DOWNLOAD HELPERS
# ══════════════════════════════════════════════════════════════════════════════

RAD_IMAGENET_URL = (
    'https://huggingface.co/Lab-Rasool/RadImageNet/resolve/main/ResNet50.pt'
)
IMAGENET_URL = (
    'https://download.pytorch.org/models/resnet50-0676ba61.pth'
)


def _download_weights(url: str, path: Path, label: str) -> bool:
    if path.exists():
        return True
    log.info(f'{label} weights not found at {path} — attempting download...')
    try:
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path, reporthook=lambda b, bs, t: None)
        log.info(f'Downloaded {label} weights -> {path}')
        return True
    except Exception as e:
        log.warning(f'Download failed: {e}')
        log.warning(f'Run manually:  wget {url} -O {path}')
        return False


def _ensure_rad_imagenet(path: Path) -> bool:
    return _download_weights(RAD_IMAGENET_URL, path, 'RadImageNet')


def _ensure_imagenet(path: Path | None) -> Path | None:
    if path is None:
        log.info('--imagenet not set; torchvision will use its cache '
                 '(~/.cache/torch/hub/checkpoints/) — requires internet on first run')
        return None
    _download_weights(IMAGENET_URL, path, 'ImageNet ResNet50')
    return path if path.exists() else None


# ══════════════════════════════════════════════════════════════════════════════
# SAVE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _save_fid_result(result: dict, name: str, out_dir: Path):
    """Save FID result dict to JSON and flat CSV."""
    with open(out_dir / f'{name}.json', 'w') as f:
        json.dump(result, f, indent=2)

    rows = [{'comparison': k, 'frechet_distance': v}
            for k, v in result.items() if isinstance(v, float)]
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / f'{name}.csv', index=False)
        log.info(f'CSV -> {out_dir / f"{name}.csv"}')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description='Evaluation of generated CT data quality',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Data paths ────────────────────────────────────────────────────────────
    p.add_argument('--generated',  required=True, nargs='+', type=Path,
                   help='One or more directories with gauss_*.json + _gen_error.raw')
    p.add_argument('--rpi',        required=True, type=Path,
                   help='RPI root dir (contains body*/ subdirs)')
    p.add_argument('--real',       required=True, type=Path,
                   help='Directory with hospital *.raw images')
    p.add_argument('--features',   required=True, type=Path,
                   help='features_norm.csv — normalized AAPM features')
    p.add_argument('--clip-stats', required=True, type=Path,
                   help='feature_clip_stats.json — normalization statistics')
    p.add_argument('--out',        required=True, type=Path,
                   help='Output directory for JSON, CSV, PNG results')
    p.add_argument('--bodies',     nargs='+', default=None, metavar='BODY',
                   help='Body variants to include (e.g. --bodies body8 body9)')
    p.add_argument('--magnet',     type=Path, default=None,
                   help='Directory with magnet/jitter images (jitter*.raw)')
    # ── Model weights ─────────────────────────────────────────────────────────
    p.add_argument('--rad-imagenet', type=Path, default=None,
                   help='RadImageNet ResNet50 weights (.pt). '
                        'Downloaded from HuggingFace if missing.')
    p.add_argument('--imagenet',     type=Path, default=None,
                   help='ImageNet ResNet50 weights (.pth). '
                        'If not set, torchvision uses ~/.cache/torch/hub/checkpoints/.')
    # ── Experiment skip flags ─────────────────────────────────────────────────
    p.add_argument('--skip-ssim',         action='store_true', help='Skip Masked SSIM')
    p.add_argument('--skip-physical-fid', action='store_true', help='Skip Physical FID')
    p.add_argument('--skip-fid-1', action='store_true', help='Skip [1] FID-CT-ImageNet')
    p.add_argument('--skip-fid-2', action='store_true', help='Skip [2] FID-Sino-ImageNet')
    p.add_argument('--skip-fid-3', action='store_true', help='Skip [3] FID-CT-RadImageNet')
    p.add_argument('--skip-fid-4', action='store_true', help='Skip [4] FID-Sino-RadImageNet')
    p.add_argument('--skip-fid-5', action='store_true', help='Skip [5] FID-Sino-Stats (270D)')
    # ── Mask mode ─────────────────────────────────────────────────────────────
    p.add_argument('--mask-mode', choices=['unmasked', 'masked', 'both'], default='unmasked',
                   help='"unmasked" = full image (default); '
                        '"masked" = only artifact region (background -> HU_LO); '
                        '"both" = run both variants.')
    # ── Sinogram subsampling ──────────────────────────────────────────────────
    p.add_argument('--sino-max-c', type=int, default=2000, metavar='N',
                   help='Max hospital (C) images for sinogram experiments [2][4][5]. '
                        'Radon transform is slow. 0 = no limit.')
    p.add_argument('--sino-max-d', type=int, default=0, metavar='N',
                   help='Max magnet (D) images for sinogram experiments. 0 = no limit.')

    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # ── Mask modes ────────────────────────────────────────────────────────────
    if args.mask_mode == 'both':
        mask_modes = [False, True]
    elif args.mask_mode == 'masked':
        mask_modes = [True]
    else:
        mask_modes = [False]

    # ── Weight availability ───────────────────────────────────────────────────
    imagenet_weights = _ensure_imagenet(args.imagenet)

    rad_available = False
    if args.rad_imagenet is not None:
        rad_available = _ensure_rad_imagenet(args.rad_imagenet)
        if not rad_available:
            log.warning('RadImageNet weights unavailable; [3] and [4] will be skipped')
    else:
        log.warning('--rad-imagenet not set; [3] FID-CT-RIN and [4] FID-Sino-RIN skipped')

    run_fid3 = rad_available and not args.skip_fid_3
    run_fid4 = rad_available and not args.skip_fid_4

    # ── Load data ─────────────────────────────────────────────────────────────
    samples = []
    for gen_dir in args.generated:
        samples += load_generated_samples(gen_dir, bodies=args.bodies)

    bodies   = args.bodies or sorted({s['body'] for s in samples})
    hospital = load_hospital_images(args.real)
    magnet   = load_magnet_images(args.magnet) if args.magnet else []

    with open(args.clip_stats) as f:
        clip_stats = json.load(f)

    group_a, group_b = build_groups(samples, args.rpi)

    run_paired = len(group_a) > 0
    if not run_paired:
        log.warning('No generated/AAPM pairs — experiments requiring paired data skipped')

    # ── Session metadata ──────────────────────────────────────────────────────
    run_meta = {
        'generated_dirs': [str(d) for d in args.generated],
        'rpi_dir':        str(args.rpi),
        'real_dir':       str(args.real),
        'features_csv':   str(args.features),
        'bodies_used':    bodies,
        'n_generated':    len(group_a),
        'n_hospital':     len(hospital),
        'n_magnet':       len(magnet),
    }
    with open(args.out / 'run_meta.json', 'w') as f:
        json.dump(run_meta, f, indent=2)

    summary = {**run_meta}
    results = {}

    # ── SSIM ─────────────────────────────────────────────────────────────────
    if run_paired and not args.skip_ssim:
        results['ssim'] = run_ssim_exp(group_a, group_b)
        with open(args.out / 'ssim_scores.json', 'w') as f:
            json.dump(results['ssim'], f, indent=2)
        if results['ssim']['scores']:
            pd.DataFrame(results['ssim']['scores']).to_csv(
                args.out / 'ssim_scores.csv', index=False)
        _plot_ssim_histogram(results['ssim'], args.out)
        if results['ssim'].get('mean') is not None:
            summary['ssim_mean'] = results['ssim']['mean']
            summary['ssim_std']  = results['ssim']['std']
            summary['ssim_n']    = results['ssim']['n']
    else:
        log.info('[SSIM] skipped')

    # ── Physical FID ─────────────────────────────────────────────────────────
    if run_paired and not args.skip_physical_fid:
        fid_result, df_gen, df_aapm = run_physical_fid(
            group_a, group_b, args.features, clip_stats,
            bodies=bodies,
            magnet_images=magnet or None,
        )
        results['physical_fid'] = fid_result
        with open(args.out / 'physical_fid.json', 'w') as f:
            json.dump(fid_result, f, indent=2)

        phys_csv_groups = {k: fid_result[k] for k in
                           ('Generated_vs_AAPM', 'Generated_vs_Magnet', 'AAPM_vs_Magnet')
                           if fid_result.get(k)}
        if phys_csv_groups:
            save_fid_csv(phys_csv_groups, args.out / 'physical_fid.csv')

        if not df_gen.empty and FEATURE_COLS[0] in df_gen.columns:
            df_gen_out  = df_gen[FEATURE_COLS].copy();  df_gen_out['dataset']  = 'Generated'
            df_aapm_out = df_aapm[FEATURE_COLS].copy(); df_aapm_out['dataset'] = 'AAPM'
            pd.concat([df_gen_out, df_aapm_out], ignore_index=True).to_csv(
                args.out / 'features_gen_vs_aapm.csv', index=False)
            _plot_distributions(df_gen, df_aapm, args.out)

        summary['phys_fd_Generated_vs_AAPM'] = (
            (fid_result.get('Generated_vs_AAPM') or {}).get('combined_6D'))
    else:
        log.info('[Phys] Physical FID skipped')

    # ── FID experiments — iterate over mask modes ─────────────────────────────
    sino_max_c = args.sino_max_c or None
    sino_max_d = args.sino_max_d or None

    for use_masked in mask_modes:
        sfx   = '_masked' if use_masked else ''
        s_pfx = 'masked_' if use_masked else ''

        # [1] FID-CT-ImageNet
        if not args.skip_fid_1:
            r = run_fid_ct_in(group_a, group_b, hospital, magnet or None,
                              use_masked, imagenet_weights)
            results[f'fid_ct_in{sfx}'] = r
            _save_fid_result(r, f'fid_ct_in{sfx}', args.out)
            for cmp in ['Generated(A)_vs_AAPM(B)',
                        'Generated(A)_vs_Hospital(C)', 'AAPM(B)_vs_Hospital(C)']:
                if isinstance(r.get(cmp), float):
                    summary[f'{s_pfx}fid1_ct_in_{cmp}'] = r[cmp]
        else:
            log.info('[1] FID-CT-ImageNet skipped')

        # [2] FID-Sino-ImageNet
        if not args.skip_fid_2:
            r = run_fid_sino_in(group_a, group_b, hospital, magnet or None,
                                use_masked, imagenet_weights, sino_max_c, sino_max_d)
            results[f'fid_sino_in{sfx}'] = r
            _save_fid_result(r, f'fid_sino_in{sfx}', args.out)
            for cmp in ['Generated(A)_vs_AAPM(B)',
                        'Generated(A)_vs_Hospital(C)', 'AAPM(B)_vs_Hospital(C)']:
                if isinstance(r.get(cmp), float):
                    summary[f'{s_pfx}fid2_sino_in_{cmp}'] = r[cmp]
        else:
            log.info('[2] FID-Sino-ImageNet skipped')

        # [3] FID-CT-RadImageNet
        if run_fid3:
            r = run_fid_ct_rin(group_a, group_b, hospital, magnet or None,
                               use_masked, args.rad_imagenet)
            results[f'fid_ct_rin{sfx}'] = r
            _save_fid_result(r, f'fid_ct_rin{sfx}', args.out)
            for cmp in ['Generated(A)_vs_AAPM(B)',
                        'Generated(A)_vs_Hospital(C)', 'AAPM(B)_vs_Hospital(C)']:
                if isinstance(r.get(cmp), float):
                    summary[f'{s_pfx}fid3_ct_rin_{cmp}'] = r[cmp]
        else:
            log.info('[3] FID-CT-RadImageNet skipped')

        # [4] FID-Sino-RadImageNet
        if run_fid4:
            r = run_fid_sino_rin(group_a, group_b, hospital, magnet or None,
                                 use_masked, args.rad_imagenet, sino_max_c, sino_max_d)
            results[f'fid_sino_rin{sfx}'] = r
            _save_fid_result(r, f'fid_sino_rin{sfx}', args.out)
            for cmp in ['Generated(A)_vs_AAPM(B)',
                        'Generated(A)_vs_Hospital(C)', 'AAPM(B)_vs_Hospital(C)']:
                if isinstance(r.get(cmp), float):
                    summary[f'{s_pfx}fid4_sino_rin_{cmp}'] = r[cmp]
        else:
            log.info('[4] FID-Sino-RadImageNet skipped')

        # [5] FID-Sino-Stats (270D, no NN)
        if not args.skip_fid_5:
            r = run_fid_sino_stats(group_a, group_b, hospital, magnet or None,
                                   use_masked, sino_max_c, sino_max_d)
            results[f'fid_sino_stats{sfx}'] = r
            _save_fid_result(r, f'fid_sino_stats{sfx}', args.out)
            for cmp in ['Generated(A)_vs_AAPM(B)',
                        'Generated(A)_vs_Hospital(C)', 'AAPM(B)_vs_Hospital(C)']:
                if isinstance(r.get(cmp), float):
                    summary[f'{s_pfx}fid5_sino_stats_{cmp}'] = r[cmp]
        else:
            log.info('[5] FID-Sino-Stats skipped')

    # ── Summary ───────────────────────────────────────────────────────────────
    with open(args.out / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame([summary]).to_csv(args.out / 'summary.csv', index=False)

    log.info('\n=== SUMMARY ===')
    for k, v in summary.items():
        if k not in run_meta:
            log.info(f'  {k:<55} {v}')
    log.info(f'\nResults -> {args.out}')


if __name__ == '__main__':
    main()
