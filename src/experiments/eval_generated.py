"""eval_generated.py — evaluation of generated CT data quality.

Groups compared:
  A = I_clean_AAPM + I_error_generated   (synthesized CT from our model)
  B = I_clean_AAPM + I_error_AAPM        (AAPM CT reconstructed without raw metal)
  C = Hospital_raw                        (real hospital CT, used directly)

Experiments:
  [1] Masked SSIM   — I_error_gen vs I_error_aapm inside artifact mask (A vs B, pixel-level)
  [2] Physical FID  — Frechet on 6 physical features (A vs B only; hospital has no pairs)
  [3] Sinogram FID  — Radon statistics 270D: A vs B vs C (no NN)
                      + RadImageNet on sinogram images (optional)
  [4] Med-FID       — RadImageNet ResNet50 on full CT images: A vs B vs C

Usage:
  python3 eval_generated.py \\
    --generated data/raw/generated \\
    --rpi       data/raw/RPI \\
    --real      data/raw/real \\
    --features  results/features_norm.csv \\
    --clip-stats results/feature_clip_stats.json \\
    --out       results/eval_generated \\
    --bodies    body8 \\
    [--rad-imagenet models/RadImageNet-ResNet50_notop.pt]
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
    imagenet_features, load_generated_samples, load_hospital_images, load_raw,
    normalize_features, radimagenet_features, save_fid_csv,
    sinogram_images, sinogram_stat_features,
)
from src.features.feature_extraction import compute_tau, extract_features, preprocess

# HU range used for sinogram / Med-FID inputs — clips out raw metal saturation
HU_LO, HU_HI = -1000.0, 3000.0
# Physical range for I_error clipping
ERR_CLIP = 3000.0


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE GROUP CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_groups(samples: list[dict], rpi_dir: Path) -> tuple[list[dict], list[dict]]:
    """Match each generated sample to its AAPM pair and build image groups.

    Returns two equal-length lists (only samples with an existing AAPM pair):
      group_a — generated: I_clean + I_error_gen
      group_b — AAPM:      I_clean + I_error_aapm  (no raw metal point)

    Each record contains:
      I_error_*  — clipped I_error in HU  (for SSIM and Physical FID)
      I_synth/I_art — full HU image       (for Physical FID feature extraction)
      I_sino     — clipped to [HU_LO, HU_HI] (fair input for Sinogram/Med-FID)
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

        # Clip I_error to physical range (removes generator artifacts beyond CT range)
        I_error_gen_clip  = np.clip(I_error_gen,     -ERR_CLIP, ERR_CLIP)
        I_error_real_clip = np.clip(I_art - I_clean, -ERR_CLIP, ERR_CLIP)

        group_a.append({
            'img_id':      s['img_id'],
            'body':        s['body'],
            'I_synth':     (I_clean + I_error_gen_clip).astype(np.float32),
            'I_clean':     I_clean,
            'I_error_gen': I_error_gen_clip,
            # Clipped to [HU_LO, HU_HI] — used as sinogram / Med-FID input
            'I_sino':      np.clip(I_clean + I_error_gen_clip, HU_LO, HU_HI).astype(np.float32),
        })
        group_b.append({
            'img_id':       s['img_id'],
            'body':         s['body'],
            'I_art':        I_art,            # raw AAPM — used for SSIM artifact mask
            'I_clean':      I_clean,
            'I_error_aapm': I_error_real_clip,
            # I_clean + I_error (no raw metal blob) clipped to [HU_LO, HU_HI]
            'I_sino':       np.clip(I_clean + I_error_real_clip, HU_LO, HU_HI).astype(np.float32),
        })

    log.info(f'A/B pairs: {len(group_a)} (from {len(samples)} generated samples)')
    return group_a, group_b


# ══════════════════════════════════════════════════════════════════════════════
# [1] MASKED SSIM
# ══════════════════════════════════════════════════════════════════════════════

def run_ssim_exp(group_a: list[dict], group_b: list[dict]) -> dict:
    """[1] Masked SSIM — I_error_gen vs I_error_aapm inside the artifact mask."""
    log.info('=== [1] Masked SSIM ===')
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

    vals = [r['masked_ssim'] for r in scores]
    result = {'scores': scores, 'mean': float(np.mean(vals)),
              'std': float(np.std(vals)), 'n': len(scores)}
    log.info(f'  mean={result["mean"]:.4f} ± {result["std"]:.4f}  n={result["n"]}')
    return result


# ══════════════════════════════════════════════════════════════════════════════
# [2] PHYSICAL FID  (A vs B only — hospital excluded, no clean reference)
# ══════════════════════════════════════════════════════════════════════════════

def run_physical_fid(group_a: list[dict], group_b: list[dict],
                     features_norm_path: Path, clip_stats: dict,
                     bodies: list[str] | None = None,
                     ) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """[2] Frechet distance on 6 physical features: Generated (A) vs AAPM (B).

    Hospital data excluded — no (I_art, I_clean) pairs available.
    """
    log.info('=== [2] Physical FID — Generated vs AAPM ===')

    # Extract features from synthesized images (A)
    records_gen = []
    for a in group_a:
        row = extract_features(a['I_synth'], a['I_clean'])
        row['img_id'] = a['img_id']
        row['body']   = a['body']
        records_gen.append(row)
    df_gen_raw = pd.DataFrame(records_gen)
    df_gen = normalize_features(df_gen_raw, clip_stats)

    # Load AAPM features from CSV (already normalized), filter by body if requested
    df_aapm = pd.read_csv(features_norm_path, index_col='img_id')
    if bodies and 'source' in df_aapm.columns:
        df_aapm = df_aapm[df_aapm['source'].isin(bodies)]
    log.info(f'  Generated: {len(df_gen)}  AAPM: {len(df_aapm)}')

    if len(df_gen) < 2:
        log.warning('  [2] Skipped — fewer than 2 generated samples with matched AAPM pairs')
        return {'Generated_vs_AAPM': None, 'n_gen': len(df_gen), 'n_aapm': len(df_aapm)}, df_gen, df_aapm

    fd = frechet_per_feature(df_gen, df_aapm, 'Generated', 'AAPM')
    return {'Generated_vs_AAPM': fd, 'n_gen': len(df_gen), 'n_aapm': len(df_aapm)}, df_gen, df_aapm


# ══════════════════════════════════════════════════════════════════════════════
# [3] SINOGRAM FID
# ══════════════════════════════════════════════════════════════════════════════

def run_sinogram_fid(group_a: list[dict], group_b: list[dict],
                     hospital: list[tuple[Path, np.ndarray]],
                     rad_imagenet_path: Path | None = None) -> dict:
    """[3] Sinogram FID — Radon-domain Frechet distance.

    A = Radon(I_clean + I_error_gen)   clipped to [HU_LO, HU_HI]
    B = Radon(I_clean + I_error_aapm)  clipped to [HU_LO, HU_HI]  (no raw metal)
    C = Radon(Hospital_raw)            clipped to [HU_LO, HU_HI]

    Variant 1: 270D statistics (mean/std/P95 per projection angle) — no NN required.
    Variant 2: RadImageNet on sinogram images — if weights are provided.
    """
    log.info('=== [3] Sinogram FID ===')

    imgs_a = [a['I_sino'] for a in group_a]
    imgs_b = [b['I_sino'] for b in group_b]
    imgs_c = [np.clip(img, HU_LO, HU_HI).astype(np.float32) for _, img in hospital]

    log.info(f'  A={len(imgs_a)}  B={len(imgs_b)}  C={len(imgs_c)}')

    # ── Variant 1: statistics (270D) ──────────────────────────────────────────
    log.info('  Computing sinogram statistics (270D)...')
    feats_a = np.array([sinogram_stat_features(img) for img in imgs_a])
    feats_b = np.array([sinogram_stat_features(img) for img in imgs_b])
    feats_c = (np.array([sinogram_stat_features(img) for img in imgs_c])
               if imgs_c else np.empty((0, 270)))

    results = {'method_stats': 'sinogram_statistics_270D'}
    for fa, fb, la, lb in [
        (feats_a, feats_b, 'Generated(A)', 'AAPM(B)'),
        (feats_a, feats_c, 'Generated(A)', 'Hospital(C)'),
        (feats_b, feats_c, 'AAPM(B)',      'Hospital(C)'),
    ]:
        key = f'stats_{la}_vs_{lb}'
        if fa.shape[0] >= 2 and fb.shape[0] >= 2:
            results[key] = frechet_distance(fa, fb)
            log.info(f'  FD_stats({la} vs {lb}) = {results[key]:.4f}')
        else:
            results[key] = None
            log.warning(f'  Not enough samples for {key}')

    # ── Variant 2: RadImageNet on sinogram images ─────────────────────────────
    if rad_imagenet_path is not None and rad_imagenet_path.exists():
        log.info('  Extracting RadImageNet features from sinogram images...')
        sinos_a = sinogram_images(imgs_a)
        sinos_b = sinogram_images(imgs_b)
        sinos_c = sinogram_images(imgs_c) if imgs_c else []

        rf_a = radimagenet_features(sinos_a, rad_imagenet_path)
        rf_b = radimagenet_features(sinos_b, rad_imagenet_path)
        rf_c = (radimagenet_features(sinos_c, rad_imagenet_path)
                if sinos_c else np.empty((0, 2048)))

        results['method_radimagenet_sino'] = 'RadImageNet_ResNet50_2048D_sinograms'
        for fa, fb, la, lb in [
            (rf_a, rf_b, 'Generated(A)', 'AAPM(B)'),
            (rf_a, rf_c, 'Generated(A)', 'Hospital(C)'),
            (rf_b, rf_c, 'AAPM(B)',      'Hospital(C)'),
        ]:
            key = f'radIN_sino_{la}_vs_{lb}'
            if fa.shape[0] >= 2 and fb.shape[0] >= 2:
                results[key] = frechet_distance(fa, fb)
                log.info(f'  FD_RadIN_sino({la} vs {lb}) = {results[key]:.4f}')
            else:
                results[key] = None
    else:
        log.info('  RadImageNet sinogram variant skipped (no --rad-imagenet)')

    return results


# ══════════════════════════════════════════════════════════════════════════════
# [4] MED-FID — RadImageNet on full CT images
# ══════════════════════════════════════════════════════════════════════════════

def run_med_fid(group_a: list[dict], group_b: list[dict],
                hospital: list[tuple[Path, np.ndarray]],
                rad_imagenet_path: Path) -> dict:
    """[4] Med-FID — Frechet distance using RadImageNet ResNet50 features.

    Images passed to the network (all clipped to [HU_LO, HU_HI]):
      A = I_clean + I_error_gen   (synthesized CT)
      B = I_clean + I_error_aapm  (AAPM CT without raw metal)
      C = Hospital_raw            (hospital CT, used directly)

    Each image: HU → [0, 1] → resize 224×224 → 3-channel → ResNet50 → 2048D.
    """
    log.info('=== [4] Med-FID (RadImageNet, full CT images) ===')

    imgs_a = [a['I_sino'] for a in group_a]
    imgs_b = [b['I_sino'] for b in group_b]
    imgs_c = [np.clip(img, HU_LO, HU_HI).astype(np.float32) for _, img in hospital]

    log.info(f'  A={len(imgs_a)}  B={len(imgs_b)}  C={len(imgs_c)}')
    log.info('  Extracting RadImageNet features from CT images...')

    rf_a = radimagenet_features(imgs_a, rad_imagenet_path)
    rf_b = radimagenet_features(imgs_b, rad_imagenet_path)
    rf_c = (radimagenet_features(imgs_c, rad_imagenet_path)
            if imgs_c else np.empty((0, 2048)))

    log.info(f'  Features: A{rf_a.shape}  B{rf_b.shape}  C{rf_c.shape}')

    results = {'method': 'RadImageNet_ResNet50_2048D_full_images'}
    for fa, fb, la, lb in [
        (rf_a, rf_b, 'Generated(A)', 'AAPM(B)'),
        (rf_a, rf_c, 'Generated(A)', 'Hospital(C)'),
        (rf_b, rf_c, 'AAPM(B)',      'Hospital(C)'),
    ]:
        key = f'{la}_vs_{lb}'
        if fa.shape[0] >= 2 and fb.shape[0] >= 2:
            results[key] = frechet_distance(fa, fb)
            log.info(f'  Med-FID({la} vs {lb}) = {results[key]:.4f}')
        else:
            results[key] = None
            log.warning(f'  Not enough samples for {key}')

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
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

RAD_IMAGENET_URL = (
    'https://huggingface.co/Lab-Rasool/RadImageNet/resolve/main/ResNet50.pt'
)


def _ensure_rad_imagenet(path: Path) -> bool:
    """Download RadImageNet weights if not present. Returns True if available."""
    if path.exists():
        return True
    log.info(f'RadImageNet weights not found at {path} — attempting download...')
    try:
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(RAD_IMAGENET_URL, path,
                                   reporthook=lambda b, bs, t: None)
        log.info(f'Downloaded RadImageNet weights → {path}')
        return True
    except Exception as e:
        log.warning(f'Download failed: {e}')
        log.warning('Run manually: wget ' + RAD_IMAGENET_URL + f' -O {path}')
        return False


def main():
    p = argparse.ArgumentParser(
        description='Evaluation of generated CT data quality',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--generated',    required=True, type=Path,
                   help='Directory with gauss_*.json + _gen_error.raw')
    p.add_argument('--rpi',          required=True, type=Path,
                   help='Parent directory containing body*/ (e.g. data/raw/RPI)')
    p.add_argument('--real',         required=True, type=Path,
                   help='Directory with hospital *.raw images (used directly, no extraction)')
    p.add_argument('--features',     required=True, type=Path,
                   help='features_norm.csv — normalized AAPM features (col: source=body*)')
    p.add_argument('--clip-stats',   required=True, type=Path,
                   help='feature_clip_stats.json — normalization statistics')
    p.add_argument('--out',          required=True, type=Path,
                   help='Output directory for JSON, CSV, PNG results')
    p.add_argument('--bodies',       nargs='+', default=None, metavar='BODY',
                   help='Body variants to include (e.g. --bodies body8). '
                        'Default: all bodies found in --generated.')
    p.add_argument('--rad-imagenet', type=Path, default=None,
                   help='RadImageNet-ResNet50_notop.pt weights. '
                        'If the file is missing it will be downloaded automatically.')
    # Coarse experiment groups (kept for backward compatibility)
    p.add_argument('--ssim-only',    action='store_true', help='Run only experiment [1]')
    p.add_argument('--fid-only',     action='store_true', help='Run only experiments [2][3][4]')
    # Per-experiment skip flags
    p.add_argument('--skip-ssim',         action='store_true', help='Skip [1] Masked SSIM')
    p.add_argument('--skip-physical-fid', action='store_true', help='Skip [2] Physical FID')
    p.add_argument('--skip-sinogram-fid', action='store_true', help='Skip [3] Sinogram FID')
    p.add_argument('--skip-med-fid',      action='store_true',
                   help='Skip [4] Med-FID (implied when --rad-imagenet is absent and download fails)')
    p.add_argument('--skip-standard-fid', action='store_true',
                   help='Skip [5] Standard FID (ImageNet ResNet50)')
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # ── Resolve experiment flags ──────────────────────────────────────────────
    run_ssim    = not args.fid_only  and not args.skip_ssim
    run_phys    = not args.ssim_only and not args.skip_physical_fid
    run_sino    = not args.ssim_only and not args.skip_sinogram_fid
    run_med     = not args.ssim_only and not args.skip_med_fid
    run_std_fid = not args.ssim_only and not args.skip_standard_fid

    # ── Ensure RadImageNet weights ────────────────────────────────────────────
    if run_med or (run_sino and args.rad_imagenet is not None):
        if args.rad_imagenet is None:
            log.warning('--rad-imagenet not specified; [3] RadIN variant and [4] Med-FID will be skipped')
            run_med = False
        elif not _ensure_rad_imagenet(args.rad_imagenet):
            log.warning('RadImageNet weights unavailable; [3] RadIN variant and [4] Med-FID skipped')
            run_med = False

    # ── Load data ────────────────────────────────────────────────────────────
    samples  = load_generated_samples(args.generated, bodies=args.bodies)
    bodies   = args.bodies or sorted({s['body'] for s in samples})
    hospital = load_hospital_images(args.real)

    with open(args.clip_stats) as f:
        clip_stats = json.load(f)

    group_a, group_b = build_groups(samples, args.rpi)

    if len(group_a) == 0:
        log.warning('No generated/AAPM pairs found — experiments requiring paired data will be skipped')
        run_ssim = False
        run_phys = False

    # ── Session metadata ─────────────────────────────────────────────────────
    run_meta = {
        'generated_dir': str(args.generated),
        'rpi_dir':       str(args.rpi),
        'real_dir':      str(args.real),
        'features_csv':  str(args.features),
        'bodies_used':   bodies,
        'n_generated':   len(group_a),
        'n_hospital':    len(hospital),
    }
    with open(args.out / 'run_meta.json', 'w') as f:
        json.dump(run_meta, f, indent=2)

    summary = {**run_meta}
    results = {}

    # ── [1] Masked SSIM ──────────────────────────────────────────────────────
    if run_ssim:
        results['ssim'] = run_ssim_exp(group_a, group_b)
        with open(args.out / 'ssim_scores.json', 'w') as f:
            json.dump(results['ssim'], f, indent=2)
        if results['ssim']['scores']:
            pd.DataFrame(results['ssim']['scores']).to_csv(
                args.out / 'ssim_scores.csv', index=False)
            log.info(f'CSV → {args.out / "ssim_scores.csv"}')
        _plot_ssim_histogram(results['ssim'], args.out)
        if results['ssim'].get('mean') is not None:
            summary['masked_ssim_mean'] = results['ssim']['mean']
            summary['masked_ssim_std']  = results['ssim']['std']
            summary['masked_ssim_n']    = results['ssim']['n']
    else:
        log.info('[1] Masked SSIM skipped')

    # ── [2] Physical FID ─────────────────────────────────────────────────────
    if run_phys:
        fid_result, df_gen, df_aapm = run_physical_fid(
            group_a, group_b, args.features, clip_stats, bodies=bodies,
        )
        results['physical_fid'] = fid_result
        with open(args.out / 'physical_fid.json', 'w') as f:
            json.dump(fid_result, f, indent=2)

        if fid_result.get('Generated_vs_AAPM'):
            save_fid_csv({'Generated_vs_AAPM': fid_result['Generated_vs_AAPM']},
                         args.out / 'physical_fid.csv')

        if not df_gen.empty and FEATURE_COLS[0] in df_gen.columns:
            df_gen_out  = df_gen[FEATURE_COLS].copy();  df_gen_out['dataset']  = 'Generated'
            df_aapm_out = df_aapm[FEATURE_COLS].copy(); df_aapm_out['dataset'] = 'AAPM'
            pd.concat([df_gen_out, df_aapm_out], ignore_index=True).to_csv(
                args.out / 'features_gen_vs_aapm.csv', index=False)
            log.info(f'CSV → {args.out / "features_gen_vs_aapm.csv"}')
            _plot_distributions(df_gen, df_aapm, args.out)

        summary['phys_fd_Generated_vs_AAPM'] = (
            (fid_result.get('Generated_vs_AAPM') or {}).get('combined_6D'))
    else:
        log.info('[2] Physical FID skipped')

    # ── [3] Sinogram FID ─────────────────────────────────────────────────────
    if run_sino:
        if len(group_a) == 0 and len(hospital) == 0:
            log.warning('[3] Sinogram FID skipped — no paired data and no hospital images')
        else:
            results['sinogram_fid'] = run_sinogram_fid(
                group_a, group_b, hospital,
                rad_imagenet_path=args.rad_imagenet if run_med else None,
            )
            with open(args.out / 'sinogram_fid.json', 'w') as f:
                json.dump(results['sinogram_fid'], f, indent=2)

            sino_rows = [{'comparison': k, 'frechet_distance': v}
                         for k, v in results['sinogram_fid'].items()
                         if isinstance(v, float)]
            if sino_rows:
                pd.DataFrame(sino_rows).to_csv(args.out / 'sinogram_fid.csv', index=False)
                log.info(f'CSV → {args.out / "sinogram_fid.csv"}')

            for cmp in ['stats_Generated(A)_vs_AAPM(B)',
                        'stats_Generated(A)_vs_Hospital(C)',
                        'stats_AAPM(B)_vs_Hospital(C)']:
                val = results['sinogram_fid'].get(cmp)
                if isinstance(val, float):
                    summary[f'sino_{cmp}'] = val
    else:
        log.info('[3] Sinogram FID skipped')

    # ── [4] Med-FID ──────────────────────────────────────────────────────────
    if run_med:
        results['med_fid'] = run_med_fid(group_a, group_b, hospital, args.rad_imagenet)
        with open(args.out / 'med_fid.json', 'w') as f:
            json.dump(results['med_fid'], f, indent=2)

        med_rows = [{'comparison': k, 'frechet_distance': v}
                    for k, v in results['med_fid'].items()
                    if isinstance(v, float)]
        if med_rows:
            pd.DataFrame(med_rows).to_csv(args.out / 'med_fid.csv', index=False)
            log.info(f'CSV → {args.out / "med_fid.csv"}')

        for cmp in ['Generated(A)_vs_AAPM(B)',
                    'Generated(A)_vs_Hospital(C)',
                    'AAPM(B)_vs_Hospital(C)']:
            val = results['med_fid'].get(cmp)
            if isinstance(val, float):
                summary[f'med_fid_{cmp}'] = val
    else:
        log.info('[4] Med-FID skipped')

    # ── [5] Standard FID — ImageNet ResNet50 ─────────────────────────────────
    if run_std_fid:
        if len(group_a) == 0 and len(hospital) == 0:
            log.warning('[5] Standard FID skipped — no data')
        else:
            log.info('=== [5] Standard FID (ImageNet ResNet50) ===')
            imgs_a = [a['I_sino'] for a in group_a]
            imgs_b = [b['I_sino'] for b in group_b]
            imgs_c = [np.clip(img, HU_LO, HU_HI).astype(np.float32) for _, img in hospital]

            rf_a = imagenet_features(imgs_a) if imgs_a else np.empty((0, 2048))
            rf_b = imagenet_features(imgs_b) if imgs_b else np.empty((0, 2048))
            rf_c = imagenet_features(imgs_c) if imgs_c else np.empty((0, 2048))

            log.info(f'  Features: A{rf_a.shape}  B{rf_b.shape}  C{rf_c.shape}')

            std_fid = {'method': 'ImageNet_ResNet50_2048D'}
            for fa, fb, la, lb in [
                (rf_a, rf_b, 'Generated(A)', 'AAPM(B)'),
                (rf_a, rf_c, 'Generated(A)', 'Hospital(C)'),
                (rf_b, rf_c, 'AAPM(B)',      'Hospital(C)'),
            ]:
                key = f'{la}_vs_{lb}'
                if fa.shape[0] >= 2 and fb.shape[0] >= 2:
                    std_fid[key] = frechet_distance(fa, fb)
                    log.info(f'  Std-FID({la} vs {lb}) = {std_fid[key]:.4f}')
                else:
                    std_fid[key] = None

            results['standard_fid'] = std_fid
            with open(args.out / 'standard_fid.json', 'w') as f:
                json.dump(std_fid, f, indent=2)

            std_rows = [{'comparison': k, 'frechet_distance': v}
                        for k, v in std_fid.items() if isinstance(v, float)]
            if std_rows:
                pd.DataFrame(std_rows).to_csv(args.out / 'standard_fid.csv', index=False)
                log.info(f'CSV → {args.out / "standard_fid.csv"}')

            for cmp in ['Generated(A)_vs_AAPM(B)',
                        'Generated(A)_vs_Hospital(C)',
                        'AAPM(B)_vs_Hospital(C)']:
                val = std_fid.get(cmp)
                if isinstance(val, float):
                    summary[f'std_fid_{cmp}'] = val
    else:
        log.info('[5] Standard FID skipped')

    # ── Summary ──────────────────────────────────────────────────────────────
    with open(args.out / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame([summary]).to_csv(args.out / 'summary.csv', index=False)

    log.info('\n=== SUMMARY ===')
    for k, v in summary.items():
        if k not in run_meta:
            log.info(f'  {k:<45} {v}')
    log.info(f'\nResults → {args.out}')


if __name__ == '__main__':
    main()
