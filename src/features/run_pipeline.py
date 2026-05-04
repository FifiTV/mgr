"""
run_pipeline.py -- Feature extraction and normalization for multiple body series
=================================================================================

Usage:
    python run_pipeline.py --data-dir data/raw/RPI --output results/features.csv
    python run_pipeline.py --data-dir data/raw/RPI --output results/features.csv --bodies body1 body2
    python run_pipeline.py --data-dir data/raw/RPI --output results/features.csv --no-cache

What it does:
    1. For each body/* -> extract features -> per-body CSV (with cache)
    2. Merge all body series -> <output> (raw)
    3. Normalize on the merged dataset -> <output_stem>_norm.csv
    4. Save clip_stats.json (required for inference)

Input directory structure:
    <data-dir>/
        body1/
            Baseline/   training_body_metalart_imgN_512x512x1.raw
            Target/     training_body_nometal_imgN_512x512x1.raw
        body2/ ...

Output structure (relative to --output):
    <output>                    -- raw features, all body series merged
    <output_stem>_norm.csv      -- normalized features
    feature_clip_stats.json     -- normalization parameters (keep this!)
    body1/features.csv          -- per-body cache
    body2/features.csv
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Supports import from both src/features/ and project root
try:
    from feature_extraction import HU_CLIP, load_raw, extract_features
except ModuleNotFoundError:
    from src.features.feature_extraction import HU_CLIP, load_raw, extract_features


FEATURE_COLS = [
    'peak_amplitude', 'spatial_extent', 'bbox_ratio',
    'dark_to_bright_ratio', 'angular_concentration', 'texture_roughness'
]

# Features with right-skewed distributions -- apply log1p before normalization
# Selection criterion: mean/median > 1.5 on body1 (verify after each new body)
LOG_FEATURES = {'peak_amplitude', 'angular_concentration'}


# ── Single body extraction ──────────────────────────────────────────────────────
def extract_one_body(body_dir: Path, results_dir: Path,
                     use_cache: bool = True) -> pd.DataFrame:
    """
    Extract features for all image pairs in a single body directory.

    Cache: if results_dir/body_name/features.csv already exists ->
    loads from disk without re-extraction. Delete the file or use --no-cache
    to force recomputation (e.g. after changing the extraction pipeline).

    Args:
        body_dir    : Path to body directory (must contain Baseline/ and Target/)
        results_dir : Path where per-body CSV is saved
        use_cache   : whether to use cached results

    Returns:
        DataFrame with features and 'source' column, index = img_id.
        Empty DataFrame if no data found or on error.
    """
    cache_path = results_dir / body_dir.name / 'features.csv'

    if use_cache and cache_path.exists():
        print(f'  {body_dir.name}: loaded from cache ({cache_path})')
        df = pd.read_csv(cache_path, index_col='img_id')
        print(f'    {len(df)} images')
        return df

    def _img_id(path: Path) -> int:
        m = re.search(r'img(\d+)', path.stem)
        return int(m.group(1)) if m else -1

    baseline_dir = body_dir / 'Baseline'
    target_dir   = body_dir / 'Target'

    if not baseline_dir.exists():
        print(f'  [SKIPPED] Baseline/ not found: {body_dir}')
        return pd.DataFrame()

    baseline_files = sorted(
        [p for p in baseline_dir.glob('*.raw')
         if re.search(r'img(\d+)', p.stem)],
        key=_img_id
    )

    if not baseline_files:
        print(f'  [SKIPPED] No .raw files in: {baseline_dir}')
        return pd.DataFrame()

    records = []
    for img_art_path in tqdm(baseline_files, desc=f'  {body_dir.name}', leave=False):
        img_id       = _img_id(img_art_path)
        i_clean_path = target_dir / f'training_body_nometal_img{img_id}_512x512x1.raw'
        if not i_clean_path.exists():
            continue
        try:
            img_art = load_raw(img_art_path)
            i_clean = load_raw(i_clean_path)
            row           = extract_features(img_art, i_clean)
            row['img_id'] = img_id
            row['source'] = body_dir.name
            records.append(row)
        except Exception as e:
            print(f'    ERROR img{img_id}: {e}')

    if not records:
        print(f'  [SKIPPED] No image pairs found in: {body_dir}')
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index('img_id').sort_index()

    n_saturated = (df['tau'] >= HU_CLIP * 0.97).sum()
    if n_saturated:
        print(f'    Warning: {n_saturated} images with tau >= {HU_CLIP*0.97:.0f} HU '
              f'(saturated -- features less precise, but retained)')
    print(f'    {body_dir.name}: {len(df)} images')

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)

    return df


# ── Normalization ───────────────────────────────────────────────────────────────
def robust_normalize(df: pd.DataFrame,
                     clip_stats: dict = None,
                     lo_pct: float = 1.0,
                     hi_pct: float = 99.0) -> tuple:
    """
    Robust Min-Max normalization with optional log1p transformation.

    Modes:
        clip_stats=None -> FIT: compute bounds from df (use on train set)
        clip_stats=dict -> TRANSFORM: use provided bounds (use on val/test)

    Pipeline per feature:
        1. [optional] log1p(x)           -- stabilizes skewed distributions
        2. clip([P_lo, P_hi])             -- removes outliers
        3. (x - lo) / (hi - lo) -> [0,1] -- MinMax

    Args:
        df         : DataFrame with raw features
        clip_stats : None (fit) or dict {feature: {lo, hi, log}} (transform)
        lo_pct     : lower clipping percentile (default 1%)
        hi_pct     : upper clipping percentile (default 99%)

    Returns:
        df_norm    : DataFrame with normalized features [0, 1]
        clip_stats : normalization parameters -- save to JSON before inference!
    """
    df_work  = df[FEATURE_COLS].copy().astype(float)
    fit_mode = clip_stats is None
    if fit_mode:
        clip_stats = {}

    mode_str = 'FIT (from this data)' if fit_mode else 'TRANSFORM (from clip_stats)'
    print(f'\nNormalization [{mode_str}]:')
    print(f'  {"Feature":<28} {"Transform":<10} {"Lo":>10} {"Hi":>10}')
    print(f'  {"-" * 62}')

    for col in FEATURE_COLS:
        use_log = col in LOG_FEATURES

        if use_log:
            df_work[col] = np.log1p(df_work[col])

        if fit_mode:
            lo = float(np.percentile(df_work[col].dropna(), lo_pct))
            hi = float(np.percentile(df_work[col].dropna(), hi_pct))
            clip_stats[col] = {'lo': lo, 'hi': hi, 'log': use_log}
        else:
            lo = clip_stats[col]['lo']
            hi = clip_stats[col]['hi']

        df_work[col] = (df_work[col].clip(lo, hi) - lo) / (hi - lo + 1e-9)
        transform_str = 'log1p' if use_log else 'linear'
        print(f'  {col:<28} {transform_str:<10} {lo:>10.4f} {hi:>10.4f}')

    return df_work, clip_stats


def denormalize_label(y_norm: np.ndarray, clip_stats: dict) -> np.ndarray:
    """
    Inverse normalization -- used during model inference.

    Transforms normalized vector y in [0,1]^6 back to original units
    (HU or dimensionless).

    Args:
        y_norm     : shape (6,) or (N, 6)
        clip_stats : loaded from feature_clip_stats.json

    Returns:
        x_orig : labels in original units
    """
    y = np.array(y_norm, dtype=float)
    x = np.zeros_like(y)
    for i, col in enumerate(FEATURE_COLS):
        lo, hi  = clip_stats[col]['lo'], clip_stats[col]['hi']
        use_log = clip_stats[col]['log']
        x_clip  = y[..., i] * (hi - lo) + lo
        x[..., i] = np.expm1(x_clip) if use_log else x_clip
    return x


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Extract metal artifact features from CT data (RPI format).'
    )
    parser.add_argument(
        '--data-dir', required=True, type=Path,
        help='Root folder containing body1/, body2/, ... subdirectories '
             '(e.g. data/raw/RPI)'
    )
    parser.add_argument(
        '--output', required=True, type=Path,
        help='Output CSV path for raw features '
             '(e.g. results/features.csv). '
             'Also creates: <stem>_norm.csv and feature_clip_stats.json'
    )
    parser.add_argument(
        '--bodies', nargs='+', default=None,
        metavar='BODY',
        help='Body series to process (e.g. body1 body3). '
             'Default: all body* subdirectories in --data-dir'
    )
    parser.add_argument(
        '--no-cache', action='store_true',
        help='Ignore existing cache files and recompute from scratch'
    )
    parser.add_argument(
        '--clip-lo', type=float, default=1.0,
        help='Lower percentile for normalization clipping (default: 1.0)'
    )
    parser.add_argument(
        '--clip-hi', type=float, default=99.0,
        help='Upper percentile for normalization clipping (default: 99.0)'
    )
    args = parser.parse_args()

    data_root   = args.data_dir
    output_path = args.output
    results_dir = output_path.parent
    use_cache   = not args.no_cache

    if not data_root.exists():
        print(f'ERROR: --data-dir does not exist: {data_root}', file=sys.stderr)
        sys.exit(1)

    results_dir.mkdir(parents=True, exist_ok=True)

    # Discover body directories
    if args.bodies is None:
        body_dirs = sorted([d for d in data_root.iterdir()
                            if d.is_dir() and d.name.startswith('body')])
    else:
        body_dirs = [data_root / name for name in args.bodies]

    if not body_dirs:
        print(f'ERROR: No body* directories found in: {data_root}', file=sys.stderr)
        sys.exit(1)

    print(f'Data:   {data_root}')
    print(f'Output: {output_path}')
    print(f'Series: {[d.name for d in body_dirs]}')
    print('=' * 60)

    # ── 1. Extraction ──────────────────────────────────────────────────────────
    print('\n[1/3] Feature extraction')
    frames = []
    for body_dir in body_dirs:
        df_body = extract_one_body(body_dir, results_dir, use_cache=use_cache)
        if not df_body.empty:
            frames.append(df_body)

    if not frames:
        print('ERROR: Failed to load any data.', file=sys.stderr)
        sys.exit(1)

    df_all = pd.concat(frames, axis=0)
    print(f'\nTotal: {len(df_all)} images')
    print('Distribution by series:')
    for src, count in df_all['source'].value_counts().items():
        print(f'  {src}: {count}')

    df_all.to_csv(output_path)
    print(f'\nSaved raw features: {output_path}')

    # ── 2. Raw statistics ──────────────────────────────────────────────────────
    print('\n[2/3] Raw statistics:')
    print(df_all[FEATURE_COLS].describe().round(4).to_string())

    # ── 3. Normalization ───────────────────────────────────────────────────────
    print('\n[3/3] Normalization')
    df_norm, clip_stats = robust_normalize(
        df_all,
        clip_stats=None,
        lo_pct=args.clip_lo,
        hi_pct=args.clip_hi,
    )
    df_norm['source'] = df_all['source'].values

    norm_path = output_path.with_name(output_path.stem + '_norm.csv')
    df_norm.to_csv(norm_path)
    print(f'\nSaved normalized: {norm_path}')

    stats_path = results_dir / 'feature_clip_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(clip_stats, f, indent=2)
    print(f'Saved clip_stats: {stats_path}')

    print('\nStatistics after normalization:')
    print(df_norm[FEATURE_COLS].describe().round(3).to_string())

    # Round-trip test
    y_test = np.array([df_norm[col].median() for col in FEATURE_COLS])
    x_back = denormalize_label(y_test, clip_stats)
    print('\nRound-trip test (normalized median -> original scale):')
    for col, val in zip(FEATURE_COLS, x_back):
        print(f'  {col:<28} {val:>10.4f}  (original median: {df_all[col].median():.4f})')

    print('\n[DONE]')
    return df_all, df_norm, clip_stats


if __name__ == '__main__':
    main()
