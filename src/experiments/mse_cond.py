"""
mse_cond.py -- Conditioning MSE for the Gaussian Vicinal DDPM model.

For each generated image:
  1. Load I_error_gen ([-1,1]) and I_clean_norm ([-1,1]) from the raw/ directory
  2. Denormalize I_clean to HU: (norm + 1) * 2000 - 1000
  3. Reconstruct I_metal_gen = I_clean_HU + I_error_gen * error_scale
  4. Extract physical features from the pair (I_metal_gen, I_clean_HU)
  5. Compare with y_target stored in the metadata JSON
  6. Compute MSE between y_pred and y_target

Supported directory layouts
---------------------------
Flat (single run):
  infer_dir/
    raw/  *.json  *_gen_error.raw  img/*_clean.raw

Multi-body (ablation runs):
  infer_dir/
    body8/  raw/  ...
    body9/  raw/  ...

Both layouts are detected automatically.

Usage:
  # flat
  python src/experiments/mse_cond.py \
      --infer-dir results/gaussian_infer/bs8_6feat \
      --clip-stats results/feature_clip_stats.json \
      --out results/mse_cond_6feat.csv

  # multi-body
  python src/experiments/mse_cond.py \
      --infer-dir results/gaussian_ablation_no_mask_no_dist_4 \
      --clip-stats results/feature_clip_stats.json \
      --out results/mse_cond_ablation.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# allows importing src.* from the mgr/ directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.feature_extraction import extract_features, load_raw  # noqa: E402

FEATURE_COLS = [
    'peak_amplitude', 'spatial_extent', 'bbox_ratio',
    'dark_to_bright_ratio', 'angular_concentration', 'texture_roughness',
]

# normalize_ct: (clip(x, -1000, 3000) - (-1000)) / 4000 * 2 - 1
# inverse:      (norm + 1) * 2000 - 1000
_CT_NORM_SCALE  = 2000.0
_CT_NORM_OFFSET = 1000.0


def denormalize_ct(norm: np.ndarray) -> np.ndarray:
    """Inverse of normalize_ct: [-1, 1] -> HU."""
    return (norm + 1.0) * _CT_NORM_SCALE - _CT_NORM_OFFSET


def normalize_feature(val: float, col: str, clip_stats: dict) -> float:
    """Feature normalization identical to training (log1p + min-max)."""
    stats   = clip_stats[col]
    lo, hi  = stats['lo'], stats['hi']
    if stats['log']:
        val = float(np.log1p(val))
    return float(np.clip((val - lo) / (hi - lo + 1e-9), 0.0, 1.0))


def _collect_raw_dirs(infer_dir: Path) -> list[Path]:
    """
    Return all raw/ directories to scan, supporting two layouts:
      - flat:       infer_dir/raw/
      - multi-body: infer_dir/bodyN/raw/  (any immediate subdirectory containing raw/)
    """
    flat = infer_dir / 'raw'
    if flat.is_dir():
        return [flat]

    raw_dirs = sorted(sub / 'raw' for sub in infer_dir.iterdir()
                      if sub.is_dir() and (sub / 'raw').is_dir())
    return raw_dirs


def compute_mse_cond(infer_dir: Path, clip_stats_path: Path) -> pd.DataFrame:
    """
    infer_dir:       directory with infer_gaussian.py outputs; supports both
                     flat (raw/ directly inside) and multi-body (bodyN/raw/) layouts
    clip_stats_path: feature_clip_stats.json (used for feature normalization)
    """
    with open(clip_stats_path) as f:
        clip_stats = json.load(f)

    raw_dirs = _collect_raw_dirs(infer_dir)
    if not raw_dirs:
        print(f'No raw/ directories found under {infer_dir}')
        return pd.DataFrame()

    print(f'Found {len(raw_dirs)} raw/ director{"y" if len(raw_dirs) == 1 else "ies"}: '
          f'{[str(d) for d in raw_dirs]}')

    records = []
    meta    = {}   # keep last meta in scope for feat_names_final
    for raw_dir in raw_dirs:
        json_files = sorted(raw_dir.glob('*.json'))
        if not json_files:
            print(f'  No JSON files in {raw_dir}, skipping')
            continue

        for meta_path in json_files:
            with open(meta_path) as f:
                meta = json.load(f)

            stem        = meta_path.stem
            error_path  = raw_dir / f'{stem}_gen_error.raw'
            clean_path  = raw_dir / 'img' / f'{stem}_clean.raw'
            error_scale = float(meta.get('error_scale', 642.8))

            if not error_path.exists() or not clean_path.exists():
                print(f'Missing files for {stem}, skipping')
                continue

            # Load normalized files (range [-1, 1])
            I_error_gen  = load_raw(error_path)   # model output, [-1, 1]
            I_clean_norm = load_raw(clean_path)    # normalize_ct output, [-1, 1]

            # Denormalize clean CT to HU (required by extract_features)
            I_clean_HU = denormalize_ct(I_clean_norm)

            # Reconstruct artifact CT in HU
            I_metal_gen = I_clean_HU + I_error_gen * error_scale

            # Extract physical features
            try:
                feats_raw = extract_features(I_metal_gen, I_clean_HU)
            except Exception as e:
                print(f'Feature extraction error for {stem}: {e}')
                continue

            # y_target from metadata (already normalized to [0,1] as during training)
            y_target_list = meta.get('y_target')
            feature_cols  = meta.get('y_target_named', {})
            # Use feature_cols from metadata if available, otherwise fall back to FEATURE_COLS
            feat_names = list(feature_cols.keys()) if feature_cols else FEATURE_COLS
            n_feats    = len(feat_names)

            if y_target_list is None or len(y_target_list) != n_feats:
                print(f'Invalid y_target in {stem}, skipping')
                continue

            y_target = np.array(y_target_list, dtype=np.float32)

            # Normalize y_pred using the same method as during training
            y_pred = np.array(
                [normalize_feature(feats_raw[col], col, clip_stats) for col in feat_names],
                dtype=np.float32,
            )

            mse_per_feat = (y_target - y_pred) ** 2
            mse_total    = float(mse_per_feat.mean())

            row = {
                'stem':      stem,
                'img_id':    meta.get('img_id'),
                'body':      meta.get('body'),
                'mse_total': mse_total,
            }
            for i, col in enumerate(feat_names):
                row[f'y_target_{col}'] = float(y_target[i])
                row[f'y_pred_{col}']   = float(y_pred[i])
                row[f'mse_{col}']      = float(mse_per_feat[i])

            records.append(row)

    df = pd.DataFrame(records)

    if df.empty:
        print('No results.')
        return df

    feat_names_final = list(meta.get('y_target_named', {}).keys()) if records else FEATURE_COLS

    print(f'\nN = {len(df)} images')
    print(f'MSE_cond (total): mean={df["mse_total"].mean():.4f}  '
          f'std={df["mse_total"].std():.4f}  '
          f'median={df["mse_total"].median():.4f}')
    print('\nMSE per feature:')
    for col in feat_names_final:
        key = f'mse_{col}'
        if key in df.columns:
            vals = df[key]
            print(f'  {col:<30} mean={vals.mean():.4f}  std={vals.std():.4f}')

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Conditioning MSE -- Gaussian Vicinal DDPM',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--infer-dir',  type=Path, required=True,
                        help='Directory with infer_gaussian.py outputs')
    parser.add_argument('--clip-stats', type=Path, required=True,
                        help='feature_clip_stats.json')
    parser.add_argument('--out',        type=Path, default=Path('results/mse_cond.csv'),
                        help='Output CSV path (default: results/mse_cond.csv)')
    args = parser.parse_args()

    df = compute_mse_cond(args.infer_dir, args.clip_stats)
    if not df.empty:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f'\nSaved: {args.out}')


if __name__ == '__main__':
    main()
