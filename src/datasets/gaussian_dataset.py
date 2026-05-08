"""
Dataset for Gaussian Vicinal DDPM — metal artifact generation.

Loads triplets (I_clean, I_metal, M_metal) from RPI .raw files and joins
with a precomputed, normalized feature vector y from a CSV file.

Normalization:
    I_clean, I_metal : clip to [-1000, 3000] HU -> scale to [-1, 1]
    I_error          : clip to [-3000,  3000] HU -> scale to [-1, 1]  (divide by 3000)
    M_metal          : binary mask [0, 1] loaded from Metal/ directory
                       Falls back to threshold on |I_metal - I_clean| if Metal/ missing.

Random metal augmentation:
    With probability p_random_metal, M_metal for sample i is replaced by a random
    metal mask from another sample in the dataset. This lets the model learn to
    combine any anatomy with any metal position.

Feature loading helpers:
    load_features_df(path, clip_stats_path=None)
        path         : path to a single CSV or a directory containing body*/features.csv
        clip_stats   : path to feature_clip_stats.json (required for raw per-body CSVs)
        Returns a DataFrame with normalized features and columns:
            img_id, source, peak_amplitude, spatial_extent, bbox_ratio,
            dark_to_bright_ratio, angular_concentration, texture_roughness
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

SHAPE       = (512, 512)
FEATURE_COLS = [
    'peak_amplitude', 'spatial_extent', 'bbox_ratio',
    'dark_to_bright_ratio', 'angular_concentration', 'texture_roughness',
]
LOG_FEATURES = {'peak_amplitude', 'angular_concentration'}
METAL_THRESHOLD_HU = 2500.0  # fallback metal mask threshold


# ── Normalization ────────────────────────────────────────────────────────────────

def normalize_ct(x: np.ndarray, hu_min: float = -1000.0,
                 hu_max: float = 3000.0) -> np.ndarray:
    """Clip to [hu_min, hu_max] and scale to [-1, 1]."""
    x = np.clip(x, hu_min, hu_max)
    return (x - hu_min) / (hu_max - hu_min) * 2.0 - 1.0


def normalize_error(x: np.ndarray, clip: float = 3000.0) -> np.ndarray:
    """Clip I_error to [-clip, clip] and scale to [-1, 1]."""
    return np.clip(x, -clip, clip) / clip


# ── Feature CSV loading ──────────────────────────────────────────────────────────

def _normalize_raw_features(df: pd.DataFrame, clip_stats: dict) -> pd.DataFrame:
    """Apply clip_stats normalization (log1p + min-max) to raw feature columns."""
    df = df.copy()
    for col in FEATURE_COLS:
        if col not in df.columns:
            continue
        stats = clip_stats[col]
        if stats.get('log', False):
            df[col] = np.log1p(df[col].astype(float))
        lo, hi = stats['lo'], stats['hi']
        df[col] = (df[col].clip(lo, hi) - lo) / (hi - lo + 1e-9)
    return df


def load_features_df(path: Union[str, Path],
                     clip_stats_path: Optional[Union[str, Path]] = None
                     ) -> pd.DataFrame:
    """
    Load and (if necessary) normalize a feature DataFrame.

    Args:
        path           : Path to:
                           - a single CSV (already normalized, e.g. features_norm.csv), OR
                           - a directory containing body*/features.csv per-body caches
        clip_stats_path: Path to feature_clip_stats.json for normalizing raw CSVs.
                         Required when loading per-body raw CSVs.
                         If None and loading raw CSVs, raises ValueError.

    Returns:
        DataFrame with columns: img_id (index), source, and FEATURE_COLS (normalized).
    """
    path = Path(path)

    if path.is_file():
        df = pd.read_csv(path, index_col='img_id')
        logger.info(f"Loaded features: {len(df)} rows from {path}")

        # Detect if normalization is needed: if any feature has values > 1.0
        needs_norm = any(
            df[col].max() > 1.05 for col in FEATURE_COLS if col in df.columns
        )
        if needs_norm:
            if clip_stats_path is None:
                raise ValueError(
                    f"Features in {path} appear raw (values > 1). "
                    "Provide clip_stats_path to normalize, or use a _norm.csv file."
                )
            with open(clip_stats_path) as f:
                clip_stats = json.load(f)
            df = _normalize_raw_features(df, clip_stats)
            logger.info("Applied normalization from clip_stats.")
        return df

    if path.is_dir():
        body_csvs = sorted(path.glob('*/features.csv'))
        if not body_csvs:
            raise FileNotFoundError(f"No body*/features.csv found in {path}")

        clip_stats = None
        stats_file = path / 'feature_clip_stats.json'
        if stats_file.exists():
            with open(stats_file) as f:
                clip_stats = json.load(f)
            logger.info(f"Loaded clip_stats from {stats_file}")
        elif clip_stats_path is not None:
            with open(clip_stats_path) as f:
                clip_stats = json.load(f)

        frames = []
        for csv_path in body_csvs:
            df_body = pd.read_csv(csv_path, index_col='img_id')
            frames.append(df_body)
            logger.info(f"  {csv_path.parent.name}: {len(df_body)} rows")

        df = pd.concat(frames, axis=0)

        # Check if normalization is needed
        needs_norm = any(
            df[col].max() > 1.05 for col in FEATURE_COLS if col in df.columns
        )
        if needs_norm:
            if clip_stats is None:
                raise ValueError(
                    "Per-body CSVs contain raw features but no clip_stats.json found. "
                    "Run run_pipeline.py first to generate normalization stats, "
                    "or provide clip_stats_path."
                )
            df = _normalize_raw_features(df, clip_stats)

        logger.info(f"Total features loaded: {len(df)} rows from {len(body_csvs)} body dirs")
        return df

    raise FileNotFoundError(f"Features path does not exist: {path}")


# ── Dataset ──────────────────────────────────────────────────────────────────────

class GaussianDataset(Dataset):
    """
    Dataset for Gaussian Vicinal DDPM.

    Each sample returns:
        i_clean  : [1, 512, 512]  clean CT image normalized to [-1, 1]
        i_error  : [1, 512, 512]  I_metal - I_clean normalized to [-1, 1]
        m_metal  : [1, 512, 512]  binary metal mask [0, 1]
        y        : [6]            normalized feature vector

    Directory structure per body variant:
        <body_dir>/Baseline/  training_body_metalart_imgN_512x512x1.raw
        <body_dir>/Target/    training_body_nometal_imgN_512x512x1.raw
        <body_dir>/Metal/     training_body_metalonly_imgN_512x512x1.raw  (optional)
    """

    def __init__(self,
                 body_dirs: List[Union[str, Path]],
                 features_df: pd.DataFrame,
                 p_random_metal: float = 0.5,
                 metal_threshold_hu: float = METAL_THRESHOLD_HU,
                 preload: bool = False):
        """
        Args:
            body_dirs        : list of paths to body variant directories
            features_df      : normalized feature DataFrame (index=img_id, col 'source')
            p_random_metal   : probability of replacing M_metal with a random sample's mask
            metal_threshold_hu : HU threshold for fallback metal mask (when Metal/ missing).
                                 Applied to |I_metal - I_clean|; at 2500 HU this isolates
                                 the implant itself, not the surrounding bloom artifacts.
            preload          : if True, all .raw files are loaded into RAM at init.
                               Fast for small datasets; skip for large ones (>50 GB).
        """
        self.p_random_metal      = p_random_metal
        self.metal_threshold_hu  = metal_threshold_hu
        self._cache: Optional[dict] = {} if preload else None

        # Build per-variant lookup: {body_name: {img_id: (art_path, clean_path, metal_path)}}
        self._records: List[dict] = []
        self._all_metal_paths: List[Optional[Path]] = []  # for random metal swap

        for body_dir in body_dirs:
            body_dir  = Path(body_dir)
            body_name = body_dir.name
            baseline  = body_dir / 'Baseline'
            target    = body_dir / 'Target'
            metal_dir = body_dir / 'Metal'

            if not baseline.exists() or not target.exists():
                logger.warning(f"Skipping {body_dir}: missing Baseline/ or Target/")
                continue

            # Filter features for this body
            body_feats = features_df[features_df['source'] == body_name]
            if body_feats.empty:
                logger.warning(f"No features found for source='{body_name}', skipping")
                continue

            feat_ids = set(body_feats.index.tolist())

            def _img_id(p: Path) -> Optional[int]:
                m = re.search(r'img(\d+)', p.stem)
                return int(m.group(1)) if m else None

            for art_path in sorted(baseline.glob('*.raw')):
                img_id = _img_id(art_path)
                if img_id is None or img_id not in feat_ids:
                    continue

                clean_path = target / f'training_body_nometal_img{img_id}_512x512x1.raw'
                if not clean_path.exists():
                    continue

                metal_path = metal_dir / f'training_body_metalonly_img{img_id}_512x512x1.raw'
                metal_path = metal_path if metal_path.exists() else None

                row = body_feats.loc[img_id, FEATURE_COLS].values.astype(np.float32)

                self._records.append({
                    'art_path':   art_path,
                    'clean_path': clean_path,
                    'metal_path': metal_path,
                    'y':          row,
                })
                self._all_metal_paths.append(metal_path)

        if not self._records:
            raise RuntimeError(
                "GaussianDataset is empty. Check body_dirs and features_df."
            )
        logger.info(f"GaussianDataset: {len(self._records)} samples from "
                    f"{len(body_dirs)} body dir(s)")

    def __len__(self) -> int:
        return len(self._records)

    def _load_raw(self, path: Path) -> np.ndarray:
        if self._cache is not None:
            if path not in self._cache:
                self._cache[path] = np.fromfile(path, dtype=np.float32).reshape(SHAPE)
            return self._cache[path].copy()  # copy: normalization must not modify cache
        return np.fromfile(path, dtype=np.float32).reshape(SHAPE)

    def _get_metal_mask(self, record: dict) -> np.ndarray:
        """Load metal mask from Metal/ dir or compute from implant position.

        Fallback threshold (metal_threshold_hu = 2500 HU) isolates the implant voxels
        only -- bloom artifacts typically stay below 2040 HU (bloom_max) so they are
        NOT included in this mask. That is the intended behaviour: M_metal encodes
        implant position, not artifact extent.
        """
        if record['metal_path'] is not None:
            mask = self._load_raw(record['metal_path'])
            # Binary: some files store the mask as soft HU values
            return (mask > 0.5).astype(np.float32)

        # Fallback: threshold on |I_metal - I_clean| to locate the implant
        i_art   = self._load_raw(record['art_path'])
        i_clean = self._load_raw(record['clean_path'])
        return (np.abs(i_art - i_clean) > self.metal_threshold_hu).astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        rec     = self._records[idx]
        i_art   = self._load_raw(rec['art_path'])
        i_clean = self._load_raw(rec['clean_path'])

        # I_error in HU before normalization (needed for proper error clipping)
        i_error = i_art - i_clean

        # Metal mask
        m_metal = self._get_metal_mask(rec)

        # Random metal swap augmentation
        if self.p_random_metal > 0.0 and np.random.rand() < self.p_random_metal:
            swap_idx    = np.random.randint(len(self._records))
            swap_rec    = self._records[swap_idx]
            m_metal_src = swap_rec['metal_path']

            if m_metal_src is not None:
                m_metal = self._load_raw(m_metal_src)
                m_metal = (m_metal > 0.5).astype(np.float32)
            else:
                # Fallback: compute swap mask from difference
                i_art_swap   = self._load_raw(swap_rec['art_path'])
                i_clean_swap = self._load_raw(swap_rec['clean_path'])
                m_metal = (np.abs(i_art_swap - i_clean_swap) >
                            self.metal_threshold_hu).astype(np.float32)

        # Normalize
        i_clean_n = normalize_ct(i_clean)
        i_error_n = normalize_error(i_error)

        return {
            'i_clean': torch.from_numpy(i_clean_n[None]).float(),  # [1, H, W]
            'i_error': torch.from_numpy(i_error_n[None]).float(),  # [1, H, W]
            'm_metal': torch.from_numpy(m_metal[None]).float(),    # [1, H, W]
            'y':       torch.from_numpy(rec['y']).float(),         # [6]
        }
