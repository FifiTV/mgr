"""
Metal-mask resolution shared by GaussianDataset, precompute_priors.py,
infer_gaussian.py and eval_localization.py.

Source order (identical everywhere):
    1. <body_dir>/<d>/training_body_metalonlymask_img{N}_512x512x1.raw
       for d in metal_mask_dirs (default: Metal/, then Mask/)
    2. fallback: |I_metal - I_clean| > threshold_hu   (config gaussian.metal_threshold_hu)

History: the original in_ch=3 variant only looked in Metal/. On the local copy of
the data the metalonlymask files live in Mask/, so Mask/ is searched too.
Set gaussian.metal_mask_dirs = ["Metal"] to reproduce the old lookup exactly.
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

SHAPE = (512, 512)
METAL_MASK_DIRS: Tuple[str, ...] = ('Metal', 'Mask')


def metal_mask_filename(img_id: int) -> str:
    return f'training_body_metalonlymask_img{img_id}_512x512x1.raw'


def find_metal_mask(body_dir: Path, img_id: int,
                    mask_dirs: Sequence[str] = METAL_MASK_DIRS) -> Optional[Path]:
    """Return path of the binary metal-only mask file, or None if absent."""
    body_dir = Path(body_dir)
    for d in mask_dirs:
        p = body_dir / d / metal_mask_filename(img_id)
        if p.exists():
            return p
    return None


def _load_raw(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(SHAPE)


def load_binary_metal_mask(metal_path: Optional[Path],
                           art_path: Optional[Path] = None,
                           clean_path: Optional[Path] = None,
                           threshold_hu: float = 500.0) -> Tuple[np.ndarray, str]:
    """Load a boolean [512, 512] metal mask.

    Returns (mask, source) where source is 'file:<dir name>' or 'threshold:<hu>'.
    Raises FileNotFoundError when neither a mask file nor an (art, clean) pair exists.
    """
    if metal_path is not None:
        return _load_raw(Path(metal_path)) > 0.5, f'file:{Path(metal_path).parent.name}'
    if art_path is None or clean_path is None:
        raise FileNotFoundError(
            'No metal mask file and no (Baseline, Target) pair for threshold fallback.')
    diff = np.abs(_load_raw(Path(art_path)) - _load_raw(Path(clean_path)))
    return diff > threshold_hu, f'threshold:{threshold_hu:g}'
