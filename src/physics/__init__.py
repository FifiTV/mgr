"""
Physics-based helpers: metal-mask resolution and the analytic artifact prior.
"""

from .metal_mask import (
    METAL_MASK_DIRS, metal_mask_filename, find_metal_mask, load_binary_metal_mask,
)
from .artifact_prior import (
    PriorParams, compute_artifact_prior, compute_artifact_prior_raw,
    normalize_prior, rescale_unit_prior, global_scale_from, prior_filename,
    apply_prior_transform, finalize_prior, unit_prior,
    DEFAULT_PRIOR_FEATURES, GEOMETRIC_FEATURES,
)

__all__ = [
    "METAL_MASK_DIRS", "metal_mask_filename", "find_metal_mask", "load_binary_metal_mask",
    "PriorParams", "compute_artifact_prior", "compute_artifact_prior_raw",
    "normalize_prior", "rescale_unit_prior", "global_scale_from", "prior_filename",
    "apply_prior_transform", "finalize_prior", "unit_prior",
    "DEFAULT_PRIOR_FEATURES", "GEOMETRIC_FEATURES",
]
