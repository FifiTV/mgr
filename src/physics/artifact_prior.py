"""
Analytic metal-artifact prior I_prior computed from a binary metal mask M.

    p_M     = radon(M * mu_scale, theta)              # metal trace in the sinogram
    delta_p = -beta * p_M**2                          # simplest beam-hardening model
    I_prior = iradon(delta_p, theta, filter='ramp')   # FBP back to image space

I_prior carries the GEOMETRY of the artifact (streak positions and directions,
including streaks connecting several implants). Amplitude and texture are left
to the feature vector y (AdaLN) — therefore the prior is normalized.

Note on mu_scale / beta:
    I_prior is exactly proportional to beta * mu_scale**2 (radon/iradon are linear,
    the only nonlinearity is the square). After per-image normalization both
    parameters cancel, and after global normalization they cancel as well, because
    the global constant is computed with the same parameters. They are kept
    configurable (and recorded in every manifest) for a future, richer physics
    model (e.g. adding a linear/water term) and for prior_norm='none'.

Normalization (prior_norm):
    'per_image' : I_prior / max|I_prior|                         -> max|.| = 1
    'global'    : I_prior / global_scale, clipped to [-1, 1]; global_scale is
                  computed on the TRAINING set (percentile of per-image max|I_prior|)
                  and stored in prior_norm.json / model_config.json.
    'none'      : raw values (debugging only)

Cache representation: unit prior (I_prior / max|I_prior|, float16) + max|I_prior|.
Every normalization / transform is derived from it at load time (finalize_prior),
so changing norm/transform does NOT require recomputing radon/iradon.

Dynamic range: max|I_prior| is reached on the implant pixel itself; outside the
implant (+5 px) the streaks are only ~1-5 % of it (p99 ~ 0.01). Optional
transform='signed_log' compresses that range (off by default).

Geometry:
    Only 'parallel' (skimage radon / iradon) is implemented. The AAPM CT-MAR data
    were simulated with a fan-beam scanner; see the report in desc/desc/.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple

import numpy as np

SHAPE = (512, 512)

PRIOR_VERSION = 1   # bump when the formula changes -> invalidates caches

# Features kept in y for the prior variant (intensity / texture only).
DEFAULT_PRIOR_FEATURES = ['peak_amplitude', 'dark_to_bright_ratio', 'texture_roughness']
# Features that duplicate the geometry carried by I_prior (shortcut risk).
GEOMETRIC_FEATURES = {'spatial_extent', 'bbox_ratio', 'angular_concentration'}

_VALID_NORMS = ('per_image', 'global', 'none')
_VALID_GEOMETRIES = ('parallel',)
_VALID_TRANSFORMS = ('none', 'signed_log')


@dataclass
class PriorParams:
    n_angles: int = 360            # projections over [0, 180) degrees
    mu_scale: float = 1.0
    beta: float = 1.0
    geometry: str = 'parallel'
    norm: str = 'per_image'        # per_image | global | none
    global_percentile: float = 99.0
    global_scale: Optional[float] = None   # filled from the training set when norm='global'
    # Optional dynamic-range compression applied AFTER normalization (load time,
    # cache-independent). max|I_prior| sits on the implant pixel itself, streaks are
    # only ~1-5 % of it. 'signed_log': sign(x)*log1p(k|x|)/log1p(k).
    transform: str = 'none'        # none | signed_log
    log_k: float = 100.0
    filter_name: str = 'ramp'
    circle: bool = True
    version: int = PRIOR_VERSION

    def __post_init__(self):
        if self.norm not in _VALID_NORMS:
            raise ValueError(f'prior norm must be one of {_VALID_NORMS}, got {self.norm!r}')
        if self.transform not in _VALID_TRANSFORMS:
            raise ValueError(f'prior transform must be one of {_VALID_TRANSFORMS}, '
                             f'got {self.transform!r}')
        if self.geometry not in _VALID_GEOMETRIES:
            raise NotImplementedError(
                f'prior geometry {self.geometry!r} not implemented '
                f'(available: {_VALID_GEOMETRIES}). skimage provides parallel beam only.')

    @classmethod
    def from_config(cls, prior_cfg: Optional[dict]) -> 'PriorParams':
        prior_cfg = prior_cfg or {}
        names = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in prior_cfg.items() if k in names})

    def to_dict(self) -> dict:
        return asdict(self)

    def geometry_key(self) -> dict:
        """Parameters that determine cached (unit-normalized) prior files."""
        return {k: getattr(self, k) for k in
                ('n_angles', 'mu_scale', 'beta', 'geometry', 'filter_name', 'circle', 'version')}

    def theta(self) -> np.ndarray:
        return np.linspace(0.0, 180.0, int(self.n_angles), endpoint=False)


def prior_filename(img_id: int) -> str:
    """Cache file name, tied to the metal-mask file name."""
    return f'training_body_metalonlymask_img{img_id}_512x512x1.prior.npy'


def _inscribed_circle(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    r = min(h, w) // 2
    return (yy - h // 2) ** 2 + (xx - w // 2) ** 2 <= r ** 2


def compute_artifact_prior_raw(mask: np.ndarray, params: PriorParams) -> np.ndarray:
    """Unnormalized I_prior (float64, [H, W]) from a binary/soft metal mask."""
    from skimage.transform import radon, iradon

    m = np.asarray(mask, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f'mask must be square 2D, got {m.shape}')
    if not m.any():
        return np.zeros_like(m)

    if params.circle:
        # radon(circle=True) assumes zero outside the inscribed circle
        m = m * _inscribed_circle(m.shape)

    theta   = params.theta()
    p_m     = radon(m * params.mu_scale, theta=theta, circle=params.circle)
    delta_p = -params.beta * p_m ** 2
    return iradon(delta_p, theta=theta, filter_name=params.filter_name,
                  circle=params.circle, output_size=m.shape[0])


def normalize_prior(raw: np.ndarray, norm: str = 'per_image',
                    global_scale: Optional[float] = None) -> np.ndarray:
    """Apply prior normalization; returns float32."""
    if norm == 'none':
        return raw.astype(np.float32)
    if norm == 'per_image':
        s = float(np.abs(raw).max())
        return (raw / s).astype(np.float32) if s > 0 else np.zeros_like(raw, np.float32)
    if norm == 'global':
        if not global_scale or global_scale <= 0:
            raise ValueError('prior_norm=global requires a positive global_scale '
                             '(computed on the training set).')
        return np.clip(raw / global_scale, -1.0, 1.0).astype(np.float32)
    raise ValueError(f'unknown prior norm {norm!r}')


def rescale_unit_prior(unit: np.ndarray, raw_max_abs: float,
                       norm: str, global_scale: Optional[float] = None) -> np.ndarray:
    """Convert a cached unit prior (max|.|=1) to the requested normalization."""
    unit = unit.astype(np.float32)
    if norm == 'per_image':
        return unit
    if norm == 'global':
        if not global_scale or global_scale <= 0:
            raise ValueError('prior_norm=global requires a positive global_scale.')
        return np.clip(unit * (raw_max_abs / global_scale), -1.0, 1.0).astype(np.float32)
    if norm == 'none':
        return (unit * raw_max_abs).astype(np.float32)
    raise ValueError(f'unknown prior norm {norm!r}')


def apply_prior_transform(x: np.ndarray, transform: str = 'none',
                          log_k: float = 100.0) -> np.ndarray:
    if transform == 'none':
        return x.astype(np.float32)
    if transform == 'signed_log':
        return (np.sign(x) * np.log1p(log_k * np.abs(x)) / np.log1p(log_k)).astype(np.float32)
    raise ValueError(f'unknown prior transform {transform!r}')


def finalize_prior(unit: np.ndarray, raw_max_abs: float, params: PriorParams) -> np.ndarray:
    """Cached unit prior -> model input (normalization + optional transform).

    Single entry point used by the dataset, inference and evaluation, so the
    network always sees identically prepared priors.
    """
    x = rescale_unit_prior(unit, raw_max_abs, params.norm, params.global_scale)
    return apply_prior_transform(x, params.transform, params.log_k)


def unit_prior(mask: np.ndarray, params: PriorParams) -> Tuple[np.ndarray, float]:
    """(I_prior / max|I_prior|, max|I_prior|) — the representation stored in the cache."""
    raw = compute_artifact_prior_raw(mask, params)
    s = float(np.abs(raw).max())
    unit = (raw / s).astype(np.float32) if s > 0 else np.zeros_like(raw, np.float32)
    return unit, s


def compute_artifact_prior(mask: np.ndarray,
                           n_angles: int = 360,
                           mu_scale: float = 1.0,
                           beta: float = 1.0,
                           geometry: str = 'parallel',
                           prior_norm: str = 'per_image',
                           global_scale: Optional[float] = None,
                           params: Optional[PriorParams] = None,
                           return_scale: bool = False):
    """Compute the normalized artifact prior from a metal mask.

    Either pass explicit keyword parameters or a PriorParams object (which wins).
    Returns float32 [H, W]; with return_scale=True returns (prior, raw_max_abs).
    """
    if params is None:
        params = PriorParams(n_angles=n_angles, mu_scale=mu_scale, beta=beta,
                             geometry=geometry, norm=prior_norm, global_scale=global_scale)
    unit, s = unit_prior(mask, params)
    out = finalize_prior(unit, s, params)
    if return_scale:
        return out, s
    return out


def global_scale_from(raw_max_abs_values, percentile: float = 99.0) -> float:
    """Global normalization constant from per-image max|I_prior| of the training set."""
    v = np.asarray([x for x in raw_max_abs_values if x > 0], dtype=np.float64)
    if v.size == 0:
        raise ValueError('cannot compute global prior scale: no non-empty priors')
    return float(np.percentile(v, percentile))
