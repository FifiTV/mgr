"""
Training module initialization.
"""

try:
    from .cyclegan_trainer import train_cyclegan
except Exception as _e:  # noqa: BLE001
    import warnings
    warnings.warn(f"CycleGAN trainer unavailable: {_e}", ImportWarning, stacklevel=1)
    train_cyclegan = None  # type: ignore[assignment]

try:
    from .diffusion_trainer import train_diffusion
except Exception as _e:  # noqa: BLE001
    import warnings
    warnings.warn(f"Diffusion trainer unavailable: {_e}", ImportWarning, stacklevel=1)
    train_diffusion = None  # type: ignore[assignment]

from .gaussian_trainer import train_gaussian

__all__ = [
    "train_cyclegan",
    "train_diffusion",
    "train_gaussian",
]
