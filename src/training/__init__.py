"""
Training module initialization.
"""

from .cyclegan_trainer import train_cyclegan
from .diffusion_trainer import train_diffusion

__all__ = [
    "train_cyclegan",
    "train_diffusion",
]
