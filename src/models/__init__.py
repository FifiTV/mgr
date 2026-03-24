"""
Model architectures for metal artifact reduction.
"""

from .cyclegan import Generator, Discriminator, ResidualBlock, weights_init_normal
from .diffusion import (
    DiffusionModel, 
    DiffusionArchitecture,
    SinusoidalPositionEmbeddings, 
    Block,
    ConditionalUnetLight,
    ConditionalUnetStandard,
    ResnetBlock,
    AttentionBlock
)

__all__ = [
    "Generator",
    "Discriminator",
    "ResidualBlock",
    "weights_init_normal",
    "DiffusionModel",
    "DiffusionArchitecture",
    "SinusoidalPositionEmbeddings",
    "Block",
    "ConditionalUnetLight",
    "ConditionalUnetStandard",
    "ResnetBlock",
    "AttentionBlock",
]
