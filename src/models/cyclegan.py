"""
CycleGAN architecture for metal artifact reduction.
"""

import torch
import torch.nn as nn
from typing import Optional


class ResidualBlock(nn.Module):
    """Residual block with reflection padding."""
    
    def __init__(self, in_features: int):
        """
        Args:
            in_features: Number of input channels
        """
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        return x + self.block(x)


class Generator(nn.Module):
    """Generator network for CycleGAN."""
    
    def __init__(self, input_nc: int, output_nc: int, n_residual_blocks: int = 9):
        """
        Args:
            input_nc: Number of input channels
            output_nc: Number of output channels
            n_residual_blocks: Number of residual blocks
        """
        super(Generator, self).__init__()

        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        # Downsampling layers
        in_features = 64
        out_features = in_features * 2

        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks (core of the network)
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling layers
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, 
                                 stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2

        # Final output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh(),  # Output range: [-1, 1]
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through generator."""
        return self.model(x)


class Discriminator(nn.Module):
    """PatchGAN Discriminator for CycleGAN."""
    
    def __init__(self, input_nc: int):
        """
        Args:
            input_nc: Number of input channels
        """
        super(Discriminator, self).__init__()

        def d_block(in_filters: int, out_filters: int, normalize: bool = True):
            """
            Create a discriminator block.
            
            Args:
                in_filters: Number of input filters
                out_filters: Number of output filters
                normalize: Whether to apply instance normalization
                
            Returns:
                List of layers
            """
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Build discriminator
        self.model = nn.Sequential(
            *d_block(input_nc, 64, normalize=False),
            *d_block(64, 128),
            *d_block(128, 256),
            *d_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through discriminator."""
        return self.model(x)


def weights_init_normal(m: nn.Module) -> None:
    """
    Initialize model weights with normal distribution.
    
    Args:
        m: Module to initialize
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
