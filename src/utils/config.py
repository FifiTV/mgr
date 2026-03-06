"""
Configuration and setup utilities.
"""

import torch
import logging
import toml
from pathlib import Path


logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """
    Load configuration from TOML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = toml.load(f)
    
    logger.info(f"Configuration loaded from {config_path}")
    return config


def setup_device() -> torch.device:
    """
    Setup GPU/CPU device.
    
    Returns:
        torch.device object
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    
    return device
