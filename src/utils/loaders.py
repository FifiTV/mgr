"""
Data loader creation and dataset utilities.
"""

import logging
from typing import Tuple, List, Dict, Optional
from torch.utils.data import DataLoader

from src.datasets import CTDataset, LabelMode, ScalingMethod
from src.utils.data_sources import load_data_source


logger = logging.getLogger(__name__)


def load_dataset_metadata(data_source: str = 'rpi', base_path: str = 'data/raw',
                          rpi_variants: Optional[List[str]] = None,
                          metal_id_min: Optional[int] = None,
                          metal_id_max: Optional[int] = None,
                          real_path: Optional[str] = None,
                          rpi_path: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Load metadata for specified data source.

    Args:
        data_source: Data source to load ('real', 'rpi', or 'both')
        base_path: Base data directory path
        rpi_variants: List of RPI variant folder names (None = default ["body1"])
        metal_id_min: Lower bound for real image metal IDs (None / 0 = no bound)
        metal_id_max: Upper bound for real image metal IDs (None / 0 = no bound)
        real_path: Explicit path to real images dir (overrides base_path/real)
        rpi_path:  Explicit path to RPI dir        (overrides base_path/RPI)

    Returns:
        List of metadata dictionaries with 'clear_path', 'art_path', 'id', 'source' keys
    """
    logger.info(f"Loading metadata from data source: {data_source}")

    try:
        metadata = load_data_source(source=data_source, base_path=base_path,
                                    rpi_variants=rpi_variants,
                                    metal_id_min=metal_id_min,
                                    metal_id_max=metal_id_max,
                                    real_path=real_path, rpi_path=rpi_path)
        logger.info(f"Loaded {len(metadata)} metadata entries from {data_source}")
        return metadata
    except Exception as e:
        logger.error(f"Failed to load data source '{data_source}': {e}")
        raise


def create_data_loaders(config: dict, dataset_metadata: list,
                       batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for soft and hard label modes.
    
    Args:
        config: Configuration dictionary
        dataset_metadata: List of image pair metadata
        batch_size: Batch size for training
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (dataloader_soft, dataloader_hard)
    """
    logger.info("Creating data loaders...")
    
    # Soft labels dataset — scale by global bloom_max when available
    bloom_max = config['data'].get('bloom_max') or None
    dataset_soft = CTDataset(
        data_list=dataset_metadata,
        label_mode=LabelMode.SOFT,
        scaling_method=ScalingMethod.LOG,
        metal_threshold_hu=config['data']['metal_threshold_hu'],
        tanh_scale=config['data']['tanh_scale'],
        bloom_max=bloom_max,
    )
    
    dataloader_soft = DataLoader(
        dataset_soft,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config['dataset']['pin_memory'],
        persistent_workers=config['dataset']['persistent_workers'] if num_workers > 0 else False
    )
    
    logger.info(f"Soft labels dataloader created with {len(dataset_soft)} samples")
    
    # Hard labels dataset
    dataset_hard = CTDataset(
        data_list=dataset_metadata,
        label_mode=LabelMode.HARD,
        scaling_method=ScalingMethod.LOG,
        metal_threshold_hu=config['data']['metal_threshold_hu'],
        hard_threshold_hu=config['data']['hard_threshold_hu']
    )
    
    dataloader_hard = DataLoader(
        dataset_hard,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config['dataset']['pin_memory'],
        persistent_workers=config['dataset']['persistent_workers'] if num_workers > 0 else False
    )
    
    logger.info(f"Hard labels dataloader created with {len(dataset_hard)} samples")
    
    return dataloader_soft, dataloader_hard
