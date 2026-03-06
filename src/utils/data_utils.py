"""
Data loading and preprocessing utilities.
"""

import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def get_id_from_filename(filename: str) -> Optional[str]:
    """
    Extract image ID from filename using regex pattern.
    
    Args:
        filename: Filename to extract ID from
        
    Returns:
        ID string or None if not found
    """
    match = re.search(r'img(\d+)', filename)
    return match.group(1) if match else None


def load_raw_image(path: str, shape: Tuple[int, int] = (512, 512), 
                   dtype: type = np.float32) -> np.ndarray:
    """
    Load raw CT image from binary file.
    
    Args:
        path: Path to raw image file
        shape: Expected image shape
        dtype: Data type of the raw image
        
    Returns:
        Loaded and reshaped image array
    """
    arr = np.fromfile(path, dtype=dtype)
    try:
        arr = arr.reshape(shape)
    except ValueError as e:
        print(f"Dimension error for {Path(path).name}. Expected {shape}, got {arr.shape}")
        return np.zeros(shape, dtype=dtype)
    return arr


def create_dataset_metadata(base_path: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """
    Create dataset metadata by matching clean and artifact images.
    
    Args:
        base_path: Path to dataset directory containing 'Target' and 'Baseline' subdirs
        limit: Maximum number of image pairs to include
        
    Returns:
        List of dictionaries with image pair metadata
    """
    base_path = Path(base_path)
    clear_dir = base_path / 'Target'
    art_dir = base_path / 'Baseline'

    if not clear_dir.exists() or not art_dir.exists():
        raise ValueError(f"Expected 'Target' and 'Baseline' directories in {base_path}")

    print("Scanning directories for image pairs...")

    # Collect clear image paths indexed by ID
    clear_paths = {}
    for file_path in clear_dir.glob('*.raw'):
        img_id = get_id_from_filename(file_path.name)
        if img_id:
            clear_paths[img_id] = file_path

    # Match with artifact images
    valid_pairs = []
    for file_path in art_dir.glob('*.raw'):
        img_id = get_id_from_filename(file_path.name)
        if img_id and img_id in clear_paths:
            valid_pairs.append({
                'id': img_id,
                'clear_path': str(clear_paths[img_id]),
                'art_path': str(file_path)
            })

    valid_pairs.sort(key=lambda x: int(x['id']))

    if limit is not None:
        print(f"Limiting dataset to {limit} pairs.")
        valid_pairs = valid_pairs[:limit]

    print(f"Found {len(valid_pairs)} image pairs.")
    return valid_pairs
