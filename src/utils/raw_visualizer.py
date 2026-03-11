"""
Raw image visualization and inspection utilities.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from PIL import Image


def load_raw_to_array(filepath: str, shape: Tuple[int, int] = (512, 512), 
                      dtype: type = np.float32) -> np.ndarray:
    """
    Load raw binary image file into numpy array.
    
    Args:
        filepath: Path to raw file
        shape: Expected image shape (H, W)
        dtype: Data type of raw file
        
    Returns:
        Loaded numpy array
    """
    data = np.fromfile(filepath, dtype=dtype)
    
    if len(data) != shape[0] * shape[1]:
        print(f"Warning: Expected {shape[0]*shape[1]} values, got {len(data)}")
        print(f"Attempting reshape anyway...")
    
    try:
        img = data.reshape(shape)
    except ValueError as e:
        print(f"Error reshaping {Path(filepath).name}: {e}")
        return None
    
    return img


def normalize_image(img: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize image for visualization.
    
    Args:
        img: Input image array
        method: Normalization method ('minmax' or 'percentile')
        
    Returns:
        Normalized image in range [0, 1]
    """
    if img is None:
        return None
    
    if method == 'minmax':
        img_min = img.min()
        img_max = img.max()
        if img_max - img_min > 1e-5:
            return (img - img_min) / (img_max - img_min)
        return img
    
    elif method == 'percentile':
        # Use 2nd and 98th percentile for robust normalization
        p2 = np.percentile(img, 2)
        p98 = np.percentile(img, 98)
        if p98 - p2 > 1e-5:
            img_norm = np.clip((img - p2) / (p98 - p2), 0, 1)
            return img_norm
        return img
    
    return img


def raw_to_png(input_raw: str, output_png: str, 
               shape: Tuple[int, int] = (512, 512),
               normalize_method: str = 'percentile',
               cmap: str = 'gray') -> bool:
    """
    Convert raw binary image to PNG.
    
    Args:
        input_raw: Path to raw file
        output_png: Path to save PNG
        shape: Image shape (H, W)
        normalize_method: Normalization method for visualization
        cmap: Colormap to use
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load raw image
        img = load_raw_to_array(input_raw, shape=shape)
        
        if img is None:
            print(f"Failed to load {input_raw}")
            return False
        
        # Normalize
        img_norm = normalize_image(img, method=normalize_method)
        
        # Convert to uint8
        img_uint8 = (img_norm * 255).astype(np.uint8)
        
        # Save using PIL
        pil_img = Image.fromarray(img_uint8, mode='L')
        pil_img.save(output_png)
        
        print(f"Successfully saved: {output_png}")
        print(f"  Shape: {img.shape}")
        print(f"  Min value: {img.min():.4f}")
        print(f"  Max value: {img.max():.4f}")
        print(f"  Mean value: {img.mean():.4f}")
        
        return True
    
    except Exception as e:
        print(f"Error converting {input_raw}: {e}")
        return False


def visualize_raw(filepath: str, shape: Tuple[int, int] = (512, 512),
                  save_path: Optional[str] = None,
                  normalize_method: str = 'percentile') -> None:
    """
    Load and display raw image in matplotlib.
    
    Args:
        filepath: Path to raw file
        shape: Image shape (H, W)
        save_path: Optional path to save figure
        normalize_method: Normalization method
    """
    img = load_raw_to_array(filepath, shape=shape)
    
    if img is None:
        return
    
    img_norm = normalize_image(img, method=normalize_method)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original values
    im1 = axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f'Raw Values\nMin: {img.min():.2f}, Max: {img.max():.2f}, Mean: {img.mean():.2f}')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Normalized
    im2 = axes[1].imshow(img_norm, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Normalized ({normalize_method})')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved: {save_path}")
    
    plt.show()


def inspect_raw_file(filepath: str, shape: Tuple[int, int] = (512, 512)) -> dict:
    """
    Inspect raw file properties.
    
    Args:
        filepath: Path to raw file
        shape: Expected image shape
        
    Returns:
        Dictionary with file statistics
    """
    try:
        img = load_raw_to_array(filepath, shape=shape)
        
        if img is None:
            return None
        
        stats = {
            'filepath': str(filepath),
            'shape': img.shape,
            'dtype': str(img.dtype),
            'min': float(img.min()),
            'max': float(img.max()),
            'mean': float(img.mean()),
            'std': float(img.std()),
            'median': float(np.median(img)),
            'file_size_mb': Path(filepath).stat().st_size / (1024 * 1024),
        }
        
        print(f"\nFile: {Path(filepath).name}")
        print(f"  Shape: {stats['shape']}")
        print(f"  Dtype: {stats['dtype']}")
        print(f"  Min: {stats['min']:.4f}")
        print(f"  Max: {stats['max']:.4f}")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std: {stats['std']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  File size: {stats['file_size_mb']:.2f} MB")
        
        return stats
    
    except Exception as e:
        print(f"Error inspecting {filepath}: {e}")
        return None
