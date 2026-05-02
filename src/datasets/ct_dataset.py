"""
CT Image Dataset for metal artifact reduction training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, List, Dict


class LabelMode(Enum):
    """Label generation mode for artifact masks."""
    SOFT = "soft"  # Continuous soft masks
    HARD = "hard"  # Binary hard masks


class ScalingMethod(Enum):
    """Scaling method for mask generation."""
    NONE = "none"       # Linear scaling (Min-Max)
    LOG = "log"         # Logarithmic scaling
    CLIPPING = "clip"   # Value clipping
    GAMMA = "gamma"     # Gamma correction
    TANH = "tanh"       # Hyperbolic tangent


class CTDataset(Dataset):
    """
    Dataset for CT images with metal artifacts.
    Loads clean and artifact images with corresponding masks.
    """
    
    def __init__(self,
                 data_list: List[Dict[str, str]],
                 label_mode: LabelMode = LabelMode.SOFT,
                 scaling_method: ScalingMethod = ScalingMethod.LOG,
                 metal_threshold_hu: float = 2500.0,
                 hard_threshold_hu: float = 100.0,
                 tanh_scale: float = 80.0,
                 bloom_max: Optional[float] = None,
                 shape: Tuple[int, int] = (512, 512)):
        """
        Args:
            data_list: List of dicts with 'clear_path' and 'art_path' keys
            label_mode: LabelMode.SOFT or LabelMode.HARD
            scaling_method: Method for scaling artifact masks
            metal_threshold_hu: Threshold for metal mask (diff > this = implant voxel)
            hard_threshold_hu: Threshold for HARD artifact mask (diff > this = artifact)
            tanh_scale: Scale factor for tanh scaling method
            bloom_max: Global maximum bloom/artifact intensity (HU) used to anchor
                       LOG soft-label scaling across all images.  When provided,
                       the log mask is divided by log1p(bloom_max) instead of the
                       per-image maximum, giving consistent label magnitudes.
                       None → fall back to per-image normalisation (old behaviour).
            shape: Expected shape of CT images
        """
        self.data_paths = data_list
        self.shape = shape
        self.label_mode = label_mode
        self.scaling_method = scaling_method
        self.tanh_scale = tanh_scale
        self.bloom_max = bloom_max
        self.metal_threshold = metal_threshold_hu
        self.hard_threshold_hu = hard_threshold_hu

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data_paths)

    def load_raw_image(self, path: str) -> np.ndarray:
        """
        Load raw CT image from binary file.
        
        Args:
            path: Path to raw image file
            
        Returns:
            Loaded image as numpy array
        """
        arr = np.fromfile(path, dtype=np.float32)
        try:
            arr = arr.reshape(self.shape)
        except ValueError:
            print(f"Error reshaping {path}")
            return np.zeros(self.shape, dtype=np.float32)
        return arr

    def normalize_hu(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize CT image from HU range to [-1, 1].
        
        Args:
            img: Input image
            
        Returns:
            Normalized image
        """
        min_val, max_val = img.min(), img.max()
        if max_val - min_val > 1e-5:
            return 2 * (img - min_val) / (max_val - min_val) - 1
        return img

    def compute_soft_mask(self, diff_raw: np.ndarray) -> np.ndarray:
        """
        Compute soft (continuous) artifact mask.
        
        Args:
            diff_raw: Absolute difference between artifact and clean images
            
        Returns:
            Soft mask in range [0, 1]
        """
        if self.scaling_method == ScalingMethod.TANH:
            return np.tanh(diff_raw / self.tanh_scale)
        elif self.scaling_method == ScalingMethod.LOG:
            weight_log = np.log1p(diff_raw)
            if self.bloom_max is not None:
                # Global anchor: consistent scale across all images
                denom = np.log1p(self.bloom_max)
            else:
                # Per-image fallback (old behaviour)
                denom = weight_log.max()
            if denom > 1e-6:
                return np.clip(weight_log / denom, 0.0, 1.0)
            return weight_log
        else:
            return diff_raw / (diff_raw.max() + 1e-6)

    def get_masks(self, clear_img: np.ndarray, 
                  art_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create metal and artifact masks from image pair.
        
        Args:
            clear_img: Clean CT image
            art_img: CT image with artifacts
            
        Returns:
            Tuple of (metal_mask, artifact_mask)
        """
        diff = np.abs(art_img - clear_img)

        # Metal mask: large local difference indicates metal implant
        metal_mask = (diff > self.metal_threshold).astype(np.float32)

        # Artifact mask based on full difference
        if self.label_mode == LabelMode.HARD:
            artifact_mask = (diff > self.hard_threshold_hu).astype(np.float32)
        else:
            artifact_mask = self.compute_soft_mask(diff).astype(np.float32)

        return metal_mask, artifact_mask

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get single dataset item.
        
        Args:
            index: Index in dataset
            
        Returns:
            Dictionary with tensors:
                - real_A: Clean image (1, H, W)
                - real_B: Image with artifacts (1, H, W)
                - mask_M: Metal mask (1, H, W)
                - mask_A: Artifact mask (1, H, W)
        """
        item_paths = self.data_paths[index]
        
        # Load images
        clear_np = self.load_raw_image(item_paths['clear_path'])
        art_np = self.load_raw_image(item_paths['art_path'])
        
        # Generate masks
        m_metal, m_art = self.get_masks(clear_np, art_np)
        
        # Normalize to [-1, 1] range
        real_A = self.normalize_hu(clear_np)
        real_B = self.normalize_hu(art_np)
        
        return {
            'real_A': torch.from_numpy(real_A).unsqueeze(0).float(),
            'real_B': torch.from_numpy(real_B).unsqueeze(0).float(),
            'mask_M': torch.from_numpy(m_metal).unsqueeze(0).float(),
            'mask_A': torch.from_numpy(m_art).unsqueeze(0).float()
        }
