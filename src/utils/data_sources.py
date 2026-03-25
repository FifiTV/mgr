"""
Multiple data source handling for real and RPI data.
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Expected file sizes (for float32 data)
EXPECTED_IMAGE_SIZE_BYTES = 512 * 512 * 4      # 1,048,576 bytes (512×512 images)
SINOGRAM_SIZE_BYTES = 900 * 1000 * 4            # 3,600,000 bytes (900×1000 sinograms)


def is_valid_image_file(filepath: Path) -> bool:
    """
    Check if file is a valid 512×512 image (not a sinogram).
    
    Filters out sinogram files (900×1000) by checking file size.
    For 512×512 float32 image: expects exactly 1,048,576 bytes
    For 900×1000 float32 sinogram: has 3,600,000 bytes (skipped)
    
    Args:
        filepath: Path to raw file
        
    Returns:
        True if file is correct size for images (512×512)
    """
    try:
        actual_size = filepath.stat().st_size
        
        if actual_size == EXPECTED_IMAGE_SIZE_BYTES:
            return True
        elif actual_size == SINOGRAM_SIZE_BYTES:
            logger.debug(f"Filtering out sinogram (900×1000): {filepath.name}")
            return False
        else:
            logger.warning(f"Skipping file with unexpected size {actual_size} bytes: {filepath.name}")
            return False
    except OSError as e:
        logger.error(f"Cannot check file: {filepath.name} ({e})")
        return False



class DataSourceManager:
    """Manage multiple data sources (real and RPI)."""

    def __init__(self, base_data_path: str = "data/raw",
                 real_path: Optional[str] = None,
                 rpi_path: Optional[str] = None):
        """
        Args:
            base_data_path: Base directory containing real/ and RPI/ subdirs.
            real_path: Explicit path to real images dir (overrides base_data_path/real).
            rpi_path:  Explicit path to RPI dir        (overrides base_data_path/RPI).
        """
        base = Path(base_data_path)
        self.real_path = Path(real_path) if real_path else base / "real"
        self.rpi_path  = Path(rpi_path)  if rpi_path  else base / "RPI"
    
    def verify_sources(self) -> Dict[str, bool]:
        """Check which data sources are available."""
        sources = {
            'real': self.real_path.exists(),
            'rpi': self.rpi_path.exists(),
        }
        
        for source, exists in sources.items():
            logger.info(f"Data source '{source}': {'✓ Available' if exists else '✗ Not found'}")
        
        return sources
    
    def get_real_data(self, metal_id_min: Optional[int] = None,
                      metal_id_max: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Load real CT images metadata.
        Real data structure: real/*.raw files (unpaired)

        Filenames are expected to contain a metal ID encoded as ``metalNN_``
        (e.g. ``metal06_slice0495_H512_W512.raw`` → metal_id = 6).
        Files whose metal ID falls outside [metal_id_min, metal_id_max] are
        excluded so that train / val / test sets never share the same scan.

        Args:
            metal_id_min: Lowest metal ID to include (None or 0 = no lower bound)
            metal_id_max: Highest metal ID to include (None or 0 = no upper bound)

        Validates that files are 512×512 images (expects 1,048,576 bytes).
        """
        if not self.real_path.exists():
            logger.warning(f"Real data path not found: {self.real_path}")
            return []

        real_files = list(self.real_path.glob("*.raw"))

        if not real_files:
            logger.warning(f"No .raw files found in {self.real_path}")
            return []

        # Normalise bounds: treat 0 / None as "no bound"
        lo = metal_id_min if metal_id_min else None
        hi = metal_id_max if metal_id_max else None

        metadata = []
        skipped_size = 0
        skipped_range = 0
        skipped_no_id = 0

        for filepath in sorted(real_files):
            if not is_valid_image_file(filepath):
                skipped_size += 1
                continue

            # Parse metal ID from filename (e.g. metal06_slice… → 6)
            match = re.search(r'metal(\d+)', filepath.stem, re.IGNORECASE)
            if match:
                metal_id = int(match.group(1))
                if lo is not None and metal_id < lo:
                    skipped_range += 1
                    continue
                if hi is not None and metal_id > hi:
                    skipped_range += 1
                    continue
            else:
                metal_id = None
                if lo is not None or hi is not None:
                    # A range filter is active but this file has no parseable ID —
                    # skip it to avoid silent leakage.
                    logger.warning(
                        f"Cannot parse metal ID from '{filepath.name}'; "
                        f"skipping (range filter is active)."
                    )
                    skipped_no_id += 1
                    continue

            metadata.append({
                'id': f'real_metal{metal_id:02d}_{filepath.stem}' if metal_id is not None
                      else f'real_{filepath.stem}',
                'clear_path': str(filepath),
                'art_path': str(filepath),  # Placeholder
                'source': 'real',
                'metal_id': metal_id,
            })

        range_str = (
            f" (metal_id {lo or '?'}–{hi or '?'})"
            if (lo is not None or hi is not None) else ""
        )
        logger.info(f"Loaded {len(metadata)} real image(s){range_str}")
        if skipped_size:
            logger.warning(f"  Skipped {skipped_size} file(s) with wrong size")
        if skipped_range:
            logger.info(f"  Skipped {skipped_range} file(s) outside metal_id range")
        if skipped_no_id:
            logger.warning(f"  Skipped {skipped_no_id} file(s) with unparseable metal ID")
        return metadata
    
    def get_rpi_data(self, rpi_variants: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """
        Load RPI simulated data metadata from one or more variant folders.
        RPI data structure: RPI/{variant}/Target, RPI/{variant}/Baseline

        Args:
            rpi_variants: List of variant folder names (e.g. ["body1", "body2"]).
                          Defaults to ["body1"] for backward compatibility.

        Filters out sinogram files (900×1000) - only includes image files (512×512).
        """
        if rpi_variants is None:
            rpi_variants = ["body1"]

        all_metadata: List[Dict[str, str]] = []

        for rpi_variant in rpi_variants:
            rpi_variant_path = self.rpi_path / rpi_variant
            target_dir = rpi_variant_path / "Target"
            baseline_dir = rpi_variant_path / "Baseline"

            if not target_dir.exists() or not baseline_dir.exists():
                logger.warning(f"RPI data structure incomplete: {rpi_variant_path}")
                continue

            # Collect Target (clean) images with size validation
            target_files = {}
            skipped_target = 0
            for filepath in sorted(target_dir.glob("*.raw")):
                if not is_valid_image_file(filepath):
                    skipped_target += 1
                    continue
                # Extract numeric ID: e.g., "1000" from "training_body_nometal_img1000_512x512x1.raw"
                match = re.search(r'img(\d+)', filepath.stem)
                if not match:
                    continue
                img_id = match.group(1)
                target_files[img_id] = str(filepath)

            # Match with Baseline (artifact) images
            skipped_baseline = 0
            variant_metadata = []
            for filepath in sorted(baseline_dir.glob("*.raw")):
                if not is_valid_image_file(filepath):
                    skipped_baseline += 1
                    continue

                match = re.search(r'img(\d+)', filepath.stem)
                if not match:
                    continue
                img_id = match.group(1)

                if img_id in target_files:
                    variant_metadata.append({
                        'id': f'rpi_{rpi_variant}_{img_id}',
                        'clear_path': target_files[img_id],
                        'art_path': str(filepath),
                        'source': 'rpi',
                        'variant': rpi_variant,
                    })

            logger.info(f"Loaded {len(variant_metadata)} RPI image pairs from '{rpi_variant}'")
            if skipped_target > 0 or skipped_baseline > 0:
                logger.info(f"  (Filtered: {skipped_target} Target, {skipped_baseline} Baseline files)")

            all_metadata.extend(variant_metadata)

        logger.info(f"Total RPI pairs loaded ({len(rpi_variants)} variant(s)): {len(all_metadata)}")
        return all_metadata
    
    def get_combined_data(self, use_real: bool = True, use_rpi: bool = True,
                         rpi_variants: Optional[List[str]] = None,
                         metal_id_min: Optional[int] = None,
                         metal_id_max: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Combine metadata from multiple sources.

        Args:
            use_real: Include real CT images
            use_rpi: Include RPI simulated images
            rpi_variants: List of RPI variant folder names to include
            metal_id_min: Lower bound for real image metal IDs
            metal_id_max: Upper bound for real image metal IDs

        Returns:
            Combined metadata list
        """
        metadata = []

        if use_real:
            metadata.extend(self.get_real_data(metal_id_min=metal_id_min,
                                               metal_id_max=metal_id_max))

        if use_rpi:
            metadata.extend(self.get_rpi_data(rpi_variants=rpi_variants))

        if not metadata:
            logger.error("No data sources available!")
            raise ValueError("No training data found")

        logger.info(f"Total metadata entries: {len(metadata)}")
        return metadata


def load_data_source(source: str, base_path: str = "data/raw",
                     rpi_variants: Optional[List[str]] = None,
                     metal_id_min: Optional[int] = None,
                     metal_id_max: Optional[int] = None,
                     real_path: Optional[str] = None,
                     rpi_path: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Convenience function to load specific data source.

    Args:
        source: Data source to load ('real', 'rpi', or 'both')
        base_path: Base data directory (used when real_path / rpi_path are not given)
        rpi_variants: List of RPI variant folder names to include (None = default ["body1"])
        metal_id_min: Lower bound for real image metal IDs (None / 0 = no bound)
        metal_id_max: Upper bound for real image metal IDs (None / 0 = no bound)
        real_path: Explicit path to real images dir (overrides base_path/real)
        rpi_path:  Explicit path to RPI dir        (overrides base_path/RPI)

    Returns:
        Metadata list
    """
    manager = DataSourceManager(base_data_path=base_path,
                                real_path=real_path, rpi_path=rpi_path)

    if source == 'real':
        return manager.get_real_data(metal_id_min=metal_id_min, metal_id_max=metal_id_max)
    elif source == 'rpi':
        return manager.get_rpi_data(rpi_variants=rpi_variants)
    elif source == 'both':
        return manager.get_combined_data(use_real=True, use_rpi=True,
                                         rpi_variants=rpi_variants,
                                         metal_id_min=metal_id_min,
                                         metal_id_max=metal_id_max)
    else:
        raise ValueError(f"Unknown data source: {source}")
