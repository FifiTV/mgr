"""
Multiple data source handling for real and RPI data.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class DataSourceManager:
    """Manage multiple data sources (real and RPI)."""
    
    def __init__(self, base_data_path: str = "data/raw"):
        self.base_path = Path(base_data_path)
        self.real_path = self.base_path / "real"
        self.rpi_path = self.base_path / "RPI"
    
    def verify_sources(self) -> Dict[str, bool]:
        """Check which data sources are available."""
        sources = {
            'real': self.real_path.exists(),
            'rpi': self.rpi_path.exists(),
        }
        
        for source, exists in sources.items():
            logger.info(f"Data source '{source}': {'✓ Available' if exists else '✗ Not found'}")
        
        return sources
    
    def get_real_data(self) -> List[Dict[str, str]]:
        """
        Load real CT images metadata.
        Real data structure: real/*.raw files (unpaired)
        For now, treat each real image as a single entry.
        """
        if not self.real_path.exists():
            logger.warning(f"Real data path not found: {self.real_path}")
            return []
        
        real_files = list(self.real_path.glob("*.raw"))
        
        if not real_files:
            logger.warning(f"No .raw files found in {self.real_path}")
            return []
        
        metadata = []
        for idx, filepath in enumerate(sorted(real_files)):
            # Real data: use same file as both clean and artifact initially
            metadata.append({
                'id': f'real_{idx:04d}',
                'clear_path': str(filepath),
                'art_path': str(filepath),  # Placeholder
                'source': 'real'
            })
        
        logger.info(f"Loaded {len(metadata)} real image(s)")
        return metadata
    
    def get_rpi_data(self, rpi_variant: str = "body1") -> List[Dict[str, str]]:
        """
        Load RPI simulated data metadata.
        RPI data structure: RPI/{variant}/Target, RPI/{variant}/Baseline
        """
        rpi_variant_path = self.rpi_path / rpi_variant
        target_dir = rpi_variant_path / "Target"
        baseline_dir = rpi_variant_path / "Baseline"
        
        if not target_dir.exists() or not baseline_dir.exists():
            logger.warning(f"RPI data structure incomplete: {rpi_variant_path}")
            return []
        
        # Collect Target (clean) images
        target_files = {}
        for filepath in target_dir.glob("*.raw"):
            # Extract ID: e.g., "img0001" from "img0001_*.raw"
            img_id = filepath.stem.split('_')[0]
            target_files[img_id] = str(filepath)
        
        # Match with Baseline (artifact) images
        metadata = []
        for filepath in baseline_dir.glob("*.raw"):
            img_id = filepath.stem.split('_')[0]
            
            if img_id in target_files:
                metadata.append({
                    'id': f'rpi_{img_id}',
                    'clear_path': target_files[img_id],
                    'art_path': str(filepath),
                    'source': 'rpi'
                })
        
        logger.info(f"Loaded {len(metadata)} RPI image pairs from '{rpi_variant}'")
        return metadata
    
    def get_combined_data(self, use_real: bool = True, use_rpi: bool = True,
                         rpi_variant: str = "body1") -> List[Dict[str, str]]:
        """
        Combine metadata from multiple sources.
        
        Args:
            use_real: Include real CT images
            use_rpi: Include RPI simulated images
            rpi_variant: RPI variant to use (e.g., 'body1', 'body2')
            
        Returns:
            Combined metadata list
        """
        metadata = []
        
        if use_real:
            metadata.extend(self.get_real_data())
        
        if use_rpi:
            metadata.extend(self.get_rpi_data(rpi_variant=rpi_variant))
        
        if not metadata:
            logger.error("No data sources available!")
            raise ValueError("No training data found")
        
        logger.info(f"Total metadata entries: {len(metadata)}")
        return metadata


def load_data_source(source: str, base_path: str = "data/raw") -> List[Dict[str, str]]:
    """
    Convenience function to load specific data source.
    
    Args:
        source: Data source to load ('real', 'rpi', or 'both')
        base_path: Base data directory path
        
    Returns:
        Metadata list
    """
    manager = DataSourceManager(base_data_path=base_path)
    
    if source == 'real':
        return manager.get_real_data()
    elif source == 'rpi':
        return manager.get_rpi_data()
    elif source == 'both':
        return manager.get_combined_data(use_real=True, use_rpi=True)
    else:
        raise ValueError(f"Unknown data source: {source}")
