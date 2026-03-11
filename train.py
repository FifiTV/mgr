"""
Metal Artifact Reduction Training Script

Train CycleGAN and Diffusion models for CT image artifact reduction.

Usage:
    python train.py --type cycle --epochs 5
    python train.py --type diff --epochs 50
    python train.py --type cycle --data-path /path/to/data --config custom_config.toml
"""

import argparse
import logging
import os
import warnings

from src.utils import (
    load_config, setup_device, load_dataset_metadata, create_data_loaders
)
from src.training import train_cyclegan, train_diffusion


warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train Metal Artifact Reduction models"
    )
    
    parser.add_argument(
        '--type', 
        choices=['cycle', 'diff'],
        default='cycle',
        help='Model type: cycle (CycleGAN) or diff (Diffusion)'
    )
    parser.add_argument(
        '--data-source',
        choices=['real', 'rpi', 'both'],
        default='rpi',
        help='Data source to use: real (CT images), rpi (simulations), both (combined)'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/raw',
        help='Path to dataset base directory (contains real/ and RPI/ subdirs)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.toml',
        help='Path to config file'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.epochs:
        config['cyclegan']['n_epochs'] = args.epochs
        config['diffusion']['n_epochs'] = args.epochs
    if args.batch_size:
        config['dataset']['batch_size'] = args.batch_size
    
    # Setup
    device = setup_device()
    os.makedirs(config['models']['model_save_dir'], exist_ok=True)
    os.makedirs(config['models']['log_dir'], exist_ok=True)
    
    logger.info(f"Loading dataset from source '{args.data_source}' ({args.data_path})...")
    dataset_metadata = load_dataset_metadata(
        data_source=args.data_source,
        base_path=args.data_path
    )
    logger.info(f"Loaded {len(dataset_metadata)} image pairs")
    
    # Create data loaders
    dataloader_soft, dataloader_hard = create_data_loaders(
        config,
        dataset_metadata,
        config['dataset']['batch_size'],
        config['dataset']['num_workers']
    )
    
    # Train
    try:
        if args.type == 'cycle':
            train_cyclegan(
                dataloader_soft, dataloader_hard,
                config, device,
                config['models']['model_save_dir']
            )
        elif args.type == 'diff':
            train_diffusion(
                dataloader_soft, dataloader_hard,
                config, device,
                config['models']['model_save_dir']
            )
        
        logger.info("Training completed successfully!")
    
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
