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
    os.makedirs(config['training']['model_save_dir'], exist_ok=True)
    os.makedirs(config['training']['log_dir'], exist_ok=True)
    
    # Read RPI folder splits from config (optional section)
    rpi_splits = config.get('rpi_splits', {})
    rpi_train_variants = rpi_splits.get('train') or None
    rpi_val_variants = rpi_splits.get('val') or None
    rpi_test_variants = rpi_splits.get('test') or None

    if rpi_train_variants:
        logger.info(f"RPI splits — train: {rpi_train_variants}")
    if rpi_val_variants:
        logger.info(f"RPI splits — val:   {rpi_val_variants}")
    if rpi_test_variants:
        logger.info(f"RPI splits — test:  {rpi_test_variants}")

    # Train
    try:
        if args.type == 'cycle':
            # CycleGAN uses mixed data strategy internally:
            # Generator: RPI only (controlled artifacts)
            # Discriminator: Real+RPI (realistic patterns)
            # Validation: Real+RPI (quality verification)
            logger.info("Training CycleGAN with mixed data strategy...")
            logger.info("  Generator:     RPI (controlled artifacts)")
            logger.info("  Discriminator: Real+RPI (realistic patterns)")
            logger.info("  Validation:    Real+RPI (quality check)")

            train_cyclegan(
                config=config,
                device=device,
                output_dir=config['training']['model_save_dir'],
                gen_data_source='rpi',
                disc_data_source='both',
                val_data_source='both',
                data_path=args.data_path,
                rpi_train_variants=rpi_train_variants,
                rpi_val_variants=rpi_val_variants,
            )
        elif args.type == 'diff':
            logger.info(f"Loading dataset from source '{args.data_source}' ({args.data_path})...")
            dataset_metadata = load_dataset_metadata(
                data_source=args.data_source,
                base_path=args.data_path,
                rpi_variants=rpi_train_variants,
            )
            logger.info(f"Loaded {len(dataset_metadata)} image pairs")
            
            # Create data loaders for Diffusion
            dataloader_soft, dataloader_hard = create_data_loaders(
                config,
                dataset_metadata,
                config['dataset']['batch_size'],
                config['dataset']['num_workers']
            )
            
            train_diffusion(
                dataloader_soft, dataloader_hard,
                config, device,
                config['training']['model_save_dir'],
                data_source=args.data_source
            )
        
        logger.info("Training completed successfully!")
    
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
