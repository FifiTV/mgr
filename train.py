"""
Metal Artifact Reduction Training Script

Train CycleGAN, Diffusion, and Gaussian Vicinal DDPM models.

Usage:
    python train.py --type cycle --epochs 5
    python train.py --type diff --epochs 50
    python train.py --type gaussian
    python train.py --type gaussian --features-dir results --data-path data/raw/RPI
"""

import argparse
import logging
import os
import warnings

from src.utils import (
    load_config, setup_device, load_dataset_metadata, create_data_loaders
)
from src.training import train_cyclegan, train_diffusion, train_gaussian


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
        choices=['cycle', 'diff', 'gaussian'],
        default='cycle',
        help='Model type: cycle (CycleGAN), diff (Diffusion), gaussian (Gaussian Vicinal DDPM)'
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
        default=None,
        help='Nadpisuje paths.raw_data_path z config.toml'
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
    parser.add_argument(
        '--features-dir',
        type=str,
        default=None,
        help='Directory with body*/features.csv (or a single _norm.csv). '
             'Defaults to results/ next to config.'
    )
    parser.add_argument(
        '--features',
        type=str,
        default=None,
        help='Comma-separated feature names for Gaussian training, e.g. '
             '"peak_amplitude,spatial_extent,tau". Overrides config feature_cols.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for checkpoints, samples, and history. '
             'Defaults to results/models/gaussian.'
    )

    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.epochs:
        config['cyclegan']['n_epochs'] = args.epochs
        config['diffusion']['n_epochs'] = args.epochs
        config.setdefault('gaussian', {})['n_epochs'] = args.epochs
    if args.batch_size:
        config['dataset']['batch_size'] = args.batch_size
        config.setdefault('gaussian', {})['batch_size'] = args.batch_size
    if args.features:
        config.setdefault('gaussian', {})['feature_cols'] = [
            f.strip() for f in args.features.split(',')
        ]

    # Resolve data paths: CLI > config [paths] > hardcoded fallback
    paths_cfg = config.get('paths', {})
    data_path = args.data_path or paths_cfg.get('raw_data_path', 'data/raw')
    real_path = paths_cfg.get('real_path') or None
    rpi_path  = paths_cfg.get('rpi_path')  or None

    logger.info(f"Data paths — base: {data_path}"
                + (f", real: {real_path}" if real_path else "")
                + (f", rpi: {rpi_path}"   if rpi_path  else ""))
    
    # Setup
    device = setup_device()
    os.makedirs(config['training']['model_save_dir'], exist_ok=True)
    os.makedirs(config['training']['log_dir'], exist_ok=True)
    
    # Read split config (optional section)
    rpi_splits = config.get('rpi_splits', {})

    rpi_train_variants  = rpi_splits.get('train') or None
    rpi_val_variants    = rpi_splits.get('val')   or None
    rpi_test_variants   = rpi_splits.get('test')  or None

    real_train_min = rpi_splits.get('real_metal_train_min') or None
    real_train_max = rpi_splits.get('real_metal_train_max') or None
    real_val_min   = rpi_splits.get('real_metal_val_min')   or None
    real_val_max   = rpi_splits.get('real_metal_val_max')   or None
    real_test_min  = rpi_splits.get('real_metal_test_min')  or None
    real_test_max  = rpi_splits.get('real_metal_test_max')  or None

    if rpi_train_variants:
        logger.info(f"RPI splits — train: {rpi_train_variants}")
    if rpi_val_variants:
        logger.info(f"RPI splits — val:   {rpi_val_variants}")
    if rpi_test_variants:
        logger.info(f"RPI splits — test:  {rpi_test_variants}")
    if real_train_max:
        logger.info(f"Real metal ID — train: {real_train_min or 1}–{real_train_max}")
    if real_val_max or real_val_min:
        logger.info(f"Real metal ID — val:   {real_val_min or 1}–{real_val_max or '∞'}")
    if real_test_min:
        logger.info(f"Real metal ID — test:  {real_test_min}–{real_test_max or '∞'}")

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
                data_path=data_path,
                rpi_train_variants=rpi_train_variants,
                rpi_val_variants=rpi_val_variants,
                real_train_metal_min=real_train_min,
                real_train_metal_max=real_train_max,
                real_val_metal_min=real_val_min,
                real_val_metal_max=real_val_max,
                real_path=real_path,
                rpi_path=rpi_path,
            )
        elif args.type == 'diff':
            logger.info(f"Loading dataset from source '{args.data_source}' ({data_path})...")
            dataset_metadata = load_dataset_metadata(
                data_source=args.data_source,
                base_path=data_path,
                rpi_variants=rpi_train_variants,
                metal_id_min=real_train_min,
                metal_id_max=real_train_max,
                real_path=real_path,
                rpi_path=rpi_path,
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
            )
        
        elif args.type == 'gaussian':
            paths_cfg  = config.get('paths', {})
            rpi_base   = rpi_path or data_path

            # Resolve body dirs for training split
            train_variants = rpi_train_variants or ['body1']
            body_dirs = [
                os.path.join(rpi_base, variant) for variant in train_variants
            ]
            logger.info(f"Gaussian — body dirs: {body_dirs}")

            # Features: --features-dir > sibling 'results/' of config > results/
            config_dir    = os.path.dirname(os.path.abspath(args.config))
            features_dir  = args.features_dir or os.path.join(config_dir, 'results')
            clip_stats    = os.path.join(features_dir, 'feature_clip_stats.json')
            clip_stats    = clip_stats if os.path.exists(clip_stats) else None
            logger.info(f"Gaussian — features_dir: {features_dir}")
            if clip_stats:
                logger.info(f"Gaussian — clip_stats:   {clip_stats}")
            else:
                logger.warning("feature_clip_stats.json not found — features must be pre-normalized")

            if args.output_dir:
                output_dir = args.output_dir
            else:
                output_dir = os.path.join(
                    paths_cfg.get('results_dir', 'results'), 'models', 'gaussian'
                )
            logger.info(f"Gaussian — output_dir: {output_dir}")

            train_gaussian(
                config=config,
                device=device,
                output_dir=output_dir,
                body_dirs=body_dirs,
                features_source=features_dir,
                clip_stats_path=clip_stats,
            )

        logger.info("Training completed successfully!")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
