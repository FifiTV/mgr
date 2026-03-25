"""
CycleGAN training module.
"""

import logging
import os
import json
from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import cycle

from src.models import (
    Generator, Discriminator, weights_init_normal
)
from src.utils.metrics import calculate_psnr
from src.utils import load_dataset_metadata, create_data_loaders


logger = logging.getLogger(__name__)


def train_cyclegan(
    config: dict,
    device: torch.device,
    output_dir: str,
    gen_data_source: str = 'rpi',
    disc_data_source: str = 'both',
    val_data_source: str = 'both',
    data_path: str = 'data/raw',
    compute_metrics: bool = False,
    rpi_train_variants=None,
    rpi_val_variants=None,
    real_train_metal_min: Optional[int] = None,
    real_train_metal_max: Optional[int] = None,
    real_val_metal_min: Optional[int] = None,
    real_val_metal_max: Optional[int] = None,
    real_path: Optional[str] = None,
    rpi_path: Optional[str] = None,
) -> None:
    """
    Train CycleGAN model with mixed data sources.
    
    Mixed data strategy:
    - Generator learns on RPI only (controlled, well-paired artifacts)
    - Discriminator learns on Real+RPI (realistic artifact diversity)
    - Validation evaluates on Real+RPI (quality verification)
    
    This approach gives the generator clean learning signals while teaching
    the discriminator to distinguish realistic patterns, resulting in more
    natural synthetic artifacts.
    
    Args:
        config: Configuration dictionary
        device: Computation device (cuda/cpu)
        output_dir: Directory to save model checkpoints
        gen_data_source: Data source for generator training ('rpi', 'real', or 'both')
        disc_data_source: Data source for discriminator training ('rpi', 'real', or 'both')
        val_data_source: Data source for validation ('rpi', 'real', or 'both')
        data_path: Path to dataset base directory
    """
    logger.info("=" * 70)
    logger.info("CycleGAN Training (Mixed Data Strategy)")
    logger.info("=" * 70)
    logger.info(f"Generator data source:     {gen_data_source.upper()}")
    logger.info(f"Discriminator data source: {disc_data_source.upper()}")
    logger.info(f"Validation data source:    {val_data_source.upper()}")
    logger.info("=" * 70)
    
    # Configuration
    epochs = config['cyclegan']['n_epochs']
    lr = config['cyclegan']['learning_rate']
    lambda_cyc = config['cyclegan']['lambda_cycle']
    lambda_id = config['cyclegan']['lambda_identity']
    lambda_sup = config['cyclegan']['lambda_supervised']
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['num_workers']
    
    # Load metadata for separate roles
    logger.info(f"\nLoading datasets...")
    gen_metadata = load_dataset_metadata(
        gen_data_source, data_path,
        rpi_variants=rpi_train_variants,
        metal_id_min=real_train_metal_min, metal_id_max=real_train_metal_max,
        real_path=real_path, rpi_path=rpi_path,
    )
    disc_metadata = load_dataset_metadata(
        disc_data_source, data_path,
        rpi_variants=rpi_train_variants,
        metal_id_min=real_train_metal_min, metal_id_max=real_train_metal_max,
        real_path=real_path, rpi_path=rpi_path,
    )
    val_metadata = load_dataset_metadata(
        val_data_source, data_path,
        rpi_variants=rpi_val_variants,
        metal_id_min=real_val_metal_min, metal_id_max=real_val_metal_max,
        real_path=real_path, rpi_path=rpi_path,
    )
    
    logger.info(f"  Generator:     {len(gen_metadata)} samples ({gen_data_source})")
    logger.info(f"  Discriminator: {len(disc_metadata)} samples ({disc_data_source})")
    logger.info(f"  Validation:    {len(val_metadata)} samples ({val_data_source})")
    
    # Create dataloaders for each role (soft and hard labels)
    gen_soft, gen_hard = create_data_loaders(
        config, gen_metadata, batch_size, num_workers
    )
    disc_soft, disc_hard = create_data_loaders(
        config, disc_metadata, batch_size, num_workers
    )
    val_soft, val_hard = create_data_loaders(
        config, val_metadata, batch_size, num_workers
    )
    
    # Initialize models
    G_AB = Generator(
        input_nc=config['models']['generator_input_nc'],
        output_nc=config['models']['generator_output_nc'],
        n_residual_blocks=config['models']['generator_n_residual']
    ).to(device)
    
    G_BA = Generator(
        input_nc=1,
        output_nc=1,
        n_residual_blocks=config['models']['generator_n_residual']
    ).to(device)
    
    D_A = Discriminator(input_nc=1).to(device)
    D_B = Discriminator(input_nc=1).to(device)
    
    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)
    
    # Optimizers
    optimizer_G = optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()),
        lr=lr, betas=(0.5, 0.999)
    )
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Loss functions
    criterion_GAN = nn.MSELoss().to(device)
    criterion_cycle = nn.L1Loss().to(device)
    criterion_identity = nn.L1Loss().to(device)
    criterion_supervised = nn.L1Loss().to(device)
    
    logger.info(f"\nModels initialized. Training for {epochs} epochs...")
    logger.info(f"Learning rate: {lr}, Lambda_cycle: {lambda_cyc}, "
                f"Lambda_identity: {lambda_id}, Lambda_supervised: {lambda_sup}")
    
    # Train both soft and hard label versions
    for label_mode, (gen_loader, disc_loader, val_loader) in [
        ("SOFT", (gen_soft, disc_soft, val_soft)),
        ("HARD", (gen_hard, disc_hard, val_hard))
    ]:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Training {label_mode} Labels Model")
        logger.info(f"{'=' * 70}")
        
        # Reset weights
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)
        
        history = {
            'G_loss': [],
            'D_loss': [],
            'Sup_loss': [],
            'PSNR': []
        }
        
        # Setup mixed precision training if CUDA available
        # init_scale=256 prevents float16 overflow in early training (default 65536 can cause NaN)
        scaler = torch.amp.GradScaler('cuda', init_scale=256.0) if torch.cuda.is_available() else None

        # Precompute discriminator patch shape to avoid redundant forward passes per batch
        with torch.no_grad():
            _dummy = torch.zeros(1, 1, config['dataset']['img_size'], config['dataset']['img_size']).to(device)
            _patch_shape = D_A(_dummy).shape[1:]

        for epoch in range(epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_sup_loss = 0.0
            epoch_psnr = 0.0
            batch_count = 0
            
            # Use cycle() to handle uneven dataloader lengths
            disc_iter = cycle(disc_loader)
            
            for batch_idx, gen_batch in enumerate(gen_loader):
                disc_batch = next(disc_iter)
                
                # Get generator training data (RPI only or specified source)
                real_A_gen = gen_batch['real_A'].to(device)
                real_B_gen = gen_batch['real_B'].to(device)
                mask_M_gen = gen_batch['mask_M'].to(device)
                mask_A_gen = gen_batch['mask_A'].to(device)
                
                # Get discriminator training data (Real+RPI or specified source)
                real_A_disc = disc_batch['real_A'].to(device)
                real_B_disc = disc_batch['real_B'].to(device)
                mask_M_disc = disc_batch['mask_M'].to(device)
                mask_A_disc = disc_batch['mask_A'].to(device)
                
                # Create valid/fake labels for discriminators
                valid_gen = torch.ones(real_A_gen.size(0), *_patch_shape).to(device)
                fake_gen = torch.zeros(real_A_gen.size(0), *_patch_shape).to(device)

                valid_disc = torch.ones(real_A_disc.size(0), *_patch_shape).to(device)
                fake_disc = torch.zeros(real_A_disc.size(0), *_patch_shape).to(device)
                
                # ==================== Train Generators ====================
                optimizer_G.zero_grad()
                
                if scaler:
                    with torch.amp.autocast(device_type='cuda'):
                        # Identity loss
                        loss_id_A = criterion_identity(G_BA(real_A_gen), real_A_gen) * lambda_id
                        
                        # Generator AB: clean image -> metallic artifact image
                        input_G_AB = torch.cat((real_A_gen, mask_M_gen, mask_A_gen), 1)
                        fake_B = G_AB(input_G_AB)
                        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid_gen)
                        loss_supervised = criterion_supervised(fake_B, real_B_gen) * lambda_sup
                        
                        # Generator BA: artifact image -> clean image
                        fake_A = G_BA(real_B_gen)
                        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid_gen)
                        
                        # Cycle consistency: clean -> artifact -> clean
                        rec_A = G_BA(fake_B)
                        loss_cycle_A = criterion_cycle(rec_A, real_A_gen) * lambda_cyc
                        
                        # Cycle consistency: artifact -> clean -> artifact
                        input_rec_B = torch.cat((fake_A, mask_M_gen, mask_A_gen), 1)
                        rec_B = G_AB(input_rec_B)
                        loss_cycle_B = criterion_cycle(rec_B, real_B_gen) * lambda_cyc
                        
                        loss_G = (
                            loss_GAN_AB + loss_GAN_BA +
                            loss_cycle_A + loss_cycle_B +
                            loss_id_A + loss_supervised
                        )
                    
                    scaler.scale(loss_G).backward()
                    scaler.step(optimizer_G)
                else:
                    # Identity loss
                    loss_id_A = criterion_identity(G_BA(real_A_gen), real_A_gen) * lambda_id
                    
                    # Generator AB
                    input_G_AB = torch.cat((real_A_gen, mask_M_gen, mask_A_gen), 1)
                    fake_B = G_AB(input_G_AB)
                    loss_GAN_AB = criterion_GAN(D_B(fake_B), valid_gen)
                    loss_supervised = criterion_supervised(fake_B, real_B_gen) * lambda_sup
                    
                    # Generator BA
                    fake_A = G_BA(real_B_gen)
                    loss_GAN_BA = criterion_GAN(D_A(fake_A), valid_gen)
                    
                    # Cycle consistency A
                    rec_A = G_BA(fake_B)
                    loss_cycle_A = criterion_cycle(rec_A, real_A_gen) * lambda_cyc
                    
                    # Cycle consistency B
                    input_rec_B = torch.cat((fake_A, mask_M_gen, mask_A_gen), 1)
                    rec_B = G_AB(input_rec_B)
                    loss_cycle_B = criterion_cycle(rec_B, real_B_gen) * lambda_cyc
                    
                    loss_G = (
                        loss_GAN_AB + loss_GAN_BA +
                        loss_cycle_A + loss_cycle_B +
                        loss_id_A + loss_supervised
                    )
                    loss_G.backward()
                    optimizer_G.step()
                
                # ==================== Train Discriminator A ====================
                optimizer_D_A.zero_grad()
                
                if scaler:
                    with torch.amp.autocast(device_type='cuda'):
                        loss_real_A = criterion_GAN(D_A(real_A_disc), valid_disc)
                        # Use fake_A from generator but detach from discriminator data
                        fake_A_disc = G_BA(real_B_disc).detach()
                        loss_fake_A = criterion_GAN(D_A(fake_A_disc), fake_disc)
                        loss_D_A = (loss_real_A + loss_fake_A) / 2
                    
                    scaler.scale(loss_D_A).backward()
                    scaler.step(optimizer_D_A)
                else:
                    loss_real_A = criterion_GAN(D_A(real_A_disc), valid_disc)
                    fake_A_disc = G_BA(real_B_disc).detach()
                    loss_fake_A = criterion_GAN(D_A(fake_A_disc), fake_disc)
                    loss_D_A = (loss_real_A + loss_fake_A) / 2
                    loss_D_A.backward()
                    optimizer_D_A.step()
                
                # ==================== Train Discriminator B ====================
                optimizer_D_B.zero_grad()
                
                if scaler:
                    with torch.amp.autocast(device_type='cuda'):
                        loss_real_B = criterion_GAN(D_B(real_B_disc), valid_disc)
                        fake_B_disc = G_AB(torch.cat((real_A_disc, mask_M_disc, mask_A_disc), 1)).detach()
                        loss_fake_B = criterion_GAN(D_B(fake_B_disc), fake_disc)
                        loss_D_B = (loss_real_B + loss_fake_B) / 2
                    
                    scaler.scale(loss_D_B).backward()
                    scaler.step(optimizer_D_B)
                    scaler.update()
                else:
                    loss_real_B = criterion_GAN(D_B(real_B_disc), valid_disc)
                    fake_B_disc = G_AB(torch.cat((real_A_disc, mask_M_disc, mask_A_disc), 1)).detach()
                    loss_fake_B = criterion_GAN(D_B(fake_B_disc), fake_disc)
                    loss_D_B = (loss_real_B + loss_fake_B) / 2
                    loss_D_B.backward()
                    optimizer_D_B.step()
                
                # Accumulate losses
                epoch_g_loss += loss_G.item()
                epoch_d_loss += (loss_D_A.item() + loss_D_B.item())
                epoch_sup_loss += loss_supervised.item()
                if compute_metrics:
                    psnr = calculate_psnr(fake_B.detach().float(), real_B_gen.float()).item()
                    epoch_psnr += psnr
                batch_count += 1

                if (batch_idx + 1) % 10 == 0:
                    if compute_metrics:
                        logger.info(
                            f"[Epoch {epoch + 1}/{epochs}] "
                            f"[Batch {batch_idx + 1}/{len(gen_loader)}] "
                            f"[G: {loss_G.item():.4f}] "
                            f"[D: {(loss_D_A.item() + loss_D_B.item()):.4f}] "
                            f"[PSNR: {psnr:.2f} dB]"
                        )
                    else:
                        logger.info(
                            f"[Epoch {epoch + 1}/{epochs}] "
                            f"[Batch {batch_idx + 1}/{len(gen_loader)}] "
                            f"[G: {loss_G.item():.4f}] "
                            f"[D: {(loss_D_A.item() + loss_D_B.item()):.4f}]"
                        )

            # Epoch statistics
            avg_g_loss = epoch_g_loss / batch_count
            avg_d_loss = epoch_d_loss / batch_count
            avg_sup_loss = epoch_sup_loss / batch_count

            history['G_loss'].append(avg_g_loss)
            history['D_loss'].append(avg_d_loss)
            history['Sup_loss'].append(avg_sup_loss)

            if compute_metrics:
                avg_psnr = epoch_psnr / batch_count
                history['PSNR'].append(avg_psnr)
                logger.info(
                    f"[Epoch {epoch + 1}/{epochs}] "
                    f"G_Loss: {avg_g_loss:.4f} | "
                    f"D_Loss: {avg_d_loss:.4f} | "
                    f"Sup_Loss: {avg_sup_loss:.4f} | "
                    f"PSNR: {avg_psnr:.2f} dB"
                )
            else:
                logger.info(
                    f"[Epoch {epoch + 1}/{epochs}] "
                    f"G_Loss: {avg_g_loss:.4f} | "
                    f"D_Loss: {avg_d_loss:.4f} | "
                    f"Sup_Loss: {avg_sup_loss:.4f}"
                )
        
        # Save models
        model_G_AB_path = os.path.join(output_dir, f'cyclegan_G_AB_{label_mode.lower()}.pth')
        model_G_BA_path = os.path.join(output_dir, f'cyclegan_G_BA_{label_mode.lower()}.pth')
        model_D_A_path = os.path.join(output_dir, f'cyclegan_D_A_{label_mode.lower()}.pth')
        model_D_B_path = os.path.join(output_dir, f'cyclegan_D_B_{label_mode.lower()}.pth')
        
        torch.save(G_AB.state_dict(), model_G_AB_path)
        torch.save(G_BA.state_dict(), model_G_BA_path)
        torch.save(D_A.state_dict(), model_D_A_path)
        torch.save(D_B.state_dict(), model_D_B_path)
        
        logger.info(f"\nModels saved:")
        logger.info(f"  {model_G_AB_path}")
        logger.info(f"  {model_G_BA_path}")
        logger.info(f"  {model_D_A_path}")
        logger.info(f"  {model_D_B_path}")
        
        # Save history
        history_path = os.path.join(output_dir, f'cyclegan_history_{label_mode.lower()}.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("CycleGAN Training Complete")
    logger.info("=" * 70)
