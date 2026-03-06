"""
CycleGAN training module.
"""

import logging
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models import (
    Generator, Discriminator, weights_init_normal
)
from src.utils.metrics import calculate_psnr


logger = logging.getLogger(__name__)


def train_cyclegan(dataloader_soft: DataLoader, dataloader_hard: DataLoader,
                   config: dict, device: torch.device,
                   output_dir: str) -> None:
    """
    Train CycleGAN model with soft and hard labels.
    
    Args:
        dataloader_soft: DataLoader with soft labels
        dataloader_hard: DataLoader with hard labels
        config: Configuration dictionary
        device: Computation device (cuda/cpu)
        output_dir: Directory to save model checkpoints
    """
    logger.info("=" * 60)
    logger.info("Starting CycleGAN Training")
    logger.info("=" * 60)
    
    # Configuration
    epochs = config['cyclegan']['n_epochs']
    lr = config['cyclegan']['learning_rate']
    lambda_cyc = config['cyclegan']['lambda_cycle']
    lambda_id = config['cyclegan']['lambda_identity']
    lambda_sup = config['cyclegan']['lambda_supervised']
    
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
    
    logger.info(f"Models initialized. Training for {epochs} epochs...")
    
    # Train both soft and hard versions
    for mode_name, dataloader in [("SOFT", dataloader_soft), ("HARD", dataloader_hard)]:
        logger.info(f"\nTraining {mode_name} Labels Model...")
        
        # Reset weights
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        
        history = {
            'G_loss': [],
            'D_loss': [],
            'Sup_loss': [],
            'PSNR': []
        }
        
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        
        for epoch in range(epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_sup_loss = 0.0
            epoch_psnr = 0.0
            
            for batch_idx, batch in enumerate(dataloader):
                real_A = batch['real_A'].to(device)
                real_B = batch['real_B'].to(device)
                mask_M = batch['mask_M'].to(device)
                mask_A = batch['mask_A'].to(device)
                
                valid = torch.ones((real_A.size(0), *D_A(real_A).shape[1:])).to(device)
                fake = torch.zeros((real_A.size(0), *D_A(real_A).shape[1:])).to(device)
                
                # Train Generators
                optimizer_G.zero_grad()
                
                if scaler:
                    with torch.amp.autocast('cuda'):
                        loss_id_A = criterion_identity(G_BA(real_A), real_A) * lambda_id
                        
                        input_G_AB = torch.cat((real_A, mask_M, mask_A), 1)
                        fake_B = G_AB(input_G_AB)
                        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
                        loss_supervised = criterion_supervised(fake_B, real_B) * lambda_sup
                        
                        fake_A = G_BA(real_B)
                        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
                        
                        rec_A = G_BA(fake_B)
                        loss_cycle_A = criterion_cycle(rec_A, real_A) * lambda_cyc
                        
                        input_rec_B = torch.cat((fake_A, mask_M, mask_A), 1)
                        rec_B = G_AB(input_rec_B)
                        loss_cycle_B = criterion_cycle(rec_B, real_B) * lambda_cyc
                        
                        loss_G = (loss_GAN_AB + loss_GAN_BA +
                                loss_cycle_A + loss_cycle_B +
                                loss_id_A + loss_supervised)
                    
                    scaler.scale(loss_G).backward()
                    scaler.step(optimizer_G)
                else:
                    loss_id_A = criterion_identity(G_BA(real_A), real_A) * lambda_id
                    
                    input_G_AB = torch.cat((real_A, mask_M, mask_A), 1)
                    fake_B = G_AB(input_G_AB)
                    loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
                    loss_supervised = criterion_supervised(fake_B, real_B) * lambda_sup
                    
                    fake_A = G_BA(real_B)
                    loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
                    
                    rec_A = G_BA(fake_B)
                    loss_cycle_A = criterion_cycle(rec_A, real_A) * lambda_cyc
                    
                    input_rec_B = torch.cat((fake_A, mask_M, mask_A), 1)
                    rec_B = G_AB(input_rec_B)
                    loss_cycle_B = criterion_cycle(rec_B, real_B) * lambda_cyc
                    
                    loss_G = (loss_GAN_AB + loss_GAN_BA +
                            loss_cycle_A + loss_cycle_B +
                            loss_id_A + loss_supervised)
                    loss_G.backward()
                    optimizer_G.step()
                
                # Train Discriminators
                optimizer_D_A.zero_grad()
                if scaler:
                    with torch.amp.autocast('cuda'):
                        loss_real_A = criterion_GAN(D_A(real_A), valid)
                        loss_fake_A = criterion_GAN(D_A(fake_A.detach()), fake)
                        loss_D_A = (loss_real_A + loss_fake_A) / 2
                    scaler.scale(loss_D_A).backward()
                    scaler.step(optimizer_D_A)
                else:
                    loss_real_A = criterion_GAN(D_A(real_A), valid)
                    loss_fake_A = criterion_GAN(D_A(fake_A.detach()), fake)
                    loss_D_A = (loss_real_A + loss_fake_A) / 2
                    loss_D_A.backward()
                    optimizer_D_A.step()
                
                optimizer_D_B.zero_grad()
                if scaler:
                    with torch.amp.autocast('cuda'):
                        loss_real_B = criterion_GAN(D_B(real_B), valid)
                        loss_fake_B = criterion_GAN(D_B(fake_B.detach()), fake)
                        loss_D_B = (loss_real_B + loss_fake_B) / 2
                    scaler.scale(loss_D_B).backward()
                    scaler.step(optimizer_D_B)
                else:
                    loss_real_B = criterion_GAN(D_B(real_B), valid)
                    loss_fake_B = criterion_GAN(D_B(fake_B.detach()), fake)
                    loss_D_B = (loss_real_B + loss_fake_B) / 2
                    loss_D_B.backward()
                    optimizer_D_B.step()
                
                if scaler:
                    scaler.update()
                
                # Metrics
                current_psnr = calculate_psnr(fake_B.detach(), real_B).item()
                epoch_g_loss += loss_G.item()
                epoch_d_loss += (loss_D_A.item() + loss_D_B.item())
                epoch_sup_loss += loss_supervised.item()
                epoch_psnr += current_psnr
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(
                        f"[Epoch {epoch + 1}/{epochs}] "
                        f"[Batch {batch_idx + 1}/{len(dataloader)}] "
                        f"[G Loss: {loss_G.item():.4f}] "
                        f"[PSNR: {current_psnr:.2f} dB]"
                    )
            
            avg_g_loss = epoch_g_loss / len(dataloader)
            avg_d_loss = epoch_d_loss / len(dataloader)
            avg_sup_loss = epoch_sup_loss / len(dataloader)
            avg_psnr = epoch_psnr / len(dataloader)
            
            history['G_loss'].append(avg_g_loss)
            history['D_loss'].append(avg_d_loss)
            history['Sup_loss'].append(avg_sup_loss)
            history['PSNR'].append(avg_psnr)
            
            logger.info(
                f"End of Epoch {epoch + 1}: "
                f"Avg PSNR: {avg_psnr:.2f} dB | "
                f"Avg Sup Loss: {avg_sup_loss:.4f}"
            )
        
        # Save model
        model_path = os.path.join(output_dir, f'cyclegan_G_AB_{mode_name.lower()}.pth')
        torch.save(G_AB.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save history
        history_path = os.path.join(output_dir, f'cyclegan_history_{mode_name.lower()}.json')
        with open(history_path, 'w') as f:
            json.dump(history, f)
        logger.info(f"History saved to {history_path}")
