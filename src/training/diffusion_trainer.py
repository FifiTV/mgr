"""
Diffusion Model training module.
"""

import logging
import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.diffusion import DiffusionModel, DiffusionArchitecture


logger = logging.getLogger(__name__)


def train_diffusion(dataloader_soft: DataLoader, dataloader_hard: DataLoader,
                    config: dict, device: torch.device,
                    output_dir: str) -> None:
    """
    Train Diffusion model with soft and hard labels.
    
    Args:
        dataloader_soft: DataLoader with soft labels
        dataloader_hard: DataLoader with hard labels
        config: Configuration dictionary
        device: Computation device (cuda/cpu)
        output_dir: Directory to save model checkpoints
    """
    logger.info("=" * 60)
    logger.info("Starting Diffusion Model Training")
    logger.info("=" * 60)
    
    # Configuration
    epochs = config['diffusion']['n_epochs']
    lr = config['diffusion']['learning_rate']
    architecture = config['diffusion'].get('architecture', 'light')  # light or standard
    
    logger.info(f"Using diffusion architecture: {architecture.upper()}")
    
    # Initialize model (DiffusionModel now creates the appropriate U-Net)
    diffusion = DiffusionModel(
        architecture=architecture,
        time_steps=1000,
        device=device,
        input_channels=config['models']['unet_input_channels'],
        output_channels=config['models']['unet_output_channels']
    )
    
    optimizer = optim.Adam(diffusion.parameters(), lr=lr)
    
    logger.info(f"Diffusion model initialized ({architecture}). Training for {epochs} epochs...")
    
    # Train both soft and hard versions
    for mode_name, dataloader in [("SOFT", dataloader_soft), ("HARD", dataloader_hard)]:
        logger.info(f"\nTraining {mode_name} Labels Diffusion Model...")
        
        history = {'loss': []}
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            diffusion.train()
            
            for batch_idx, batch in enumerate(dataloader):
                real_A = batch['real_A'].to(device)
                real_B = batch['real_B'].to(device)
                mask_M = batch['mask_M'].to(device)
                mask_A = batch['mask_A'].to(device)
                
                if mode_name == "HARD":
                    mask_A = (mask_A > 0.2).float()
                
                condition = torch.cat((real_A, mask_M, mask_A), dim=1)
                
                optimizer.zero_grad()
                
                if scaler:
                    with torch.amp.autocast('cuda'):
                        loss = diffusion.compute_losses(real_B, condition)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = diffusion.compute_losses(real_B, condition)
                    loss.backward()
                    optimizer.step()
                
                epoch_loss += loss.item()
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(
                        f"[Epoch {epoch + 1}/{epochs}] "
                        f"[Batch {batch_idx + 1}/{len(dataloader)}] "
                        f"[Loss: {loss.item():.4f}]"
                    )
            
            avg_loss = epoch_loss / len(dataloader)
            history['loss'].append(avg_loss)
            logger.info(f"End of Epoch {epoch + 1}: Avg Loss: {avg_loss:.4f}")
        
        # Save model
        model_path = os.path.join(output_dir, f'diffusion_unet_{mode_name.lower()}.pth')
        torch.save(diffusion.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save history
        history_path = os.path.join(output_dir, f'diffusion_history_{mode_name.lower()}.json')
        with open(history_path, 'w') as f:
            json.dump(history, f)
        logger.info(f"History saved to {history_path}")
