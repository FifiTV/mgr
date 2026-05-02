"""
Diffusion Model training module.
"""

import copy
import logging
import os
import json
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.models.diffusion import DiffusionModel, DiffusionArchitecture


logger = logging.getLogger(__name__)


class EMA:
    """
    Exponential Moving Average of model weights.

    EMA weights are smoother than raw weights and produce significantly
    better samples in diffusion models. Decay=0.9999 is standard.
    Use ema.state_dict() to save the EMA weights for inference.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        # Store EMA params as CPU tensors to save GPU memory
        self.shadow = {k: v.detach().float().cpu().clone()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1.0 - self.decay) * v.float().cpu()

    def state_dict(self) -> dict:
        return self.shadow


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

    diff_cfg            = config['diffusion']
    epochs              = diff_cfg['n_epochs']
    lr                  = diff_cfg['learning_rate']
    ema_decay           = diff_cfg.get('ema_decay', 0.9999)
    warmup              = diff_cfg.get('lr_warmup_steps', 500)
    architecture        = diff_cfg.get('architecture', 'light')
    mask_dropout        = diff_cfg.get('mask_dropout_prob', 0.0)
    checkpoint_interval = config['training'].get('checkpoint_interval', 10)

    logger.info(f"Architecture: {architecture.upper()}")
    logger.info(f"Epochs: {epochs}  LR: {lr}  EMA decay: {ema_decay}  Warmup: {warmup}")
    logger.info(f"Mask dropout probability: {mask_dropout}")

    # Single model shared across both modes (HARD continues from SOFT weights)
    diffusion = DiffusionModel(
        architecture=architecture,
        time_steps=1000,
        device=device,
        input_channels=config['models']['unet_input_channels'],
        output_channels=config['models']['unet_output_channels'],
    )

    for mode_name, dataloader in [("SOFT", dataloader_soft), ("HARD", dataloader_hard)]:
        logger.info(f"\n{'='*40}")
        logger.info(f"Training {mode_name} Labels Diffusion Model  ({epochs} epochs)")
        logger.info(f"{'='*40}")

        optimizer = optim.Adam(diffusion.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)
        scaler    = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        ema       = EMA(diffusion.model, decay=ema_decay)

        history = {'loss': []}
        total_steps = 0

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

                # Mask dropout: zero out masks for the whole batch with probability
                # mask_dropout_prob. This teaches the model to generate artifacts
                # conditioned on the clean image alone (no mask at inference time).
                if mask_dropout > 0.0 and torch.rand(1).item() < mask_dropout:
                    mask_M = torch.zeros_like(mask_M)
                    mask_A = torch.zeros_like(mask_A)

                condition = torch.cat((real_A, mask_M, mask_A), dim=1)

                # LR warmup: linear ramp for first `warmup` steps
                if total_steps < warmup:
                    scale = (total_steps + 1) / warmup
                    for pg in optimizer.param_groups:
                        pg['lr'] = lr * scale

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

                ema.update(diffusion.model)
                total_steps += 1

                if (batch_idx + 1) % 10 == 0:
                    logger.info(
                        f"[{mode_name}][Epoch {epoch+1}/{epochs}]"
                        f"[Batch {batch_idx+1}/{len(dataloader)}]"
                        f"  loss={loss.item():.6f}"
                        f"  lr={optimizer.param_groups[0]['lr']:.2e}"
                    )

            avg_loss = epoch_loss / max(len(dataloader), 1)
            history['loss'].append(avg_loss)

            # Step scheduler only after warmup
            if total_steps >= warmup:
                scheduler.step()

            logger.info(
                f"[{mode_name}] Epoch {epoch+1}/{epochs}  avg_loss={avg_loss:.6f}"
                f"  lr={optimizer.param_groups[0]['lr']:.2e}"
            )

            # Save checkpoint every checkpoint_interval epochs (0 = disabled)
            if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
                ckpt_dir = os.path.join(
                    output_dir, 'checkpoints', 'diffusion',
                    mode_name.lower(), f'epoch_{epoch+1:04d}'
                )
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(diffusion.model.state_dict(),
                           os.path.join(ckpt_dir, 'model.pth'))
                torch.save(ema.state_dict(),
                           os.path.join(ckpt_dir, 'model_ema.pth'))
                logger.info(f"  Checkpoint saved: {ckpt_dir}")

        # Save raw model weights
        raw_path = os.path.join(output_dir, f'diffusion_unet_{mode_name.lower()}.pth')
        torch.save(diffusion.model.state_dict(), raw_path)
        logger.info(f"Raw weights saved to {raw_path}")

        # Save EMA weights (use these for inference -- better sample quality)
        ema_path = os.path.join(output_dir, f'diffusion_unet_{mode_name.lower()}_ema.pth')
        torch.save(ema.state_dict(), ema_path)
        logger.info(f"EMA weights saved to {ema_path}")

        # Save history
        history_path = os.path.join(output_dir, f'diffusion_history_{mode_name.lower()}.json')
        with open(history_path, 'w') as f:
            json.dump(history, f)
        logger.info(f"History saved to {history_path}")
