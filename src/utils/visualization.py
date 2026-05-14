"""
Visualization utilities for training results and inference.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import torch


def plot_training_history_cyclegan(history: Dict[str, List[float]], 
                                    save_path: Optional[str] = None) -> None:
    """
    Plot CycleGAN training history.
    
    Args:
        history: Dictionary with training metrics
        save_path: Path to save the plot (optional)
    """
    epochs = range(len(history.get('G_loss', [])))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Generator Loss
    if 'G_loss' in history:
        axes[0, 0].plot(epochs, history['G_loss'], 'b-', linewidth=2)
        axes[0, 0].set_title('Generator Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Discriminator Loss
    if 'D_loss' in history:
        axes[0, 1].plot(epochs, history['D_loss'], 'r-', linewidth=2)
        axes[0, 1].set_title('Discriminator Loss', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Supervised Loss
    if 'Sup_loss' in history:
        axes[1, 0].plot(epochs, history['Sup_loss'], 'g-', linewidth=2)
        axes[1, 0].set_title('Supervised Reconstruction Loss (L1)', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
    
    # PSNR
    if 'PSNR' in history:
        axes[1, 1].plot(epochs, history['PSNR'], 'purple', linewidth=2)
        axes[1, 1].set_title('PSNR (Image Quality)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('dB')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_training_history_diffusion(history: Dict[str, List[float]], 
                                     save_path: Optional[str] = None) -> None:
    """
    Plot Diffusion model training history.
    
    Args:
        history: Dictionary with training metrics (should contain 'loss')
        save_path: Path to save the plot (optional)
    """
    epochs = range(1, len(history.get('loss', [])) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'loss' in history:
        ax.plot(epochs, history['loss'], 'b-', linewidth=2, marker='o')
        ax.set_title('Diffusion Model Training Loss', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('MSE Loss', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def visualize_predictions(clean_img: torch.Tensor, 
                          target_img: torch.Tensor,
                          predicted_img: torch.Tensor,
                          mask_metal: torch.Tensor,
                          mask_artifact: torch.Tensor,
                          title: str = "Prediction Results",
                          save_path: Optional[str] = None) -> None:
    """
    Visualize model predictions with multiple views.
    
    Args:
        clean_img: Clean CT image (input)
        target_img: Target/ground truth image
        predicted_img: Model prediction
        mask_metal: Metal mask
        mask_artifact: Artifact mask
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # Convert to numpy and squeeze batch/channel dims
    clean = clean_img.squeeze().cpu().detach().numpy()
    target = target_img.squeeze().cpu().detach().numpy()
    predicted = predicted_img.squeeze().cpu().detach().numpy()
    mask_m = mask_metal.squeeze().cpu().detach().numpy()
    mask_a = mask_artifact.squeeze().cpu().detach().numpy()
    
    # Clean Image
    axes[0].imshow(clean, cmap='gray', vmin=-1, vmax=1)
    axes[0].set_title('Input (Clean)', fontweight='bold')
    axes[0].axis('off')
    
    # Metal Mask
    axes[1].imshow(mask_m, cmap='Reds')
    axes[1].set_title('Metal Mask', fontweight='bold')
    axes[1].axis('off')
    
    # Artifact Mask
    axes[2].imshow(mask_a, cmap='hot')
    axes[2].set_title('Artifact Mask', fontweight='bold')
    axes[2].axis('off')
    
    # Predicted
    axes[3].imshow(predicted, cmap='gray', vmin=-1, vmax=1)
    axes[3].set_title('Generated', fontweight='bold')
    axes[3].axis('off')
    
    # Target
    axes[4].imshow(target, cmap='gray', vmin=-1, vmax=1)
    axes[4].set_title('Ground Truth', fontweight='bold')
    axes[4].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()


def save_cyclegan_samples(epoch: int,
                          label_mode: str,
                          real_A: torch.Tensor,
                          real_B: torch.Tensor,
                          fake_B: torch.Tensor,
                          mask_M: torch.Tensor,
                          mask_A: torch.Tensor,
                          save_dir: str,
                          n_samples: int = 4) -> None:
    """
    Save training-progress image grids for one epoch.

    Saves up to n_samples PNGs from the batch, each with layout:
        Input (clean) | Metal Mask | Artifact Mask | Generated | Ground Truth

    Args:
        epoch:      Current epoch number (1-based).
        label_mode: "SOFT" or "HARD" — used in filename.
        real_A:     Clean CT images  (B, 1, H, W), range [-1, 1].
        real_B:     Artifact images  (B, 1, H, W), range [-1, 1].
        fake_B:     Generated images (B, 1, H, W), range [-1, 1].
        mask_M:     Metal mask       (B, 1, H, W), range [0, 1].
        mask_A:     Artifact mask    (B, 1, H, W), range [0, 1].
        save_dir:   Directory where PNGs are written.
        n_samples:  How many samples from the batch to save (default: 4).
    """
    os.makedirs(save_dir, exist_ok=True)

    batch_size = real_A.shape[0]
    n = min(n_samples, batch_size)

    def to_np(t: torch.Tensor, i: int) -> np.ndarray:
        return t[i].squeeze().cpu().detach().float().numpy()

    for i in range(n):
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        fig.suptitle(
            f"Epoch {epoch} | {label_mode} | sample {i+1}/{n}",
            fontsize=13, fontweight='bold'
        )

        axes[0].imshow(to_np(real_A, i), cmap='gray', vmin=-1, vmax=1)
        axes[0].set_title('Input (clean)',  fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(to_np(mask_M, i),  cmap='Reds', vmin=0, vmax=1)
        axes[1].set_title('Metal mask',    fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(to_np(mask_A, i),  cmap='hot',  vmin=0, vmax=1)
        axes[2].set_title('Artifact mask', fontweight='bold')
        axes[2].axis('off')

        axes[3].imshow(to_np(fake_B, i),  cmap='gray', vmin=-1, vmax=1)
        axes[3].set_title('Generated',     fontweight='bold')
        axes[3].axis('off')

        axes[4].imshow(to_np(real_B, i),  cmap='gray', vmin=-1, vmax=1)
        axes[4].set_title('Ground truth',  fontweight='bold')
        axes[4].axis('off')

        plt.tight_layout()
        fname = os.path.join(
            save_dir,
            f"sample_{label_mode.lower()}_epoch{epoch:04d}_s{i:02d}.png"
        )
        plt.savefig(fname, dpi=130, bbox_inches='tight')
        plt.close(fig)


def save_diffusion_samples(epoch: int,
                           label_mode: str,
                           real_A: torch.Tensor,
                           real_B: torch.Tensor,
                           fake_B: torch.Tensor,
                           mask_M: torch.Tensor,
                           mask_A: torch.Tensor,
                           save_dir: str,
                           n_samples: int = 4) -> None:
    """
    Save diffusion training-progress image grids for one epoch.

    Layout per sample:
        Input (clean) | Metal mask | Artifact mask | Generated | Ground truth

    Args:
        epoch:      Current epoch number (1-based).
        label_mode: "SOFT" or "HARD" — used in filename.
        real_A:     Clean CT images      (B, 1, H, W), range [-1, 1].
        real_B:     Artifact images      (B, 1, H, W), range [-1, 1].
        fake_B:     Diffusion samples    (B, 1, H, W), range [-1, 1].
        mask_M:     Metal mask           (B, 1, H, W), range [0, 1].
        mask_A:     Artifact mask        (B, 1, H, W), range [0, 1].
        save_dir:   Directory where PNGs are written.
        n_samples:  How many samples from the batch to save (default: 4).
    """
    os.makedirs(save_dir, exist_ok=True)

    n = min(n_samples, real_A.shape[0])

    def to_np(t: torch.Tensor, i: int) -> np.ndarray:
        return t[i].squeeze().cpu().detach().float().numpy()

    for i in range(n):
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        fig.suptitle(
            f"Diffusion | Epoch {epoch} | {label_mode} | sample {i+1}/{n}",
            fontsize=13, fontweight='bold'
        )

        axes[0].imshow(to_np(real_A, i), cmap='gray', vmin=-1, vmax=1)
        axes[0].set_title('Input (clean)',  fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(to_np(mask_M, i),  cmap='Reds', vmin=0, vmax=1)
        axes[1].set_title('Metal mask',    fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(to_np(mask_A, i),  cmap='hot',  vmin=0, vmax=1)
        axes[2].set_title('Artifact mask', fontweight='bold')
        axes[2].axis('off')

        axes[3].imshow(to_np(fake_B, i),  cmap='gray', vmin=-1, vmax=1)
        axes[3].set_title('Generated',     fontweight='bold')
        axes[3].axis('off')

        axes[4].imshow(to_np(real_B, i),  cmap='gray', vmin=-1, vmax=1)
        axes[4].set_title('Ground truth',  fontweight='bold')
        axes[4].axis('off')

        plt.tight_layout()
        fname = os.path.join(
            save_dir,
            f"diff_sample_{label_mode.lower()}_epoch{epoch:04d}_s{i:02d}.png"
        )
        plt.savefig(fname, dpi=130, bbox_inches='tight')
        plt.close(fig)
