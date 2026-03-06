"""
Visualization utilities for training results and inference.
"""

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
