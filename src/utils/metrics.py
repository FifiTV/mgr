"""
Metrics for model evaluation (PSNR and others).
"""

import torch


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        img1: Predicted image tensor
        img2: Ground truth image tensor
        
    Returns:
        PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'), device=img1.device)
    return 20 * torch.log10(2.0 / torch.sqrt(mse))


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, 
                   kernel_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1: Predicted image tensor
        img2: Ground truth image tensor
        kernel_size: Size of the Gaussian kernel
        sigma: Standard deviation of the Gaussian kernel
        
    Returns:
        SSIM value in range [0, 1]
    """
    # Simplified SSIM implementation
    # For production, consider using torchvision or pytorch-ssim
    channels = img1.shape[1]
    
    # Ensure tensors are in correct shape for comparison
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes must match: {img1.shape} vs {img2.shape}")
    
    # Constants for numerical stability
    c1 = 0.01**2
    c2 = 0.03**2
    
    # Mean
    mean1 = torch.mean(img1, dim=[2, 3], keepdim=True)
    mean2 = torch.mean(img2, dim=[2, 3], keepdim=True)
    
    # Variance
    var1 = torch.mean((img1 - mean1)**2, dim=[2, 3], keepdim=True)
    var2 = torch.mean((img2 - mean2)**2, dim=[2, 3], keepdim=True)
    
    # Covariance
    cov = torch.mean((img1 - mean1) * (img2 - mean2), dim=[2, 3], keepdim=True)
    
    # SSIM
    ssim = ((2 * mean1 * mean2 + c1) * (2 * cov + c2)) / (
        (mean1**2 + mean2**2 + c1) * (var1 + var2 + c2)
    )
    
    return torch.mean(ssim)
