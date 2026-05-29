"""
Gaussian Vicinal DDPM -- Training module.

Key components:
    GaussianVicinalLoss : weighted MSE with Gaussian kernel in feature space
    sigma_schedule      : cosine annealing of sigma from sigma_max to sigma_min
    train_gaussian      : main training loop
"""

import copy
import csv
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.models.gaussian_unet import GaussianUNet, GaussianDDPM
from src.datasets.gaussian_dataset import GaussianDataset, load_features_df, FEATURE_COLS

logger = logging.getLogger(__name__)


# ── EMA ──────────────────────────────────────────────────────────────────────────

class EMA:
    """Exponential Moving Average of model weights (stored on CPU)."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay  = decay
        self.shadow = {k: v.detach().float().cpu().clone()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.float().cpu()

    def state_dict(self) -> dict:
        return self.shadow


# ── Sigma schedule ────────────────────────────────────────────────────────────────

def sigma_schedule(epoch: int, total_epochs: int,
                   sigma_max: float = 0.3, sigma_min: float = 0.02) -> float:
    """
    Cosine annealing of the Gaussian neighborhood bandwidth sigma.

    Large sigma at the start: broad neighborhood (many samples contribute).
    Small sigma at the end:   narrow neighborhood (precise feature targeting).
    """
    cosine = 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
    return sigma_min + (sigma_max - sigma_min) * cosine


# ── Loss ─────────────────────────────────────────────────────────────────────────

def gaussian_vicinal_loss(eps: torch.Tensor,
                          eps_pred: torch.Tensor,
                          y_i: torch.Tensor,
                          y_target: torch.Tensor,
                          sigma: float) -> tuple:
    """
    Gaussian Vicinal Loss.

    Weights each sample by how close its feature vector y_i is to the target
    y_target sampled uniformly from [0,1]^6. The model is conditioned on
    y_target for the whole batch.

    L = mean_i( w_i * ||eps_i - eps_pred_i||^2 )
    w_i = exp(-||y_target - y_i||^2 / (2*sigma^2))

    Args:
        eps       : [B, 1, H, W]  true noise
        eps_pred  : [B, 1, H, W]  predicted noise
        y_i       : [B, 6]        per-sample feature vectors
        y_target  : [6] or [B, 6] batch target feature vector (same for all i)
        sigma     : current bandwidth

    Returns:
        (loss, mean_weight)  -- mean_weight in [0, 1] shows neighborhood coverage
    """
    if y_target.dim() == 1:
        y_target = y_target.unsqueeze(0)              # [1, 6]

    diff    = y_target - y_i                           # [B, 6]
    weights = torch.exp(-diff.pow(2).sum(dim=1) / (2 * sigma ** 2))  # [B]
    mse     = (eps - eps_pred).pow(2).mean(dim=[1, 2, 3])             # [B]
    loss    = (weights * mse).mean()

    return loss, weights.mean().item()


# ── Sample generation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_samples(ddpm: GaussianDDPM,
                     ema_state: dict,
                     condition: torch.Tensor,
                     save_dir: str,
                     epoch: int,
                     device: torch.device,
                     y_dim: int = 6) -> None:
    """
    Generate visual sanity-check samples using EMA weights for two fixed y_target:
        y_low  = [0.2, 0.5, ..., 0.5]  -- low peak_amplitude
        y_high = [0.8, 0.5, ..., 0.5]  -- high peak_amplitude

    EMA weights are temporarily swapped in and restored after sampling.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    # Swap in EMA weights -- better sample quality than raw training weights
    orig_state = {k: v.clone() for k, v in ddpm.unet.state_dict().items()}
    ddpm.unet.load_state_dict({k: v.to(device) for k, v in ema_state.items()})
    ddpm.eval()

    B = min(condition.shape[0], 2)
    cond = condition[:B].to(device)

    for label, amp in [('low', 0.2), ('high', 0.8)]:
        y_vals      = [0.5] * y_dim
        y_vals[0]   = amp  # peak_amplitude is always first feature
        y_target    = torch.tensor(y_vals, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1)
        i_error_gen = ddpm.sample(cond, y_target)          # [B, 1, H, W]

        for i in range(B):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            fig.suptitle(f'Epoch {epoch} | peak_amplitude={label} | sample {i}',
                         fontsize=11)

            axes[0].imshow(cond[i, 0].cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
            axes[0].set_title('I_clean')
            axes[0].axis('off')

            axes[1].imshow(cond[i, 1].cpu().numpy(), cmap='Reds', vmin=0, vmax=1)
            axes[1].set_title('M_metal')
            axes[1].axis('off')

            axes[2].imshow(i_error_gen[i, 0].cpu().numpy(), cmap='RdBu',
                           vmin=-1, vmax=1)
            axes[2].set_title(f'Generated I_error ({label})')
            axes[2].axis('off')

            plt.tight_layout()
            fname = os.path.join(save_dir,
                                 f'sample_epoch{epoch:04d}_{label}_s{i:02d}.png')
            plt.savefig(fname, dpi=100, bbox_inches='tight')
            plt.close(fig)

    # Restore raw training weights
    ddpm.unet.load_state_dict(orig_state)
    ddpm.train()
    logger.info(f"Samples saved to {save_dir}")


# ── Main training function ────────────────────────────────────────────────────────

def train_gaussian(config: dict,
                   device: torch.device,
                   output_dir: str,
                   body_dirs: List[Union[str, Path]],
                   features_source: Union[str, Path],
                   clip_stats_path: Optional[Union[str, Path]] = None) -> None:
    """
    Train Gaussian Vicinal DDPM.

    Args:
        config          : full config dict (reads from config['gaussian'])
        device          : torch device
        output_dir      : directory for checkpoints, history, samples
        body_dirs       : list of body variant directories (each with Baseline/Target/Metal/)
        features_source : path to a single CSV or a directory with body*/features.csv
        clip_stats_path : path to feature_clip_stats.json (needed if features are raw)
    """
    logger.info("=" * 60)
    logger.info("Gaussian Vicinal DDPM Training")
    logger.info("=" * 60)

    gaus_cfg            = config.get('gaussian', {})
    T                   = gaus_cfg.get('T', 1000)
    beta_schedule       = gaus_cfg.get('beta_schedule', 'linear')
    beta_start          = gaus_cfg.get('beta_start', 1e-4)
    beta_end            = gaus_cfg.get('beta_end', 0.02)
    batch_size          = gaus_cfg.get('batch_size', 4)
    lr                  = gaus_cfg.get('learning_rate', 2e-4)
    n_epochs            = gaus_cfg.get('n_epochs', 100)
    sigma_max           = gaus_cfg.get('sigma_max', 0.3)
    sigma_min           = gaus_cfg.get('sigma_min', 0.02)
    p_random_metal      = gaus_cfg.get('p_random_metal', 0.5)
    ema_decay           = gaus_cfg.get('ema_decay', 0.9999)
    base_ch             = gaus_cfg.get('base_channels', 64)
    t_emb_dim           = gaus_cfg.get('t_emb_dim', 256)
    attn_heads          = gaus_cfg.get('attn_heads', 8)
    metal_threshold_hu  = gaus_cfg.get('metal_threshold_hu',
                              config.get('data', {}).get('metal_threshold_hu', 500.0))
    metal_dt_radius     = gaus_cfg.get('metal_dt_radius', 200.0)
    metal_dt_enabled    = gaus_cfg.get('metal_dt_enabled', True)
    error_scale         = gaus_cfg.get('error_scale', 310.5)
    checkpoint_interval = config.get('training', {}).get('checkpoint_interval', 10)
    num_workers         = config.get('dataset', {}).get('num_workers', 2)
    sample_interval     = gaus_cfg.get('sample_interval', 1)
    feature_cols        = gaus_cfg.get('feature_cols', None) or FEATURE_COLS
    y_dim               = len(feature_cols)

    logger.info(f"T={T}, beta_schedule={beta_schedule}, epochs={n_epochs}, "
                f"batch={batch_size}, lr={lr}")
    logger.info(f"sigma: {sigma_max} -> {sigma_min} (cosine)")
    logger.info(f"p_random_metal={p_random_metal}, ema_decay={ema_decay}")
    logger.info(f"feature_cols ({y_dim}): {feature_cols}")

    os.makedirs(output_dir, exist_ok=True)
    sample_dir = os.path.join(output_dir, 'samples', 'gaussian')

    # ── Features ──────────────────────────────────────────────────────────────────
    features_df = load_features_df(features_source, clip_stats_path)
    logger.info(f"Features loaded: {len(features_df)} rows")

    # ── Dataset / DataLoader ──────────────────────────────────────────────────────
    logger.info(f"metal_dt_enabled={metal_dt_enabled}  metal_dt_radius={metal_dt_radius}")
    dataset = GaussianDataset(
        body_dirs=body_dirs,
        features_df=features_df,
        p_random_metal=p_random_metal,
        metal_threshold_hu=metal_threshold_hu,
        metal_dt_radius=metal_dt_radius,
        use_distance_transform=metal_dt_enabled,
        error_scale=error_scale,
        feature_cols=feature_cols,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    logger.info(f"DataLoader: {len(dataset)} samples, {len(dataloader)} batches/epoch")

    # ── Model ─────────────────────────────────────────────────────────────────────
    unet = GaussianUNet(
        in_ch=3, out_ch=1,
        base_ch=base_ch,
        t_emb_dim=t_emb_dim,
        y_dim=y_dim,
        attn_heads=attn_heads,
    ).to(device)

    ddpm = GaussianDDPM(
        unet=unet,
        T=T,
        beta_schedule=beta_schedule,
        beta_start=beta_start,
        beta_end=beta_end,
    ).to(device)

    param_count = sum(p.numel() for p in unet.parameters()) / 1e6
    logger.info(f"GaussianUNet parameters: {param_count:.1f}M")

    # ── Optimizer / Scheduler / EMA ───────────────────────────────────────────────
    optimizer = optim.Adam(ddpm.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.1)
    scaler    = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    ema       = EMA(unet, decay=ema_decay)

    history = {'loss': [], 'sigma': [], 'mean_weight': [], 'avg_dist': []}

    hist_csv_path = os.path.join(output_dir, 'gaussian_history.csv')
    with open(hist_csv_path, 'w', newline='') as _f:
        csv.writer(_f).writerow(['epoch', 'loss', 'sigma', 'mean_weight', 'avg_dist', 'lr'])
    logger.info(f"Metrics CSV: {hist_csv_path}")

    # Fixed condition for visualization — built WITHOUT random metal swap so each
    # I_clean is paired with its own M_metal, not a randomly swapped one.
    _n_vis = min(4, len(dataset))
    _p_save = dataset.p_random_metal
    dataset.p_random_metal = 0.0
    _vis = [dataset[i] for i in range(_n_vis)]
    dataset.p_random_metal = _p_save
    _i_clean_vis = torch.stack([s['i_clean'] for s in _vis])
    _m_metal_vis = torch.stack([s['m_metal'] for s in _vis])
    sample_condition = torch.cat([_i_clean_vis, _m_metal_vis], dim=1).cpu()
    del _vis, _i_clean_vis, _m_metal_vis

    # ── Training loop ─────────────────────────────────────────────────────────────
    for epoch in range(n_epochs):
        sigma   = sigma_schedule(epoch, n_epochs, sigma_max, sigma_min)
        ep_loss = 0.0
        ep_wt   = 0.0
        ep_dist = 0.0
        ddpm.train()

        for batch_idx, batch in enumerate(dataloader):
            i_clean  = batch['i_clean'].to(device)   # [B, 1, H, W]
            i_error  = batch['i_error'].to(device)   # [B, 1, H, W]
            m_metal  = batch['m_metal'].to(device)   # [B, 1, H, W]
            y_i      = batch['y'].to(device)         # [B, 6]

            condition = torch.cat([i_clean, m_metal], dim=1)  # [B, 2, H, W]

            # Sample y_target as a random row from the current batch's feature vectors.
            # Using torch.rand would land in empty regions of feature space (U[0,1]^F
            # has no samples near it), driving all Gaussian weights to ~0 and loss to 0.
            rand_idx = torch.randint(y_i.size(0), (1,)).item()
            y_target = y_i[rand_idx].unsqueeze(0).expand(y_i.size(0), -1).contiguous()  # [B, F]

            # ── Diagnostic log (first batch of each epoch) ────────────────────────
            if batch_idx == 0:
                with torch.no_grad():
                    yi_mean = y_i.mean(dim=0).cpu().numpy().round(3)
                    yi_std  = y_i.std(dim=0).cpu().numpy().round(3)
                    dist    = (y_target - y_i).pow(2).sum(dim=1).sqrt()
                    dist_mean = dist.mean().item()
                    logger.info(
                        f"  [DIAG E{epoch+1}] y_i mean={yi_mean}  std={yi_std}"
                    )
                    logger.info(
                        f"  [DIAG E{epoch+1}] dist(y_target→y_i): mean={dist_mean:.4f}"
                        f"  min={dist.min().item():.4f}  max={dist.max().item():.4f}"
                    )
                    logger.info(
                        f"  [DIAG E{epoch+1}] i_error: min={i_error.min():.4f}"
                        f"  max={i_error.max():.4f}  mean={i_error.mean():.5f}"
                    )

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    eps, eps_pred = ddpm(i_error, condition, y_target)
                    loss, mean_wt = gaussian_vicinal_loss(
                        eps, eps_pred, y_i, y_target, sigma
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                eps, eps_pred = ddpm(i_error, condition, y_target)
                loss, mean_wt = gaussian_vicinal_loss(
                    eps, eps_pred, y_i, y_target[0], sigma
                )
                loss.backward()
                nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
                optimizer.step()

            ema.update(unet)
            ep_loss += loss.item()
            ep_wt   += mean_wt
            with torch.no_grad():
                ep_dist += (y_target - y_i).pow(2).sum(dim=1).sqrt().mean().item()

            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"[Epoch {epoch+1}/{n_epochs}][Batch {batch_idx+1}/{len(dataloader)}]"
                    f"  loss={loss.item():.5f}"
                    f"  sigma={sigma:.4f}"
                    f"  mean_weight={mean_wt:.4f}"
                    f"  lr={optimizer.param_groups[0]['lr']:.2e}"
                )

        scheduler.step()

        avg_loss = ep_loss / max(len(dataloader), 1)
        avg_wt   = ep_wt   / max(len(dataloader), 1)
        avg_dist = ep_dist / max(len(dataloader), 1)
        history['loss'].append(avg_loss)
        history['sigma'].append(sigma)
        history['mean_weight'].append(avg_wt)
        history['avg_dist'].append(avg_dist)

        current_lr = optimizer.param_groups[0]['lr']
        wt_warn = '  *** mean_weight HIGH — sigma too large or features clustered? ***' if avg_wt > 0.3 else ''
        logger.info(
            f"[Epoch {epoch+1}/{n_epochs}] avg_loss={avg_loss:.5f}"
            f"  sigma={sigma:.4f}  mean_weight={avg_wt:.4f}"
            f"  avg_dist={avg_dist:.4f}  lr={current_lr:.2e}{wt_warn}"
        )

        with open(hist_csv_path, 'a', newline='') as _f:
            csv.writer(_f).writerow(
                [epoch + 1, f'{avg_loss:.6f}', f'{sigma:.6f}',
                 f'{avg_wt:.6f}', f'{avg_dist:.6f}', f'{current_lr:.2e}']
            )

        # ── Checkpoint ────────────────────────────────────────────────────────────
        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            ckpt_dir = os.path.join(
                output_dir, 'checkpoints', 'gaussian', f'epoch_{epoch+1:04d}'
            )
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                'epoch':          epoch + 1,
                'model_state':    unet.state_dict(),
                'ema_state':      ema.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'sigma':          sigma,
                'loss':           avg_loss,
            }, os.path.join(ckpt_dir, 'checkpoint.pth'))
            logger.info(f"  Checkpoint saved: {ckpt_dir}")

        # ── Sample generation ─────────────────────────────────────────────────────
        if sample_interval > 0 and (epoch + 1) % sample_interval == 0:
            cond_batch = sample_condition.to(device)
            generate_samples(ddpm, ema.state_dict(), cond_batch, sample_dir, epoch + 1, device, y_dim=y_dim)

    # ── Final weights ─────────────────────────────────────────────────────────────
    raw_path = os.path.join(output_dir, 'gaussian_unet.pth')
    torch.save(unet.state_dict(), raw_path)
    logger.info(f"Raw weights saved: {raw_path}")

    ema_path = os.path.join(output_dir, 'gaussian_unet_ema.pth')
    torch.save(ema.state_dict(), ema_path)
    logger.info(f"EMA weights saved: {ema_path}")

    hist_path = os.path.join(output_dir, 'gaussian_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"History saved: {hist_path}")

    logger.info("Gaussian Vicinal DDPM training complete.")
