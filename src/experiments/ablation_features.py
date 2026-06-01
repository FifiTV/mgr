"""ablation_features.py — Feature sensitivity analysis for Gaussian Vicinal DDPM.

Generates a F×N grid where each row sweeps one feature from 0.0→1.0
while all other features are frozen at the dataset median.
Same noise seed, same I_clean, same M_metal across every cell in the grid.

Usage:
  # Full ablation (all model features), first image from body1
  python src/experiments/ablation_features.py --model-dir results/models/gaussian/

  # Only 3 features, specific anchor image, faster sampling
  python src/experiments/ablation_features.py \\
      --model-dir results/models/gaussian/ \\
      --features peak_amplitude angular_concentration dark_to_bright_ratio \\
      --body body3 --img-id 42 --stride 10

  # Explicit model file, custom output
  python src/experiments/ablation_features.py \\
      --model results/models/gaussian/gaussian_unet_ema.pth \\
      --out results/ablation/run1.png

  # Different model with fewer features
  python src/experiments/ablation_features.py \\
      --model results/models/exp2/gaussian_unet_ema.pth \\
      --config results/models/exp2/config.toml \\
      --features peak_amplitude spatial_extent
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

_ROOT = Path(__file__).parent.parent.parent   # mgr/
sys.path.insert(0, str(_ROOT))

from src.datasets.gaussian_dataset import load_features_df, normalize_ct
from infer_utils import SHAPE, load_config, load_raw, load_model, ddpm_sample, GaussianDDPM

_FEATURE_LABELS: dict[str, str] = {
    'peak_amplitude':        'Peak amplitude',
    'spatial_extent':        'Spatial extent',
    'bbox_ratio':            'BBox ratio',
    'dark_to_bright_ratio':  'Dark/bright ratio',
    'angular_concentration': 'Angular conc.',
    'texture_roughness':     'Texture roughness',
}


# ── Model path resolution ──────────────────────────────────────────────────────

def resolve_model_path(model_arg: Path | None, model_dir_arg: Path | None,
                       cfg: dict) -> Path:
    if model_arg is not None:
        if not model_arg.exists():
            sys.exit(f'Model not found: {model_arg}')
        return model_arg

    if model_dir_arg is not None:
        d = model_dir_arg
    else:
        paths = cfg.get('paths', {})
        d = Path(paths.get('results_dir', str(_ROOT / 'results')), 'models', 'gaussian')

    for name in ('gaussian_unet_ema.pth', 'gaussian_unet.pth'):
        candidate = d / name
        if candidate.exists():
            return candidate
    sys.exit(f'No model found in {d}. Provide --model or --model-dir.')


# ── Anchor image ───────────────────────────────────────────────────────────────

def pick_anchor(body_dir: Path, img_id: int | None,
                metal_threshold_hu: float,
                device: torch.device,
                use_metal_mask: bool = True) -> tuple[torch.Tensor, np.ndarray, int]:
    """Load one (I_clean, M_metal) pair as the fixed anchor for all cells.

    Returns:
        condition  — [1, 2, H, W] tensor (I_clean_norm cat M_metal) when use_metal_mask=True
                     [1, 1, H, W] tensor (I_clean_norm only)         when use_metal_mask=False
        i_clean_n  — [H, W] numpy for optional reference display
        img_id_used
    """
    target_dir   = body_dir / 'Target'
    baseline_dir = body_dir / 'Baseline'

    if not target_dir.exists():
        sys.exit(f'Target directory not found: {target_dir}')

    def _id(p: Path) -> int | None:
        m = re.search(r'img(\d+)', p.stem)
        return int(m.group(1)) if m else None

    available = sorted(
        [(p, _id(p)) for p in target_dir.glob('*.raw') if _id(p) is not None],
        key=lambda x: x[1],
    )
    if not available:
        sys.exit(f'No .raw files in {target_dir}')

    if img_id is not None:
        matches = [(p, iid) for p, iid in available if iid == img_id]
        if not matches:
            sys.exit(f'img_id={img_id} not found in {target_dir}')
        clean_path, iid = matches[0]
    else:
        clean_path, iid = available[0]

    art_path  = baseline_dir / f'training_body_metalart_img{iid}_512x512x1.raw'
    i_clean_n = normalize_ct(load_raw(clean_path))

    if art_path.exists():
        i_art = load_raw(art_path)
        m_metal_n = (np.abs(i_art - load_raw(clean_path)) > metal_threshold_hu).astype(np.float32)
    else:
        m_metal_n = np.zeros(SHAPE, dtype=np.float32)

    def t(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(a[None, None]).float().to(device)

    condition = torch.cat([t(i_clean_n), t(m_metal_n)], dim=1) if use_metal_mask else t(i_clean_n)
    return condition, i_clean_n, iid


# ── Median y computation ───────────────────────────────────────────────────────

def compute_medians(features_df, feature_cols: list[str], body: str) -> np.ndarray:
    """Compute per-feature medians from the features CSV, filtered by body."""
    if 'source' in features_df.columns:
        body_df = features_df[features_df['source'] == body]
        if body_df.empty:
            print(f'  WARNING: no rows for source={body!r} — using full-dataset medians.')
            body_df = features_df
    else:
        body_df = features_df

    medians = np.full(len(feature_cols), 0.5, dtype=np.float32)
    for i, col in enumerate(feature_cols):
        if col in body_df.columns:
            medians[i] = float(body_df[col].median())
        else:
            print(f'  WARNING: feature {col!r} not in CSV — using median=0.5')

    parts = '  '.join(f'{c[:8]}={v:.3f}' for c, v in zip(feature_cols, medians))
    print(f'Medians ({body}): {parts}')
    return medians


# ── Reference image ───────────────────────────────────────────────────────────

def generate_reference(
    ddpm: GaussianDDPM,
    condition: torch.Tensor,
    feature_cols: list[str],
    y_vec: np.ndarray,
    seed: int,
    stride: int,
    device: torch.device,
    out_path: Path,
    title: str,
    csv_label: str = 'reference',
    y_label: str = 'y',
    imgs_dir: Path | None = None,
) -> tuple[np.ndarray, dict]:
    """Generate one image for a fixed feature vector.

    Saves a 3-panel PNG: I_clean | Gen I_error | feature values.
    If imgs_dir is given, also saves the raw I_error to imgs_dir/reference/<csv_label>.png.
    Returns (gen_array, row_dict) for CSV inclusion.
    """
    y_t = torch.from_numpy(y_vec[None]).float().to(device)
    gen = ddpm_sample(ddpm, condition, y_t, stride=stride, seed=seed)

    i_clean_n = condition[0, 0].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))

    axes[0].imshow(i_clean_n, cmap='gray', vmin=-1, vmax=1)
    axes[0].set_title('I_clean (anchor)', fontsize=9, fontweight='bold')
    axes[0].axis('off')

    im = axes[1].imshow(gen, cmap='RdBu', vmin=-1, vmax=1)
    axes[1].set_title(f'Gen I_error\n({y_label})', fontsize=9, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)

    feat_text = f'{y_label}:\n\n' + '\n'.join(f'{c}: {v:.4f}' for c, v in zip(feature_cols, y_vec))
    axes[2].text(0.05, 0.5, feat_text,
                 transform=axes[2].transAxes, fontsize=9, va='center', family='monospace')
    axes[2].axis('off')
    axes[2].set_title('Feature values', fontsize=9, fontweight='bold')

    fig.suptitle(title, fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')

    if imgs_dir is not None:
        cell_path = imgs_dir / 'reference' / f'{csv_label}.png'
        cell_path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(cell_path, gen, cmap='RdBu', vmin=-1, vmax=1)
        print(f'Saved: {cell_path}')

    pos_thr, neg_thr = 0.05, -0.05
    row: dict = {
        'feature_ablated': csv_label,
        'feature_value':   None,
        **{fc: round(float(fv), 6) for fc, fv in zip(feature_cols, y_vec)},
        'gen_mean':     round(float(gen.mean()), 6),
        'gen_std':      round(float(gen.std()), 6),
        'gen_abs_mean': round(float(np.abs(gen).mean()), 6),
        'gen_max':      round(float(gen.max()), 6),
        'gen_min':      round(float(gen.min()), 6),
        'gen_pos_frac': round(float((gen > pos_thr).mean()), 6),
        'gen_neg_frac': round(float((gen < neg_thr).mean()), 6),
    }
    return gen, row


# ── Grid rendering ─────────────────────────────────────────────────────────────

def render_ablation_grid(
    ddpm: GaussianDDPM,
    condition: torch.Tensor,
    feature_cols: list[str],
    ablate_features: list[str],
    median_y: np.ndarray,
    n_steps: int,
    stride: int,
    seed: int,
    device: torch.device,
    out_path: Path,
    title: str,
    imgs_dir: Path | None = None,
) -> list[dict]:
    """Render and save the F × N_steps ablation grid.

    Returns a list of result dicts (one per generated cell) for CSV export.
    """
    col_vals = np.linspace(0.0, 1.0, n_steps + 1)[1:]   # 0.1 … 1.0
    n_feats  = len(ablate_features)
    total    = n_feats * n_steps
    rows: list[dict] = []

    fig, axes = plt.subplots(
        n_feats, n_steps,
        figsize=(n_steps * 1.85, n_feats * 1.9 + 0.6),
        squeeze=False,
    )

    done = 0
    for fi, feat_name in enumerate(ablate_features):
        label = _FEATURE_LABELS.get(feat_name, feat_name)

        if feat_name not in feature_cols:
            for ci in range(n_steps):
                ax = axes[fi, ci]
                ax.axis('off')
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='gray')
            axes[fi, 0].set_ylabel(label, fontsize=9, rotation=90,
                                   va='center', labelpad=4)
            print(f'  [{fi+1}/{n_feats}] {feat_name}: not in model features — skipped')
            continue

        feat_idx = feature_cols.index(feat_name)
        print(f'  [{fi+1}/{n_feats}] {feat_name}  (idx={feat_idx})', flush=True)

        for ci, val in enumerate(col_vals):
            y_vec           = median_y.copy()
            y_vec[feat_idx] = float(val)
            y_t = torch.from_numpy(y_vec[None]).float().to(device)

            gen = ddpm_sample(ddpm, condition, y_t, stride=stride, seed=seed)

            ax = axes[fi, ci]
            ax.imshow(gen, cmap='RdBu', vmin=-1, vmax=1, interpolation='nearest')
            ax.axis('off')

            if fi == 0:
                ax.set_title(f'{val:.1f}', fontsize=8)

            # ── Save individual image ──────────────────────────────────────────
            if imgs_dir is not None:
                cell_path = imgs_dir / feat_name / f'{val:.3f}.png'
                cell_path.parent.mkdir(parents=True, exist_ok=True)
                plt.imsave(cell_path, gen, cmap='RdBu', vmin=-1, vmax=1)

            # ── Collect stats for CSV ──────────────────────────────────────────
            pos_thr, neg_thr = 0.05, -0.05
            row: dict = {
                'feature_ablated': feat_name,
                'feature_value':   round(float(val), 6),
            }
            # all feature values in this cell
            for fc, fv in zip(feature_cols, y_vec):
                row[fc] = round(float(fv), 6)
            # generated output statistics
            row['gen_mean']     = round(float(gen.mean()), 6)
            row['gen_std']      = round(float(gen.std()), 6)
            row['gen_abs_mean'] = round(float(np.abs(gen).mean()), 6)
            row['gen_max']      = round(float(gen.max()), 6)
            row['gen_min']      = round(float(gen.min()), 6)
            row['gen_pos_frac'] = round(float((gen > pos_thr).mean()), 6)
            row['gen_neg_frac'] = round(float((gen < neg_thr).mean()), 6)
            rows.append(row)

            done += 1
            print(f'\r    {done}/{total}', end='', flush=True)

        print()
        axes[fi, 0].set_ylabel(label, fontsize=9, rotation=90, va='center', labelpad=4)

    fig.suptitle(title, fontsize=9, y=1.003)
    plt.tight_layout(rect=[0.05, 0, 1, 1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {out_path}')
    return rows


# ── CSV export ─────────────────────────────────────────────────────────────────

def save_ablation_csv(rows: list[dict], meta: dict, csv_path: Path) -> None:
    """Write ablation results to CSV.

    Each row = one generated cell. Run-level metadata (model, anchor, seed…)
    is prepended to every row so the file is self-contained.
    """
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Column order: metadata first, then feature values, then stats
    meta_keys  = list(meta.keys())
    cell_keys  = [k for k in rows[0] if k not in meta_keys]
    fieldnames = meta_keys + cell_keys

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**meta, **row})

    print(f'Saved: {csv_path}  ({len(rows)} rows)')


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Feature sensitivity ablation for Gaussian Vicinal DDPM.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--model', type=Path, default=None,
        help='Path to .pth model file (overrides --model-dir).')
    parser.add_argument('--model-dir', type=Path, default=None,
        help='Directory with gaussian_unet_ema.pth / gaussian_unet.pth.\n'
             'Default: results/models/gaussian/ (from config).')
    parser.add_argument('--features', nargs='+', default=None,
        help='Feature names to ablate (default: all from model config).\n'
             'Example: --features peak_amplitude angular_concentration')
    parser.add_argument('--body', default='body1',
        help='Body variant for anchor image and median computation (default: body1).')
    parser.add_argument('--img-id', type=int, default=None,
        help='Specific img_id to use as anchor (default: first available).')
    parser.add_argument('--features-csv', type=Path, default=None,
        help='Path to features_norm.csv for median computation.\n'
             'Default: results/features_norm.csv (from config).')
    parser.add_argument('--data-path', type=str, default=None,
        help='Path to RPI directory (default: config paths.rpi_path).')
    parser.add_argument('--config', type=Path, default=None,
        help='Path to config.toml (default: <repo-root>/config.toml).')
    parser.add_argument('--out', type=Path, default=None,
        help='Output PNG path.\n'
             'Default: results/ablation/ablation_{body}_img{id}.png')
    parser.add_argument('--n-steps', type=int, default=10,
        help='Steps per feature sweep (default: 10 → values 0.1, 0.2 … 1.0).')
    parser.add_argument('--stride', type=int, default=1,
        help='DDPM sampling stride (1=full quality, 10≈10x faster; default: 1).')
    parser.add_argument('--seed', type=int, default=42,
        help='Fixed noise seed applied to every cell (default: 42).')
    parser.add_argument('--cpu', action='store_true',
        help='Force CPU (for debugging without GPU).')

    args = parser.parse_args()

    config_path = args.config or (_ROOT / 'config.toml')
    cfg    = load_config(config_path)
    device = torch.device('cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Device: {device}')

    gaus_cfg = cfg.get('gaussian', {})
    metal_thr = gaus_cfg.get('metal_threshold_hu',
                cfg.get('data', {}).get('metal_threshold_hu', 2500.0))

    paths_cfg   = cfg.get('paths', {})
    results_dir = Path(paths_cfg.get('results_dir', str(_ROOT / 'results')))
    rpi_base    = Path(args.data_path or paths_cfg.get('rpi_path',
                       str(_ROOT / 'data' / 'raw' / 'RPI')))

    # ── Model ──────────────────────────────────────────────────────────────────
    model_path            = resolve_model_path(args.model, args.model_dir, cfg)
    ddpm, feature_cols, in_ch = load_model(model_path, cfg, device)
    use_metal_mask        = (in_ch == 3)
    ablate_features       = args.features if args.features else feature_cols

    # ── Features CSV → medians ─────────────────────────────────────────────────
    features_csv = args.features_csv or (results_dir / 'features_norm.csv')
    if not features_csv.exists():
        sys.exit(f'Features CSV not found: {features_csv}\nProvide --features-csv.')
    features_df = load_features_df(features_csv)
    median_y    = compute_medians(features_df, feature_cols, args.body)

    # ── Anchor image ───────────────────────────────────────────────────────────
    body_dir = rpi_base / args.body
    if not body_dir.exists():
        sys.exit(f'Body directory not found: {body_dir}')
    condition, _, iid = pick_anchor(body_dir, args.img_id, metal_thr, device,
                                    use_metal_mask=use_metal_mask)
    print(f'Anchor: {args.body}/img{iid}')

    # ── Output path ────────────────────────────────────────────────────────────
    out_path = args.out or (results_dir / 'ablation' / f'ablation_{args.body}_img{iid}.png')
    csv_path = out_path.with_suffix('.csv')
    imgs_dir = out_path.parent / (out_path.stem + '_imgs')

    model_tag = model_path.stem
    feat_tag  = ','.join(f[:4] for f in ablate_features)
    title = (f'Feature sensitivity — {model_tag}  |  anchor: {args.body}/img{iid}'
             f'  |  seed={args.seed}  stride={args.stride}\n'
             f'features: {feat_tag}  |  other features frozen at body median')

    # Run-level metadata written into every CSV row
    meta = {
        'model':        model_tag,
        'anchor_body':  args.body,
        'anchor_img_id': iid,
        'seed':         args.seed,
        'stride':       args.stride,
        'n_steps':      args.n_steps,
        'feature_cols': ','.join(feature_cols),
    }

    ref_path = out_path.with_name(out_path.stem + '_reference.png')
    print('\nGenerating reference image (all features at median)...')
    _, ref_row = generate_reference(
        ddpm=ddpm,
        condition=condition,
        feature_cols=feature_cols,
        y_vec=median_y,
        seed=args.seed,
        stride=args.stride,
        device=device,
        out_path=ref_path,
        title=f'Reference (median) — {model_tag}  |  anchor: {args.body}/img{iid}',
        csv_label='reference_median',
        y_label='median',
        imgs_dir=imgs_dir,
    )

    # Actual feature values for this specific image from the CSV
    body_feats = features_df[features_df['source'] == args.body]
    extra_rows: list[dict] = []
    if iid in body_feats.index:
        actual_y = body_feats.loc[iid, feature_cols].values.astype(np.float32)
        actual_path = out_path.with_name(out_path.stem + '_actual.png')
        print('Generating reference image (actual features for this image)...')
        _, actual_row = generate_reference(
            ddpm=ddpm,
            condition=condition,
            feature_cols=feature_cols,
            y_vec=actual_y,
            seed=args.seed,
            stride=args.stride,
            device=device,
            out_path=actual_path,
            title=f'Reference (actual) — {model_tag}  |  anchor: {args.body}/img{iid}',
            csv_label='reference_actual',
            y_label='actual',
            imgs_dir=imgs_dir,
        )
        extra_rows.append(actual_row)
    else:
        print(f'  WARNING: img_id={iid} not in features CSV — skipping actual reference.')

    print(f'\nGenerating {len(ablate_features)}×{args.n_steps} grid '
          f'({len(ablate_features) * args.n_steps} samples)...\n')

    rows = render_ablation_grid(
        ddpm=ddpm,
        condition=condition,
        feature_cols=feature_cols,
        ablate_features=ablate_features,
        median_y=median_y,
        n_steps=args.n_steps,
        stride=args.stride,
        seed=args.seed,
        device=device,
        out_path=out_path,
        title=title,
        imgs_dir=imgs_dir,
    )
    save_ablation_csv([ref_row] + extra_rows + rows, meta, csv_path)


if __name__ == '__main__':
    main()
