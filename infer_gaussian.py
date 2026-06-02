"""
infer_gaussian.py -- Gaussian Vicinal DDPM inference and visualisation.

Generates I_error for selected samples and saves PNG panels:
  I_clean | M_metal | Gen I_error | Gen CT | Overlay | [Real I_error | Real CT]

Examples:

  # 5 images from body1, default paths
  python infer_gaussian.py --body body1 --n 5

  # Multiple bodies, more images, custom output
  python infer_gaussian.py --body body1 body2 --n 10 --out results/gaussian_vis

  # Faster sampling (every 10 steps instead of 1000)
  python infer_gaussian.py --body body1 --n 5 --stride 10

  # Custom model and features
  python infer_gaussian.py --body body1 --n 5 \
      --model results/models/gaussian/gaussian_unet_ema.pth \
      --features results/features_norm.csv

  # Fixed y_target (6 values in [0,1] comma-separated)
  # peak_amplitude=0.8, rest=0.5 -> strong artifact
  python infer_gaussian.py --body body1 --n 5 --y-target 0.8,0.5,0.5,0.5,0.5,0.5

  # CPU
  python infer_gaussian.py --body body1 --n 3 --cpu
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.datasets.gaussian_dataset import load_features_df, normalize_ct, normalize_error
from infer_utils import SHAPE, load_config, load_raw, load_model, ddpm_sample, GaussianDDPM


# ── Raw output saving ─────────────────────────────────────────────────────────

def save_raw_outputs(
    out_dir: Path,
    stem: str,
    i_error_gen: np.ndarray,
    i_metal_gen_hu: np.ndarray,
    i_clean_raw: np.ndarray,
    i_art_raw: np.ndarray | None,
) -> None:
    """Save generated maps as float32 .raw files (512x512 each).

    All CT images normalized via global clip [-1000, 3000] -> [-1, 1] (same as
    infer.py and gaussian_dataset.normalize_ct), so files from all models are
    directly comparable.

    Layout:
      raw/           — model outputs
        stem_gen_error.raw   error map  (already [-1,1] from model)
        stem_gen_ct.raw      reconstructed artifact CT, normalized [-1,1]
      raw/img/       — inputs / references (global HU -> [-1,1])
        stem_clean.raw
        stem_real_art.raw
    """
    raw_dir = out_dir / 'raw'
    img_dir = raw_dir / 'img'
    img_dir.mkdir(parents=True, exist_ok=True)

    (raw_dir / f'{stem}_gen_error.raw').write_bytes(i_error_gen.astype(np.float32).tobytes())
    (raw_dir / f'{stem}_gen_ct.raw').write_bytes(normalize_ct(i_metal_gen_hu).tobytes())
    (img_dir / f'{stem}_clean.raw').write_bytes(normalize_ct(i_clean_raw).tobytes())
    if i_art_raw is not None:
        (img_dir / f'{stem}_real_art.raw').write_bytes(normalize_ct(i_art_raw).tobytes())


def save_metadata(
    out_dir: Path,
    stem: str,
    img_id: int,
    body_name: str,
    y_vec: np.ndarray,
    feature_cols: list[str],
    error_scale: float,
    seed: int,
    metalinfo: dict | None,
) -> None:
    """Save inference metadata alongside the raw model output."""
    raw_dir = out_dir / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    y_target = [round(float(v), 6) for v in y_vec]

    # peak_amplitude maps to HU via error_scale; other features are dimensionless
    y_target_hu: dict[str, float] = {}
    for col, val in zip(feature_cols, y_vec):
        if col == 'peak_amplitude':
            y_target_hu[col] = round(float(val) * error_scale, 2)
        else:
            y_target_hu[col] = round(float(val), 6)

    meta: dict = {
        'img_id':          img_id,
        'source':          'generated',
        'body':            body_name,
        'metal_source':    f'{body_name}/img{img_id}',
        'seed':            seed,
        'error_scale':     error_scale,
        'y_target':        y_target,
        'y_target_named':  {c: round(float(v), 6) for c, v in zip(feature_cols, y_vec)},
        'y_target_hu':     y_target_hu,
    }
    if metalinfo is not None:
        meta['metalinfo'] = metalinfo

    path = raw_dir / f'{stem}.json'
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'  -> {path}')


# ── Visualisation ──────────────────────────────────────────────────────────────

def save_figure(sample_panels: list[tuple[np.ndarray, str, str, tuple]],
                title: str,
                out_path: Path) -> None:
    """
    Save a row of panels. Each entry: (image, label, cmap, (vmin, vmax)).
    """
    n = len(sample_panels)
    fig, axes = plt.subplots(1, n, figsize=(n * 4, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (img, label, cmap, (vmin, vmax)) in zip(axes, sample_panels):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=8, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(title, fontsize=9, fontweight='bold')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out_path}")


# ── Per-sample processing ──────────────────────────────────────────────────────

def process_sample(img_id: int,
                   body_name: str,
                   clean_path: Path,
                   art_path: Path | None,
                   metal_path: Path | None,
                   metalinfo: dict | None,
                   y_vec: np.ndarray,
                   feature_cols: list[str],
                   ddpm: GaussianDDPM,
                   error_scale: float,
                   metal_threshold_hu: float,
                   stride: int,
                   device: torch.device,
                   out_dir: Path,
                   use_metal_mask: bool = True,
                   seed: int = -1) -> None:

    i_clean_raw = load_raw(clean_path)
    i_art_raw   = load_raw(art_path)   if art_path   else None
    m_metal_raw = load_raw(metal_path) if metal_path else None

    i_clean_n = normalize_ct(i_clean_raw)
    if m_metal_raw is not None:
        m_metal_n = (m_metal_raw > 0.5).astype(np.float32)
    elif i_art_raw is not None:
        m_metal_n = (np.abs(i_art_raw - i_clean_raw) > metal_threshold_hu).astype(np.float32)
    else:
        m_metal_n = np.zeros(SHAPE, dtype=np.float32)

    def t(arr): return torch.from_numpy(arr[None, None]).float().to(device)

    condition = torch.cat([t(i_clean_n), t(m_metal_n)], dim=1) if use_metal_mask else t(i_clean_n)
    y_t       = torch.from_numpy(y_vec[None]).float().to(device)

    if seed >= 0:
        torch.manual_seed(seed)
    actual_seed = int(torch.initial_seed())

    i_error_gen    = ddpm_sample(ddpm, condition, y_t, stride=stride, seed=seed if seed >= 0 else None)
    i_metal_gen_hu = i_clean_raw + i_error_gen * error_scale

    stem = f'gauss_{img_id:04d}_{body_name}_img{img_id}'
    save_raw_outputs(out_dir, stem, i_error_gen, i_metal_gen_hu, i_clean_raw, i_art_raw)
    save_metadata(out_dir, stem, img_id, body_name, y_vec, feature_cols,
                  error_scale, actual_seed, metalinfo)

    panels = []

    panels.append((i_clean_n,   'I_clean\n(input)',        'gray', (-1, 1)))
    panels.append((m_metal_n,   'M_metal\n(metal mask)',   'hot',  (0, 1)))
    panels.append((i_error_gen, 'Gen I_error\n(model output)', 'RdBu', (-1, 1)))

    panels.append((normalize_ct(i_metal_gen_hu), 'Gen CT\n(I_clean + gen artifact)', 'gray', (-1, 1)))

    overlay_base     = (i_clean_n + 1) / 2
    artifact_overlay = np.clip(i_error_gen, 0, 1)
    overlay_rgb = np.stack([
        np.clip(overlay_base + artifact_overlay * 0.6, 0, 1),
        np.clip(overlay_base - artifact_overlay * 0.3, 0, 1),
        np.clip(overlay_base - artifact_overlay * 0.3, 0, 1),
    ], axis=-1)
    panels.append((overlay_rgb, 'Overlay\n(red = gen artifact)', None, (0, 1)))

    if i_art_raw is not None:
        i_error_real   = i_art_raw - i_clean_raw
        i_error_real_n = np.clip(i_error_real, -error_scale, error_scale) / error_scale
        panels.append((i_error_real_n,         'Real I_error\n(ground truth)', 'RdBu', (-1, 1)))
        panels.append((normalize_ct(i_art_raw), 'Real CT\n(I_metal gt)',        'gray',  (-1, 1)))

    # ── Feature subtitle ───────────────────────────────────────────────────────
    feat_str  = '  '.join(f'{c[:6]}={v:.2f}' for c, v in zip(feature_cols, y_vec))
    metal_str = format_metalinfo(metalinfo)
    title = f'{body_name}/img{img_id}  |  {metal_str}\ny=[{feat_str}]'

    # ── Overlay panel needs special handling (RGB) ─────────────────────────────
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(n * 4, 4.8))
    if n == 1:
        axes = [axes]

    for ax, (img, label, cmap, (vmin, vmax)) in zip(axes, panels):
        if cmap is None:
            ax.imshow(img)           # RGB, no colormap
        else:
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax.set_title(label, fontsize=8, fontweight='bold')
        ax.axis('off')

    fig.suptitle(title, fontsize=8)
    plt.tight_layout()

    fname = out_dir / f'gauss_{img_id:04d}_{body_name}_img{img_id}.png'
    fname.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fname, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {fname}")


# ── Metal info ─────────────────────────────────────────────────────────────────

_MAT_ABBREV = {
    'stainless_steel_316L': 'SS316L',
    'Co':                   'Co',
    'Ti':                   'Ti',
    'Ti6Al4V':              'Ti6Al4V',
    'Fe':                   'Fe',
    'Au':                   'Au',
    'Pt':                   'Pt',
    'W':                    'W',
}


def load_metalinfo(path: Path | None) -> dict | None:
    """Load metal object metadata from Mask/metalinfo JSON."""
    if path is None or not path.exists():
        return None
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    names = [_MAT_ABBREV.get(m, m[:8]) for m in data.get('mat_name', [])]
    diams = [round(d, 1) for d in data.get('diameter', [])]
    return {
        'n':         data.get('n_materials', 0),
        'names':     names,
        'diameters': diams,
    }


def format_metalinfo(info: dict | None) -> str:
    if info is None:
        return 'no metalinfo'
    names_str = ', '.join(info['names'])
    diam_str  = ', '.join(f'o{d}' for d in info['diameters'])
    return f"n={info['n']} | {names_str} | {diam_str} mm"


# ── Pair discovery ─────────────────────────────────────────────────────────────

def find_pairs(body_dir: Path) -> list[dict]:
    """Return sorted list of {img_id, clean_path, art_path, metal_path, metalinfo_path}."""
    baseline = body_dir / 'Baseline'
    target   = body_dir / 'Target'
    metal    = body_dir / 'Metal'
    mask_dir = body_dir / 'Mask'

    if not target.exists():
        print(f"  WARNING: {body_dir}/Target not found — skipping.")
        return []

    def img_id(p: Path):
        m = re.search(r'img(\d+)', p.stem)
        return int(m.group(1)) if m else None

    pairs = []
    for clean_p in sorted(target.glob('*.raw')):
        iid = img_id(clean_p)
        if iid is None:
            continue

        art_p      = baseline / f'training_body_metalart_img{iid}_512x512x1.raw'
        metal_p    = metal    / f'training_body_metalonlymask_img{iid}_512x512x1.raw'
        metalinfo_p = mask_dir / f'training_body_metalinfo{iid}.json'

        pairs.append({
            'img_id':       iid,
            'clean_path':   clean_p,
            'art_path':     art_p       if art_p.exists()       else None,
            'metal_path':   metal_p     if metal_p.exists()     else None,
            'metalinfo_path': metalinfo_p if metalinfo_p.exists() else None,
        })

    return pairs


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Gaussian Vicinal DDPM — inference and visualisation',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        '--body', nargs='+', default=['body1'],
        help='Body variant name(s) (e.g. body1 body2). Default: body1'
    )
    parser.add_argument(
        '--n', type=int, default=0,
        help='Maximum number of images to process. 0 = all (default: 0)'
    )
    parser.add_argument(
        '--data-path', type=str, default=None,
        help='Path to RPI directory (overrides config paths.rpi_path)'
    )
    parser.add_argument(
        '--model', type=Path, default=None,
        help='Path to .pth model file. Default: results/models/gaussian/gaussian_unet_ema.pth'
    )
    parser.add_argument(
        '--features', type=Path, default=None,
        help='Path to normalized features CSV. Default: results/features_norm.csv'
    )
    parser.add_argument(
        '--out', type=Path, default=Path('results/gaussian_infer'),
        help='Output directory for PNGs. Default: results/gaussian_infer'
    )
    parser.add_argument(
        '--config', default='config.toml',
        help='Path to config.toml'
    )
    parser.add_argument(
        '--stride', type=int, default=1,
        help=(
            'DDPM sampling stride (default: 1 = full 1000 steps).\n'
            'stride=10 -> 100 steps, ~10x faster, slightly lower quality.\n'
            'stride=50 -> 20 steps, fast, for debugging.'
        )
    )
    parser.add_argument(
        '--y-target', type=str, default=None,
        help=(
            '6 comma-separated values in [0,1] — overrides per-image features from CSV.\n'
            'Order: peak_amplitude,spatial_extent,bbox_ratio,\n'
            '       dark_to_bright_ratio,angular_concentration,texture_roughness\n'
            'Example: --y-target 0.8,0.5,0.5,0.5,0.5,0.5  (strong artifact)'
        )
    )
    parser.add_argument(
        '--cpu', action='store_true',
        help='Force CPU even when CUDA is available'
    )
    parser.add_argument(
        '--seed', type=int, default=-1,
        help=(
            'Base random seed for reproducible sampling.\n'
            'Sample i gets seed (base + i). -1 = random (default).\n'
            'Saved in per-sample metadata JSON.'
        )
    )
    parser.add_argument(
        '--start-img', type=int, default=0,
        help=(
            'Skip images with img_id below this value (default: 0 = start from first).\n'
            'img_id numbering: body1=1-1000, body2=1001-2000, body3=2001-3000, …\n'
            'Example: --start-img 500  (skip body1 imgs 1-499)\n'
            '         --start-img 1500 (skip body2 imgs 1001-1499)'
        )
    )
    parser.add_argument(
        '--feature-cols', nargs='+', default=None, metavar='COL',
        help=(
            'Feature columns to use (overrides config and checkpoint auto-detection).\n'
            'Use for ablation models trained with a feature subset, e.g.:\n'
            '  --feature-cols peak_amplitude spatial_extent bbox_ratio'
        )
    )

    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device('cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Device: {device}')

    gaus_cfg           = cfg.get('gaussian', {})
    error_scale        = gaus_cfg.get('error_scale', 642.8)
    metal_threshold_hu = gaus_cfg.get('metal_threshold_hu',
                         cfg.get('data', {}).get('metal_threshold_hu', 2500.0))

    # ── Resolve paths ──────────────────────────────────────────────────────────
    paths_cfg = cfg.get('paths', {})
    rpi_base  = Path(args.data_path or paths_cfg.get('rpi_path', 'data/raw/RPI'))

    model_path = args.model or Path(
        paths_cfg.get('results_dir', 'results'),
        'models', 'gaussian', 'gaussian_unet_ema.pth'
    )
    # fallback to raw weights if EMA not found
    if not model_path.exists():
        raw_path = model_path.parent / 'gaussian_unet.pth'
        if raw_path.exists():
            model_path = raw_path
        else:
            sys.exit(f'Model not found: {model_path}\nProvide --model or train first.')

    features_path = args.features or Path(
        paths_cfg.get('results_dir', 'results'), 'features_norm.csv'
    )
    if not features_path.exists():
        sys.exit(f'Features file not found: {features_path}\nProvide --features.')

    # ── Load model and features ────────────────────────────────────────────────
    print(f'Loading model: {model_path}')
    ddpm, feature_cols, in_ch = load_model(model_path, cfg, device)
    use_metal_mask = (in_ch == 3)

    # --feature-cols overrides auto-detected feature_cols (use for ablation models)
    if args.feature_cols:
        if len(args.feature_cols) != len(feature_cols):
            sys.exit(f'--feature-cols has {len(args.feature_cols)} entries but checkpoint '
                     f'y_dim={len(feature_cols)}. Must match.')
        feature_cols = args.feature_cols
        print(f'  feature_cols overridden: {feature_cols}')

    print(f'Loading features: {features_path}')
    features_df = load_features_df(features_path)

    fixed_y = None
    if args.y_target:
        vals = [float(v) for v in args.y_target.split(',')]
        if len(vals) != len(feature_cols):
            sys.exit(f'--y-target requires {len(feature_cols)} values, got {len(vals)}.')
        fixed_y = np.array(vals, dtype=np.float32)
        print(f'Using fixed y_target: {dict(zip(feature_cols, fixed_y))}')

    all_samples = []
    for body_name in args.body:
        body_dir = rpi_base / body_name
        if not body_dir.exists():
            print(f'WARNING: {body_dir} does not exist — skipping.')
            continue

        body_feats = features_df[features_df['source'] == body_name]
        if body_feats.empty:
            print(f'WARNING: no features found for source={body_name!r} — skipping.')
            continue

        feat_ids = set(body_feats.index.tolist())
        pairs    = find_pairs(body_dir)
        pairs    = [p for p in pairs if p['img_id'] in feat_ids]
        if args.start_img > 0:
            pairs = [p for p in pairs if p['img_id'] >= args.start_img]

        if not pairs:
            print(f'WARNING: no matching pairs found for {body_name}.')
            continue

        # Limit per body proportionally (0 = all)
        per_body = len(pairs) if args.n == 0 else max(1, args.n // len(args.body))
        for rec in pairs[:per_body]:
            iid   = rec['img_id']
            y_vec = fixed_y if fixed_y is not None else \
                    body_feats.loc[iid, feature_cols].values.astype(np.float32)
            all_samples.append({**rec, 'body': body_name, 'y': y_vec})

    if args.n > 0:
        all_samples = all_samples[:args.n]

    if not all_samples:
        sys.exit('No samples found. Check --body and --data-path.')

    print(f'\nProcessing {len(all_samples)} samples (stride={args.stride}) -> {args.out}/\n')

    for idx, s in enumerate(all_samples):
        metalinfo = load_metalinfo(s['metalinfo_path'])
        print(f'[{idx+1}/{len(all_samples)}] {s["body"]}/img{s["img_id"]}  {format_metalinfo(metalinfo)}')
        process_sample(
            img_id=s['img_id'],
            body_name=s['body'],
            clean_path=s['clean_path'],
            art_path=s['art_path'],
            metal_path=s['metal_path'],
            metalinfo=metalinfo,
            y_vec=s['y'],
            feature_cols=feature_cols,
            ddpm=ddpm,
            error_scale=error_scale,
            metal_threshold_hu=metal_threshold_hu,
            stride=args.stride,
            device=device,
            out_dir=args.out,
            use_metal_mask=use_metal_mask,
            seed=args.seed + idx if args.seed >= 0 else -1,
        )

    print(f'\nDone. Results saved to: {args.out}/')


if __name__ == '__main__':
    main()
