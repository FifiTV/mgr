"""
Compute global I_error scale for normalize_error.

Scans training pairs (I_metal - I_clean) across all body* subdirectories,
collects per-image P99.5 of |I_error| non-zero pixels, then reports the
global statistics (median, mean, various percentiles).

The median of per-image P99.5 values is recommended as GLOBAL_SCALE —
it is robust to outlier images and represents a typical strong artifact.

Usage:
    python compute_error_scale.py data/RPI
    python compute_error_scale.py data/RPI --bodies body1 body2 body3
    python compute_error_scale.py data/RPI --out results/error_scale.json

Output (stdout):
    Per-body summary and final global statistics table.

Output (JSON, if --out provided):
    {
        "global_scale_p50":   <median of per-image P99.5>,   # recommended
        "global_scale_p75":   <75th pct of per-image P99.5>, # conservative
        "global_scale_p25":   <25th pct of per-image P99.5>, # aggressive
        "raw_clip":           3000.0,
        "n_images":           <total pairs processed>,
        "per_image_p99_5":    [<list of per-image P99.5 values>]
    }
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

SHAPE    = (512, 512)
CLIP_HU  = 3000.0


def load_raw(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(SHAPE)


def img_id(p: Path):
    m = re.search(r'img(\d+)', p.stem)
    return int(m.group(1)) if m else None


def process_body(body_dir: Path) -> list[float]:
    """Return list of per-image P99.5 of |I_error| non-zero values."""
    baseline = body_dir / 'Baseline'
    target   = body_dir / 'Target'

    if not baseline.exists() or not target.exists():
        logger.warning(f"  Skipping {body_dir.name}: missing Baseline/ or Target/")
        return []

    per_image = []
    art_files = sorted(baseline.glob('*.raw'))

    for art_path in art_files:
        iid = img_id(art_path)
        if iid is None:
            continue

        clean_path = target / f'training_body_nometal_img{iid}_512x512x1.raw'
        if not clean_path.exists():
            continue

        i_art   = load_raw(art_path)
        i_clean = load_raw(clean_path)
        i_error = np.clip(i_art - i_clean, -CLIP_HU, CLIP_HU)

        nonzero = np.abs(i_error[i_error != 0])
        if len(nonzero) == 0:
            continue

        p99_5 = float(np.percentile(nonzero, 99.5))
        per_image.append(p99_5)

    logger.info(f"  {body_dir.name}: {len(per_image)} pairs, "
                f"P99.5 median={np.median(per_image):.1f} HU  "
                f"(range {np.min(per_image):.1f}–{np.max(per_image):.1f})")
    return per_image


def main():
    parser = argparse.ArgumentParser(description="Compute global I_error scale.")
    parser.add_argument('rpi_dir', type=str,
                        help='Base RPI directory containing body1/, body2/, etc.')
    parser.add_argument('--bodies', nargs='*', default=None,
                        help='Specific body subdirs to include (default: all body* dirs).')
    parser.add_argument('--out', type=str, default=None,
                        help='Optional path for JSON output (e.g. results/error_scale.json).')
    args = parser.parse_args()

    rpi_dir = Path(args.rpi_dir)
    if not rpi_dir.exists():
        logger.error(f"Directory not found: {rpi_dir}")
        sys.exit(1)

    if args.bodies:
        body_dirs = [rpi_dir / b for b in args.bodies]
    else:
        body_dirs = sorted(rpi_dir.glob('body*'))

    if not body_dirs:
        logger.error(f"No body* subdirectories found in {rpi_dir}")
        sys.exit(1)

    logger.info(f"Processing {len(body_dirs)} body dir(s) in {rpi_dir}")

    all_p99_5: list[float] = []
    for body_dir in body_dirs:
        all_p99_5.extend(process_body(body_dir))

    if not all_p99_5:
        logger.error("No valid image pairs found. Check directory structure.")
        sys.exit(1)

    arr = np.array(all_p99_5)

    p25  = float(np.percentile(arr, 25))
    p50  = float(np.percentile(arr, 50))   # median — recommended GLOBAL_SCALE
    p75  = float(np.percentile(arr, 75))
    p90  = float(np.percentile(arr, 90))
    mean = float(arr.mean())
    std  = float(arr.std())

    print()
    print("=" * 55)
    print("  Global I_error scale statistics")
    print("=" * 55)
    print(f"  Images processed : {len(arr)}")
    print(f"  Clip applied     : ±{CLIP_HU:.0f} HU")
    print()
    print(f"  P25  of per-img P99.5 : {p25:8.1f} HU  (aggressive — clips more artefacts)")
    print(f"  P50  of per-img P99.5 : {p50:8.1f} HU  << RECOMMENDED GLOBAL_SCALE")
    print(f"  P75  of per-img P99.5 : {p75:8.1f} HU  (conservative)")
    print(f"  P90  of per-img P99.5 : {p90:8.1f} HU")
    print(f"  Mean ± std            : {mean:8.1f} ± {std:.1f} HU")
    print("=" * 55)
    print()
    print("  Add to config.toml [gaussian]:")
    print(f'    error_scale = {p50:.1f}  # global P99.5 median over training set')
    print()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "global_scale_p50":  p50,
            "global_scale_p75":  p75,
            "global_scale_p25":  p25,
            "global_scale_p90":  p90,
            "mean":              mean,
            "std":               std,
            "raw_clip":          CLIP_HU,
            "n_images":          len(arr),
            "per_image_p99_5":   arr.tolist(),
        }
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results saved to {out_path}")


if __name__ == '__main__':
    main()
