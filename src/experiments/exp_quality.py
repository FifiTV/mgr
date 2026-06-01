"""
exp_quality.py — Quantitative evaluation of generated CT images.

Computes SSIM, PSNR, MAE, GMSD and optionally LPIPS (VGG16, torchvision).
All helper functions live in eval_utils.py.

Generated images come from variant directories (infer.py output).
Reference images come from the RPI dataset tree (--rpi-dir) or from the
variant's raw/ subfolder (legacy, when --rpi-dir is omitted).

Body mapping (RPI naming convention):
  img_id → body_num = (img_id - 1) // 1000 + 1
  e.g.  img7001 → body8,  img8001 → body9

Generated file layout (auto-detected):
  Layout A (infer.py):  variant/raw/{stem}_{model_type}.raw
  Layout B (flat):      variant/{stem}_{model_type}.raw

Usage:
    # Explicit model dirs + RPI ground truth
    python src/experiments/exp_quality.py \\
        --cycle-soft  results/inference/cycle_soft \\
        --cycle-hard  results/inference/cycle_hard \\
        --diff-soft   results/inference/diff_soft  \\
        --diff-hard   results/inference/diff_hard  \\
        --rpi-dir     data/raw/RPI \\
        --out         results/eval

    # Auto-discover variants + RPI
    python src/experiments/exp_quality.py \\
        --data-dir  results/inference \\
        --rpi-dir   data/raw/RPI \\
        --out       results/eval

    # Legacy: references from raw/ subfolder
    python src/experiments/exp_quality.py \\
        --data-dir  data/raw/generated_c_d \\
        --no-lpips
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.experiments.eval_utils import (
    DATA_RANGE, MODEL_SUFFIXES,
    METRIC_NAMES_BASE, METRIC_NAMES_LPIPS,
    load_raw, normalize_hu,
    body_from_img_id,
    find_eval_pairs,
    compute_metrics, aggregate_metrics,
    build_lpips,
)

# Map explicit CLI arg → model types relevant for that variant
EXPLICIT_VARIANTS = {
    "cycle_soft": ["cyclegan_ab", "cyclegan_ba"],
    "cycle_hard": ["cyclegan_ab", "cyclegan_ba"],
    "diff_soft":  ["diffusion",   "cyclegan_ab"],
    "diff_hard":  ["diffusion",   "cyclegan_ab"],
}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated CT images: SSIM, PSNR, MAE, GMSD, LPIPS",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    grp_models = parser.add_argument_group(
        "Explicit model directories (optional, override --data-dir)"
    )
    grp_models.add_argument("--cycle-soft", type=Path, default=None,
        help="Directory with CycleGAN SOFT outputs")
    grp_models.add_argument("--cycle-hard", type=Path, default=None,
        help="Directory with CycleGAN HARD outputs")
    grp_models.add_argument("--diff-soft",  type=Path, default=None,
        help="Directory with Diffusion SOFT outputs")
    grp_models.add_argument("--diff-hard",  type=Path, default=None,
        help="Directory with Diffusion HARD outputs")

    grp_data = parser.add_argument_group("Data sources")
    grp_data.add_argument("--data-dir", type=Path, default=None,
        help="Root directory with variant subfolders (auto-discover).")
    grp_data.add_argument("--variants", nargs="+", default=None,
        help="Variant subfolder names within --data-dir.\nDefault: all subdirs.")
    grp_data.add_argument("--rpi-dir", type=Path, default=None,
        help="RPI data root (e.g. data/raw/RPI).\n"
             "When set, references are loaded from RPI and normalized per-image\n"
             "(identical to infer.py). Omit to use variant/raw/ subfolder.")

    grp_eval = parser.add_argument_group("Evaluation options")
    grp_eval.add_argument("--model-types", nargs="+",
        default=["cyclegan_ab", "diffusion"],
        choices=list(MODEL_SUFFIXES.keys()),
        help="Output types to evaluate (default: cyclegan_ab diffusion).")
    grp_eval.add_argument("--out", type=Path, default=Path("results/eval"),
        help="Output directory for CSV results (default: results/eval)")
    grp_eval.add_argument("--no-lpips", action="store_true",
        help="Skip LPIPS (no internet/GPU needed).")
    grp_eval.add_argument("--cpu", action="store_true",
        help="Force CPU for LPIPS.")

    args = parser.parse_args()

    # Build task list: (variant_dir, label, model_types)
    explicit = {
        "cycle_soft": args.cycle_soft,
        "cycle_hard": args.cycle_hard,
        "diff_soft":  args.diff_soft,
        "diff_hard":  args.diff_hard,
    }
    tasks: list[tuple[Path, str, list[str]]] = []

    for label, path in explicit.items():
        if path is not None:
            if not path.exists():
                print(f"WARNING: {label} dir not found: {path} — skipping")
                continue
            relevant = [t for t in EXPLICIT_VARIANTS[label] if t in args.model_types]
            tasks.append((path, label, relevant or args.model_types))

    if not tasks:
        if args.data_dir is None:
            parser.error("Provide --data-dir or at least one of "
                         "--cycle-soft/--cycle-hard/--diff-soft/--diff-hard.")
        if not args.data_dir.exists():
            sys.exit(f"ERROR: --data-dir not found: {args.data_dir}")
        dirs = ([args.data_dir / v for v in args.variants]
                if args.variants
                else sorted(p for p in args.data_dir.iterdir() if p.is_dir()))
        if not dirs:
            sys.exit(f"No subdirectories found in {args.data_dir}")
        for vd in dirs:
            tasks.append((vd, vd.name, args.model_types))

    # LPIPS setup
    lpips_fn, metric_names = None, METRIC_NAMES_BASE
    if not args.no_lpips:
        try:
            import torch
            dev = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            dev = "cpu"
        lpips_fn     = build_lpips(dev)
        metric_names = METRIC_NAMES_LPIPS if lpips_fn else METRIC_NAMES_BASE

    rpi_dir = args.rpi_dir
    if rpi_dir and not rpi_dir.exists():
        sys.exit(f"ERROR: --rpi-dir not found: {rpi_dir}")

    args.out.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    agg_rows: list[dict] = []

    for variant_dir, label, model_types in tasks:
        ref_src = f"RPI ({rpi_dir})" if rpi_dir else "variant/raw/"
        print(f"\n{'='*65}")
        print(f"  Variant  : {label}")
        print(f"  Dir      : {variant_dir}")
        print(f"  Models   : {model_types}")
        print(f"  Refs from: {ref_src}")
        print(f"{'='*65}")

        pairs = find_eval_pairs(variant_dir, model_types, rpi_dir)
        if not pairs:
            print("  No pairs found — skipping.")
            continue

        per_model: dict[str, list[dict]] = {}

        for p in pairs:
            gen = load_raw(p["generated"])
            ref = load_raw(p["reference"])
            if p["ref_is_rpi"]:
                ref = normalize_hu(ref)

            mets = compute_metrics(gen, ref, lpips_fn)
            body = body_from_img_id(p["img_id"]) if rpi_dir else ""
            row  = {
                "variant":    label,
                "model_type": p["model_type"],
                "img_id":     p["img_id"],
                "body":       body,
                **mets,
            }
            all_rows.append(row)
            per_model.setdefault(p["model_type"], []).append(mets)

            lpips_str = f"  LPIPS={mets['lpips']:.4f}" if "lpips" in mets else ""
            print(f"  img{p['img_id']:>6}  {p['model_type']:<14}"
                  f"  SSIM={mets['ssim']:.4f}"
                  f"  PSNR={mets['psnr']:6.2f}dB"
                  f"  MAE={mets['mae']:.4f}"
                  f"  GMSD={mets['gmsd']:.4f}"
                  + lpips_str)

        for mtype, score_list in per_model.items():
            agg = aggregate_metrics(score_list, metric_names)
            n   = len(score_list)
            agg_rows.append({"variant": label, "model_type": mtype, "n": n, **agg})
            print(f"\n  --- {mtype}  n={n} ---")
            for k in metric_names:
                if f"{k}_mean" in agg:
                    print(f"    {k.upper():<6}  "
                          f"mean={agg[f'{k}_mean']:.4f}  "
                          f"std={agg[f'{k}_std']:.4f}  "
                          f"median={agg[f'{k}_median']:.4f}")

    if not all_rows:
        sys.exit("No results computed. Check paths and --model-types.")

    # Per-sample CSV
    sample_fields = ["variant", "model_type", "img_id", "body"] + metric_names
    per_sample_path = args.out / "eval_per_sample.csv"
    with open(per_sample_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sample_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved: {per_sample_path}")

    # Aggregated CSV
    agg_fields = ["variant", "model_type", "n"]
    for k in metric_names:
        agg_fields += [f"{k}_mean", f"{k}_std", f"{k}_median"]
    agg_path = args.out / "eval_aggregated.csv"
    with open(agg_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(agg_rows)
    print(f"Saved: {agg_path}")

    # Summary table
    has_lpips = lpips_fn is not None
    lpips_hdr = "  LPIPS(lo)" if has_lpips else ""
    print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
    print(f"{'Variant':<16} {'Type':<14} {'N':>4}  "
          f"{'SSIM(hi)':>9}  {'PSNR(hi)':>9}  {'MAE(lo)':>8}  {'GMSD(lo)':>9}"
          + lpips_hdr)
    print("-" * 80)
    for row in agg_rows:
        lpips_col = f"  {row.get('lpips_mean', float('nan')):>9.4f}" if has_lpips else ""
        print(f"{row['variant']:<16} {row['model_type']:<14} {row['n']:>4}  "
              f"{row.get('ssim_mean', float('nan')):>9.4f}  "
              f"{row.get('psnr_mean', float('nan')):>9.2f}  "
              f"{row.get('mae_mean',  float('nan')):>8.4f}  "
              f"{row.get('gmsd_mean', float('nan')):>9.4f}"
              + lpips_col)
    print(f"{'='*80}")
    print("(hi) = higher is better   (lo) = lower is better")


if __name__ == "__main__":
    main()
