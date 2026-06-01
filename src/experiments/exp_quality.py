"""
eval_generated.py — Evaluate generated CT images vs ground truth.

Computes SSIM, PSNR, MAE, GMSD and optionally LPIPS (VGG16-based, no extra
packages needed — uses torchvision which is already a project dependency).

All raw files are float32 binary, 512×512, values in [-1, 1] (normalized HU).
This matches exactly how infer.py / save_infer_raw() writes them.

Two directory layouts are auto-detected:

  Layout A — current infer.py output:
    variant_dir/
      raw/
        infer_NNNN_stem_cyclegan_ab.raw     ← generated
        infer_NNNN_stem_diffusion.raw       ← generated
        img/
          infer_NNNN_stem_real_art.raw      ← ground truth
          infer_NNNN_stem_clean.raw         ← clean input

  Layout B — flat reference data (generated_c_d):
    variant_dir/
      infer_NNNN_stem_cyclegan_ab.raw       ← generated
      raw/
        infer_NNNN_stem_real_art.raw        ← ground truth
        infer_NNNN_stem_clean.raw           ← clean input

Usage:
    python eval_generated.py --data-dir data/raw/generated_c_d
    python eval_generated.py --data-dir results/inference --out results/eval
    python eval_generated.py \\
        --data-dir results/inference \\
        --variants cycle_soft cycle_hard diff_soft diff_hard \\
        --model-types cyclegan_ab diffusion \\
        --out results/eval \\
        --no-lpips
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


SHAPE      = (512, 512)
DATA_RANGE = 2.0    # images normalized to [-1, 1]

# generated suffix → reference suffix to compare against
MODEL_SUFFIXES = {
    "cyclegan_ab": "real_art",   # artifact generation  → vs ground-truth artifact
    "diffusion":   "real_art",   # artifact generation  → vs ground-truth artifact
    "cyclegan_ba": "clean",      # artifact removal     → vs clean CT
}


# ---------------------------------------------------------------------------
# I/O — identical to infer.py and src/experiments/utils.py
# ---------------------------------------------------------------------------

def load_raw(path: Path, shape: tuple = SHAPE) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(shape)


# ---------------------------------------------------------------------------
# SSIM  (scipy only — no skimage needed)
# ---------------------------------------------------------------------------

def ssim(img1: np.ndarray, img2: np.ndarray,
         data_range: float = DATA_RANGE, sigma: float = 1.5) -> float:
    """
    SSIM with Gaussian window (sigma=1.5).

    Numerically equivalent to skimage.metrics.structural_similarity(
        gaussian_weights=True, sigma=1.5, data_range=data_range).
    Implemented via scipy.ndimage.gaussian_filter.
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    a  = img1.astype(np.float64)
    b  = img2.astype(np.float64)

    mu_a  = gaussian_filter(a,   sigma=sigma)
    mu_b  = gaussian_filter(b,   sigma=sigma)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sig_a2 = gaussian_filter(a * a, sigma=sigma) - mu_a2
    sig_b2 = gaussian_filter(b * b, sigma=sigma) - mu_b2
    sig_ab = gaussian_filter(a * b, sigma=sigma) - mu_ab

    num = (2.0 * mu_ab + C1) * (2.0 * sig_ab + C2)
    den = (mu_a2 + mu_b2 + C1) * (sig_a2 + sig_b2 + C2)
    return float(np.mean(num / den))


# ---------------------------------------------------------------------------
# PSNR / MAE / GMSD
# ---------------------------------------------------------------------------

def psnr(img1: np.ndarray, img2: np.ndarray,
         data_range: float = DATA_RANGE) -> float:
    """Peak Signal-to-Noise Ratio [dB]. Higher = better."""
    mse = float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * np.log10(data_range ** 2 / mse)


def mae(img1: np.ndarray, img2: np.ndarray) -> float:
    return float(np.mean(np.abs(img1.astype(np.float64) - img2.astype(np.float64))))


def gmsd(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Gradient Magnitude Similarity Deviation. Lower = more similar.

    The stability constant c is derived from the original paper formula
    c = 0.0026 * T^2, where T = data_range = 2.0 (images in [-1,1]).
    This gives c = 0.0026 * 4 = 0.0104, which is the correct scale for
    normalized images (original paper used c≈170 for 8-bit [0,255]).
    """
    c = 0.0026 * (DATA_RANGE ** 2)   # ≈ 0.0104 for [-1,1]

    def gm(img: np.ndarray) -> np.ndarray:
        gx = np.abs(np.diff(img, axis=1, prepend=img[:, :1]))
        gy = np.abs(np.diff(img, axis=0, prepend=img[:1, :]))
        return np.sqrt(gx ** 2 + gy ** 2)

    g1, g2  = gm(img1), gm(img2)
    gms_map = (2.0 * g1 * g2 + c) / (g1 ** 2 + g2 ** 2 + c)
    return float(gms_map.std())


# ---------------------------------------------------------------------------
# LPIPS  (VGG16 features from torchvision — no extra package needed)
# ---------------------------------------------------------------------------

class LPIPS_VGG:
    """
    Simplified LPIPS using pretrained VGG16 features.

    Computes normalised L2 distance between multi-scale VGG16 activations,
    which correlates strongly with perceptual similarity.

    Input images: numpy [H, W] float32 in [-1, 1].
    Preprocessing: expand to 3-channel, rescale to [0,1], apply ImageNet stats.

    Note: The official lpips package adds a learned linear layer per feature
    map; this implementation omits that layer and uses equal weighting across
    all 4 feature levels. Results are not numerically identical to lpips.alex
    or lpips.vgg but capture the same perceptual information.
    """

    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD  = [0.229, 0.224, 0.225]
    # VGG16 features: relu1_2, relu2_2, relu3_3, relu4_3
    _LAYER_ENDS = [4, 9, 16, 23]

    def __init__(self, device="cpu"):
        import torch
        import torchvision.models as tvm

        self.device = torch.device(device)

        vgg_feat = tvm.vgg16(
            weights=tvm.VGG16_Weights.IMAGENET1K_V1
        ).features.to(self.device).eval()

        for p in vgg_feat.parameters():
            p.requires_grad = False

        # Split into 4 sequential blocks
        children = list(vgg_feat.children())
        import torch.nn as nn
        prev = 0
        self.blocks = []
        for end in self._LAYER_ENDS:
            block = nn.Sequential(*children[prev:end]).to(self.device).eval()
            for p in block.parameters():
                p.requires_grad = False
            self.blocks.append(block)
            prev = end

        mean = torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1).to(self.device)
        std  = torch.tensor(self._IMAGENET_STD ).view(1, 3, 1, 1).to(self.device)
        self.register = (mean, std)

    def _preprocess(self, img_np: np.ndarray):
        import torch
        mean, std = self.register
        t = torch.from_numpy(img_np).float().to(self.device)
        t = (t + 1.0) / 2.0                        # [-1, 1] → [0, 1]
        t = t.unsqueeze(0).expand(3, -1, -1)        # [H,W] → [3,H,W]
        t = t.unsqueeze(0)                          # → [1,3,H,W]
        return (t - mean) / std

    @staticmethod
    def _normalize(feat):
        """Channel-wise L2 normalisation of a feature map [1,C,H,W]."""
        norm = feat.pow(2).sum(dim=1, keepdim=True).sqrt().clamp(min=1e-8)
        return feat / norm

    def __call__(self, img1_np: np.ndarray, img2_np: np.ndarray) -> float:
        import torch
        x = self._preprocess(img1_np)
        y = self._preprocess(img2_np)

        total = 0.0
        with torch.no_grad():
            for block in self.blocks:
                x = block(x)
                y = block(y)
                xn = self._normalize(x)
                yn = self._normalize(y)
                total += float((xn - yn).pow(2).mean())

        return total / len(self.blocks)   # average over 4 levels → scale ~[0,1]


def _try_build_lpips(device: str) -> "LPIPS_VGG | None":
    """Returns LPIPS_VGG instance or None if torchvision unavailable."""
    try:
        inst = LPIPS_VGG(device=device)
        print(f"  LPIPS: VGG16 loaded on {device}")
        return inst
    except Exception as e:
        print(f"  LPIPS unavailable: {e}")
        return None


# ---------------------------------------------------------------------------
# Per-pair metric computation
# ---------------------------------------------------------------------------

METRIC_NAMES_BASE  = ["ssim", "psnr", "mae", "gmsd"]
METRIC_NAMES_LPIPS = METRIC_NAMES_BASE + ["lpips"]


def compute_metrics(gen: np.ndarray, ref: np.ndarray,
                    lpips_fn=None) -> dict:
    result = {
        "ssim": ssim(gen, ref),
        "psnr": psnr(gen, ref),
        "mae":  mae(gen, ref),
        "gmsd": gmsd(gen, ref),
    }
    if lpips_fn is not None:
        result["lpips"] = lpips_fn(gen, ref)
    return result


# ---------------------------------------------------------------------------
# Directory layout detection
# ---------------------------------------------------------------------------

def detect_layout(variant_dir: Path) -> tuple[Path, Path]:
    """
    Returns (gen_dir, ref_dir).

    Layout A (current infer.py): generated in raw/, refs in raw/img/
    Layout B (generated_c_d):    generated in root, refs in raw/
    """
    raw_dir = variant_dir / "raw"
    img_dir = raw_dir / "img"

    if img_dir.exists() and any(img_dir.glob("*_real_art.raw")):
        return raw_dir, img_dir      # Layout A

    if raw_dir.exists() and any(raw_dir.glob("*_real_art.raw")):
        return variant_dir, raw_dir  # Layout B

    return raw_dir, img_dir          # fallback (will warn on missing)


# ---------------------------------------------------------------------------
# Pair discovery
# ---------------------------------------------------------------------------

def find_pairs(variant_dir: Path, model_types: list[str]) -> list[dict]:
    gen_dir, ref_dir = detect_layout(variant_dir)

    if not gen_dir.exists():
        print(f"  WARNING: generated dir not found: {gen_dir}")
        return []

    pairs = []
    for gen_path in sorted(gen_dir.glob("*.raw")):
        stem = gen_path.stem

        matched_type = None
        for mtype in model_types:
            if stem.endswith(f"_{mtype}"):
                matched_type = mtype
                break
        if matched_type is None:
            continue

        prefix   = stem[: -(len(matched_type) + 1)]
        ref_path = ref_dir / f"{prefix}_{MODEL_SUFFIXES[matched_type]}.raw"

        if not ref_path.exists():
            print(f"  WARNING: reference not found: {ref_path.name}")
            continue

        m      = re.search(r"img(\d+)", stem)
        img_id = m.group(1) if m else stem

        pairs.append({
            "variant":    variant_dir.name,
            "img_id":     img_id,
            "model_type": matched_type,
            "generated":  gen_path,
            "reference":  ref_path,
        })

    return pairs


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(scores: list[dict], metric_names: list[str]) -> dict:
    result = {}
    for k in metric_names:
        vals = np.array([s[k] for s in scores if k in s], dtype=np.float64)
        if len(vals) == 0:
            continue
        result[f"{k}_mean"]   = float(vals.mean())
        result[f"{k}_std"]    = float(vals.std())
        result[f"{k}_median"] = float(np.median(vals))
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated CT images: SSIM, PSNR, MAE, GMSD, LPIPS",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Root directory with variant subfolders\n"
             "  e.g. data/raw/generated_c_d  or  results/inference",
    )
    parser.add_argument(
        "--variants", nargs="+", default=None,
        help="Variant subfolder names to evaluate.\n"
             "Default: auto-discover all subdirectories.",
    )
    parser.add_argument(
        "--model-types", nargs="+",
        default=["cyclegan_ab", "diffusion"],
        choices=list(MODEL_SUFFIXES.keys()),
        help="Which model outputs to evaluate.\n"
             "Default: cyclegan_ab diffusion\n"
             "Choices: cyclegan_ab  diffusion  cyclegan_ba",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/eval"),
        help="Output directory for CSV results (default: results/eval)",
    )
    parser.add_argument(
        "--no-lpips", action="store_true",
        help="Skip LPIPS computation (faster, no GPU needed).",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU for LPIPS (default: use CUDA if available).",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        sys.exit(f"ERROR: --data-dir not found: {args.data_dir}")

    # Discover variants
    if args.variants:
        variant_dirs = [args.data_dir / v for v in args.variants]
    else:
        variant_dirs = sorted(p for p in args.data_dir.iterdir() if p.is_dir())

    if not variant_dirs:
        sys.exit(f"No variant directories found in {args.data_dir}")

    # Set up LPIPS
    lpips_fn = None
    metric_names = METRIC_NAMES_BASE
    if not args.no_lpips:
        try:
            import torch
            dev = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            dev = "cpu"
        lpips_fn    = _try_build_lpips(dev)
        metric_names = METRIC_NAMES_LPIPS if lpips_fn else METRIC_NAMES_BASE

    args.out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    agg_rows = []

    for variant_dir in variant_dirs:
        if not variant_dir.exists():
            print(f"WARNING: variant dir not found — {variant_dir}")
            continue

        gen_dir, ref_dir = detect_layout(variant_dir)
        layout = "A (raw/ + raw/img/)" if ref_dir.name == "img" else "B (root + raw/)"
        print(f"\n{'='*60}")
        print(f"  Variant : {variant_dir.name}  [{layout}]")
        print(f"  gen_dir : {gen_dir}")
        print(f"  ref_dir : {ref_dir}")
        print(f"{'='*60}")

        pairs = find_pairs(variant_dir, args.model_types)
        if not pairs:
            print(f"  No pairs found — skipping.")
            continue

        per_model: dict[str, list[dict]] = {}

        for p in pairs:
            gen  = load_raw(p["generated"])
            ref  = load_raw(p["reference"])
            mets = compute_metrics(gen, ref, lpips_fn)

            row = {
                "variant":    p["variant"],
                "model_type": p["model_type"],
                "img_id":     p["img_id"],
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
            agg = aggregate(score_list, metric_names)
            n   = len(score_list)
            agg_rows.append({
                "variant": variant_dir.name,
                "model_type": mtype,
                "n": n,
                **agg,
            })
            print(f"\n  --- {mtype}  n={n} ---")
            for k in metric_names:
                if f"{k}_mean" in agg:
                    print(f"    {k.upper():<6}  "
                          f"mean={agg[f'{k}_mean']:.4f}  "
                          f"std={agg[f'{k}_std']:.4f}  "
                          f"median={agg[f'{k}_median']:.4f}")

    if not all_rows:
        sys.exit("No results computed. Check --data-dir and --variants.")

    # Per-sample CSV
    per_sample_path = args.out / "eval_per_sample.csv"
    sample_fields   = ["variant", "model_type", "img_id"] + metric_names
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
    header = (f"{'Variant':<16} {'Type':<14} {'N':>4}  "
              f"{'SSIM↑':>8}  {'PSNR↑':>8}  {'MAE↓':>8}  {'GMSD↓':>8}"
              + ("  {'LPIPS↓':>8}" if has_lpips else ""))
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
