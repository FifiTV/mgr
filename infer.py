"""
infer.py -- CycleGAN / Diffusion inference.

Examples:

  # Single pair -- clean CT + artifact CT (G_AB + Diffusion)
  python infer.py --clean data/raw/RPI/body1/Target/training_body_nometal_img1000_512x512x1.raw \
                  --artifact data/raw/RPI/body1/Baseline/training_body_metalart_img1000_512x512x1.raw \
                  --out results/inference

  # Whole paired folder (Target + Baseline)
  python infer.py --target-dir data/raw/RPI/body1/Target \
                  --baseline-dir data/raw/RPI/body1/Baseline \
                  --out results/inference --n 5

  # G_BA only -- artifact reduction (artifact image only, no pair needed)
  python infer.py --artifact data/raw/real/metal01_slice0010_H512_W512.raw \
                  --out results/inference

  # SDEdit mode: start from noisy clean image instead of pure noise
  # (recommended when the model was trained on limited data)
  python infer.py --clean ... --artifact ... --out results/inference \
                  --diffusion results/models/diffusion_unet_soft.pth \
                  --sampler ddpm --t-start 700 --steps 200

  # Explicit model weights and label mode
  python infer.py --target-dir ... --baseline-dir ... --out results/inference \
                  --cyclegan-ab results/models/cyclegan_G_AB_soft.pth \
                  --cyclegan-ba results/models/cyclegan_G_BA_soft.pth \
                  --diffusion   results/models/diffusion_unet_soft.pth \
                  --label-mode soft --steps 200
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent))

import tomllib
from src.models.cyclegan import Generator
from src.models.diffusion import DiffusionModel


# -- Helpers -------------------------------------------------------------------

def load_config(path: str = "config.toml") -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_raw(path: Path, shape: tuple = (512, 512)) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    return arr.reshape(shape)


def normalize_hu(img: np.ndarray) -> np.ndarray:
    lo, hi = img.min(), img.max()
    if hi - lo > 1e-5:
        return 2.0 * (img - lo) / (hi - lo) - 1.0
    return img


def to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """numpy (H,W) -> (1,1,H,W) float tensor."""
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float().to(device)


def compute_masks(clear: np.ndarray, art: np.ndarray, cfg: dict,
                  bloom_max: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Compute mask_M and mask_A identically to CTDataset."""
    diff = np.abs(art - clear)
    metal_mask = (diff > cfg["data"]["metal_threshold_hu"]).astype(np.float32)

    # Soft artifact mask -- log-scaled, anchored to global bloom_max
    w = np.log1p(diff)
    denom = np.log1p(bloom_max) if bloom_max else w.max()
    artifact_mask = np.clip(w / (denom + 1e-6), 0.0, 1.0).astype(np.float32)

    return metal_mask, artifact_mask


# -- Model loaders -------------------------------------------------------------

def load_cyclegan_generator(path: Path, input_nc: int, cfg: dict,
                            device: torch.device) -> Generator:
    gen = Generator(
        input_nc=input_nc,
        output_nc=cfg["models"]["generator_output_nc"],
        n_residual_blocks=cfg["models"]["generator_n_residual"],
    ).to(device)
    gen.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    gen.eval()
    return gen


def load_diffusion(path: Path, cfg: dict, device: torch.device) -> DiffusionModel:
    # Prefer EMA weights if they exist next to the given path (better sample quality)
    ema_path = path.parent / (path.stem + "_ema" + path.suffix)
    load_path = ema_path if ema_path.exists() else path
    if load_path != path:
        print(f"  Using EMA weights: {load_path}")

    diff = DiffusionModel(
        architecture=cfg["diffusion"].get("architecture", "standard"),
        time_steps=cfg["diffusion"].get("time_steps", 1000),
        device=device,
        input_channels=cfg["models"]["unet_input_channels"],
        output_channels=cfg["models"]["unet_output_channels"],
    )
    state = torch.load(load_path, map_location=device, weights_only=True)
    # EMA state dict has CPU tensors -- move to device
    state = {k: v.to(device) for k, v in state.items()}
    diff.model.load_state_dict(state)
    diff.model.eval()
    return diff


# -- Inference -----------------------------------------------------------------

@torch.no_grad()
def run_cyclegan_ab(gen_ab: Generator,
                    clean_t: torch.Tensor,
                    mask_m_t: torch.Tensor,
                    mask_a_t: torch.Tensor) -> np.ndarray:
    """G_AB: (clean + mask_M + mask_A) -> generated artifact CT."""
    out = gen_ab(torch.cat([clean_t, mask_m_t, mask_a_t], dim=1))
    return out.squeeze().cpu().numpy()


@torch.no_grad()
def run_cyclegan_ba(gen_ba: Generator, art_t: torch.Tensor) -> np.ndarray:
    """G_BA: artifact CT -> clean CT (artifact reduction)."""
    return gen_ba(art_t).squeeze().cpu().numpy()


@torch.no_grad()
def run_diffusion(diff_model: DiffusionModel,
                  clean_t: torch.Tensor,
                  mask_m_t: torch.Tensor,
                  mask_a_t: torch.Tensor,
                  steps: int,
                  sampler: str = "ddpm",
                  t_start: int | None = None) -> np.ndarray:
    """
    Reverse diffusion inference.

    sampler:
      "ddpm" -- stochastic DDPM (Ho et al. 2020). Adds noise at each step;
                more robust when the model is not perfectly trained, because
                accumulated DDIM errors don't build up into pure noise.
      "ddim" -- deterministic DDIM (Song et al. 2020). Fewer steps, but
                requires a well-trained model. Starting from pure Gaussian
                noise with an undertrained model often produces noise output.

    t_start (SDEdit, Meng et al. 2021):
      When set, forward-diffuses clean_t to that noise level first, then
      denoises from t_start to 0. Gives the model a structured starting
      point instead of pure Gaussian noise -- more reliable on limited data.
      Recommended range: 600-800.
      None = start from pure Gaussian noise (standard generative mode).
    """
    condition = torch.cat([clean_t, mask_m_t, mask_a_t], dim=1)
    device = clean_t.device
    T = diff_model.time_steps

    if t_start is not None and 0 < t_start < T:
        # SDEdit: forward-diffuse clean image to t_start, then denoise to 0
        noise = torch.randn(1, 1, clean_t.shape[2], clean_t.shape[3], device=device)
        ab_start = diff_model.alphas_cumprod[t_start]
        x = ab_start.sqrt() * clean_t + (1.0 - ab_start).sqrt() * noise
        stride = max(1, (t_start + 1) // steps)
        step_seq = list(reversed(range(0, t_start + 1, stride)))
        print(f"  SDEdit: t_start={t_start}, {len(step_seq)} denoising steps")
    else:
        x = torch.randn(1, 1, clean_t.shape[2], clean_t.shape[3], device=device)
        stride = max(1, T // steps)
        step_seq = list(reversed(range(0, T, stride)))

    for i, t_val in enumerate(step_seq):
        t = torch.full((1,), t_val, device=device, dtype=torch.long)
        eps_pred = diff_model.model(torch.cat([x, condition], dim=1), t)
        alpha_bar_t = diff_model.alphas_cumprod[t_val]

        if sampler == "ddpm":
            # DDPM reverse step: mu = (x - beta/sqrt(1-abar) * eps) / sqrt(alpha)
            alpha_t = diff_model.alphas[t_val]
            beta_t  = diff_model.betas[t_val]
            mu = (x - beta_t / (1.0 - alpha_bar_t).sqrt() * eps_pred) / alpha_t.sqrt()
            x = mu + (beta_t.sqrt() * torch.randn_like(x) if t_val > 0 else 0.0)
        else:
            # DDIM deterministic update
            x0_pred = (x - (1.0 - alpha_bar_t).sqrt() * eps_pred) / alpha_bar_t.sqrt()
            x0_pred = x0_pred.clamp(-1.0, 1.0)
            if i + 1 < len(step_seq):
                alpha_bar_prev = diff_model.alphas_cumprod[step_seq[i + 1]]
            else:
                alpha_bar_prev = torch.ones(1, device=device)
            x = alpha_bar_prev.sqrt() * x0_pred + (1.0 - alpha_bar_prev).sqrt() * eps_pred

    return x.squeeze().cpu().numpy()


# -- Visualisation -------------------------------------------------------------

def save_figure(panels: list[tuple[np.ndarray, str]],
                title: str,
                out_path: Path) -> None:
    """Save a row of (image, label) panels as a single PNG."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(n * 4, 4))
    if n == 1:
        axes = [axes]

    for ax, (img, label) in zip(axes, panels):
        vmin, vmax = (-1, 1) if img.min() < 0 else (0, 1)
        cmap = "gray" if img.min() < 0 else "hot"
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.axis("off")

    fig.suptitle(title, fontsize=10, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


# -- Pair matching -------------------------------------------------------------

def match_pairs(target_dir: Path, baseline_dir: Path) -> list[tuple[Path, Path]]:
    """Match Target <-> Baseline files by img number. Returns list of (clean, art) paths."""
    def img_id(p: Path) -> str | None:
        m = re.search(r'img(\d+)', p.stem)
        return m.group(1) if m else None

    target_map = {img_id(p): p for p in sorted(target_dir.glob("*.raw")) if img_id(p)}
    pairs = []
    for p in sorted(baseline_dir.glob("*.raw")):
        i = img_id(p)
        if i and i in target_map:
            pairs.append((target_map[i], p))
    return pairs


# -- Per-sample processing -----------------------------------------------------

def process_pair(clean_path: Path | None,
                 art_path: Path | None,
                 gen_ab: Generator | None,
                 gen_ba: Generator | None,
                 diff_model: DiffusionModel | None,
                 cfg: dict,
                 bloom_max: float | None,
                 device: torch.device,
                 diff_steps: int,
                 diff_sampler: str,
                 diff_t_start: int | None,
                 no_masks: bool,
                 out_dir: Path,
                 idx: int) -> None:

    has_clean = clean_path is not None
    has_art   = art_path is not None

    clean_np = normalize_hu(load_raw(clean_path)) if has_clean else None
    art_np   = normalize_hu(load_raw(art_path))   if has_art   else None

    clean_t = to_tensor(clean_np, device) if has_clean else None
    art_t   = to_tensor(art_np,   device) if has_art   else None

    # Compute masks from the paired images (needed unless --no-masks is set)
    if no_masks:
        # Zero masks: model generates artifacts using only the clean image
        if has_clean:
            mask_m_t = torch.zeros(1, 1, clean_t.shape[2], clean_t.shape[3], device=device)
            mask_a_t = torch.zeros_like(mask_m_t)
        else:
            mask_m_t = mask_a_t = None
        mask_m_np = mask_a_np = None
    elif has_clean and has_art:
        mask_m_np, mask_a_np = compute_masks(
            load_raw(clean_path), load_raw(art_path), cfg, bloom_max
        )
        mask_m_t = to_tensor(mask_m_np, device)
        mask_a_t = to_tensor(mask_a_np, device)
    else:
        mask_m_np = mask_a_np = None
        mask_m_t  = mask_a_t  = None

    panels = []

    if has_clean:
        panels.append((clean_np, "Input (clean)"))
    if mask_m_np is not None:
        panels.append((mask_m_np, "Metal mask"))
        panels.append((mask_a_np, "Artifact mask (soft)"))

    if not no_masks:
        need_masks = (gen_ab is not None) or (diff_model is not None)
        if need_masks and has_clean and mask_m_t is None:
            print("  WARNING: G_AB and Diffusion require --artifact (or --no-masks). Skipping.")

    # G_AB: clean + masks -> generated artifact
    if gen_ab is not None and has_clean and mask_m_t is not None:
        label = "CycleGAN G_AB\n(no-mask mode)" if no_masks else "CycleGAN G_AB\n(artifact gen.)"
        panels.append((run_cyclegan_ab(gen_ab, clean_t, mask_m_t, mask_a_t), label))

    # Diffusion: clean + masks -> generated artifact
    if diff_model is not None and has_clean and mask_m_t is not None:
        tag = f"{diff_sampler.upper()}"
        if diff_t_start is not None:
            tag += f" SDEdit t={diff_t_start}"
        if no_masks:
            tag += " (no-mask)"
        print(f"  Diffusion {tag} ({diff_steps} steps)...")
        panels.append((run_diffusion(diff_model, clean_t, mask_m_t, mask_a_t,
                                     diff_steps, diff_sampler, diff_t_start),
                       f"Diffusion {tag}\n({diff_steps} steps)"))

    # G_BA: artifact -> clean (artifact reduction, no masks needed)
    if gen_ba is not None and has_art:
        panels.append((run_cyclegan_ba(gen_ba, art_t),
                       "CycleGAN G_BA\n(artifact removal)"))

    if has_art:
        panels.append((art_np, "Ground truth\n(artifact CT)"))

    if not panels:
        print("  No models ran -- nothing to save.")
        return

    stem = (art_path or clean_path).stem[:30]
    save_figure(panels, stem, out_dir / f"infer_{idx:04d}_{stem}.png")


# -- CLI -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CycleGAN / Diffusion inference",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    grp = parser.add_argument_group("Input (single files or paired folders)")
    grp.add_argument("--clean",        type=Path, help="Single .raw file -- clean CT")
    grp.add_argument("--artifact",     type=Path, help="Single .raw file -- artifact CT")
    grp.add_argument("--target-dir",   type=Path, help="Folder of clean CTs (Target/)")
    grp.add_argument("--baseline-dir", type=Path, help="Folder of artifact CTs (Baseline/)")
    grp.add_argument("--n",            type=int, default=5,
                     help="Max number of pairs to process (default: 5)")

    grp2 = parser.add_argument_group("Model weights (.pth paths)")
    grp2.add_argument("--cyclegan-ab", type=Path,
                      help="G_AB weights (clean->artifact). Default: results/models/cyclegan_G_AB_{label_mode}.pth")
    grp2.add_argument("--cyclegan-ba", type=Path,
                      help="G_BA weights (artifact->clean). Default: results/models/cyclegan_G_BA_{label_mode}.pth")
    grp2.add_argument("--diffusion",   type=Path,
                      help="Diffusion U-Net weights. Default: results/models/diffusion_unet_{label_mode}.pth")

    parser.add_argument("--out",        type=Path, default=Path("results/inference"),
                        help="Output directory for PNGs (default: results/inference)")
    parser.add_argument("--config",     default="config.toml")
    parser.add_argument("--label-mode", choices=["soft", "hard"], default="soft",
                        help="Label mode matching the trained model (default: soft)")
    parser.add_argument("--steps",      type=int, default=200,
                        help="Diffusion denoising steps at inference (default: 200)")
    parser.add_argument("--sampler",    choices=["ddpm", "ddim"], default="ddpm",
                        help=(
                            "Diffusion sampler (default: ddpm).\n"
                            "  ddpm -- stochastic; more robust for undertrained models.\n"
                            "  ddim -- deterministic; faster, needs well-trained model.\n"
                            "If ddim produces noise, switch to ddpm or use --t-start."
                        ))
    parser.add_argument("--t-start",    type=int, default=None,
                        help=(
                            "SDEdit starting timestep (0-1000, default: None).\n"
                            "When set, the diffusion starts from the clean image\n"
                            "noised to this level instead of pure Gaussian noise.\n"
                            "Recommended: 600-800. Use when generation produces noise."
                        ))
    parser.add_argument("--no-masks",   action="store_true",
                        help=(
                            "Zero out mask inputs for G_AB and Diffusion.\n"
                            "Lets the model generate artifacts from the clean image alone\n"
                            "(requires mask_dropout_prob > 0 during training)."
                        ))
    parser.add_argument("--cpu",        action="store_true",
                        help="Force CPU even when CUDA is available")

    args = parser.parse_args()

    cfg       = load_config(args.config)
    bloom_max = cfg["data"].get("bloom_max")
    device    = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    lm         = args.label_mode
    models_dir = Path(cfg.get("training", {}).get("model_save_dir", "results/models"))
    path_ab    = args.cyclegan_ab or models_dir / f"cyclegan_G_AB_{lm}.pth"
    path_ba    = args.cyclegan_ba or models_dir / f"cyclegan_G_BA_{lm}.pth"
    path_diff  = args.diffusion   or models_dir / f"diffusion_unet_{lm}.pth"

    gen_ab = gen_ba = diff_model = None

    if path_ab.exists():
        print(f"Loading G_AB:       {path_ab}")
        gen_ab = load_cyclegan_generator(path_ab, input_nc=cfg["models"]["generator_input_nc"],
                                         cfg=cfg, device=device)
    else:
        print(f"G_AB not found:     {path_ab}")

    if path_ba.exists():
        print(f"Loading G_BA:       {path_ba}")
        gen_ba = load_cyclegan_generator(path_ba, input_nc=1, cfg=cfg, device=device)
    else:
        print(f"G_BA not found:     {path_ba}")

    if path_diff.exists():
        print(f"Loading Diffusion:  {path_diff}")
        diff_model = load_diffusion(path_diff, cfg, device)
    else:
        print(f"Diffusion not found: {path_diff}")

    if gen_ab is None and gen_ba is None and diff_model is None:
        sys.exit("No models loaded -- check paths.")

    pairs: list[tuple[Path | None, Path | None]] = []

    if args.clean or args.artifact:
        pairs.append((args.clean, args.artifact))
    elif args.target_dir and args.baseline_dir:
        all_pairs = match_pairs(args.target_dir, args.baseline_dir)
        if not all_pairs:
            sys.exit(f"No matching pairs found in {args.target_dir} <-> {args.baseline_dir}")
        pairs = all_pairs[: args.n]
        print(f"Found {len(all_pairs)} pairs, processing {len(pairs)}.")
    elif args.target_dir:
        pairs = [(p, None) for p in sorted(args.target_dir.glob("*.raw"))[: args.n]]
    elif args.baseline_dir:
        pairs = [(None, p) for p in sorted(args.baseline_dir.glob("*.raw"))[: args.n]]
    else:
        parser.print_help()
        sys.exit("\nProvide --clean/--artifact or --target-dir/--baseline-dir.")

    print(f"\nProcessing {len(pairs)} pair(s) -> {args.out}/\n")
    for idx, (clean_p, art_p) in enumerate(pairs):
        print(f"[{idx+1}/{len(pairs)}] {(art_p or clean_p).name}")
        process_pair(
            clean_path=clean_p, art_path=art_p,
            gen_ab=gen_ab, gen_ba=gen_ba, diff_model=diff_model,
            cfg=cfg, bloom_max=bloom_max, device=device,
            diff_steps=args.steps,
            diff_sampler=args.sampler,
            diff_t_start=args.t_start,
            no_masks=args.no_masks,
            out_dir=args.out, idx=idx,
        )

    print(f"\nDone. Results in: {args.out}/")


if __name__ == "__main__":
    main()
