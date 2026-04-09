"""
Scan all Baseline/Target pairs under a given RPI root, compute the bloom
(rozbłysk) as Baseline - Target per pixel — excluding metal pixels — and
write the global maximum to config.toml under [data] bloom_max.

Metal pixels are defined as abs(Baseline - Target) > metal_threshold_hu
(same logic as CTDataset), so the bloom max reflects only the artifact
halo/streak region, not the metal implant itself.

Usage:
    python get_max.py                  # uses rpi_path from config.toml
    python get_max.py data/raw/RPI     # override root path
"""

import re
import sys
import tomllib
from pathlib import Path

import numpy as np


# ── config ────────────────────────────────────────────────────────────────────

CONFIG_PATH    = Path(__file__).parent / "config.toml"
TOML_KEY       = "bloom_max"
DTYPE          = np.float32
BASELINE_GLOB  = "training_body_metalart_img*"
IDX_RE         = re.compile(r"training_body_metalart_img(\d+)_")


# ── helpers ───────────────────────────────────────────────────────────────────

def read_raw(path: Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype=DTYPE)


def find_pairs(rpi_root: Path) -> list[tuple[Path, Path]]:
    """Return (baseline, target) path pairs matched by image index."""
    pairs = []
    for baseline in sorted(rpi_root.rglob(f"Baseline/{BASELINE_GLOB}")):
        m = IDX_RE.search(baseline.name)
        if not m:
            continue
        idx = m.group(1)
        target_dir = baseline.parent.parent / "Target"
        targets = list(target_dir.glob(f"training_body_nometal_img{idx}_*"))
        if not targets:
            print(f"  WARNING: no Target found for {baseline.name}, skipping")
            continue
        pairs.append((baseline, targets[0]))
    return pairs


def compute_bloom_max(pairs: list[tuple[Path, Path]], metal_threshold: float) -> float:
    global_max = -np.inf
    for i, (bl, tg) in enumerate(pairs, 1):
        baseline = read_raw(bl)
        target   = read_raw(tg)
        # Metal is identified directly from Baseline HU values — metal implants
        # (Ti, steel) appear as extremely high HU (>>1000), independent of Target.
        # The diff is used only to measure bloom, not to detect metal.
        metal_mask   = baseline > metal_threshold
        bloom_pixels = (baseline - target)[~metal_mask]
        if bloom_pixels.size == 0:
            continue
        local_max = float(bloom_pixels.max())
        if local_max > global_max:
            global_max = local_max
        if i % 100 == 0 or i == len(pairs):
            print(f"  [{i}/{len(pairs)}]  bloom max so far = {global_max:.4f}")
    return global_max


def patch_toml(config_path: Path, key: str, value: float) -> None:
    """Update or insert `key = value` in the [data] section, preserving formatting."""
    lines = config_path.read_text(encoding="utf-8").splitlines(keepends=True)

    in_data = False
    key_idx = None
    section_end = None

    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("[") and not s.startswith("[["):
            if in_data:
                section_end = i
                break
            in_data = (s == "[data]")
            continue
        if in_data and s.startswith(key):
            rest = s[len(key):].lstrip()
            if rest.startswith("="):
                key_idx = i

    new_line = f"{key} = {value}\n"

    if key_idx is not None:
        lines[key_idx] = new_line
    elif in_data:
        insert_at = section_end if section_end is not None else len(lines)
        lines.insert(insert_at, new_line)
    else:
        lines.append(f"\n[data]\n{new_line}")

    config_path.write_text("".join(lines), encoding="utf-8")
    print(f"  Written  [data] {key} = {value}  →  {config_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    with open(CONFIG_PATH, "rb") as f:
        cfg = tomllib.load(f)

    if len(sys.argv) > 1:
        rpi_root = Path(sys.argv[1])
    else:
        project_root = CONFIG_PATH.parent
        rpi_root = project_root / cfg["paths"]["rpi_path"]

    if not rpi_root.exists():
        sys.exit(f"ERROR: RPI root not found: {rpi_root}")

    print(f"Scanning: {rpi_root}")
    pairs = find_pairs(rpi_root)

    if not pairs:
        sys.exit("ERROR: no Baseline/Target pairs found")

    metal_threshold = cfg["data"]["metal_threshold_hu"]
    print(f"Found {len(pairs)} pairs. Metal threshold = {metal_threshold} HU")
    print("Computing bloom (Baseline − Target, metal pixels excluded) …")
    bloom_max = compute_bloom_max(pairs, metal_threshold)

    print(f"\nBloom max = {bloom_max}")
    patch_toml(CONFIG_PATH, TOML_KEY, bloom_max)


if __name__ == "__main__":
    main()
