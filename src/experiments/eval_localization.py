"""eval_localization.py — do generated streaks follow the given implants?

For every generated sample (infer_gaussian.py output: <stem>.json + <stem>_gen_error.raw):
  1. streak centers   : find_streak_centers (eval_methods.py) on I_error_gen [HU]
  2. implant centroids: connected components of the metal mask the model was
                        conditioned on (raw/img/<stem>_metal_mask.raw if saved,
                        otherwise the RPI metalonlymask file / HU-threshold fallback)
  3. I_prior          : raw/img/<stem>_prior.raw if saved (prior models), otherwise
                        computed from the same mask with --prior-params / config
                        (so old and new models are scored against the same region)

Metrics per sample:
  hit_rate          fraction of detected centers within --hit-radius px (15) of an implant
  implant_recall    fraction of implants with a detected center within --hit-radius
  mean_dist         mean distance detected center -> nearest implant centroid [px]
  energy_in_prior   sum|I_error| over {|I_prior| > thr * max|I_prior|} / sum|I_error|   (body mask,
                    thr default 0.01)
  prior_area_frac   area fraction of that region inside the body (chance level)
  energy_lift       energy_in_prior / prior_area_frac   (1 = no better than chance)

Pooled metrics (summary): total hits / total centers, total recalled / total implants.

Usage:
  # compare old (mask) and new (prior) model in one table
  python src/experiments/eval_localization.py \\
      --run old=results/gaussian_infer/baseline  new=results/gaussian_infer/prior \\
      --rpi data/raw/RPI \\
      --prior-params results/models/gaussian_prior/model_config.json \\
      --include-aapm \\
      --out results/eval_localization

  Each run dir may be flat (*.json in it), have raw/, or bodyN/raw/ sub-dirs.
Outputs: <out>/localization_per_sample.csv, <out>/localization_summary.{json,csv}
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes, center_of_mass, label

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.experiments.eval_methods import find_streak_centers           # noqa: E402
from src.experiments.eval_utils import find_aapm_pair, load_raw        # noqa: E402
from src.physics.artifact_prior import PriorParams, finalize_prior, unit_prior  # noqa: E402
from src.physics.metal_mask import (METAL_MASK_DIRS, find_metal_mask,  # noqa: E402
                                    load_binary_metal_mask)

log = logging.getLogger('eval_localization')

ERROR_SCALE_DEFAULT = 642.8


# ── Discovery ──────────────────────────────────────────────────────────────────

def _raw_dirs(run_dir: Path) -> list[Path]:
    if list(run_dir.glob('*.json')):
        return [run_dir]
    if (run_dir / 'raw').is_dir():
        return [run_dir / 'raw']
    return sorted(d / 'raw' for d in run_dir.iterdir() if (d / 'raw').is_dir())


def collect_samples(run_dir: Path, bodies: list[str] | None) -> list[dict]:
    out = []
    for rd in _raw_dirs(run_dir):
        for jp in sorted(rd.glob('*.json')):
            err = rd / f'{jp.stem}_gen_error.raw'
            if not err.exists():
                continue
            meta = json.loads(jp.read_text(encoding='utf-8'))
            if bodies and meta.get('body') not in bodies:
                continue
            img = rd / 'img'
            out.append({
                'stem': jp.stem, 'img_id': int(meta['img_id']), 'body': meta['body'],
                'error_scale': float(meta.get('error_scale', ERROR_SCALE_DEFAULT)),
                'gen_error': err,
                'mask_saved':  img / f'{jp.stem}_metal_mask.raw',
                'prior_saved': img / f'{jp.stem}_prior.raw',
            })
    return out


# ── Metrics ────────────────────────────────────────────────────────────────────

def implant_centroids(mask: np.ndarray) -> np.ndarray:
    lab, n = label(mask)
    if n == 0:
        return np.empty((0, 2))
    return np.array(center_of_mass(mask, lab, range(1, n + 1)), dtype=np.float64)


def localization_metrics(i_error_hu: np.ndarray, body_mask: np.ndarray,
                         metal_mask: np.ndarray, i_prior: np.ndarray,
                         hit_radius: float, prior_thr: float) -> dict:
    centers, _ = find_streak_centers(i_error_hu, body_mask)
    implants   = implant_centroids(metal_mask)
    n_c, n_i   = len(centers), len(implants)

    res = {'n_centers': n_c, 'n_implants': n_i}
    if n_c and n_i:
        d = np.linalg.norm(centers[:, None, :].astype(float) - implants[None, :, :], axis=-1)
        res['n_hits']         = int((d.min(axis=1) <= hit_radius).sum())
        res['n_recalled']     = int((d.min(axis=0) <= hit_radius).sum())
        res['hit_rate']       = res['n_hits'] / n_c
        res['implant_recall'] = res['n_recalled'] / n_i
        res['mean_dist']      = float(d.min(axis=1).mean())
    else:
        res.update(n_hits=0, n_recalled=0, hit_rate=np.nan,
                   implant_recall=0.0 if n_i else np.nan, mean_dist=np.nan)

    e = np.abs(i_error_hu) * body_mask
    pmax = float(np.abs(i_prior).max())
    region = (np.abs(i_prior) > prior_thr * pmax) & body_mask if pmax > 0 else np.zeros_like(body_mask)
    total = float(e.sum())
    res['energy_in_prior'] = float(e[region].sum() / total) if total > 0 else np.nan
    res['prior_area_frac'] = float(region.sum() / max(body_mask.sum(), 1))
    res['energy_lift'] = (res['energy_in_prior'] / res['prior_area_frac']
                          if res['prior_area_frac'] > 0 and np.isfinite(res['energy_in_prior'])
                          else np.nan)
    return res


# ── Main ───────────────────────────────────────────────────────────────────────

def _load_prior_params(path: Path | None, config: Path) -> PriorParams:
    if path is not None:
        d = json.loads(path.read_text(encoding='utf-8'))
        d = d.get('prior') or d          # model_config.json or a bare params dict
        return PriorParams(**{k: v for k, v in d.items() if k in PriorParams.__dataclass_fields__})
    if config.exists():
        import tomllib
        with open(config, 'rb') as f:
            cfg = tomllib.load(f)
        return PriorParams.from_config(cfg.get('gaussian', {}).get('prior'))
    return PriorParams()


def summarize(df: pd.DataFrame) -> dict:
    s = {'n_samples': int(len(df))}
    for k in ('hit_rate', 'implant_recall', 'mean_dist',
              'energy_in_prior', 'prior_area_frac', 'energy_lift'):
        v = df[k].astype(float)
        s[f'{k}_mean']   = float(np.nanmean(v)) if v.notna().any() else None
        s[f'{k}_median'] = float(np.nanmedian(v)) if v.notna().any() else None
    s['pooled_hit_rate'] = (float(df['n_hits'].sum() / df['n_centers'].sum())
                            if df['n_centers'].sum() else None)
    s['pooled_implant_recall'] = (float(df['n_recalled'].sum() / df['n_implants'].sum())
                                  if df['n_implants'].sum() else None)
    s['frac_no_centers'] = float((df['n_centers'] == 0).mean())
    return s


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s', force=True)
    ap = argparse.ArgumentParser(description='Streak localization vs implant masks',
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--run', nargs='+', required=True, metavar='TAG=DIR',
                    help='One or more inference output dirs, e.g. old=results/a new=results/b')
    ap.add_argument('--rpi', type=Path, required=True, help='RPI root with body*/')
    ap.add_argument('--out', type=Path, default=Path('results/eval_localization'))
    ap.add_argument('--bodies', nargs='+', default=None)
    ap.add_argument('--prior-params', type=Path, default=None,
                    help='model_config.json (or prior params JSON) used when a sample has no\n'
                         'saved _prior.raw. Default: [gaussian.prior] of --config.')
    ap.add_argument('--config', type=Path, default=_ROOT / 'config.toml')
    ap.add_argument('--hit-radius', type=float, default=15.0)
    ap.add_argument('--prior-thr', type=float, default=0.01,
                    help='Prior region: |I_prior| > thr * max|I_prior| (default 0.01).\n'
                         'max sits on the implant; streaks are ~1-5%% of it, so 0.1 covers\n'
                         'almost only the implant core. On real AAPM I_error (body10):\n'
                         'thr=0.01 -> region 1.7%% of body, 9.4%% of energy, lift ~11.')
    ap.add_argument('--metal-threshold-hu', type=float, default=500.0,
                    help='Fallback mask threshold when no metalonlymask file exists')
    ap.add_argument('--include-aapm', action='store_true',
                    help='Also score the real AAPM I_error (tag "aapm_reference", upper bound)')
    args = ap.parse_args()

    params = _load_prior_params(args.prior_params, args.config)
    log.info(f'Prior params (for samples without saved prior): {params.to_dict()}')
    args.out.mkdir(parents=True, exist_ok=True)

    runs = []
    for item in args.run:
        tag, _, d = item.partition('=')
        if not d:
            tag, d = Path(item).name, item
        runs.append((tag, Path(d)))

    rows, aapm_done, prior_cache = [], set(), {}

    def get_mask_prior(s: dict, art_p: Path, clean_p: Path):
        if s['mask_saved'].exists():
            mask, msrc = load_raw(s['mask_saved']) > 0.5, 'saved'
        else:
            mp = find_metal_mask(args.rpi / s['body'], s['img_id'], METAL_MASK_DIRS)
            mask, msrc = load_binary_metal_mask(mp, art_p, clean_p, args.metal_threshold_hu)
        if s['prior_saved'].exists():
            return mask, msrc, load_raw(s['prior_saved']), 'saved'
        key = (s['body'], s['img_id'], msrc, int(mask.sum()))
        if key not in prior_cache:
            unit, sc = unit_prior(mask, params)
            prior_cache[key] = finalize_prior(unit, sc, params)
        return mask, msrc, prior_cache[key], 'computed'

    for tag, run_dir in runs:
        samples = collect_samples(run_dir, args.bodies)
        log.info(f'[{tag}] {len(samples)} samples from {run_dir}')
        for k, s in enumerate(samples):
            pair = find_aapm_pair(args.rpi, s['body'], s['img_id'])
            if pair is None:
                log.warning(f'  [{tag}] no RPI pair for {s["body"]}/img{s["img_id"]}, skipped')
                continue
            art_p, clean_p = pair
            i_clean   = load_raw(clean_p)
            body_mask = binary_fill_holes(i_clean > -500)
            mask, msrc, i_prior, psrc = get_mask_prior(s, art_p, clean_p)
            if not mask.any():
                log.warning(f'  [{tag}] empty metal mask for {s["stem"]}, skipped')
                continue

            i_err = load_raw(s['gen_error']) * s['error_scale']
            m = localization_metrics(i_err, body_mask, mask, i_prior,
                                     args.hit_radius, args.prior_thr)
            rows.append({'tag': tag, 'body': s['body'], 'img_id': s['img_id'], 'stem': s['stem'],
                         'mask_source': msrc, 'prior_source': psrc, **m})

            akey = (s['body'], s['img_id'])
            if args.include_aapm and akey not in aapm_done:
                aapm_done.add(akey)
                i_err_real = np.clip(load_raw(art_p) - i_clean, -3000, 3000)
                ma = localization_metrics(i_err_real, body_mask, mask, i_prior,
                                          args.hit_radius, args.prior_thr)
                rows.append({'tag': 'aapm_reference', 'body': s['body'], 'img_id': s['img_id'],
                             'stem': s['stem'], 'mask_source': msrc, 'prior_source': psrc, **ma})
            if (k + 1) % 20 == 0:
                log.info(f'  [{tag}] {k + 1}/{len(samples)}')

    if not rows:
        sys.exit('No samples evaluated.')

    df = pd.DataFrame(rows)
    df.to_csv(args.out / 'localization_per_sample.csv', index=False)

    summary = {tag: summarize(g) for tag, g in df.groupby('tag', sort=False)}
    meta = {'hit_radius': args.hit_radius, 'prior_thr': args.prior_thr,
            'prior_params': params.to_dict(), 'runs': {t: str(d) for t, d in runs}}
    (args.out / 'localization_summary.json').write_text(
        json.dumps({'meta': meta, 'summary': summary}, indent=2), encoding='utf-8')
    pd.DataFrame([{'tag': t, **v} for t, v in summary.items()]).to_csv(
        args.out / 'localization_summary.csv', index=False)

    cols = ['n_samples', 'pooled_hit_rate', 'pooled_implant_recall', 'mean_dist_mean',
            'energy_in_prior_mean', 'energy_lift_mean']
    print('\n' + pd.DataFrame(summary).T[cols].to_string(float_format=lambda x: f'{x:.3f}'))
    print(f'\nSaved: {args.out}')


if __name__ == '__main__':
    main()
