"""
precompute_priors.py -- cache the analytic artifact prior I_prior for every RPI image.

radon/iradon (360 angles, 512x512) takes ~5 s per image, so priors are computed
once here and only loaded by GaussianDataset (never computed in __getitem__).

Usage:
    python precompute_priors.py --data-dir data/raw/RPI --out-dir results/priors
    python precompute_priors.py --data-dir data/raw/RPI --out-dir results/priors \
        --bodies body1 body2 --workers 16
    python precompute_priors.py ... --overwrite          # recompute everything

Output (per body):
    <out-dir>/prior_params.json                       geometry parameters of the cache
    <out-dir>/<body>/training_body_metalonlymask_img{N}_512x512x1.prior.npy
                                                      float16 unit prior (max|.|=1)
    <out-dir>/<body>/prior_scales.json                {img_id: {raw_max_abs, mask_source, n_mask_px}}

Metal mask source (same as GaussianDataset): <body>/{Metal,Mask}/...metalonlymask...,
fallback |I_metal - I_clean| > gaussian.metal_threshold_hu. Counts per source are logged.

Idempotent: existing .npy files with a recorded scale are skipped. If prior_params.json
exists with different geometry parameters the script refuses to mix caches unless
--overwrite is given.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.physics.artifact_prior import PriorParams, prior_filename, unit_prior
from src.physics.metal_mask import METAL_MASK_DIRS, find_metal_mask, load_binary_metal_mask

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s  %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger('precompute_priors')

SCALES_FILE = 'prior_scales.json'
PARAMS_FILE = 'prior_params.json'


def _load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    import tomllib
    with open(path, 'rb') as f:
        return tomllib.load(f)


def _img_ids(body_dir: Path) -> list[int]:
    ids = []
    for p in sorted((body_dir / 'Baseline').glob('*.raw')):
        m = re.search(r'img(\d+)', p.stem)
        if not m:
            continue
        iid = int(m.group(1))
        if (body_dir / 'Target' / f'training_body_nometal_img{iid}_512x512x1.raw').exists():
            ids.append(iid)
    return sorted(ids)


def _worker(task: tuple) -> tuple:
    """Compute and save one prior. Returns (img_id, info dict or None, error str or None)."""
    body_dir, img_id, out_path, params_dict, mask_dirs, threshold_hu = task
    try:
        params = PriorParams(**params_dict)
        metal_path = find_metal_mask(body_dir, img_id, mask_dirs)
        mask, source = load_binary_metal_mask(
            metal_path,
            body_dir / 'Baseline' / f'training_body_metalart_img{img_id}_512x512x1.raw',
            body_dir / 'Target' / f'training_body_nometal_img{img_id}_512x512x1.raw',
            threshold_hu,
        )
        unit, scale = unit_prior(mask, params)
        tmp = out_path.parent / (out_path.name + '.tmp.npy')
        np.save(tmp, unit.astype(np.float16))
        os.replace(tmp, out_path)          # atomic: no half-written cache files
        return img_id, {'raw_max_abs': scale, 'mask_source': source,
                        'n_mask_px': int(mask.sum())}, None
    except Exception as e:                  # noqa: BLE001 -- report and continue
        return img_id, None, f'{type(e).__name__}: {e}'


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(obj, indent=1), encoding='utf-8')
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description='Precompute artifact priors (radon/iradon).',
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--data-dir', type=Path, required=True,
                    help='RPI root containing body*/ (Baseline, Target, Metal|Mask)')
    ap.add_argument('--out-dir', type=Path, required=True, help='Cache root directory')
    ap.add_argument('--bodies', nargs='+', default=None,
                    help='Body folders to process (default: all body* in --data-dir)')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--overwrite', action='store_true',
                    help='Recompute existing files (and allow changed parameters)')
    ap.add_argument('--config', type=Path, default=Path(__file__).parent / 'config.toml',
                    help='config.toml; prior parameters from [gaussian.prior]')
    ap.add_argument('--n-angles', type=int, default=None)
    ap.add_argument('--mu-scale', type=float, default=None)
    ap.add_argument('--beta', type=float, default=None)
    ap.add_argument('--geometry', default=None, choices=['parallel'])
    args = ap.parse_args()

    cfg       = _load_config(args.config)
    gaus_cfg  = cfg.get('gaussian', {})
    prior_cfg = dict(gaus_cfg.get('prior', {}))
    for k, v in (('n_angles', args.n_angles), ('mu_scale', args.mu_scale),
                 ('beta', args.beta), ('geometry', args.geometry)):
        if v is not None:
            prior_cfg[k] = v
    params       = PriorParams.from_config(prior_cfg)
    mask_dirs    = tuple(gaus_cfg.get('metal_mask_dirs', METAL_MASK_DIRS))
    threshold_hu = float(gaus_cfg.get('metal_threshold_hu', 500.0))

    if not args.data_dir.exists():
        sys.exit(f'--data-dir not found: {args.data_dir}')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Parameter guard: never mix caches computed with different geometry ───
    params_path = args.out_dir / PARAMS_FILE
    key = params.geometry_key()
    if params_path.exists():
        old = json.loads(params_path.read_text(encoding='utf-8'))
        if old != key and not args.overwrite:
            sys.exit(f'{params_path} has different parameters:\n  cache:  {old}\n  now:    {key}\n'
                     'Use another --out-dir or --overwrite.')
    _write_json(params_path, key)

    bodies = args.bodies or sorted(d.name for d in args.data_dir.iterdir()
                                   if d.is_dir() and d.name.startswith('body'))
    log.info(f'Data: {args.data_dir}   Cache: {args.out_dir}   workers={args.workers}')
    log.info(f'Params: {key}   mask_dirs={mask_dirs}   fallback threshold={threshold_hu} HU')
    log.info(f'Bodies: {bodies}')

    grand = Counter()
    for body in bodies:
        body_dir = args.data_dir / body
        if not (body_dir / 'Baseline').exists():
            log.warning(f'{body}: no Baseline/ -- skipped')
            continue
        out_body = args.out_dir / body
        out_body.mkdir(parents=True, exist_ok=True)
        scales_path = out_body / SCALES_FILE
        scales = (json.loads(scales_path.read_text(encoding='utf-8'))
                  if scales_path.exists() and not args.overwrite else {})

        ids, tasks = _img_ids(body_dir), []
        for iid in ids:
            out_path = out_body / prior_filename(iid)
            if not args.overwrite and out_path.exists() and str(iid) in scales:
                continue
            tasks.append((body_dir, iid, out_path, params.to_dict(), mask_dirs, threshold_hu))

        n_skip = len(ids) - len(tasks)
        log.info(f'{body}: {len(ids)} images, {n_skip} cached, {len(tasks)} to compute')
        if not tasks:
            continue

        t0, done, sources, errors = time.time(), 0, Counter(), 0
        with Pool(processes=args.workers) as pool:
            for iid, info, err in pool.imap_unordered(_worker, tasks, chunksize=2):
                done += 1
                if err:
                    errors += 1
                    log.error(f'  {body}/img{iid}: {err}')
                else:
                    scales[str(iid)] = info
                    sources[info['mask_source'].split(':')[0]] += 1
                if done % 50 == 0 or done == len(tasks):
                    el = time.time() - t0
                    eta = el / done * (len(tasks) - done)
                    log.info(f'  {body}: {done}/{len(tasks)}  {done/el:.2f} img/s  ETA {eta/60:.1f} min')
                    _write_json(scales_path, scales)      # periodic, crash-safe
        _write_json(scales_path, scales)

        log.info(f'{body}: done. mask source: {dict(sources)}  errors: {errors}')
        if sources.get('threshold'):
            log.warning(f'{body}: {sources["threshold"]} prior(s) from THRESHOLD fallback '
                        f'(no metalonlymask file in {mask_dirs}) -- mask includes bloom, not only implant')
        grand.update(sources)
        grand['errors'] += errors

    log.info(f'All done. Totals: {dict(grand)}')


if __name__ == '__main__':
    main()
