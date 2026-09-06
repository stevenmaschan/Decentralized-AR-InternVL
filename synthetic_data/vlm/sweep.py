#!/usr/bin/env python3
"""Run the three-part sweep, two jobs at a time (one per GPU).

  1. size    -- train size 1k/2k/5k/10k/20k, nested subsets of the main dataset,
                so val/test stay literally the same 5000 images throughout.
  2. mass    -- lo/hi expert OCR proportions 0.05/0.10/0.25/0.50, one dataset per
                point (the 0.10 point reuses the main dataset at 10k).
  3. context -- at 10k, split by question type instead of alpha.

Every run gets the *same optimizer budget* (--budget updates, default 7800)
rather than the same epoch count: at fixed epochs the 1k configs would simply be
undertrained and the sweep would measure that instead of data scale. Dense at
batch 128 over N examples and each expert at batch 64 over ~N/2 have identical
updates per epoch, so all three models in a config share an epoch count.

    python -m synthetic_data.vlm.sweep --dry-run     # print the plan
    python -m synthetic_data.vlm.sweep               # run it
"""

import os
import json
import time
import queue
import argparse
import threading
import subprocess
from pathlib import Path

MAIN = 'synthetic_data'
SWEEP_DS = 'synthetic_data/sweep_datasets'
RUNS = 'synthetic_data/vlm/runs/sweep'

DENSE_BATCH = 128        # single GPU; matches the 64x2 DDP effective batch
EXPERT_BATCH = 64


def dataset_paths(root):
    """(annotations, features, expert_lo, expert_hi) for a dataset root."""
    r = Path(root)
    feats = r / 'features_vit_base_patch16.npy'
    if not feats.exists():                      # the main dataset keeps them in clip/
        feats = r / 'clip' / 'features_vit_base_patch16.npy'
    return (str(r / 'annotations.jsonl'), str(feats),
            str(r / 'experts_alpha' / 'expert_alpha_lo' / 'annotations.jsonl'),
            str(r / 'experts_alpha' / 'expert_alpha_hi' / 'annotations.jsonl'))


def build_jobs(args):
    jobs = []

    def add(group, config, model, ann, feats, batch, max_train, train_n):
        # equal optimizer budget -> epochs from updates per epoch
        upd = max(1, train_n // batch)
        epochs = max(1, round(args.budget / upd))
        jobs.append({
            'group': group, 'config': config, 'model': model,
            'name': f'{group}__{config}__{model}',
            'annotations': ann, 'features': feats, 'batch': batch,
            'max_train': max_train, 'epochs': epochs,
            'eval_every': max(1, epochs // 10),
            'train_n': train_n, 'updates_per_epoch': upd,
        })

    # --- 1. size sweep on the main dataset --------------------------------- #
    ann, feats, lo, hi = dataset_paths(MAIN)
    for n in args.sizes:
        add('size', f'{n // 1000}k', 'dense', ann, feats, DENSE_BATCH, n, n)
        add('size', f'{n // 1000}k', 'expert_lo', lo, feats, EXPERT_BATCH, n, n // 2)
        add('size', f'{n // 1000}k', 'expert_hi', hi, feats, EXPERT_BATCH, n, n // 2)

    # --- 2. mass sweep at 10k (0.10 point is the main dataset) -------------- #
    for tag, root in (('p05', f'{SWEEP_DS}/mass_p05'),
                      ('p25', f'{SWEEP_DS}/mass_p25'),
                      ('p50', f'{SWEEP_DS}/mass_p50')):
        a, f, l, h = dataset_paths(root)
        add('mass', tag, 'dense', a, f, DENSE_BATCH, 10000, 10000)
        add('mass', tag, 'expert_lo', l, f, EXPERT_BATCH, 10000, 5000)
        add('mass', tag, 'expert_hi', h, f, EXPERT_BATCH, 10000, 5000)

    # --- 3. context split at 10k on the main dataset ------------------------ #
    t = Path(MAIN) / 'experts_task'
    add('context', 'p10', 'expert_ocr', str(t / 'expert_ocr' / 'annotations.jsonl'),
        feats, EXPERT_BATCH, 10000, 5000)
    add('context', 'p10', 'expert_shape', str(t / 'expert_shape' / 'annotations.jsonl'),
        feats, EXPERT_BATCH, 10000, 5000)
    return jobs


def run_job(job, gpu, args):
    out = Path(RUNS) / job['name']
    ckpt = out / 'best.pt'
    if ckpt.exists() and not args.force:
        return job['name'], 0, 'skipped (checkpoint exists)'
    out.mkdir(parents=True, exist_ok=True)
    if ckpt.exists():
        ckpt.unlink()
    cmd = [
        'python', '-m', 'synthetic_data.vlm.train',
        '--annotations', job['annotations'], '--features', job['features'],
        '--out-dir', str(out), '--epochs', str(job['epochs']),
        '--eval-every', str(job['eval_every']), '--batch-size', str(job['batch']),
        '--max-train', str(job['max_train']), '--num-workers', '4',
        '--device', f'cuda:{gpu}',
    ]
    log = Path(RUNS) / f'{job["name"]}.log'
    t0 = time.time()
    with open(log, 'w') as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if rc != 0 or not ckpt.exists():
        return job['name'], 1, f'FAILED rc={rc} after {dt:.0f}s (see {log})'
    return job['name'], 0, f'ok in {dt:.0f}s'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sizes', type=int, nargs='*',
                    default=[1000, 2000, 5000, 10000, 20000])
    ap.add_argument('--budget', type=int, default=7800,
                    help='optimizer updates per run (held constant across configs)')
    ap.add_argument('--gpus', type=int, nargs='*', default=[0, 1])
    ap.add_argument('--force', action='store_true',
                    help='retrain configs that already have a checkpoint')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    jobs = build_jobs(args)
    print(f'{len(jobs)} runs, budget {args.budget} updates each, '
          f'{len(args.gpus)} GPUs\n')
    print(f"{'run':<34}{'train':>7}{'batch':>7}{'upd/ep':>8}{'epochs':>8}{'eval@':>7}")
    for j in jobs:
        print(f"{j['name']:<34}{j['train_n']:>7}{j['batch']:>7}"
              f"{j['updates_per_epoch']:>8}{j['epochs']:>8}{j['eval_every']:>7}")
    for j in jobs:
        for key in ('annotations', 'features'):
            if not Path(j[key]).exists():
                raise SystemExit(f"missing {key} for {j['name']}: {j[key]}")
    if args.dry_run:
        est = len(jobs) * args.budget * 0.0385 / max(1, len(args.gpus)) / 60
        print(f'\nestimated wall time: ~{est:.0f} min')
        return

    Path(RUNS).mkdir(parents=True, exist_ok=True)
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    results, lock = [], threading.Lock()
    t0 = time.time()

    def worker(gpu):
        while True:
            try:
                job = q.get_nowait()
            except queue.Empty:
                return
            name, rc, msg = run_job(job, gpu, args)
            with lock:
                results.append((name, rc, msg))
                done = len(results)
                print(f'[{done}/{len(jobs)}] gpu{gpu} {name}: {msg} '
                      f'({(time.time() - t0) / 60:.0f} min elapsed)', flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    failed = [r for r in results if r[1] != 0]
    print(f'\n{len(results) - len(failed)}/{len(results)} runs ok in '
          f'{(time.time() - t0) / 60:.0f} min')
    for name, _, msg in failed:
        print(f'  FAILED {name}: {msg}')
    Path(RUNS, 'sweep_summary.json').write_text(json.dumps(
        {'jobs': jobs, 'results': results}, indent=2))


if __name__ == '__main__':
    main()
