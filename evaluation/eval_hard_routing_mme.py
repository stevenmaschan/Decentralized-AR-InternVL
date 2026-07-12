"""End-to-end HARD-ROUTED evaluation of cluster experts on MME.

Every MME question is routed to EXACTLY ONE expert -- argmax cosine similarity between the
question's image CLIP feature and that expert's cluster centroid -- each expert is evaluated
on its own disjoint subset (sequentially), and the predictions are merged and scored as one.

MME is structurally unlike the VQA benchmarks handled by `eval_hard_routing.py`, so it gets its
own script:
  * The test set is 14 per-category `.txt` files under `internvl_chat/eval/mme/Your_Results/`,
    each line `image<TAB>question<TAB>gt`, with the TWO questions of an image on consecutive
    lines. Routing is per IMAGE, so an image's two lines always land on the SAME expert and the
    consecutive pairing that MME's `acc_plus` relies on is preserved.
  * The upstream runner `eval/mme/eval.py` is a single-process `model.chat()` loop (NOT torchrun
    / InferenceSampler), used here ONLY as a prediction generator, once per expert.
  * The metric is MME's own acc + acc_plus per category (`eval/mme/calculation.py`), summed into
    Perception (max 2000) and Cognition (max 800) totals. All scoring here reuses that module's
    exact answer parsing (`parse_pred_ans`) and category grouping (`eval_type_dict`).

Routing (identical math to clustering/partition_jsonl_by_balanced_kmeans.assign_to_clusters,
reused verbatim from eval_hard_routing.py):
    feature = CLIP ViT-B/16 get_image_features(image)   # fp16
    feature -= clustering_global_mean.npy               # if present (mean-subtracted clusterings)
    cluster = argmax_k  cos(normalize(feature), normalize(centroid_k))
Expert k MUST align with centroid k, so pass --expert-checkpoints in cluster order; the script
hard-fails if the counts disagree.

Because routing partitions images (never splitting a pair), the merged per-category tallies are
the exact integer union of the per-expert tallies. The script recomputes MME on each expert's
subset AND on the merged union, and ASSERTS integer-exact consistency (merged n_questions,
correct, n_images, acc_plus_correct == sum over experts) -- this catches routing leaks (an image
scored under two experts) or mis-ordered merges. Also hard-fails on: #checkpoints != #centroids,
a consecutive pair spanning two images, per-expert output count mismatches, and missing
data/images (preflight, before any model load). Never passes --dynamic (see CLAUDE.md).

Example:
    python evaluation/eval_hard_routing_mme.py \
        --clustering-dir clustering/balanced_kmeans_unique_features_base-patch16_2_clusters_seed42_mean-subtracted \
        --expert-checkpoints \
            work_dirs/internvl2_5_1b/clusters-2_..._mean-subtracted/cluster-0 \
            work_dirs/internvl2_5_1b/clusters-2_..._mean-subtracted/cluster-1

Run from the repo root with the clustering venv active. Useful flags: --skip-eval (re-route and
re-score existing per-expert predictions), --clip-batch-size, --num-workers, --images-dir.
"""
import argparse
import json
import os
import subprocess
import sys
from collections import deque
from pathlib import Path

# Sibling import: reuse the exact routing implementation so this can never drift from it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_hard_routing import (CHAT, REPO, assign_to_clusters,  # noqa: E402
                               clip_features, load_centroids)

MME_IMAGES = CHAT / 'data/mme/MME_Benchmark_release_version'
TEMPLATES = CHAT / 'eval/mme/Your_Results'
MME_EVAL = CHAT / 'eval/mme/eval.py'
IMG_SUBPATH = 'data/mme/MME_Benchmark_release_version'  # <cat>/<img> lives under here (rel to CHAT)


# --------------------------------------------------------------------------- scoring
def _scorer():
    """Reuse MME's own answer parsing and category grouping so we never drift from calculation.py."""
    p = str(CHAT / 'eval/mme')
    if p not in sys.path:
        sys.path.insert(0, p)
    from calculation import calculate_metrics, eval_type_dict  # noqa: E402
    return calculate_metrics(), eval_type_dict


def score_task_file(path, cm):
    """MME tallies for one category file (lines: image<TAB>question<TAB>gt<TAB>pred).

    Returns None for an absent/empty file. Consecutive lines must share an image (MME's acc_plus
    pairing); the file must have an even line count.
    """
    if not Path(path).exists():
        return None
    lines = [l.rstrip('\n') for l in open(path, encoding='utf-8') if l.strip()]
    if not lines:
        return None
    if len(lines) % 2:
        raise SystemExit(f'{path}: odd line count {len(lines)} -- MME pairs 2 questions per image')

    n_q = correct = n_img = acc_plus_correct = 0
    for i in range(0, len(lines), 2):
        n_img += 1
        img_correct = 0
        imgs = set()
        for item in lines[i:i + 2]:
            parts = item.split('\t')
            if len(parts) < 4:
                raise SystemExit(f'{path}: malformed line (need 4 tab fields): {item!r}')
            img, gt, pred = parts[0], parts[2].strip().lower(), parts[3].strip().lower()
            imgs.add(img)
            if gt not in ('yes', 'no'):
                raise SystemExit(f'{path}: gt is not yes/no: {gt!r}')
            n_q += 1
            if gt == cm.parse_pred_ans(pred):
                correct += 1
                img_correct += 1
        if len(imgs) != 1:
            raise SystemExit(f'{path}: a consecutive pair spans images {imgs} -- acc_plus broken')
        if img_correct == 2:
            acc_plus_correct += 1

    return dict(n_q=n_q, correct=correct, n_img=n_img, acc_plus_correct=acc_plus_correct,
                acc=correct / n_q, acc_plus=acc_plus_correct / n_img,
                score=(correct / n_q + acc_plus_correct / n_img) * 100)


def score_dir(results_dir, cm, eval_type_dict):
    """MME per-task scores + Perception/Cognition/Total for a results directory."""
    per_task, totals = {}, {}
    for etype, tasks in eval_type_dict.items():
        s = 0.0
        for t in tasks:
            d = score_task_file(Path(results_dir) / f'{t}.txt', cm)
            if d is None:
                continue
            per_task[t] = d
            s += d['score']
        totals[etype] = s
    totals['Total'] = sum(totals[e] for e in eval_type_dict)
    return per_task, totals


# --------------------------------------------------------------------------- routing
def load_templates(tasks):
    """Read the 14 template files; validate the consecutive-pair structure MME assumes."""
    tmpl = {}
    for t in tasks:
        f = TEMPLATES / f'{t}.txt'
        lines = [l.rstrip('\n') for l in open(f, encoding='utf-8') if l.strip()]
        if len(lines) % 2:
            raise SystemExit(f'{f}: odd line count {len(lines)}')
        for i in range(0, len(lines), 2):
            if lines[i].split('\t')[0] != lines[i + 1].split('\t')[0]:
                raise SystemExit(f'{f}: lines {i}/{i + 1} are not the same image -- '
                                 'template is not in MME 2-questions-per-image order')
        tmpl[t] = lines
    return tmpl


def route(tmpl, centroids, mean_vector, args, n_experts, out_dir):
    """Route every template line to an expert (by its image) and write per-expert root dirs.

    Returns (assign, roots) where assign[task] is the per-line expert list (template order) and
    roots[k] is the directory of that expert's routed category files.
    """
    images = sorted({f'{IMG_SUBPATH}/{t}/{l.split(chr(9))[0]}'
                     for t, lines in tmpl.items() for l in lines})
    print(f'  routing {sum(len(v) for v in tmpl.values())} questions over {len(images)} images')

    feats = clip_features(images, args.clip_model, args.clip_batch_size, args.num_workers)
    img2exp = dict(zip(images, assign_to_clusters(feats, centroids, mean_vector).tolist()))

    roots = [out_dir / 'roots' / f'expert{k}' for k in range(n_experts)]
    for r in roots:
        r.mkdir(parents=True, exist_ok=True)

    assign, counts = {}, {k: 0 for k in range(n_experts)}
    for t, lines in tmpl.items():
        buckets = {k: [] for k in range(n_experts)}
        line_experts = []
        for l in lines:
            e = img2exp[f'{IMG_SUBPATH}/{t}/{l.split(chr(9))[0]}']
            line_experts.append(e)
            buckets[e].append(l)
            counts[e] += 1
        assign[t] = line_experts
        for k in range(n_experts):
            if buckets[k]:  # only write non-empty files (empty -> eval.py/scorer would choke)
                (roots[k] / f'{t}.txt').write_text('\n'.join(buckets[k]) + '\n', encoding='utf-8')
    total = sum(counts.values())
    for k in range(n_experts):
        print(f'    expert{k}: {counts[k]} questions ({100 * counts[k] / total:.1f}%)')
    return assign, roots


# --------------------------------------------------------------------------- inference
def raw_out_dir(raw_base, checkpoint):
    """Where eval/mme/eval.py writes its outputs: basename(checkpoint) under its cwd."""
    return raw_base / Path(checkpoint).resolve().name


def run_expert(checkpoint, root, raw_base, images_dir, k):
    """Run the upstream single-process MME runner on one expert's routed subset."""
    ckpt = Path(checkpoint)
    if not ckpt.is_absolute():
        ckpt = REPO / ckpt
    ckpt = ckpt.resolve()
    if not (ckpt / 'config.json').is_file():
        raise SystemExit(f'no config.json in checkpoint {ckpt}')

    raw_base.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['PYTHONPATH'] = f"{CHAT}:{env.get('PYTHONPATH', '')}"
    # eval/mme/eval.py is single-process (model.chat); run it with plain python, not torchrun.
    # It writes to basename(checkpoint) relative to cwd, so cwd=raw_base isolates each expert.
    cmd = [sys.executable, str(MME_EVAL),
           '--checkpoint', str(ckpt),
           '--root', str(root.resolve()),
           '--images-dir', str(Path(images_dir).resolve())]
    log = raw_base / 'eval.log'
    print(f'\n===== MME expert-{k}: {ckpt.name} (root={root.name}) =====', flush=True)
    print(f'  logging inference to {log}', flush=True)
    with open(log, 'w') as lf:
        rc = subprocess.run(cmd, cwd=raw_base, env=env,
                            stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise SystemExit(f'eval/mme/eval.py failed for expert {k}; see {log}')
    out = raw_out_dir(raw_base, ckpt)
    if not out.is_dir():
        raise SystemExit(f'expert {k}: expected output dir {out} not created')
    return out


# --------------------------------------------------------------------------- merge
def merge(tmpl, assign, raw_outs, n_experts, merged_dir):
    """Recombine the per-expert predictions into full per-category files in ORIGINAL order.

    calculation.py pairs consecutive lines, so the merged file must preserve template order.
    """
    merged_dir.mkdir(parents=True, exist_ok=True)
    for t, lines in tmpl.items():
        queues = {}
        for k in range(n_experts):
            f = raw_outs[k] / f'{t}.txt'
            queues[k] = deque(l.rstrip('\n') for l in open(f, encoding='utf-8')) if f.exists() \
                else deque()
        out_lines = []
        for e in assign[t]:  # assign[t] is in template order
            if not queues[e]:
                raise SystemExit(f'{t}: expert {e} produced fewer predictions than routed to it')
            out_lines.append(queues[e].popleft())
        for k in range(n_experts):
            if queues[k]:
                raise SystemExit(f'{t}: expert {k} produced {len(queues[k])} extra predictions')
        if len(out_lines) != len(lines):
            raise SystemExit(f'{t}: merged {len(out_lines)} lines, template has {len(lines)}')
        # img (field 0) and gt (field 2) must line up with the template (question is prompt-appended)
        for outl, tl in zip(out_lines, lines):
            op, tp = outl.split('\t'), tl.split('\t')
            if op[0] != tp[0] or op[2].strip().lower() != tp[2].strip().lower():
                raise SystemExit(f'{t}: merged line does not match template\n  out={outl}\n  tmpl={tl}')
        (merged_dir / f'{t}.txt').write_text('\n'.join(out_lines) + '\n', encoding='utf-8')


def assert_consistency(merged_pt, expert_pts, n_experts):
    """Merged integer tallies must equal the sum over experts (routing partitions images)."""
    for t, m in merged_pt.items():
        agg = dict(n_q=0, correct=0, n_img=0, acc_plus_correct=0)
        for k in range(n_experts):
            d = expert_pts[k].get(t)
            if d:
                for key in agg:
                    agg[key] += d[key]
        for key in agg:
            if agg[key] != m[key]:
                raise SystemExit(f'{t}: merged {key}={m[key]} != sum over experts {agg[key]} '
                                 '-- routing leak or scoring inconsistency')


# --------------------------------------------------------------------------- driver
def check_data(tasks, expert_checkpoints, clustering_dir, prefix, images_dir):
    """Preflight: fail before loading a model if templates, images, or centroids are absent."""
    missing = []
    for t in tasks:
        f = TEMPLATES / f'{t}.txt'
        if not f.is_file():
            missing.append(f'eval/mme/Your_Results/{t}.txt')
            continue
        with open(f, encoding='utf-8') as fh:
            first = fh.readline().split('\t')
        if first and first[0]:
            img = Path(images_dir) / t / first[0]
            if not img.is_file():
                missing.append(f'{img.relative_to(CHAT)}  (images for {t})')
    if missing:
        raise SystemExit('MME eval data missing:\n'
                         + '\n'.join(f'  - internvl_chat/{m}' for m in missing)
                         + '\n\nSee internvl_chat/eval/mme/README.md for download instructions.')

    d = Path(clustering_dir)
    if not d.is_absolute():
        d = REPO / d
    if not (d / f'{prefix}_centroids.npy').is_file():
        raise SystemExit(f'no centroids at {d / f"{prefix}_centroids.npy"}')
    for c in expert_checkpoints:
        ck = Path(c)
        if not ck.is_absolute():
            ck = REPO / ck
        if not (ck / 'config.json').is_file():
            raise SystemExit(f'no config.json in checkpoint {ck}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--clustering-dir', required=True)
    ap.add_argument('--expert-checkpoints', nargs='+', required=True,
                    help='expert checkpoint dirs IN CLUSTER ORDER (expert k <-> centroid k)')
    ap.add_argument('--prefix', default='clustering')
    ap.add_argument('--out-dir', default='results/hardroute/mme',
                    help='relative to internvl_chat/ (default results/hardroute/mme)')
    ap.add_argument('--images-dir', default=str(MME_IMAGES),
                    help='MME image root containing <category>/<image> (default: the bundled MME dir)')
    ap.add_argument('--clip-model', default='openai/clip-vit-base-patch16')
    ap.add_argument('--clip-batch-size', type=int, default=256)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--skip-eval', action='store_true',
                    help='re-route and re-score using existing per-expert predictions')
    args = ap.parse_args()

    cm, eval_type_dict = _scorer()
    tasks = [t for tlist in eval_type_dict.values() for t in tlist]
    check_data(tasks, args.expert_checkpoints, args.clustering_dir, args.prefix, args.images_dir)

    out_dir = CHAT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    n_experts = len(args.expert_checkpoints)
    print(f'benchmark=mme  metric=acc+acc_plus  experts={n_experts}')
    centroids, mean_vector = load_centroids(args, n_experts)

    tmpl = load_templates(tasks)
    assign, roots = route(tmpl, centroids, mean_vector, args, n_experts, out_dir)

    raw_outs = []
    for k, ckpt in enumerate(args.expert_checkpoints):
        raw_base = out_dir / f'expert{k}_raw'
        if args.skip_eval:
            out = raw_out_dir(raw_base, Path(ckpt) if Path(ckpt).is_absolute() else REPO / ckpt)
            if not out.is_dir():
                raise SystemExit(f'--skip-eval but no predictions at {out}')
            print(f'  --skip-eval: reusing {out}')
        else:
            out = run_expert(ckpt, roots[k], raw_base, args.images_dir, k)
        raw_outs.append(out)

    merged_dir = out_dir / 'merged'
    merge(tmpl, assign, raw_outs, n_experts, merged_dir)

    expert_pts, expert_totals = [], []
    for k in range(n_experts):
        pt, tot = score_dir(raw_outs[k], cm, eval_type_dict)
        expert_pts.append(pt)
        expert_totals.append(tot)
    merged_pt, merged_tot = score_dir(merged_dir, cm, eval_type_dict)

    assert_consistency(merged_pt, expert_pts, n_experts)
    n_merged = sum(d['n_q'] for d in merged_pt.values())
    n_tmpl = sum(len(v) for v in tmpl.values())
    if n_merged != n_tmpl:
        raise SystemExit(f'merged {n_merged} questions, template has {n_tmpl}')

    # -------------------------------------------------------------------- report
    print('\n' + '=' * 74)
    print(f'HARD-ROUTED MME  (acc+acc_plus, {n_experts} experts)')
    print('=' * 74)
    header = f'{"task":<24}' + ''.join(f'{"expert" + str(k):<13}' for k in range(n_experts)) + 'MERGED'
    print('\n' + header)
    for t in tasks:
        if t not in merged_pt:
            continue
        row = f'{t:<24}'
        for k in range(n_experts):
            d = expert_pts[k].get(t)
            cell = '-' if d is None else f'{d["score"]:.2f}'
            row += f'{cell:<13}'
        row += f'{merged_pt[t]["score"]:.2f}'
        print(row)
    print('-' * 74)
    for label in ('Perception', 'Cognition', 'Total'):
        row = f'{label:<24}'
        for k in range(n_experts):
            row += f'{expert_totals[k][label]:<13.2f}'
        row += f'{merged_tot[label]:.2f}'
        print(row)
    print(f'\nMME (merged)  Perception={merged_tot["Perception"]:.2f}  '
          f'Cognition={merged_tot["Cognition"]:.2f}  Total={merged_tot["Total"]:.2f}')

    summary = {
        'benchmark': 'mme', 'metric': 'acc+acc_plus',
        'clustering_dir': args.clustering_dir,
        'expert_checkpoints': list(args.expert_checkpoints),
        'n_questions': n_tmpl,
        'experts': [{'expert': k, 'per_task': expert_pts[k], 'totals': expert_totals[k]}
                    for k in range(n_experts)],
        'merged': {'per_task': merged_pt, 'totals': merged_tot},
    }
    dest = out_dir / 'combined_hardroute_mme.json'
    json.dump(summary, open(dest, 'w'), indent=2)
    print(f'\nsaved -> {dest}')


if __name__ == '__main__':
    main()
