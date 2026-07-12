"""End-to-end hard-routed TextVQA evaluation of cluster experts.

Routes every TextVQA-val question to EXACTLY ONE expert (argmax cosine similarity of the
question's image CLIP feature to that expert's cluster centroid), evaluates each expert on
its own disjoint subset, then merges the predictions and scores the union.

Pipeline:
  1. Read `textvqa_val.jsonl`, collect unique images.
  2. Compute CLIP ViT-B/16 features (`get_image_features`, fp16) -- same model/settings as
     `clustering/extract_clip_features.py`.
  3. Subtract `clustering_global_mean.npy` (if present), L2-normalize features + centroids,
     cosine similarity, argmax -> cluster k. Identical math to
     `clustering/partition_jsonl_by_balanced_kmeans.assign_to_clusters`.
  4. Write `data/textvqa/textvqa_val_hardroute_expert{k}.jsonl` (a question inherits its
     image's expert).
  5. Run `eval/vqa/evaluate_vqa.py` under torchrun once per expert, SEQUENTIALLY, with that
     expert's checkpoint. Never passes --dynamic (see CLAUDE.md).
  6. Merge the per-expert prediction files, assert the union is disjoint and covers the whole
     val set, then score it with the official `TextVQAAccuracyEvaluator`.

Expert k MUST correspond to centroid k (pass --expert-checkpoints in cluster order).

Example:
    python evaluation/eval_hard_routing_textvqa.py \
        --clustering-dir clustering/balanced_kmeans_unique_features_base-patch16_2_clusters_seed42_mean-subtracted \
        --expert-checkpoints \
            work_dirs/internvl2_5_1b/clusters-2_balanced-kmeans_vit-b-16_seed42_mean-subtracted/cluster-0 \
            work_dirs/internvl2_5_1b/clusters-2_balanced-kmeans_vit-b-16_seed42_mean-subtracted/cluster-1
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPImageProcessor, CLIPModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

REPO = Path(__file__).resolve().parent.parent
CHAT = REPO / 'internvl_chat'          # image root + evaluate_vqa.py cwd
MAX_EXPERTS = 4                        # ds_collections has hardroute entries expert0..expert3


class _ImageDataset(Dataset):
    def __init__(self, paths, processor):
        self.paths, self.processor = paths, processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(CHAT / self.paths[i]).convert('RGB')
        return self.processor(images=img, return_tensors='pt')['pixel_values'][0]


def clip_features(paths, model_name, batch_size, num_workers):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading {model_name} on {device} (fp16)...', flush=True)
    processor = CLIPImageProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device).half().eval()

    loader = DataLoader(_ImageDataset(paths, processor), batch_size=batch_size,
                        num_workers=num_workers, pin_memory=True)
    feats = []
    with torch.no_grad():
        for i, px in enumerate(loader, 1):
            f = model.get_image_features(pixel_values=px.to(device).half())
            feats.append(f.float().cpu().numpy())
            if i % 5 == 0 or i == len(loader):
                print(f'  CLIP batch {i}/{len(loader)}', flush=True)
    del model
    torch.cuda.empty_cache()
    return np.concatenate(feats, 0)


def assign_to_clusters(features, centroids, mean_vector=None):
    """Same math as clustering/partition_jsonl_by_balanced_kmeans.assign_to_clusters."""
    if mean_vector is not None:
        features = features - mean_vector
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    return np.dot(features, centroids.T).argmax(1)


def route(args, n_experts):
    val_rows = [json.loads(l) for l in open(CHAT / args.val_jsonl)]
    unique_images = sorted({r['image'] for r in val_rows})
    print(f'{len(val_rows)} questions over {len(unique_images)} unique images')

    clustering_dir = Path(args.clustering_dir)
    if not clustering_dir.is_absolute():
        clustering_dir = REPO / clustering_dir
    centroids = np.load(clustering_dir / f'{args.prefix}_centroids.npy')
    if centroids.shape[0] != n_experts:
        raise SystemExit(f'{centroids.shape[0]} centroids but {n_experts} expert checkpoints '
                         '-- expert k must align with centroid k')

    mean_file = clustering_dir / f'{args.prefix}_global_mean.npy'
    mean_vector = np.load(mean_file) if mean_file.exists() else None
    print(f'centroids {centroids.shape}; global mean: '
          + (f'||mean||={np.linalg.norm(mean_vector):.4f} (subtracting)' if mean_vector is not None
             else 'none (older non-centered run)'))

    feats = clip_features(unique_images, args.clip_model, args.clip_batch_size, args.num_workers)
    assignments = assign_to_clusters(feats, centroids, mean_vector)
    image2expert = dict(zip(unique_images, assignments.tolist()))

    buckets = {k: [] for k in range(n_experts)}
    for r in val_rows:
        buckets[image2expert[r['image']]].append(r)

    ds_names, out_paths = [], []
    for k in range(n_experts):
        p = CHAT / 'data/textvqa' / f'textvqa_val_hardroute_expert{k}.jsonl'
        with open(p, 'w') as fh:
            for r in buckets[k]:
                fh.write(json.dumps(r) + '\n')
        ds_names.append(f'textvqa_val_hardroute_expert{k}')
        out_paths.append(p)
        print(f'  expert{k}: {len(buckets[k])} questions '
              f'({100 * len(buckets[k]) / len(val_rows):.1f}%) -> {p.name}')

    assert sum(len(v) for v in buckets.values()) == len(val_rows)
    json.dump({str(r['question_id']): int(image2expert[r['image']]) for r in val_rows},
              open(CHAT / 'data/textvqa/textvqa_val_hardroute_map.json', 'w'))
    return ds_names, len(val_rows)


def run_eval(ds_name, checkpoint, args):
    """Invoke the official evaluator under torchrun, sequentially. Never passes --dynamic."""
    ckpt = Path(checkpoint)
    if not ckpt.is_absolute():
        ckpt = REPO / ckpt
    if not (ckpt / 'config.json').is_file():
        raise SystemExit(f'no config.json in checkpoint {ckpt}')

    env = dict(os.environ)
    env['PYTHONPATH'] = f"{CHAT}:{env.get('PYTHONPATH', '')}"
    env['MASTER_PORT'] = str(args.master_port)
    cmd = [
        'torchrun', '--nnodes=1', '--node_rank=0', '--master_addr=127.0.0.1',
        f'--nproc_per_node={args.gpus}', f'--master_port={args.master_port}',
        'eval/vqa/evaluate_vqa.py',
        '--checkpoint', str(ckpt),
        '--datasets', ds_name,
        '--out-dir', args.out_dir,
        '--num-workers', str(args.num_workers),
    ]
    print(f'\n===== evaluating {ds_name} with {ckpt.name} =====', flush=True)
    r = subprocess.run(cmd, cwd=CHAT, env=env)
    if r.returncode != 0:
        raise SystemExit(f'evaluate_vqa.py failed for {ds_name} (exit {r.returncode})')


def combine(ds_names, n_questions, args):
    sys.path.insert(0, str(CHAT / 'eval/vqa'))
    from textvqa_eval import TextVQAAccuracyEvaluator

    out_dir = CHAT / args.out_dir
    per_expert = []
    for ds in ds_names:
        matches = sorted(glob.glob(str(out_dir / f'{ds}_*.json')))
        if not matches:
            raise SystemExit(f'no results file for {ds} in {out_dir}')
        per_expert.append(json.load(open(matches[-1])))  # latest run

    merged = [x for preds in per_expert for x in preds]
    ids = [x['question_id'] for x in merged]
    if len(set(ids)) != len(ids):
        raise SystemExit('routing is not disjoint: duplicate question_ids across experts')
    if len(merged) != n_questions:
        raise SystemExit(f'merged {len(merged)} predictions but val set has {n_questions}')

    annotation = json.load(open(CHAT / args.annotation))['annotations']
    qid2gt = {a['question_id']: [x['answer'] for x in a['answers']] for a in annotation}

    evaluator = TextVQAAccuracyEvaluator()

    def score(preds):
        rows = [dict(p) for p in preds]
        for r in rows:
            r['pred_answer'] = r['answer']
            r['gt_answers'] = qid2gt[r['question_id']]
        return evaluator.eval_pred_list(rows)

    accs = [score(p) for p in per_expert]
    combined = score(merged)

    print('\n' + '=' * 72)
    print(f'HARD-ROUTED TEXTVQA  ({len(ds_names)} experts, {n_questions} questions)')
    print('=' * 72)
    for k, (preds, acc) in enumerate(zip(per_expert, accs)):
        n = len(preds)
        print(f'  expert-{k}: n={n:5d} ({100 * n / n_questions:5.1f}%)   acc={100 * acc:6.2f}%')
    print('-' * 72)
    print(f'  COMBINED (union, disjoint & complete)     : {100 * combined:6.2f}%')

    summary = {
        'n_questions': n_questions,
        'experts': [{'expert': k, 'n': len(p), 'accuracy': a}
                    for k, (p, a) in enumerate(zip(per_expert, accs))],
        'combined_accuracy': combined,
    }
    dest = out_dir / 'combined_hardroute_textvqa.json'
    json.dump(summary, open(dest, 'w'), indent=2)
    print(f'\nsaved -> {dest}')
    return combined


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--clustering-dir', required=True,
                    help='clustering-results dir with <prefix>_centroids.npy (+ _global_mean.npy)')
    ap.add_argument('--expert-checkpoints', nargs='+', required=True,
                    help='expert checkpoint dirs IN CLUSTER ORDER (expert k <-> centroid k)')
    ap.add_argument('--prefix', default='clustering')
    ap.add_argument('--val-jsonl', default='data/textvqa/textvqa_val.jsonl')
    ap.add_argument('--annotation', default='data/textvqa/textvqa_val_annotations.json')
    ap.add_argument('--out-dir', default='results/hardroute',
                    help='relative to internvl_chat/')
    ap.add_argument('--clip-model', default='openai/clip-vit-base-patch16')
    ap.add_argument('--clip-batch-size', type=int, default=256)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--gpus', type=int, default=1)
    ap.add_argument('--master-port', type=int, default=63669)
    ap.add_argument('--skip-routing', action='store_true',
                    help='reuse existing textvqa_val_hardroute_expert*.jsonl')
    ap.add_argument('--skip-eval', action='store_true',
                    help='only merge/score existing prediction files')
    args = ap.parse_args()

    n_experts = len(args.expert_checkpoints)
    if n_experts > MAX_EXPERTS:
        raise SystemExit(f'only {MAX_EXPERTS} hardroute datasets registered in evaluate_vqa.py')

    (CHAT / args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.skip_routing:
        ds_names = [f'textvqa_val_hardroute_expert{k}' for k in range(n_experts)]
        n_questions = sum(1 for _ in open(CHAT / args.val_jsonl))
        print('skipping routing; reusing existing routed jsonls')
    else:
        ds_names, n_questions = route(args, n_experts)

    if not args.skip_eval:
        for ds, ckpt in zip(ds_names, args.expert_checkpoints):
            run_eval(ds, ckpt, args)

    combine(ds_names, n_questions, args)


if __name__ == '__main__':
    main()
