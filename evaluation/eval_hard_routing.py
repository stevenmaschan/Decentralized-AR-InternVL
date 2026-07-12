"""End-to-end HARD-ROUTED evaluation of cluster experts across benchmarks.

Every test question is routed to EXACTLY ONE expert -- argmax cosine similarity between the
question's image CLIP feature and that expert's cluster centroid -- each expert is evaluated
on its own disjoint subset (sequentially), and the predictions are merged and scored as one.

Supported benchmarks:
    vqav2, gqa, ai2d, chartqa, docvqa, infovqa, textvqa, scienceqa, refcoco, pope, mme
    (refcoco = 8 RefCOCO/+/g splits, headline = mean; pope = Overall F1; mme = perception+cognition.
     pope/mme scores are aggregates, not per-sample means, so they are union-only like gqa.)

Routing (identical math to clustering/partition_jsonl_by_balanced_kmeans.assign_to_clusters):
    feature = CLIP ViT-B/16 get_image_features(image)   # fp16, same as extract_clip_features.py
    feature -= clustering_global_mean.npy               # if present (mean-subtracted clusterings)
    cluster = argmax_k  cos(normalize(feature), normalize(centroid_k))
A question inherits its image's expert. Expert k MUST align with centroid k, so pass
--expert-checkpoints in cluster order; the script hard-fails if the counts disagree.

Scoring: the upstream evaluator is used ONLY as a prediction generator (via its --test-file
override). All metrics are recomputed here -- per expert AND on the merged union -- reusing the
repo's own scorers, so a subset can never be silently scored against the full ground truth.
Every supported metric is a mean over samples, so the union score must equal the size-weighted
mean of the per-expert scores; the script asserts this as a consistency check.

Never passes --dynamic (see CLAUDE.md).

Example:
    python evaluation/eval_hard_routing.py --benchmark textvqa \
        --clustering-dir clustering/balanced_kmeans_unique_features_base-patch16_2_clusters_seed42_mean-subtracted \
        --expert-checkpoints \
            work_dirs/internvl2_5_1b/clusters-2_..._mean-subtracted/cluster-0 \
            work_dirs/internvl2_5_1b/clusters-2_..._mean-subtracted/cluster-1
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPImageProcessor, CLIPModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

REPO = Path(__file__).resolve().parent.parent
CHAT = REPO / 'internvl_chat'     # image root, and cwd for the upstream evaluators


@dataclass
class Benchmark:
    ds_name: str                       # ds_collections key in the upstream evaluator
    test: str                          # test jsonl (relative to internvl_chat/)
    metric: str                        # vqa_score | anls | relaxed_accuracy | exact_match | gqa |
                                       # sqa | refcoco_p1 | pope_f1
    runner: str = 'vqa'                # 'vqa'|'sqa'|'pope'|'refcoco' -> the RUNNERS mapping
    annotation: Optional[str] = None   # ground truth, for metrics that need it
    extra_tests: List[str] = field(default_factory=list)  # extra (ds_name, test) pairs, e.g. chartqa
    needs: List[str] = field(default_factory=list)        # extra files that must exist
    image_root: Optional[str] = None   # prefix (rel to internvl_chat/) joined to each row's image
                                       # for CLIP routing when the jsonl stores a bare basename (POPE)
    has_ids: bool = True               # False when predictions carry no stable per-question id
                                       # (RefCOCO): disjointness is guaranteed by construction instead


# ChartQA's headline number is the mean of the human and augmented splits, so it is
# modelled as two sub-datasets that are routed and evaluated independently.
BENCHMARKS = {
    'textvqa': Benchmark(
        ds_name='textvqa_val', test='data/textvqa/textvqa_val.jsonl', metric='vqa_score',
        annotation='data/textvqa/textvqa_val_annotations.json'),
    'vqav2': Benchmark(
        ds_name='vqav2_val', test='data/vqav2/vqav2_val.jsonl', metric='vqa_score',
        annotation='data/vqav2/v2_mscoco_val2014_annotations.json'),
    'gqa': Benchmark(
        ds_name='gqa_testdev_llava',
        test='data/gqa/llava_gqa_testdev_balanced_qwen_format.jsonl',
        metric='gqa',
        # GQA's official eval.py (from eval.zip) loads all three of these for --tier
        # testdev_balanced; eval.zip itself only ships eval.py + val/train choices.
        needs=['data/gqa/eval.py',
               'data/gqa/testdev_balanced_sceneGraphs.json',
               'data/gqa/testdev_balanced_all_questions.json',
               'data/gqa/testdev_balanced_choices.json']),
    'ai2d': Benchmark(
        ds_name='ai2diagram_test', test='data/ai2diagram/test_vlmevalkit.jsonl',
        metric='exact_match'),
    # ChartQA predictions carry questionId=-1 for every row (no real ids ship with the split),
    # so has_ids=False -> disjointness is proven by count instead (routing buckets each row once).
    'chartqa': Benchmark(
        ds_name='chartqa_test_human', test='data/chartqa/test_human.jsonl',
        metric='relaxed_accuracy', has_ids=False,
        extra_tests=[('chartqa_test_augmented', 'data/chartqa/test_augmented.jsonl')]),
    'docvqa': Benchmark(
        ds_name='docvqa_val', test='data/docvqa/val.jsonl', metric='anls',
        annotation='data/docvqa/val/val_v1.0.json'),
    'infovqa': Benchmark(
        ds_name='infographicsvqa_val', test='data/infographicsvqa/val.jsonl', metric='anls',
        annotation='data/infographicsvqa/infographicsVQA_val_v1.0_withQT.json'),
    'scienceqa': Benchmark(
        ds_name='sqa_test', test='data/scienceqa/scienceqa_test_img.jsonl',
        metric='sqa', runner='sqa'),
    # RefCOCO/+/g: 8 splits routed and scored independently; headline = mean of the 8. P@1
    # (IoU@0.5) is a per-sample mean, so the union==weighted-mean check applies. Predictions
    # carry no question id (only answer/gt_bbox/hw), so has_ids=False.
    'refcoco': Benchmark(
        ds_name='refcoco_val', test='data/refcoco/refcoco_val.jsonl', metric='refcoco_p1',
        runner='refcoco', has_ids=False,
        extra_tests=[('refcoco_testA', 'data/refcoco/refcoco_testA.jsonl'),
                     ('refcoco_testB', 'data/refcoco/refcoco_testB.jsonl'),
                     ('refcoco+_val', 'data/refcoco/refcoco+_val.jsonl'),
                     ('refcoco+_testA', 'data/refcoco/refcoco+_testA.jsonl'),
                     ('refcoco+_testB', 'data/refcoco/refcoco+_testB.jsonl'),
                     ('refcocog_val', 'data/refcoco/refcocog_val.jsonl'),
                     ('refcocog_test', 'data/refcoco/refcocog_test.jsonl')]),
    # POPE: Overall F1 over 3 categories (mean of per-category F1). F1 is NOT a per-sample mean,
    # so it is a union-only metric (per-expert = n/a, like GQA). The test jsonl stores a bare
    # image basename, so image_root is joined for CLIP routing only.
    'pope': Benchmark(
        ds_name='pope', test='data/pope/llava_pope_test.jsonl', metric='pope_f1',
        runner='pope', image_root='data/pope/val2014',
        needs=['eval/pope/eval_pope.py', 'data/pope/coco/coco_pope_random.json',
               'data/pope/coco/coco_pope_popular.json', 'data/pope/coco/coco_pope_adversarial.json']),
}

RUNNERS = {'vqa': 'eval/vqa/evaluate_vqa.py', 'sqa': 'eval/scienceqa/evaluate_scienceqa.py',
           'pope': 'eval/pope/evaluate_pope.py', 'refcoco': 'eval/refcoco/evaluate_grounding.py'}

# Benchmarks whose predictions nest under a per-model subdir, and the prediction file extension.
NESTED_RUNNERS = {'sqa', 'pope'}

# MME is handled by a dedicated path (evaluate_mme): its "test set" is 14 category txt files, not
# a jsonl, and its score (perception+cognition) is not a per-sample mean. Listed here only so
# --benchmark mme is accepted; it never goes through the generic BENCHMARKS/route_split flow.
MME_IMAGES = 'data/mme/MME_Benchmark_release_version'
MME_QUESTIONS = 'eval/mme/Your_Results'


# --------------------------------------------------------------------------- routing
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
    print(f'  CLIP: {model_name} on {device} (fp16), {len(paths)} images', flush=True)
    processor = CLIPImageProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device).half().eval()
    loader = DataLoader(_ImageDataset(paths, processor), batch_size=batch_size,
                        num_workers=num_workers, pin_memory=True)
    feats = []
    with torch.no_grad():
        for i, px in enumerate(loader, 1):
            feats.append(model.get_image_features(pixel_values=px.to(device).half())
                         .float().cpu().numpy())
            if i % 20 == 0 or i == len(loader):
                print(f'    batch {i}/{len(loader)}', flush=True)
    del model
    torch.cuda.empty_cache()
    return np.concatenate(feats, 0)


def assign_to_clusters(features, centroids, mean_vector=None):
    """Identical to clustering/partition_jsonl_by_balanced_kmeans.assign_to_clusters."""
    if mean_vector is not None:
        features = features - mean_vector
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    return np.dot(features, centroids.T).argmax(1)


def load_centroids(args, n_experts):
    d = Path(args.clustering_dir)
    if not d.is_absolute():
        d = REPO / d
    centroids = np.load(d / f'{args.prefix}_centroids.npy')
    if centroids.shape[0] != n_experts:
        raise SystemExit(f'{centroids.shape[0]} centroids but {n_experts} expert checkpoints; '
                         'expert k must align with centroid k')
    mean_file = d / f'{args.prefix}_global_mean.npy'
    mean_vector = np.load(mean_file) if mean_file.exists() else None
    print(f'  centroids {centroids.shape}; global mean: '
          + (f'||mean||={np.linalg.norm(mean_vector):.4f} (subtracting)'
             if mean_vector is not None else 'none (older non-centered clustering)'))
    return centroids, mean_vector


def route_split(test_rel, centroids, mean_vector, n_experts, args, tag, image_root=None):
    """Route one test jsonl into n_experts disjoint subsets. Returns list of subset paths.

    `image_root` (rel to internvl_chat/) is joined to each row's image ONLY to locate the file
    for CLIP feature extraction; the rows written to the subsets keep their original image value
    so the upstream runner (which applies its own root) is unaffected.
    """
    rows = [json.loads(l) for l in open(CHAT / test_rel)]
    images = sorted({r[args.image_key] for r in rows})
    print(f'  {tag}: {len(rows)} questions over {len(images)} unique images')

    paths_for_clip = [str(Path(image_root) / im) if image_root else im for im in images]
    feats = clip_features(paths_for_clip, args.clip_model, args.clip_batch_size, args.num_workers)
    image2expert = dict(zip(images, assign_to_clusters(feats, centroids, mean_vector).tolist()))

    buckets = {k: [] for k in range(n_experts)}
    for r in rows:
        buckets[image2expert[r[args.image_key]]].append(r)

    src = Path(test_rel)
    paths = []
    for k in range(n_experts):
        p = CHAT / src.parent / f'{src.stem}_hardroute_expert{k}.jsonl'
        with open(p, 'w') as fh:
            for r in buckets[k]:
                fh.write(json.dumps(r) + '\n')
        paths.append(p)
        print(f'    expert{k}: {len(buckets[k])} ({100 * len(buckets[k]) / len(rows):.1f}%)')
    assert sum(len(v) for v in buckets.values()) == len(rows)
    return paths, len(rows)


# --------------------------------------------------------------------------- inference
def run_expert(runner, ds_name, test_file, checkpoint, out_dir, args, tag):
    ckpt = Path(checkpoint)
    if not ckpt.is_absolute():
        ckpt = REPO / ckpt
    if not (ckpt / 'config.json').is_file():
        raise SystemExit(f'no config.json in checkpoint {ckpt}')

    env = dict(os.environ)
    env['PYTHONPATH'] = f"{CHAT}:{env.get('PYTHONPATH', '')}"
    cmd = ['torchrun', '--nnodes=1', '--node_rank=0', '--master_addr=127.0.0.1',
           f'--nproc_per_node={args.gpus}', f'--master_port={args.master_port}',
           RUNNERS[runner],
           '--checkpoint', str(ckpt),
           '--datasets', ds_name,
           '--test-file', str(Path(test_file).relative_to(CHAT)),
           '--out-dir', out_dir,
           '--num-workers', str(args.num_workers)]
    # evaluate_vqa.py / evaluate_pope.py / evaluate_grounding.py support batched generation
    # (batch_chat); evaluate_scienceqa.py does not and asserts batch size 1. Forward --batch-size
    # to every batched runner so the experts match the dense baseline's batch size (CLAUDE.md:
    # never mix batched and batch-1 predictions in a dense-vs-expert comparison).
    if runner != 'sqa':
        cmd += ['--batch-size', str(args.batch_size)]
    print(f'\n===== {tag}: {ds_name} @ {ckpt.name} =====', flush=True)
    if subprocess.run(cmd, cwd=CHAT, env=env).returncode != 0:
        raise SystemExit(f'{RUNNERS[runner]} failed for {tag}')


def stable_path(out_dir, ds_name, expert, runner):
    """Deterministic per-expert predictions path, so --skip-eval can find them again."""
    ext = 'jsonl' if runner == 'sqa' else 'json'
    return CHAT / out_dir / f'{ds_name}_hardroute_expert{expert}.{ext}'


def harvest(out_dir, ds_name, runner, checkpoint, expert):
    """Move the file the upstream evaluator just wrote to its deterministic per-expert path.

    Both experts run the same ds_name, so upstream writes `{ds_name}_{timestamp}` twice;
    without this the second harvest would re-read the first expert's predictions.
    """
    base = CHAT / out_dir
    if runner in NESTED_RUNNERS:  # scienceqa/pope nest results under a per-model subdir
        base = base / '_'.join(str(Path(checkpoint)).split('/')[-2:])
    ext = 'jsonl' if runner == 'sqa' else 'json'
    pattern = f'{ds_name}_*.{ext}'
    hits = [h for h in glob.glob(str(base / pattern)) if 'hardroute_expert' not in h]
    if not hits:
        raise SystemExit(f'no predictions found: {base / pattern}')
    newest = max(hits, key=os.path.getmtime)
    dest = stable_path(out_dir, ds_name, expert, runner)
    shutil.move(newest, dest)
    return dest


def load_preds(path):
    if path.endswith('.jsonl'):
        return [json.loads(l) for l in open(path)]
    return json.load(open(path))


def pred_id(row):
    """Stable per-question identity, for disjointness/completeness checks."""
    for k in ('question_id', 'questionId'):
        if k in row:
            return row[k]
    if 'image_path' in row:                       # scienceqa
        return (row['image_path'], row['question'])
    return (row.get('image'), row.get('question'))  # ai2d


# --------------------------------------------------------------------------- scoring
def _vqa_scorers():
    """Reuse the repo's own scorers so this script can never drift from the official metric."""
    for p in (str(CHAT), str(CHAT / 'eval/vqa')):  # CHAT: the `internvl` package evaluate_vqa imports
        if p not in sys.path:
            sys.path.insert(0, p)
    from evaluate_vqa import (evaluate_exact_match_accuracy,  # noqa: E402
                              evaluate_relaxed_accuracy)
    from textvqa_eval import TextVQAAccuracyEvaluator  # noqa: E402
    return TextVQAAccuracyEvaluator, evaluate_relaxed_accuracy, evaluate_exact_match_accuracy


def score(preds, bm, args, is_union):
    """Recompute the benchmark metric on an arbitrary subset of predictions."""
    if not preds:
        return None

    if bm.metric == 'vqa_score':
        TextVQAAccuracyEvaluator, _, _ = _vqa_scorers()
        ann = json.load(open(CHAT / bm.annotation))['annotations']
        qid2gt = {a['question_id']: [x['answer'] for x in a['answers']] for a in ann}
        rows = [dict(p) for p in preds]
        for r in rows:
            r['pred_answer'] = r['answer']
            r['gt_answers'] = qid2gt[r['question_id']]
        return TextVQAAccuracyEvaluator().eval_pred_list(rows)

    if bm.metric == 'relaxed_accuracy':
        _, evaluate_relaxed_accuracy, _ = _vqa_scorers()
        return evaluate_relaxed_accuracy([dict(p) for p in preds])

    if bm.metric == 'exact_match':
        _, _, evaluate_exact_match_accuracy = _vqa_scorers()
        return evaluate_exact_match_accuracy([dict(p) for p in preds])

    if bm.metric == 'sqa':
        return sum(p['answer'] == p['gt_answers'] for p in preds) / len(preds)

    if bm.metric == 'anls':
        return _score_anls(preds, bm)

    if bm.metric == 'gqa':
        if not is_union:
            return None  # GQA's harness scores against its full question set; subsets are N/A
        return _score_gqa(preds)

    if bm.metric == 'refcoco_p1':
        return _score_refcoco_p1(preds)

    if bm.metric == 'pope_f1':
        # F1 is an aggregate (not a per-sample mean); only the merged union is meaningful.
        return _score_pope_f1(preds, bm) if is_union else None

    raise SystemExit(f'unknown metric {bm.metric}')


def _score_anls(preds, bm):
    """Run the repo's ANLS scorer, filtering the GT to exactly the questions in `preds`."""
    gt = json.load(open(CHAT / bm.annotation))
    ids = {int(pred_id(p)) for p in preds}
    sub = {'dataset_name': gt.get('dataset_name', 'docvqa'),
           'data': [x for x in gt['data'] if int(x['questionId']) in ids]}
    if len(sub['data']) != len(ids):
        raise SystemExit(f'ANLS ground truth covers {len(sub["data"])} of {len(ids)} predictions')
    with tempfile.TemporaryDirectory() as td:
        g, s = Path(td) / 'gt.json', Path(td) / 'pred.json'
        json.dump(sub, open(g, 'w'))
        json.dump([dict(p) for p in preds], open(s, 'w'), ensure_ascii=False)
        out = subprocess.run([sys.executable, 'eval/vqa/infographicsvqa_eval.py',
                              '-g', str(g), '-s', str(s)],
                             cwd=CHAT, capture_output=True, text=True)
        for line in (out.stdout + out.stderr).splitlines():
            if 'ANLS' in line.upper():
                for tok in line.replace('=', ' ').replace(':', ' ').split():
                    try:
                        return float(tok)
                    except ValueError:
                        continue
        raise SystemExit(f'could not parse ANLS from scorer output:\n{out.stdout}\n{out.stderr}')


def _score_gqa(preds):
    dst = CHAT / 'data/gqa/testdev_balanced_predictions.json'
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / 'merged.json'
        json.dump([dict(p) for p in preds], open(src, 'w'))
        subprocess.run([sys.executable, 'eval/vqa/convert_gqa_for_eval.py',
                        '--src', str(src), '--dst', str(dst)], cwd=CHAT, check=True)
    out = subprocess.run([sys.executable, 'eval.py', '--tier', 'testdev_balanced'],
                         cwd=CHAT / 'data/gqa', capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if 'Accuracy' in line:
            for tok in line.replace('%', ' ').split():
                try:
                    return float(tok) / 100.0
                except ValueError:
                    continue
    raise SystemExit(f'could not parse GQA accuracy:\n{out.stdout}\n{out.stderr}')


def _box_iou(boxes1, boxes2):
    """Identical to eval/refcoco/evaluate_grounding.box_iou (copied to avoid importing that module,
    which pulls in the whole internvl model stack just for this 10-line function)."""
    from torchvision.ops.boxes import box_area
    area1, area2 = box_area(boxes1), box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union, union


def _score_refcoco_p1(preds):
    """Precision@1 (IoU@0.5) -- reimplements eval/refcoco/evaluate_grounding.py's scoring loop
    exactly (same box_iou, same 1000-scale + hw denorm), so it cannot drift from the official
    metric. A per-sample mean, so the union==weighted-mean check applies."""
    import re
    box_iou = _box_iou
    pattern = re.compile(r'\[*\[(.*?),(.*?),(.*?),(.*?)\]\]*')
    correct = total = 0
    for output in preds:
        m = re.findall(pattern, output['answer'])
        try:
            predict_bbox = (float(m[0][0]), float(m[0][1]), float(m[0][2]), float(m[0][3]))
        except Exception:
            predict_bbox = (0., 0., 0., 0.)
        target_bbox = torch.tensor(output['gt_bbox'], dtype=torch.float32).view(-1, 4)
        predict_bbox = torch.tensor(predict_bbox, dtype=torch.float32).view(-1, 4)
        if predict_bbox.sum() >= 4:
            predict_bbox = predict_bbox / 1000
        predict_bbox[:, 0::2] *= output['hw'][1]
        predict_bbox[:, 1::2] *= output['hw'][0]
        iou, _ = box_iou(predict_bbox, target_bbox)
        total += 1
        if iou.item() >= 0.5:
            correct += 1
    return correct / total


def _score_pope_f1(preds, bm):
    """Overall F1 via the repo's own eval/pope/eval_pope.py, on the merged predictions. Uses the
    full test jsonl as the question file (union covers every question_id) and the 3 coco_pope_*
    label files. F1 is an aggregate, so this is only computed on the union."""
    with tempfile.TemporaryDirectory() as td:
        s = Path(td) / 'pred.json'
        json.dump([dict(p) for p in preds], open(s, 'w'), ensure_ascii=False)
        out = subprocess.run([sys.executable, 'eval/pope/eval_pope.py',
                              '--annotation-dir', './data/pope/coco',
                              '--question-file', bm.test,
                              '--result-file', str(s)],
                             cwd=CHAT, capture_output=True, text=True)
    for line in (out.stdout + out.stderr).splitlines():
        if 'Overall F1' in line:
            for tok in line.replace(':', ' ').split():
                try:
                    return float(tok) / 100.0
                except ValueError:
                    continue
    raise SystemExit(f'could not parse POPE F1:\n{out.stdout}\n{out.stderr}')


# --------------------------------------------------------------------------- driver
def check_data(bm, image_key):
    """Preflight: fail before loading a model if any test/GT file or the images are absent."""
    tests = [bm.test, *[t for _, t in bm.extra_tests]]
    missing = [p for p in [*tests, bm.annotation, *bm.needs]
               if p and not (CHAT / p).exists()]

    # Images are referenced from inside the jsonls, so a present jsonl says nothing about them.
    root = Path(bm.image_root) if bm.image_root else None
    for t in tests:
        if (CHAT / t).exists():
            with open(CHAT / t) as fh:
                first = json.loads(fh.readline())
            img = first.get(image_key)
            if img and not (CHAT / (str(root / img) if root else img)).is_file():
                missing.append(f'{img}  (images for {t})')

    if missing:
        raise SystemExit(
            'eval data missing for this benchmark:\n'
            + '\n'.join(f'  - internvl_chat/{m}' for m in missing)
            + '\n\nSee internvl_chat/eval/vqa/README.md (and eval/scienceqa/README.md) for the '
              'download instructions for this benchmark.')


def evaluate_split(ds_name, test_rel, bm, centroids, mean_vector, args, n_experts):
    subsets, n_questions = route_split(test_rel, centroids, mean_vector, n_experts, args, ds_name,
                                       image_root=bm.image_root)
    if args.skip_eval:
        print('  --skip-eval: reusing existing predictions')
    per_expert = []
    for k, (sub, ckpt) in enumerate(zip(subsets, args.expert_checkpoints)):
        # Degenerate routing: an expert can receive zero questions (e.g. every image is closest
        # to one centroid). The upstream evaluator asserts a non-empty dataset, so skip it and
        # treat this expert's predictions as empty; the merge below is unaffected.
        if sum(1 for _ in open(sub)) == 0:
            print(f'  expert-{k}: empty subset -- no questions routed here, skipping inference')
            per_expert.append([])
            continue
        if args.skip_eval:
            path = stable_path(args.out_dir, ds_name, k, bm.runner)
            if not path.exists():
                raise SystemExit(f'--skip-eval but no predictions at {path}')
        else:
            run_expert(bm.runner, ds_name, sub, ckpt, args.out_dir, args,
                       tag=f'{ds_name} expert-{k}')
            path = harvest(args.out_dir, ds_name, bm.runner, ckpt, k)
        per_expert.append(load_preds(str(path)))

    merged = [p for preds in per_expert for p in preds]
    if bm.has_ids:
        ids = [pred_id(p) for p in merged]
        if len(set(ids)) != len(ids):
            raise SystemExit(f'{ds_name}: routing not disjoint -- duplicate question ids')
    # RefCOCO predictions carry no id, but routing buckets each row exactly once (all referring
    # expressions of an image share its expert), so a matching total count proves disjoint cover.
    if len(merged) != n_questions:
        raise SystemExit(f'{ds_name}: merged {len(merged)} predictions, expected {n_questions}')

    accs = [score(p, bm, args, is_union=False) for p in per_expert]
    combined = score(merged, bm, args, is_union=True)

    # every per-sample-mean metric satisfies union == size-weighted mean. ANLS is parsed from the
    # scorer's 4-decimal printed output, so per-expert vs union rounding can differ by ~1e-4;
    # give it a looser tolerance while keeping the exact-computed metrics tight (catches leaks).
    if combined is not None and all(a is not None for a in accs):
        weighted = sum(len(p) * a for p, a in zip(per_expert, accs)) / n_questions
        tol = 2e-3 if bm.metric == 'anls' else 1e-6
        if abs(weighted - combined) > tol:
            raise SystemExit(f'{ds_name}: union score {combined:.6f} != weighted mean '
                             f'{weighted:.6f} -- routing or scoring is inconsistent')
    return {'ds_name': ds_name, 'n_questions': n_questions,
            'experts': [{'expert': k, 'n': len(p), 'accuracy': a}
                        for k, (p, a) in enumerate(zip(per_expert, accs))],
            'combined_accuracy': combined}


# --------------------------------------------------------------------------- MME (bespoke)
def _read_mme_questions():
    """{category: [(img, question, gt, raw_line)]} from eval/mme/Your_Results/*.txt.

    MME stores two questions per image on adjacent lines; calculation.py relies on that
    adjacency (it chunks each category file into consecutive pairs)."""
    cats = {}
    for f in sorted((CHAT / MME_QUESTIONS).glob('*.txt')):
        rows = []
        for line in open(f):
            if not line.strip():
                continue
            img, question, gt = line.rstrip('\n').split('\t')
            rows.append((img, question, gt, line if line.endswith('\n') else line + '\n'))
        cats[f.stem] = rows
    return cats


def run_mme(args):
    """Hard-routed MME. Routes each (category, image) to one expert, runs eval/mme/eval.py per
    expert on its subset, merges the per-expert prediction dirs (block concat keeps question pairs
    adjacent), and scores the union with eval/mme/calculation.py. Perception+Cognition are
    aggregate scores (not per-sample means), so only the union is reported."""
    out_dir = args.out_dir or 'results/hardroute/mme'
    (CHAT / out_dir).mkdir(parents=True, exist_ok=True)
    for need in [MME_QUESTIONS, MME_IMAGES, 'eval/mme/eval.py', 'eval/mme/calculation.py']:
        if not (CHAT / need).exists():
            raise SystemExit(f'MME data missing: internvl_chat/{need}')

    n_experts = len(args.expert_checkpoints)
    print(f'benchmark=mme  metric=perception+cognition  experts={n_experts}')
    centroids, mean_vector = load_centroids(args, n_experts)

    cats = _read_mme_questions()
    keys = sorted({(cat, img) for cat, rows in cats.items() for img, *_ in rows})
    rel_paths = [str(Path(MME_IMAGES) / cat / img) for cat, img in keys]
    total = sum(len(v) for v in cats.values())
    print(f'  MME: {total} questions over {len(keys)} unique images across {len(cats)} categories')
    feats = clip_features(rel_paths, args.clip_model, args.clip_batch_size, args.num_workers)
    key2expert = dict(zip(keys, assign_to_clusters(feats, centroids, mean_vector).tolist()))

    work = CHAT / out_dir
    roots, counts = [], [0] * n_experts
    for k in range(n_experts):
        root = work / f'expert{k}_root'
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        for cat, rows in cats.items():
            sel = [raw for (img, q, gt, raw) in rows if key2expert[(cat, img)] == k]
            counts[k] += len(sel)
            open(root / f'{cat}.txt', 'w').writelines(sel)
        roots.append(root)
        print(f'    expert{k}: {counts[k]} ({100 * counts[k] / total:.1f}%)')
    assert sum(counts) == total, 'MME routing dropped or duplicated questions'

    env = dict(os.environ)
    env['PYTHONPATH'] = f"{CHAT}:{env.get('PYTHONPATH', '')}"
    mme_dir = CHAT / 'eval/mme'
    pred_dirs = []
    for k, ckpt in enumerate(args.expert_checkpoints):
        ckptp = Path(ckpt) if Path(ckpt).is_absolute() else REPO / ckpt
        pred_dir = mme_dir / os.path.basename(str(ckptp))  # eval.py writes to basename(checkpoint)
        pred_dirs.append(pred_dir)
        if counts[k] == 0:
            print(f'  expert-{k}: empty subset -- skipping inference')
            continue
        if args.skip_eval and pred_dir.exists():
            print(f'  expert-{k}: --skip-eval, reusing {pred_dir}')
            continue
        print(f'\n===== mme expert-{k} @ {ckptp.name} =====', flush=True)
        cmd = [sys.executable, 'eval.py', '--checkpoint', str(ckptp),
               '--root', str(roots[k]), '--images-dir', str(CHAT / MME_IMAGES)]
        if subprocess.run(cmd, cwd=mme_dir, env=env).returncode != 0:
            raise SystemExit(f'eval/mme/eval.py failed for expert-{k}')

    # Merge per-expert predictions: per category, concat expert blocks (each even-length, so the
    # question pairs calculation.py expects stay adjacent).
    merged = work / 'merged_predictions'
    if merged.exists():
        shutil.rmtree(merged)
    merged.mkdir(parents=True)
    merged_total = 0
    for cat in cats:
        lines = []
        for k in range(n_experts):
            f = pred_dirs[k] / f'{cat}.txt'
            if f.exists():
                lines += open(f).readlines()
        open(merged / f'{cat}.txt', 'w').writelines(lines)
        merged_total += len(lines)
    if merged_total != total:
        raise SystemExit(f'MME merged {merged_total} predictions, expected {total}')

    out = subprocess.run([sys.executable, 'calculation.py', '--results_dir', str(merged)],
                         cwd=mme_dir, capture_output=True, text=True)
    print(out.stdout)
    section, totals = None, {}
    for line in out.stdout.splitlines():
        if 'Perception' in line:
            section = 'perception'
        elif 'Cognition' in line:
            section = 'cognition'
        elif 'total score' in line and section:
            totals[section] = float(line.split(':')[1].strip())
    if 'perception' not in totals or 'cognition' not in totals:
        raise SystemExit(f'could not parse MME scores:\n{out.stdout}\n{out.stderr}')
    combined = totals['perception'] + totals['cognition']

    print('\n' + '=' * 74)
    print(f'HARD-ROUTED MME  (perception+cognition, {n_experts} experts)')
    print('=' * 74)
    for k in range(n_experts):
        print(f'  expert-{k}: n={counts[k]:6d} ({100 * counts[k] / total:5.1f}%)   (scored on union only)')
    print(f'  Perception: {totals["perception"]:.2f}   Cognition: {totals["cognition"]:.2f}')
    print(f'  MME TOTAL : {combined:.2f}')

    summary = {'benchmark': 'mme', 'metric': 'perception+cognition',
               'clustering_dir': args.clustering_dir,
               'expert_checkpoints': list(args.expert_checkpoints),
               'expert_counts': counts, 'n_questions': total,
               'perception': totals['perception'], 'cognition': totals['cognition'],
               'mme_total': combined}
    dest = work / 'combined_hardroute_mme.json'
    json.dump(summary, open(dest, 'w'), indent=2)
    print(f'\nsaved -> {dest}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--benchmark', required=True, choices=sorted(list(BENCHMARKS) + ['mme']))
    ap.add_argument('--clustering-dir', required=True)
    ap.add_argument('--expert-checkpoints', nargs='+', required=True,
                    help='expert checkpoint dirs IN CLUSTER ORDER (expert k <-> centroid k)')
    ap.add_argument('--prefix', default='clustering')
    ap.add_argument('--out-dir', default=None, help='relative to internvl_chat/ '
                                                    '(default results/hardroute/<benchmark>)')
    ap.add_argument('--image-key', default='image')
    ap.add_argument('--clip-model', default='openai/clip-vit-base-patch16')
    ap.add_argument('--clip-batch-size', type=int, default=256)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--batch-size', type=int, default=1,
                    help='per-GPU generation batch size for the vqa runner (batched batch_chat; '
                         'default 1 preserves legacy behavior). Ignored for the sqa runner, which '
                         'is not batched and requires batch size 1.')
    ap.add_argument('--gpus', type=int, default=1)
    ap.add_argument('--master-port', type=int, default=63669)
    ap.add_argument('--skip-eval', action='store_true',
                    help='re-route and re-score using existing prediction files')
    args = ap.parse_args()

    if args.benchmark == 'mme':
        run_mme(args)
        return

    bm = BENCHMARKS[args.benchmark]
    check_data(bm, args.image_key)
    if args.out_dir is None:
        args.out_dir = f'results/hardroute/{args.benchmark}'
    (CHAT / args.out_dir).mkdir(parents=True, exist_ok=True)

    n_experts = len(args.expert_checkpoints)
    print(f'benchmark={args.benchmark}  metric={bm.metric}  experts={n_experts}')
    centroids, mean_vector = load_centroids(args, n_experts)

    splits = [(bm.ds_name, bm.test)] + bm.extra_tests
    results = [evaluate_split(ds, test, bm, centroids, mean_vector, args, n_experts)
               for ds, test in splits]

    print('\n' + '=' * 74)
    print(f'HARD-ROUTED {args.benchmark.upper()}  ({bm.metric}, {n_experts} experts)')
    print('=' * 74)
    for r in results:
        print(f'\n{r["ds_name"]}  (n={r["n_questions"]})')
        for e in r['experts']:
            a = 'n/a' if e['accuracy'] is None else f'{100 * e["accuracy"]:6.2f}%'
            print(f'  expert-{e["expert"]}: n={e["n"]:6d} '
                  f'({100 * e["n"] / r["n_questions"]:5.1f}%)   {a}')
        print(f'  {"COMBINED":<12}                  {100 * r["combined_accuracy"]:6.2f}%')

    headline = sum(r['combined_accuracy'] for r in results) / len(results)
    if len(results) > 1:  # e.g. chartqa = mean(human, augmented)
        print(f'\n{args.benchmark.upper()} headline (mean of splits): {100 * headline:.2f}%')

    summary = {'benchmark': args.benchmark, 'metric': bm.metric,
               'clustering_dir': args.clustering_dir,
               'expert_checkpoints': list(args.expert_checkpoints),
               'splits': results, 'headline_accuracy': headline}
    dest = CHAT / args.out_dir / f'combined_hardroute_{args.benchmark}.json'
    json.dump(summary, open(dest, 'w'), indent=2)
    print(f'\nsaved -> {dest}')


if __name__ == '__main__':
    main()
