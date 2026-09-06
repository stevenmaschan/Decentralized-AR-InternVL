# Synthetic shapes + text dataset

A small synthetic image/text dataset for VLM experiments. Each image holds up to 10
non-overlapping geometric shapes and (usually) a short random letter sequence
(3-6 letters), and each image yields
questions from two task families:

* **OCR** — read the letter sequence on the image.
* **Shape reasoning** — count/name shapes by type, color, and number of angles.

## Generate

```bash
python synthetic_data/generate_shapes_dataset.py --output-dir synthetic_data --num-samples 10000 --seed 0
```

Takes ~20 s for 10000 images (CPU only, PIL, 32 worker processes by default;
`--num-workers` to change). Each image is seeded independently from `--seed`, so
output is byte-identical for any worker count — verified across 1 vs 8 workers.

## The `alpha` parameter

Every image is parametrised by a single scalar `alpha ~ Beta(b, b)` (`--alpha-beta`,
default `b = 0.5`) that trades text prominence against shape prominence:

| | `alpha = 0` | `alpha = 1` |
|---|---|---|
| text size (448 px canvas) | 14–24 px | 42–78 px |
| text opacity | 0.25–0.45 | 0.85–1.00 |
| shape radius (fraction of grid cell) | 0.55–0.95 | 0.22–0.45 |
| shape opacity | 0.75–1.00 | 0.30–0.55 |

So low `alpha` images favour shape reasoning and high `alpha` images favour OCR.

`b = 0.5` (the arcsine distribution) is U-shaped, concentrating mass at both ends:
40.7% of the 10k set falls in the outer 20% (`alpha < 0.1` or `> 0.9`) versus 20%
under a uniform prior, and only 12.2% in the middle 20%. Decile counts:
`2025 917 720 688 614 609 690 747 942 2048`. Pass `--alpha-beta 1.0` for uniform.
At the populated extremes the parametrisation is stark:

| | `alpha < 0.05` (n=1436) | `alpha > 0.95` (n=1456) |
|---|---|---|
| mean shape radius / opacity | 46.9 px / 0.87 | 21.3 px / 0.43 |
| mean text size / opacity | 19.7 px / 0.36 | 59.1 px / 0.91 |
`alpha` is stored on every image and on every QA line, so it can be used to
stratify training/eval or to study routing behaviour as a function of one knob.

## Tasks

| `task_id` | `task_type` | Question | Answer |
|---|---|---|---|
| 1 | `ocr` | `Recognise text on the image.` | the letter sequence, or `""` when the image has no text |
| 2 | `shape_reasoning` | `How many {plural type} are on the image?` | count |
| 3 | `shape_reasoning` | `How many shapes of {color} color are on the image?` | count |
| 4 | `shape_reasoning` | `Name types of shapes you see on the image.` | singular types, comma-separated |
| 5 | `shape_reasoning` | `How many shapes have at least {n} angles?` | count, `n in {1, 3, 4, 6}` |

### Question sampling

By default (`--question-sampling alpha`) **one question is sampled per image**:
it is the OCR question with probability `alpha`, otherwise a uniformly chosen one
of the four shape questions. Because `alpha ~ Beta(b, b)`, the per-image OCR
probability is itself Beta-distributed across the dataset — text-prominent images
get asked about text, shape-prominent images about shapes.

Measured on the 10k set (10000 QA pairs, 5077 OCR / 4923 shape reasoning):

| `alpha` decile | 0–.1 | .1–.2 | .2–.3 | .3–.4 | .4–.5 | .5–.6 | .6–.7 | .7–.8 | .8–.9 | .9–1 |
|---|---|---|---|---|---|---|---|---|---|---|
| P(OCR question) | .03 | .16 | .27 | .35 | .44 | .57 | .67 | .75 | .85 | .98 |
| mean `alpha` | .03 | .15 | .25 | .35 | .45 | .55 | .65 | .75 | .85 | .97 |

P(OCR) tracks `alpha` decile by decile; overall P(OCR) = 0.508 against
E[`alpha`] = 0.503. Mean `alpha` is 0.751 for OCR questions and 0.247 for shape
questions. The four shape tasks come out uniform (24.5–25.5% each).

`--questions-per-image N` samples N questions instead of one.
`--question-sampling all` restores the full fixed battery — 8 QA pairs per image
(1 + 2 + 2 + 1 + 2), every image with the same task mix — which is what you want
for evaluation, where full coverage beats a sample.

Shape types: circle, triangle, ellipse, square, rectangle, hexagon (circles and
ellipses count as having 0 angles). Colors: red, green, blue, yellow, orange,
purple, pink, brown, gray, black.

Roughly a quarter of the counting questions deliberately ask about a type or color
that is absent, so `0` is a well-represented answer (`--p-absent-query`). Task 4
lists types in a fixed canonical order. Task 1's answer is case-sensitive and
matches the rendered string exactly.

## Outputs

```
images/000000.png ...   448x448 RGB renders
annotations.jsonl       one line per QA pair, InternVL chat format
metadata.jsonl          one line per image: alpha, text (string/font/size/opacity/rotation/bbox),
                        every shape (type, color, center, radius, rotation, opacity, bbox, n_angles),
                        per-type and per-color counts, and the image's QA list
meta.json               InternVL meta file, usable as --meta_path
preview.png             sanity-check grid
```

An `annotations.jsonl` line:

```json
{"id": 0, "image_id": 0, "image": "images/000000.png", "width": 448, "height": 448,
 "alpha": 0.8444, "task_id": 1, "task_type": "ocr",
 "conversations": [{"from": "human", "value": "<image>\nRecognise text on the image."},
                   {"from": "gpt", "value": "Wonder"}]}
```

## Notes and knobs

* Shapes are placed on a jittered grid and never overlap (verified: 0 overlapping
  bounding-box pairs, 0 out-of-bounds shapes in the 1k set), so counts are
  unambiguous. Text is placed in a free region when one is found within 60 tries
  (~5% of images end up with the text bbox touching a shape bbox).
* `--p-no-text` (default `0.15`) controls how many images carry no text. For those,
  the OCR answer is the **empty string** — "produce nothing", as specified. If your
  trainer chokes on a zero-token target, pass e.g.
  `--no-text-answer "There is no text on the image."`.
* `--min-shapes` / `--max-shapes` (default 1/10), `--n-type-questions`,
  `--n-color-questions`, `--n-angle-questions`, `--image-size`, `--text-max-rotation`,
  `--image-prefix` (prefix for the image path written into `annotations.jsonl`),
  `--root` (the `root` field of `meta.json`).

## CLIP features + t-SNE

Extract CLIP ViT-B/16 features with the repo's existing extractor, then embed and
color by `alpha`:

```bash
python clustering/extract_clip_features.py --input synthetic_data/clip/images.jsonl --image-root synthetic_data --output synthetic_data/clip/features_vit_base_patch16 --batch-size 256 --num-workers 8
```

```bash
python synthetic_data/tsne_alpha.py
```

Outputs under `synthetic_data/clip/`: `features_vit_base_patch16.npy` (10000×512),
aligned `_keys.json`/`_paths.json`, `tsne_alpha.png`, `tsne_alpha_factors.png`,
`tsne_alpha_coords.npy`.

**Text is random letter sequences, so nothing clusters by string identity.**
8397 of the 8531 rendered strings in the 10k set are distinct (the 134 repeats are
chance collisions among random 3-6 letter sequences), and 1-NN string purity is
0.7% — chance level. This removes the repeated categorical signal that an earlier
dictionary-word vocabulary created, where 126 words repeated ~7x each gave 96%
1-NN word purity, 60% of feature variance, and a t-SNE of ~126 word micro-clusters.

With that gone, `alpha` becomes visible, and more so as the sample count grows:

| statistic | words, 1k, uniform | letters, 1k, uniform | letters, 10k, uniform | letters, 10k, Beta(.5,.5) |
|---|---|---|---|---|
| \|Spearman(`alpha`, best t-SNE axis)\| | 0.03 | 0.18 | 0.31 | **0.46** |
| k-NN (k=10) 5-fold CV R² for `alpha` | 0.17 | 0.42 | 0.47 | **0.62** |
| 1-NN shares the same string | 96% | n/a | 0.7% | 0.7% |

Share of CLIP feature variance explained (10k set, U-shaped `alpha`):

| factor | share |
|---|---|
| shape count | 5.5% |
| `alpha` (5 equal bins) | 3.1% |
| text length | 3.2% |
| presence of text | 2.6% |

Concentrating `alpha` at the ends makes it markedly easier to read off the CLIP
features (R² 0.47 -> 0.62) and gives it a clear t-SNE axis, because the ambiguous
middle images that previously blurred the two regimes are now rare.

Note that string-identity variance-explained is degenerate when strings are
effectively unique (nearly every group is a singleton, so the statistic is ~100%
by construction). `tsne_alpha.py` reports it only when the average string group
has at least two members; 1-NN string purity is always reported.

## Balanced spherical k-means (k=2)

Using the repo's clusterer on the same features (cosine distance, global mean
subtracted, balanced cluster sizes):

```bash
python clustering/single_stage_balanced_kmeans.py synthetic_data/clip/features_vit_base_patch16.npy --output-dir synthetic_data/clip/balanced-kmeans_vit-b-16_2-coarse --n-clusters 2 --device cuda:0
```

Note the script prepends `clustering/` to `--output-dir`; the results here were
moved to `synthetic_data/clip/balanced-kmeans_2/`. It needs the `balanced-kmeans`
package (`git clone https://github.com/kernelmachine/balanced-kmeans /tmp/balanced-kmeans`,
then `pip install --no-deps -e /tmp/balanced-kmeans` plus `numba`).

Describe the partition against the ground truth:

```bash
python synthetic_data/analyze_clusters.py
```

**The k=2 split is a text-prominence split, i.e. it recovers `alpha`.** Converged
in 4 iterations, exactly 5000/5000.

| factor | cluster 0 | cluster 1 |
|---|---|---|
| `alpha` | 0.401 | 0.607 |
| text size (px) | 32.6 | 44.3 |
| text opacity | 0.55 | 0.70 |
| mean shape radius | 41.4 | 37.6 |
| mean shape opacity | 0.70 | 0.60 |
| images with text | 70.6% | 100% |
| **shape count** | **5.62** | **5.33** |

How well each factor predicts the partition (AUC, 0.5 = no signal):

| factor | AUC |
|---|---|
| text size | 0.761 |
| text opacity | 0.749 |
| `alpha` | 0.706 |
| mean shape opacity | 0.298 (i.e. 0.702 inverted) |
| mean shape radius | 0.390 (i.e. 0.610 inverted) |
| shape count | 0.471 |

All 1469 text-free images land in cluster 0, none in cluster 1. Shape count is the
one factor the split ignores (AUC 0.47, chance). So the two experts this partition
would produce are roughly **"faint/absent text, bold shapes"** (cluster 0, low
`alpha`) and **"prominent text, faint shapes"** (cluster 1, high `alpha`) — which
lines up with the OCR / shape-reasoning task split the dataset was built around,
though it is a soft split rather than a clean one (`alpha` AUC 0.71, and the two
`alpha` histograms overlap substantially in the middle).

## Balanced spherical k-means (k=2)

```bash
python clustering/single_stage_balanced_kmeans.py synthetic_data/clip/features_vit_base_patch16.npy --output-dir ../synthetic_data/clip/balanced_kmeans_k2 --n-clusters 2 --plot-tsne
```

```bash
python synthetic_data/cluster_report.py
```

(`single_stage_balanced_kmeans.py` always writes under `clustering/`, hence the
`../` in `--output-dir`.) Artifacts land in `synthetic_data/clip/balanced_kmeans_k2/`:
assignments, centroids, global mean, model pickle, the repo-standard
`clustering_tsne_clustering.png`, and `cluster_report.png`.

The partition is exactly balanced (5000 / 5000) and **splits the dataset by text
prominence, i.e. by `alpha`** — not by shape content (cluster indices are
arbitrary; here c0 is the high-`alpha` side):

| factor (mean) | cluster 0 | cluster 1 |
|---|---|---|
| `alpha` | 0.670 | 0.336 |
| has text | 99.9% | 69.8% |
| text size (px) | 46.8 | 29.1 |
| text opacity | 0.735 | 0.500 |
| shape radius (px) | 36.4 | 42.0 |
| shape opacity | 0.573 | 0.725 |
| n_shapes | 5.17 | 5.92 |

Mutual information with the cluster label (bits, max 1.0): text length 0.182,
`alpha` 0.169, has_text 0.167, **n_shapes 0.026**. Nearly all 1512 text-free
images land in cluster 1, and the `alpha` histograms are strongly opposed — each
cluster peaks at its own end of the U. Silhouette is 0.120 (cosine): the split is
a cut through one continuous cloud rather than two separated blobs.

Concentrating `alpha` at the ends roughly doubled how much the partition tracks
it (MI 0.095 -> 0.169, cluster `alpha` means 0.40/0.61 -> 0.34/0.67) while leaving
shape count almost as irrelevant as before (MI 0.011 -> 0.026).

**Effect on expert routing.** With `alpha`-weighted question sampling the two
clusters carry genuinely different task mixes, so routing by image cluster also
partially specialises experts by task family:

| | cluster 0 | cluster 1 |
|---|---|---|
| OCR pairs | 3373 | 1704 |
| shape-reasoning pairs | 1627 | 3296 |
| share OCR | **67.5%** | **34.1%** |

(Under the old `--question-sampling all` battery both clusters sat at 12.5% OCR,
since every image carried the same 8 questions.) The partition is still driven by
image appearance rather than by the question text, so the specialisation is
partial, not a clean task split.

## Training a small VLM from scratch

`synthetic_data/vlm/` holds a self-contained pipeline: frozen CLIP embedding ->
learned projector -> visual prefix tokens -> causal transformer decoder trained
from scratch, generating the answer character by character.

```bash
python -m synthetic_data.vlm.train --out-dir synthetic_data/vlm/runs/base --epochs 120 --eval-every 10 --device cuda:0
```

```bash
python -m synthetic_data.vlm.train --out-dir synthetic_data/vlm/runs/blind --no-vision --epochs 120 --eval-every 10 --device cuda:1
```

* `data.py` — character tokenizer (70 symbols), image-level train/val/test split
  (80/10/10, so no image spans splits), feature standardisation fitted on train only.
* `model.py` — `Projector` (512-d CLIP vector -> `n_prefix` tokens) + pre-norm
  decoder blocks with causal SDPA attention, weight-tied output head.
* `train.py` — AdamW + cosine schedule, bf16 autocast, greedy decoding, metrics
  per task and per `alpha` bin, plus a majority-answer baseline.

Sequence layout: `[visual prefix] [BOS] question [SEP] answer [EOS]`, with loss on
the answer tokens and the final EOS only. Nothing is pretrained; the CLIP tower is
never loaded at train time (features come off disk).

**Model: 1.34M trainable parameters** (`d_model=128`, 4 layers, 4 heads,
`n_prefix=4`): blocks 793k (59%), projector 526k (39%), embeddings/LayerNorms 23k.

### Results (test split, 1000 QA pairs, 120 epochs, ~4 min per run)

| | with CLIP features | blind (no vision) | majority answer |
|---|---|---|---|
| **exact match** | **39.9%** | 24.6% | 24.5% |
| task 1 OCR | 28.1% (CER 0.373) | 14.5% (CER 0.855) | 14.5% |
| task 2 count by type | 63.3% (MAE 0.49) | 54.7% (MAE 0.52) | 54.7% |
| task 3 count by color | 41.8% (MAE 0.67) | 56.0% (MAE 0.45) | 56.0% |
| task 4 name types | 47.2% (set-F1 0.867) | 5.7% (F1 0.724) | 5.7% |
| task 5 count by angles | 56.1% (MAE 0.61) | 17.1% (MAE 1.63) | 16.3% |

The blind model lands on the majority baseline (24.6% vs 24.5%), confirming it
learns answer priors and nothing else, so the +15.3 points is what the pooled CLIP
vector actually contributes.

**`alpha` behaves exactly as designed.** Splitting test accuracy by `alpha` bin
shows the two task families trading places:

| `alpha` bin | 0–.2 | .2–.4 | .4–.6 | .6–.8 | .8–1 |
|---|---|---|---|---|---|
| shape-reasoning EM | **54.9%** | 51.8% | 53.3% | 39.6% | **42.9%** |
| OCR EM | **11.1%** | 34.0% | 22.2% | 29.3% | **28.8%** |

Shape accuracy falls as shapes shrink and fade; OCR accuracy rises as text grows
and darkens.

**Where OCR actually stands.** 74 of the 509 test OCR questions are on text-free
images, and the empty answer is free (100%). On the 435 images that do carry text
the model gets 15.9% exactly right (CER 0.436), and that collapses with string
length — 44.2% at 3 letters, 14.3% at 4, 4.5% at 5, 1.8% at 6. A single pooled
512-d CLIP vector carries roughly a few characters' worth of legible text, no more.

**Two caveats worth reading before trusting these numbers.**

1. Colour counting is *worse* than the blind baseline (41.8% vs 56.0%). The pooled
   feature apparently misleads more than it informs here, and the model spends
   capacity on it rather than falling back on the prior.
2. The model overfits: final train loss 0.022 against val loss 0.836, best
   checkpoint at epoch 80 of 120. With 8000 training pairs and 1.34M parameters
   that is expected. Generating with `--question-sampling all` (80k pairs from the
   same 10k images) is the cheapest fix.

The ceiling here is the *representation*, not the decoder: one global vector cannot
localise or count. Feeding CLIP patch tokens (196 x 768 from ViT-B/16) into the same
projector is the natural next step and needs no change to `model.py` beyond letting
the projector accept a token sequence.
