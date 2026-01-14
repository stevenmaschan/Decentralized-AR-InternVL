# Decentralized Autoregressive Generation

Implementation of decentralized autoregressive generation for multimodal language models using expert routing based on dataset clustering. Built on [InternVL 2.5](https://github.com/OpenGVLab/InternVL).

Paper: [Decentralized Autoregressive Generation](https://arxiv.org/abs/2601.03184)

## Overview

This work implements expert-based training for InternVL 2.5-1B, where the model is partitioned into multiple experts trained on different data clusters. We use **CLIP-ViT-B/16** for dataset partitioning and routing, with **top-1 approximation** for expert selection during inference.

## Clustering Strategies

Two clustering approaches are supported:

1. **Two-stage clustering**: Hierarchical approach using FAISS spherical k-means for fine clustering (default: 1024 clusters), followed by balanced k-means for coarse clustering (default: 2 clusters).

   ![Two-stage clustering t-SNE](kmeans_vit-base-patch-16_1024-fine_2-coarse/clustering_tsne_combined_clustering.png)

2. **Single-stage spherical k-means**: Direct balanced k-means clustering on CLIP features using cosine distance.

   ![Single-stage clustering t-SNE](balanced-kmeans_vit-base-patch-16_2-coarse/clustering_tsne_clustering.png)

## Training Configuration

- **Model**: InternVL 2.5-1B (full-parameter fine-tuning: ViT+MLP+LLM)
- **Initialization**: InternVL-2.5-1B-Pretrained checkpoint (post-Stage 1.5)
- **Data**: Subset of InternVL 2.5 Stage 2 fine-tuning data mixture
- **Training**: 2 GPUs, batch size 1 per device, context length 8192 tokens, dynamic resolution disabled

## Usage

### Clustering

**Two-stage:**
```bash
python clustering/two_stage_kmeans.py features.npy \
    --output-dir clustering/kmeans_vit-base-patch-16_1024-fine_2-coarse \
    --n-fine-clusters 1024 --n-coarse-clusters 2
```

**Single-stage:**
```bash
python clustering/single_stage_balanced_kmeans.py features.npy \
    --output-dir clustering/balanced-kmeans_vit-base-patch-16_2-coarse \
    --n-clusters 2
```

### Split Dataset

```bash
python clustering/split_dataset.py \
    --input-jsonl data/docvqa/val.jsonl \
    --output-dir data/docvqa/val_2experts \
    --clustering-results-dir clustering/kmeans_vit-base-patch-16_1024-fine_2-coarse \
    --images-dir data/docvqa/val \
    --clip-model openai/clip-vit-base-patch16
```

### Combine Results

```bash
python clustering/combine_cluster_results_unified.py \
    --cluster0 results/cluster0_results.json \
    --cluster1 results/cluster1_results.json \
    --benchmark textvqa \
    --output results/combined_results.json
```

## Citation

```bibtex
@article{maschan2026decentralized,
  title={Decentralized Autoregressive Generation},
  author={Maschan, Stepan and Qu, Haoxuan and Liu, Jun},
  journal={arXiv preprint arXiv:2601.03184},
  year={2026}
}
```
