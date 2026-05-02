#!/usr/bin/env python3
"""
Partition dense_combined_clean.jsonl by SimCSE clustering assignments.

Reads:
  - data/dense_combined_clean.jsonl (one line per sample, has dataset_source)
  - clustering/simcse_balanced_2_clusters/clustering_assignments.npy (same order)

Writes:
  - output_dir/cluster-{c}/{dataset_name}.jsonl for each cluster and dataset.

Usage:
  python partition_jsonl_by_simcse_assignments.py \\
    --assignments clustering/simcse_balanced_2_clusters/clustering_assignments.npy \\
    --jsonl data/dense_combined_clean.jsonl \\
    --output-dir data/clusters-2_balanced_kmeans_simcse
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Partition combined JSONL by SimCSE cluster assignments")
    parser.add_argument("--assignments", type=str, default="clustering/simcse_balanced_2_clusters/clustering_assignments.npy",
                        help="Path to .npy assignments (same length as jsonl)")
    parser.add_argument("--jsonl", type=str, default="data/dense_combined_clean.jsonl",
                        help="Path to combined JSONL (must have dataset_source per line)")
    parser.add_argument("--output-dir", type=str, default="data/clusters-2_balanced_kmeans_simcse",
                        help="Output directory for cluster-0, cluster-1, ...")
    args = parser.parse_args()

    import numpy as np
    assignments = np.load(args.assignments)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # (cluster_id, dataset_name) -> open file handle; create on first use
    open_files = {}
    counts = defaultdict(lambda: defaultdict(int))

    def get_file(c, name):
        if (c, name) not in open_files:
            cluster_dir = output_dir / f"cluster-{c}"
            cluster_dir.mkdir(parents=True, exist_ok=True)
            open_files[(c, name)] = open(cluster_dir / f"{name}.jsonl", "w")
        return open_files[(c, name)]

    with open(args.jsonl, "r") as f:
        for i, line in enumerate(tqdm(f, desc="Partitioning JSONL")):
            if not line.strip():
                continue
            if i >= len(assignments):
                break
            c = int(assignments[i])
            try:
                obj = json.loads(line)
                src = obj.get("dataset_source", "unknown")
                name = Path(src).stem if src else "unknown"
                get_file(c, name).write(line)
                counts[c][name] += 1
            except json.JSONDecodeError:
                continue

    for (c, name), h in open_files.items():
        h.close()
    for c in sorted(counts.keys()):
        for name in sorted(counts[c].keys()):
            print(f"cluster-{c}/{name}.jsonl: {counts[c][name]:,} entries")
    print(f"Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
