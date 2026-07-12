#!/usr/bin/env bash
# Soft-mixture eval for ScienceQA (image-only, letter exact-match) and POPE (yes/no, Overall F1).
# Fitted production T*=21.79, batched (short answers -> bs=48 is ample). Run with the clustering
# venv ACTIVATED from the repo root. Continues on per-benchmark failure.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CLU=clustering/balanced_kmeans_unique_features_base-patch16_2_clusters_seed42_mean-subtracted
C0=work_dirs/internvl2_5_1b/clusters-2_balanced-kmeans_vit-b-16_seed42_mean-subtracted/cluster-0
C1=work_dirs/internvl2_5_1b/clusters-2_balanced-kmeans_vit-b-16_seed42_mean-subtracted/cluster-1
T=21.79190274519012
TAG=T21.79_2experts_ms
DRIVER=evaluation/_softroute_sqa_pope_${TAG}_driver.log
: > "$DRIVER"

for bench in scienceqa pope; do
  log=evaluation/softroute_${bench}_${TAG}.log
  echo "START $bench -> $log" | tee -a "$DRIVER"
  python evaluation/eval_soft_routing.py --benchmark "$bench" --temperature "$T" \
      --clustering-dir "$CLU" --expert-checkpoints "$C0" "$C1" --batch-size 48 \
      > "$log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    line=$(grep -E "sqa_test|pope " "$log" | tail -1)
    echo "END   $bench rc=0  ${line}" | tee -a "$DRIVER"
  else
    echo "FAIL  $bench rc=$rc (see $log)" | tee -a "$DRIVER"
  fi
done
echo "ALL DONE" | tee -a "$DRIVER"
