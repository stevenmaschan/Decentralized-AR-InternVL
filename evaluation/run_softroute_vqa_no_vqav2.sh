#!/usr/bin/env bash
# Soft-mixture (per-token expert mixture) evaluation over the VQA-runner benchmarks EXCEPT vqav2.
# ai2d is intentionally omitted here (already run separately at the same T). Uses the fitted
# production temperature T*=21.79 for the 2 trained mean-subtracted experts. Run with the
# clustering venv ACTIVATED, from the repo root. Continues on per-benchmark failure.
set -u
CLU=clustering/balanced_kmeans_unique_features_base-patch16_2_clusters_seed42_mean-subtracted
C0=work_dirs/internvl2_5_1b/clusters-2_balanced-kmeans_vit-b-16_seed42_mean-subtracted/cluster-0
C1=work_dirs/internvl2_5_1b/clusters-2_balanced-kmeans_vit-b-16_seed42_mean-subtracted/cluster-1
T=21.79190274519012
TAG=T21.79_2experts_ms
DRIVER=evaluation/_softroute_${TAG}_driver.log
: > "$DRIVER"

# benchmark:batch_size  (short-answer max_new_tokens=10 -> 16; max_new_tokens=100 -> 8), cost order
for spec in infovqa:8 chartqa:8 textvqa:16 docvqa:8 gqa:16; do
  bench=${spec%%:*}; bs=${spec##*:}
  log=evaluation/softroute_${bench}_${TAG}.log
  echo "START $bench (bs=$bs) -> $log" | tee -a "$DRIVER"
  python evaluation/eval_soft_routing.py --benchmark "$bench" --temperature "$T" \
      --clustering-dir "$CLU" --expert-checkpoints "$C0" "$C1" --batch-size "$bs" \
      > "$log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    line=$(grep -E "headline|ai2diagram_test|_test |_val |testdev" "$log" | tail -1)
    echo "END   $bench rc=0  ${line}" | tee -a "$DRIVER"
  else
    echo "FAIL  $bench rc=$rc (see $log)" | tee -a "$DRIVER"
  fi
done
echo "ALL DONE" | tee -a "$DRIVER"
