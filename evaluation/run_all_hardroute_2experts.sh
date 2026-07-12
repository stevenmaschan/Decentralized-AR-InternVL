#!/bin/bash
# Hard-routed evaluation of a 2-expert set across ALL 11 benchmarks, via the single script
# evaluation/eval_hard_routing.py (one --benchmark per call). Every test question is routed to
# exactly one expert (argmax cosine to its cluster centroid, mean-subtracted space); each expert
# runs on its disjoint subset; predictions are merged and scored as one. Never passes --dynamic.
#
# Run from the repo root with the clustering venv ACTIVATED (not just its python) so the
# per-expert torchrun subprocesses find transformers on PATH (see CLAUDE.md).
#   source /lambda/nfs/virginia/clip-feat-venv/bin/activate
#   bash evaluation/run_all_hardroute_2experts.sh
set -u

REPO=/lambda/nfs/virginia/Decentralized-AR-InternVL
LOGDIR=$REPO/evaluation
BS=32

# --- what to evaluate (override by exporting before calling) ---------------------------------
CLU=${CLU:-clustering/balanced_kmeans_unique_features_base-patch16_2_clusters_seed42_mean-subtracted}
C0=${C0:-work_dirs/internvl2_5_1b/clusters-2_balanced-kmeans_vit-b-16_seed42_mean-subtracted/cluster-0}
C1=${C1:-work_dirs/internvl2_5_1b/clusters-2_balanced-kmeans_vit-b-16_seed42_mean-subtracted/cluster-1}
TAG=${TAG:-clusters-2_ms}          # log-file prefix, so different expert sets don't collide

cd "$REPO" || exit 1

# Benchmarks in ascending cost order. batch-size is forwarded to the vqa/pope/refcoco runners;
# scienceqa (batch-1) and mme (its own batch-1 eval.py) ignore it. Distinct master ports.
# Override BENCHES to run a subset, e.g. BENCHES="ai2d chartqa pope" bash <this>.
BENCHES=${BENCHES:-"mme pope ai2d chartqa scienceqa textvqa docvqa infovqa gqa refcoco vqav2"}
PORT=63700

run () {  # $1 = benchmark
  echo "===== [$(date +%T)] START $1 =====" >> "$LOGDIR/_hardroute_${TAG}_driver.log"
  python evaluation/eval_hard_routing.py --benchmark "$1" \
      --clustering-dir "$CLU" --expert-checkpoints "$C0" "$C1" \
      --batch-size $BS --master-port $PORT \
      > "$LOGDIR/hardroute_${TAG}_$1.log" 2>&1
  rc=$?   # capture BEFORE any command substitution ($(date) below would overwrite $?)
  local mark="END  "; [ "$rc" -ne 0 ] && mark="FAIL "
  echo "===== [$(date +%T)] $mark $1 (exit $rc) =====" >> "$LOGDIR/_hardroute_${TAG}_driver.log"
  PORT=$((PORT + 1))
}

: > "$LOGDIR/_hardroute_${TAG}_driver.log"
for b in $BENCHES; do run "$b"; done
echo "===== [$(date +%T)] ALL DONE =====" >> "$LOGDIR/_hardroute_${TAG}_driver.log"
