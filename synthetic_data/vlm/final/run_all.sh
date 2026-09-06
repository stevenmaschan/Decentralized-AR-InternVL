#!/usr/bin/env bash
# Final experiment suite: every system at 100k train, identical budget
# (39,050 optimizer updates), identical val/test, results under vlm/final/.
#
# Batch size halves as the training shard halves, so every run gets 781
# updates/epoch x 50 epochs regardless of how the data was partitioned.
set -u
cd "$(dirname "$0")/../../.." || exit 1
source /venv/main/bin/activate >/dev/null 2>&1

D=synthetic_data/sweep_datasets/scale_100k
F=$D/features_vit_base_patch16.npy
OUT=synthetic_data/vlm/final
TOK=synthetic_data/vlm/final/shared_tokenizer.json
mkdir -p "$OUT/logs"

# name : module : extra args : batch : gpu
JOBS=(
  "dense_seed0|train|--seed 0|128"
  "dense_seed1|train|--seed 1|128"
  "dense_seed2|train|--seed 2|128"
  "alpha_lo|train|--annotations $D/experts_alpha/expert_alpha_lo/annotations.jsonl|64"
  "alpha_hi|train|--annotations $D/experts_alpha/expert_alpha_hi/annotations.jsonl|64"
  "context_ocr|train|--annotations $D/experts_task/expert_ocr/annotations.jsonl --tokenizer $TOK|64"
  "context_shape|train|--annotations $D/experts_task/expert_shape/annotations.jsonl --tokenizer $TOK|64"
  "random_a|train|--annotations $D/experts_random/rand_a/annotations.jsonl --tokenizer $TOK|64"
  "random_b|train|--annotations $D/experts_random/rand_b/annotations.jsonl --tokenizer $TOK|64"
  "moe_hard_full|train_moe_full|--routing sparse --jitter 0.01|128"
  "moe_layerwise|train|--moe-experts 2 --moe-aux-coef 0.01|128"
)

run_job () {
  local spec="$1" gpu="$2"
  local name="${spec%%|*}"; local rest="${spec#*|}"
  local mod="${rest%%|*}"; rest="${rest#*|}"
  local extra="${rest%|*}"; local bs="${rest##*|}"
  local out="$OUT/$name" tmp="$OUT/$name.tmp"
  if [ -f "$out/best.pt" ]; then echo "$name SKIP (exists)"; return 0; fi
  rm -rf "$tmp"; mkdir -p "$tmp"
  # shellcheck disable=SC2086
  python -m synthetic_data.vlm.$mod \
      --annotations $D/annotations.jsonl --features $F \
      $extra --out-dir "$tmp" --max-train 100000 --epochs 50 --eval-every 5 \
      --batch-size "$bs" --num-workers 4 --device "cuda:$gpu" \
      > "$OUT/logs/$name.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -f "$tmp/best.pt" ]; then
    rm -rf "$out"; mv "$tmp" "$out"; echo "$name OK"
  else
    echo "$name FAILED rc=$rc"; tail -4 "$OUT/logs/$name.log"; rm -rf "$tmp"
  fi
}

i=0
while [ $i -lt ${#JOBS[@]} ]; do
  run_job "${JOBS[$i]}" 0 &
  p0=$!
  if [ $((i+1)) -lt ${#JOBS[@]} ]; then run_job "${JOBS[$((i+1))]}" 1 & p1=$!; else p1=""; fi
  wait $p0; [ -n "$p1" ] && wait $p1
  i=$((i+2))
done
echo "ALL TRAINING DONE"
