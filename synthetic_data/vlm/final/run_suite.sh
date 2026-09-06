#!/usr/bin/env bash
# Parameterised version of run_all.sh: same 11 systems, any dataset / scale.
#   DATASET=<dir> OUTDIR=<dir> MAXTRAIN=<n> EPOCHS=<n> bash run_suite.sh
set -u
cd "$(dirname "$0")/../../.." || exit 1
source /venv/main/bin/activate >/dev/null 2>&1

D="${DATASET:?set DATASET}"
OUT="${OUTDIR:?set OUTDIR}"
MAXTRAIN="${MAXTRAIN:?set MAXTRAIN}"
EPOCHS="${EPOCHS:-50}"
F=$D/features_vit_base_patch16.npy
TOK=$OUT/shared_tokenizer.json
mkdir -p "$OUT/logs"

python - "$D" "$TOK" <<'EOF'
import json, sys
from synthetic_data.vlm.data import CharTokenizer
d, out = sys.argv[1], sys.argv[2]
q, a = [], []
for l in open(f'{d}/annotations.jsonl'):
    r = json.loads(l)['conversations']
    q.append(r[0]['value'].split('\n', 1)[-1]); a.append(r[1]['value'])
CharTokenizer.from_texts(q + a).save(out)
print(f'shared tokenizer -> {out}')
EOF

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
  local o="$OUT/$name" tmp="$OUT/$name.tmp"
  if [ -f "$o/best.pt" ]; then echo "$name SKIP"; return 0; fi
  rm -rf "$tmp"; mkdir -p "$tmp"
  # shellcheck disable=SC2086
  python -m synthetic_data.vlm.$mod --annotations $D/annotations.jsonl --features $F \
      $extra --out-dir "$tmp" --max-train "$MAXTRAIN" --epochs "$EPOCHS" \
      --eval-every 5 --batch-size "$bs" --num-workers 4 --device "cuda:$gpu" \
      > "$OUT/logs/$name.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -f "$tmp/best.pt" ]; then
    rm -rf "$o"; mv "$tmp" "$o"; echo "$name OK"
  else
    echo "$name FAILED rc=$rc"; tail -4 "$OUT/logs/$name.log"; rm -rf "$tmp"
  fi
}

i=0
while [ $i -lt ${#JOBS[@]} ]; do
  run_job "${JOBS[$i]}" 0 & p0=$!
  if [ $((i+1)) -lt ${#JOBS[@]} ]; then run_job "${JOBS[$((i+1))]}" 1 & p1=$!; else p1=""; fi
  wait $p0; [ -n "$p1" ] && wait $p1
  i=$((i+2))
done
echo "SUITE DONE ($OUT)"
