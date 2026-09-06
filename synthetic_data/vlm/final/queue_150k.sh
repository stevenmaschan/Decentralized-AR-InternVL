#!/usr/bin/env bash
# Waits for the 100k suite to finish, then builds a 150k dataset and runs the
# same 11 systems on it. Aborts if the 100k suite has not finished in 4 hours.
set -u
cd "$(dirname "$0")/../../.." || exit 1
source /venv/main/bin/activate >/dev/null 2>&1

WAIT_LOG=synthetic_data/vlm/final/logs/driver.log
echo "[queue] waiting for the 100k suite to finish..."
deadline=$(( $(date +%s) + 4*3600 ))
until grep -q "ALL TRAINING DONE" "$WAIT_LOG" 2>/dev/null; do
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "[queue] ABORT: 100k suite did not finish within 4h"; exit 1
  fi
  sleep 60
done
echo "[queue] 100k suite finished at $(date -u '+%H:%M:%S'); starting 150k build"

D=synthetic_data/sweep_datasets/scale_150k
rm -rf "$D"; mkdir -p "$D"
python synthetic_data/generate_shapes_dataset.py --output-dir "$D" \
  --num-samples 160000 --seed 0 --preview 0 --exclude-tasks 3 \
  --alpha-beta 0.197 --val-count 5000 --test-count 5000 || exit 1
python - "$D" <<'EOF'
import json, sys
d = sys.argv[1]
with open(f'{d}/images.jsonl', 'w') as o:
    for l in open(f'{d}/metadata.jsonl'):
        m = json.loads(l); o.write(json.dumps({'image': m['image'], 'key': str(m['image_id'])}) + '\n')
EOF
python clustering/extract_clip_features.py --input "$D/images.jsonl" --image-root "$D" \
  --output "$D/features_vit_base_patch16" --batch-size 512 --num-workers 16 || exit 1
python synthetic_data/split_by_alpha.py --annotations "$D/annotations.jsonl" \
  --metadata "$D/metadata.jsonl" --output-dir "$D/experts_alpha" >/dev/null || exit 1
python synthetic_data/split_by_task.py --annotations "$D/annotations.jsonl" \
  --metadata "$D/metadata.jsonl" --output-dir "$D/experts_task" >/dev/null || exit 1
python synthetic_data/split_random.py --annotations "$D/annotations.jsonl" \
  --metadata "$D/metadata.jsonl" --output-dir "$D/experts_random" \
  --names rand_a rand_b --seed 0 >/dev/null || exit 1
echo "[queue] dataset ready; training"

DATASET="$D" OUTDIR=synthetic_data/vlm/final_150k MAXTRAIN=150000 EPOCHS=50 \
  bash synthetic_data/vlm/final/run_suite.sh
echo "[queue] ALL DONE"
