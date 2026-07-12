#!/bin/bash
# Evaluate dense_seed44 on the full benchmark suite, sequentially (1 GPU).
# Mirrors exactly how dense_seed43 was evaluated: no --dynamic, bs=32 for the
# batched runners (vqa/pope/refcoco), batch-1 for scienceqa + mme.
set -u

REPO=/lambda/nfs/virginia/Decentralized-AR-InternVL
IC=$REPO/internvl_chat
CKPT=$REPO/work_dirs/internvl2_5_1b/dense_seed44
LOGDIR=$REPO/evaluation
BS=32

source /lambda/nfs/virginia/clip-feat-venv/bin/activate
export PYTHONPATH=$IC:${PYTHONPATH:-}

run_vqa () {  # $1=logname  $2=datasets  $3=master_port
  cd "$IC" || exit 1
  echo "===== [$(date +%T)] START $1 ($2) =====" >> "$LOGDIR/_seed44_driver.log"
  GPUS=1 torchrun --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=1 \
    --master_port="$3" eval/vqa/evaluate_vqa.py --checkpoint "$CKPT" \
    --datasets "$2" --batch-size $BS > "$LOGDIR/dense_seed44_$1.log" 2>&1
  echo "===== [$(date +%T)] END   $1 (exit $?) =====" >> "$LOGDIR/_seed44_driver.log"
}

# 1. VQAv2 (~65 min)
run_vqa vqav2 vqav2_val 29570
# 2. GQA
run_vqa gqa gqa_testdev_llava 29571
# 3. TextVQA
run_vqa textvqa textvqa_val 29572
# 4. ChartQA (two splits)
run_vqa chartqa "chartqa_test_human,chartqa_test_augmented" 29573
# 5. AI2D
run_vqa ai2d ai2diagram_test 29574
# 6. DocVQA
run_vqa docvqa docvqa_val 29575
# 7. InfoVQA
run_vqa infovqa infographicsvqa_val 29576

# 8. ScienceQA (separate runner, batch-1)
cd "$IC" || exit 1
echo "===== [$(date +%T)] START scienceqa =====" >> "$LOGDIR/_seed44_driver.log"
GPUS=1 torchrun --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=1 \
  --master_port=29577 eval/scienceqa/evaluate_scienceqa.py --checkpoint "$CKPT" \
  --datasets sqa_test > "$LOGDIR/dense_seed44_scienceqa.log" 2>&1
echo "===== [$(date +%T)] END   scienceqa (exit $?) =====" >> "$LOGDIR/_seed44_driver.log"

# 9. POPE (batched runner)
cd "$IC" || exit 1
echo "===== [$(date +%T)] START pope =====" >> "$LOGDIR/_seed44_driver.log"
GPUS=1 torchrun --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=1 \
  --master_port=29578 eval/pope/evaluate_pope.py --checkpoint "$CKPT" \
  --datasets pope --batch-size $BS > "$LOGDIR/dense_seed44_pope.log" 2>&1
echo "===== [$(date +%T)] END   pope (exit $?) =====" >> "$LOGDIR/_seed44_driver.log"

# 10. RefCOCO / + / g (8 subsets, batched runner)
cd "$IC" || exit 1
echo "===== [$(date +%T)] START refcoco =====" >> "$LOGDIR/_seed44_driver.log"
GPUS=1 torchrun --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=1 \
  --master_port=29579 eval/refcoco/evaluate_grounding.py --checkpoint "$CKPT" \
  --datasets 'refcoco_val,refcoco_testA,refcoco_testB,refcoco+_val,refcoco+_testA,refcoco+_testB,refcocog_val,refcocog_test' \
  --batch-size $BS > "$LOGDIR/dense_seed44_refcoco.log" 2>&1
echo "===== [$(date +%T)] END   refcoco (exit $?) =====" >> "$LOGDIR/_seed44_driver.log"

# 11. MME (eval.py + calculation.py, batch-1, must run from eval/mme)
cd "$IC/eval/mme" || exit 1
echo "===== [$(date +%T)] START mme =====" >> "$LOGDIR/_seed44_driver.log"
python eval.py --checkpoint "$CKPT" > "$LOGDIR/dense_seed44_mme.log" 2>&1
echo "=== CALCULATION ===" >> "$LOGDIR/dense_seed44_mme.log"
python calculation.py --results_dir "$(basename "$CKPT")" >> "$LOGDIR/dense_seed44_mme.log" 2>&1
echo "===== [$(date +%T)] END   mme (exit $?) =====" >> "$LOGDIR/_seed44_driver.log"

echo "===== [$(date +%T)] ALL DONE =====" >> "$LOGDIR/_seed44_driver.log"
