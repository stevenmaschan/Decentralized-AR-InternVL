set -x

# Use CUDA devices 1 and 2
export CUDA_VISIBLE_DEVICES=1,2

GPUS=${GPUS:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}

export PYTHONPATH="${PYTHONPATH}:$(pwd)/internvl_chat:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR='work_dirs/internvl_chat_v2_5/internvl2_5_1b_test_stage2'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# Stage: Stage 2 (Full Model Instruction Tuning) - torchrun version for 2 GPUs (test)
# Architecture: InternViT-300M-448px-V2_5 + MLP + internlm2_5-7b-chat (using 1B pretrain model)
# Trainable Components: ViT + MLP + LLM
# Number of GPUs: 2 (CUDA 1,2)
# Packed Batch Size: 2 (per_device_batch_size=1 * 2 GPUs)
# Learning Rate: 4e-5
# Context Length: 16384
# Image Tile Threshold: 48
# ViT Drop Path: 0.1
# Weight Decay: 0.05
# Training: max_steps based
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl_chat/internvl/train/internvl_chat_pretrain.py \
  --model_name_or_path "/media/data2/maschan/internvl/models/InternVL2_5-1B-Pretrain/InternVL2_5-1B-Pretrain" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "/media/data2/maschan/internvl/data/dataset_mixture.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --min_num_frame 8 \
  --max_num_frame 32 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --max_steps 100 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 50 \
  --save_total_limit 2 \
  --learning_rate 4e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 16384 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "internvl_chat/zero_stage1_config.json" \
  --report_to "tensorboard" \
  --use_packed_ds True \
  --num_images_expected 48 \
  --max_packed_tokens 16384 \
  --max_buffer_size 20 \
  --log_freq 1000 \
  --strict_mode False \
  --replacement False \
  --allow_overflow False \
  --remove_unused_columns False \
  --loss_reduction "square" \
  --loss_reduction_all_gather True \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

