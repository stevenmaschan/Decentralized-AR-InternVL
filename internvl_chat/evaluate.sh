set -x

CHECKPOINT=${1}
DATASET=${2}

# Activate internvl conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate internvl
# Handle checkpoint path: if already absolute, use as-is; otherwise make absolute
if [[ "$CHECKPOINT" == /* ]]; then
    CHECKPOINT="${CHECKPOINT}"
else
CHECKPOINT="$(pwd)/${CHECKPOINT}"
fi
export PYTHONPATH="$(pwd):${PYTHONPATH}"
echo "CHECKPOINT: ${CHECKPOINT}"

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

# Save original arguments
ARGS=("$@")

# Parse options
while [[ $# -gt 0 ]]; do
  case "$1" in
    --auto)
      GPUS=1
      shift
      ;;
    *)
      shift
      ;;
  esac
done
echo "GPUS: ${GPUS}"

if  [ ${DATASET} == "mme" ]; then
  cd eval/mme/
  DIRNAME=`basename ${CHECKPOINT}`
  python eval.py --checkpoint ${CHECKPOINT} "${ARGS[@]:2}"
  python calculation.py --results_dir ${DIRNAME}
  cd ../../
fi

if  [ ${DATASET} == "mme-2experts-cluster0" ]; then
  cd eval/mme/
  DIRNAME=`basename ${CHECKPOINT}`
  export PYTHONPATH="$(pwd)/../..:${PYTHONPATH}"
  python eval.py --checkpoint ${CHECKPOINT} --root ../../data/mme/test_2experts/cluster0_txt_files "${ARGS[@]:2}"
  python calculation.py --results_dir ${DIRNAME}
  cd ../../
fi

if  [ ${DATASET} == "mme-2experts-cluster1" ]; then
  cd eval/mme/
  DIRNAME=`basename ${CHECKPOINT}`
  export PYTHONPATH="$(pwd)/../..:${PYTHONPATH}"
  python eval.py --checkpoint ${CHECKPOINT} --root ../../data/mme/test_2experts/cluster1_txt_files "${ARGS[@]:2}"
  python calculation.py --results_dir ${DIRNAME}
  cd ../../
fi

if  [ ${DATASET} == "mme-balanced-cluster0" ]; then
  cd eval/mme/
  DIRNAME=`basename ${CHECKPOINT}`
  # Backup 1B results if running 2B model
  if [[ ${CHECKPOINT} == *"internvl2_5_2b"* ]] && [ -d "${DIRNAME}" ] && [ ! -d "internvl2_5_1b_${DIRNAME}" ]; then
    mv "${DIRNAME}" "internvl2_5_1b_${DIRNAME}"
  fi
  export PYTHONPATH="$(pwd)/../..:${PYTHONPATH}"
  source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && python eval.py --checkpoint ${CHECKPOINT} --root ../../data/mme/test_2experts_balanced/cluster0_txt_files "${ARGS[@]:2}"
  python calculation.py --results_dir ${DIRNAME}
  cd ../../
fi

if  [ ${DATASET} == "mme-balanced-cluster1" ]; then
  cd eval/mme/
  DIRNAME=`basename ${CHECKPOINT}`
  # Backup 1B results if running 2B model
  if [[ ${CHECKPOINT} == *"internvl2_5_2b"* ]] && [ -d "${DIRNAME}" ] && [ ! -d "internvl2_5_1b_${DIRNAME}" ]; then
    mv "${DIRNAME}" "internvl2_5_1b_${DIRNAME}"
  fi
  export PYTHONPATH="$(pwd)/../..:${PYTHONPATH}"
  source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && python eval.py --checkpoint ${CHECKPOINT} --root ../../data/mme/test_2experts_balanced/cluster1_txt_files "${ARGS[@]:2}"
  python calculation.py --results_dir ${DIRNAME}
  cd ../../
fi

if  [ ${DATASET} == "mme-balanced-vitl14-cluster0" ]; then
  cd eval/mme/
  DIRNAME=`basename ${CHECKPOINT}`
  # Backup 1B results if running 2B model
  if [[ ${CHECKPOINT} == *"internvl2_5_2b"* ]] && [ -d "${DIRNAME}" ] && [ ! -d "internvl2_5_1b_${DIRNAME}" ]; then
    mv "${DIRNAME}" "internvl2_5_1b_${DIRNAME}"
  fi
  export PYTHONPATH="$(pwd)/../..:${PYTHONPATH}"
  source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && python eval.py --checkpoint ${CHECKPOINT} --root ../../data/mme/test_2experts_balanced_vitl14/cluster0_txt_files "${ARGS[@]:2}"
  python calculation.py --results_dir ${DIRNAME}
  cd ../../
fi

if  [ ${DATASET} == "mme-balanced-vitl14-cluster1" ]; then
  cd eval/mme/
  DIRNAME=`basename ${CHECKPOINT}`
  # Backup 1B results if running 2B model
  if [[ ${CHECKPOINT} == *"internvl2_5_2b"* ]] && [ -d "${DIRNAME}" ] && [ ! -d "internvl2_5_1b_${DIRNAME}" ]; then
    mv "${DIRNAME}" "internvl2_5_1b_${DIRNAME}"
  fi
  export PYTHONPATH="$(pwd)/../..:${PYTHONPATH}"
  source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && python eval.py --checkpoint ${CHECKPOINT} --root ../../data/mme/test_2experts_balanced_vitl14/cluster1_txt_files "${ARGS[@]:2}"
  python calculation.py --results_dir ${DIRNAME}
  cd ../../
fi

if  [ ${DATASET} == "mme-balanced-vitb16-4experts-cluster0" ]; then
  cd eval/mme/
  DIRNAME=`basename ${CHECKPOINT}`
  # Backup 1B results if running 2B model
  if [[ ${CHECKPOINT} == *"internvl2_5_2b"* ]] && [ -d "${DIRNAME}" ] && [ ! -d "internvl2_5_1b_${DIRNAME}" ]; then
    mv "${DIRNAME}" "internvl2_5_1b_${DIRNAME}"
  fi
  export PYTHONPATH="$(pwd)/../..:${PYTHONPATH}"
  source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && python eval.py --checkpoint ${CHECKPOINT} --root ../../data/mme/test_4experts_balanced_vitb16/cluster0_txt_files "${ARGS[@]:2}"
  python calculation.py --results_dir ${DIRNAME}
  cd ../../
fi

if  [ ${DATASET} == "mme-balanced-vitb16-4experts-cluster1" ]; then
  cd eval/mme/
  DIRNAME=`basename ${CHECKPOINT}`
  # Backup 1B results if running 2B model
  if [[ ${CHECKPOINT} == *"internvl2_5_2b"* ]] && [ -d "${DIRNAME}" ] && [ ! -d "internvl2_5_1b_${DIRNAME}" ]; then
    mv "${DIRNAME}" "internvl2_5_1b_${DIRNAME}"
  fi
  export PYTHONPATH="$(pwd)/../..:${PYTHONPATH}"
  source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && python eval.py --checkpoint ${CHECKPOINT} --root ../../data/mme/test_4experts_balanced_vitb16/cluster1_txt_files "${ARGS[@]:2}"
  python calculation.py --results_dir ${DIRNAME}
  cd ../../
fi

if  [ ${DATASET} == "mme-balanced-vitb16-4experts-cluster2" ]; then
  cd eval/mme/
  DIRNAME=`basename ${CHECKPOINT}`
  # Backup 1B results if running 2B model
  if [[ ${CHECKPOINT} == *"internvl2_5_2b"* ]] && [ -d "${DIRNAME}" ] && [ ! -d "internvl2_5_1b_${DIRNAME}" ]; then
    mv "${DIRNAME}" "internvl2_5_1b_${DIRNAME}"
  fi
  export PYTHONPATH="$(pwd)/../..:${PYTHONPATH}"
  source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && python eval.py --checkpoint ${CHECKPOINT} --root ../../data/mme/test_4experts_balanced_vitb16/cluster2_txt_files "${ARGS[@]:2}"
  python calculation.py --results_dir ${DIRNAME}
  cd ../../
fi

if  [ ${DATASET} == "mme-balanced-vitb16-4experts-cluster3" ]; then
  cd eval/mme/
  DIRNAME=`basename ${CHECKPOINT}`
  # Backup 1B results if running 2B model
  if [[ ${CHECKPOINT} == *"internvl2_5_2b"* ]] && [ -d "${DIRNAME}" ] && [ ! -d "internvl2_5_1b_${DIRNAME}" ]; then
    mv "${DIRNAME}" "internvl2_5_1b_${DIRNAME}"
  fi
  export PYTHONPATH="$(pwd)/../..:${PYTHONPATH}"
  source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && python eval.py --checkpoint ${CHECKPOINT} --root ../../data/mme/test_4experts_balanced_vitb16/cluster3_txt_files "${ARGS[@]:2}"
  python calculation.py --results_dir ${DIRNAME}
  cd ../../
fi

if  [ ${DATASET} == "caption" ]; then
  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/caption/evaluate_caption.py --checkpoint ${CHECKPOINT} "${ARGS[@]:2}"
fi

if  [ ${DATASET} == "caption-coco" ]; then
  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/caption/evaluate_caption.py --checkpoint ${CHECKPOINT} --datasets coco "${ARGS[@]:2}"
fi

if  [ ${DATASET} == "caption-flickr30k" ]; then
  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/caption/evaluate_caption.py --checkpoint ${CHECKPOINT} --datasets flickr30k "${ARGS[@]:2}"
fi

if  [ ${DATASET} == "caption-nocaps" ]; then
  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/caption/evaluate_caption.py --checkpoint ${CHECKPOINT} --datasets nocaps "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-okvqa-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets okvqa_val "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-okvqa-val-cluster0" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets okvqa_val_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-okvqa-val-cluster1" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets okvqa_val_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-textvqa-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets textvqa_val "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-textvqa-val-cluster0" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets textvqa_val_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-textvqa-val-cluster1" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets textvqa_val_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-textvqa-val-ocr" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets textvqa_val_ocr "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-textvqa-val-balanced-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets textvqa_val_balanced_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-textvqa-val-balanced-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets textvqa_val_balanced_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-textvqa-val-balanced-vitl14-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets textvqa_val_balanced_vitl14_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-textvqa-val-balanced-vitl14-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets textvqa_val_balanced_vitl14_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-textvqa-val-balanced-vitb16-4experts-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets textvqa_val_balanced_vitb16_4experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-textvqa-val-balanced-vitb16-4experts-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets textvqa_val_balanced_vitb16_4experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-textvqa-val-balanced-vitb16-4experts-cluster2" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets textvqa_val_balanced_vitb16_4experts_cluster2 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-textvqa-val-balanced-vitb16-4experts-cluster3" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets textvqa_val_balanced_vitb16_4experts_cluster3 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-vizwiz-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets vizwiz_val "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-vizwiz-test" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets vizwiz_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-vqav2-testdev" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets vqav2_testdev "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-ai2d-test" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets ai2diagram_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-ai2d-test-balanced-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets ai2diagram_test_balanced_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-ai2d-test-balanced-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets ai2diagram_test_balanced_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-ai2d-test-balanced-vitl14-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets ai2diagram_test_balanced_vitl14_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-ai2d-test-balanced-vitl14-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets ai2diagram_test_balanced_vitl14_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-vqav2-val" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets vqav2_val "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-vqav2-val-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets vqav2_val_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-vqav2-val-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets vqav2_val_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-vqav2-val-balanced-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets vqav2_val_balanced_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-vqav2-val-balanced-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets vqav2_val_balanced_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-vqav2-val-balanced-vitl14-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets vqav2_val_balanced_vitl14_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-vqav2-val-balanced-vitl14-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets vqav2_val_balanced_vitl14_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-vqav2-val-balanced-vitb16-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets vqav2_val_balanced_vitb16_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-vqav2-val-balanced-vitb16-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets vqav2_val_balanced_vitb16_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-vqav2-val-balanced-vitb16-4experts-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets vqav2_val_balanced_vitb16_4experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-vqav2-val-balanced-vitb16-4experts-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets vqav2_val_balanced_vitb16_4experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-vqav2-val-balanced-vitb16-4experts-cluster2" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets vqav2_val_balanced_vitb16_4experts_cluster2 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-vqav2-val-balanced-vitb16-4experts-cluster3" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets vqav2_val_balanced_vitb16_4experts_cluster3 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-gqa-testdev" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets gqa_testdev_llava "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-gqa-testdev-balanced-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets gqa_testdev_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-gqa-testdev-balanced-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets gqa_testdev_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-gqa-testdev-balanced-vitl14-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets gqa_testdev_balanced_vitl14_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-gqa-testdev-balanced-vitl14-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets gqa_testdev_balanced_vitl14_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-docvqa-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets docvqa_val "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-docvqa-val-cluster0" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets docvqa_val_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-docvqa-val-cluster1" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets docvqa_val_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-docvqa-test" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets docvqa_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-docvqa-val-balanced-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets docvqa_val_balanced_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-docvqa-val-balanced-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets docvqa_val_balanced_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-docvqa-val-balanced-vitl14-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets docvqa_val_balanced_vitl14_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-docvqa-val-balanced-vitl14-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets docvqa_val_balanced_vitl14_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-mpdocvqa-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/mpdocvqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets mpdocvqa_val "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-mpdocvqa-test" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/mpdocvqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets mpdocvqa_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-chartqa-test" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets chartqa_test_human,chartqa_test_augmented "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-infovqa-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets infographicsvqa_val "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-infovqa-val-cluster0" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets infographicsvqa_val_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-infovqa-val-cluster1" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets infographicsvqa_val_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-infovqa-test" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets infographicsvqa_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-infovqa-val-balanced-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets infographicsvqa_val_balanced_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-infovqa-val-balanced-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets infographicsvqa_val_balanced_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-infovqa-val-balanced-vitl14-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets infographicsvqa_val_balanced_vitl14_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-infovqa-val-balanced-vitl14-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets infographicsvqa_val_balanced_vitl14_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-chartqa-test-human" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets chartqa_test_human "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-chartqa-test-human-cluster0" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets chartqa_test_human_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-chartqa-test-human-cluster1" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets chartqa_test_human_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-chartqa-test-augmented" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets chartqa_test_augmented "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-chartqa-test-augmented-cluster0" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets chartqa_test_augmented_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-chartqa-test-augmented-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets chartqa_test_augmented_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-chartqa-test-human-balanced-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets chartqa_test_human_balanced_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-chartqa-test-human-balanced-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets chartqa_test_human_balanced_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-chartqa-test-augmented-balanced-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets chartqa_test_augmented_balanced_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-chartqa-test-augmented-balanced-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets chartqa_test_augmented_balanced_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-chartqa-test-human-balanced-vitl14-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets chartqa_test_human_balanced_vitl14_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-chartqa-test-human-balanced-vitl14-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets chartqa_test_human_balanced_vitl14_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-chartqa-test-augmented-balanced-vitl14-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets chartqa_test_augmented_balanced_vitl14_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-chartqa-test-augmented-balanced-vitl14-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets chartqa_test_augmented_balanced_vitl14_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-ocrvqa-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets ocrvqa_val "${ARGS[@]:2}"
fi

if [ ${DATASET} == "vqa-ocrvqa-test" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets ocrvqa_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_val "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco-testA" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_testA "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco-testB" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_testB "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco-val-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_val_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco-val-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_val_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco-testA-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_testA_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco-testA-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_testA_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco-testA-balanced-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_testA_balanced_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco-testA-balanced-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_testA_balanced_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco-testB-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_testB_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco-testB-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_testB_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco-testB-balanced-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_testB_balanced_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco-testB-balanced-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_testB_balanced_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco-val-balanced-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_val_balanced_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco-val-balanced-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_val_balanced_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco+-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco+_val "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco+-val-balanced-vitb16-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco+_val_balanced_vitb16_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco+-val-balanced-vitb16-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco+_val_balanced_vitb16_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco+-testA-balanced-vitb16-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco+_testA_balanced_vitb16_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco+-testA-balanced-vitb16-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco+_testA_balanced_vitb16_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco+-testB-balanced-vitb16-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco+_testB_balanced_vitb16_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco+-testB-balanced-vitb16-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco+_testB_balanced_vitb16_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco+-testA" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco+_testA "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco+-testB" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco+_testB "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcocog-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcocog_val "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcocog-val-balanced-vitb16-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcocog_val_balanced_vitb16_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcocog-val-balanced-vitb16-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcocog_val_balanced_vitb16_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcocog-test-balanced-vitb16-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcocog_test_balanced_vitb16_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcocog-test-balanced-vitb16-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcocog_test_balanced_vitb16_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcocog-test" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcocog_test "${ARGS[@]:2}"
fi

# RefCOCO 4-expert balanced vitb16 clusters
if [ ${DATASET} == "refcoco-val-balanced-vitb16-4experts-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_val_balanced_vitb16_4experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco-val-balanced-vitb16-4experts-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_val_balanced_vitb16_4experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco-val-balanced-vitb16-4experts-cluster2" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_val_balanced_vitb16_4experts_cluster2 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco-val-balanced-vitb16-4experts-cluster3" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_val_balanced_vitb16_4experts_cluster3 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco-testA-balanced-vitb16-4experts-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_testA_balanced_vitb16_4experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco-testA-balanced-vitb16-4experts-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_testA_balanced_vitb16_4experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco-testA-balanced-vitb16-4experts-cluster2" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_testA_balanced_vitb16_4experts_cluster2 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco-testA-balanced-vitb16-4experts-cluster3" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_testA_balanced_vitb16_4experts_cluster3 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco-testB-balanced-vitb16-4experts-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_testB_balanced_vitb16_4experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco-testB-balanced-vitb16-4experts-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_testB_balanced_vitb16_4experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco-testB-balanced-vitb16-4experts-cluster2" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_testB_balanced_vitb16_4experts_cluster2 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "refcoco-testB-balanced-vitb16-4experts-cluster3" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_testB_balanced_vitb16_4experts_cluster3 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "llava-bench" ]; then
    rm -rf results/llava_bench_results_review.jsonl
    python eval/llava_bench/evaluate_llava_bench.py --checkpoint ${CHECKPOINT} "${ARGS[@]:2}"
    python -u eval/llava_bench/eval_gpt_review_bench.py \
      --question data/llava-bench-in-the-wild/questions.jsonl \
      --context data/llava-bench-in-the-wild/context.jsonl \
      --rule eval/llava_bench/rule.json \
      --answer-list \
          data/llava-bench-in-the-wild/answers_gpt4.jsonl \
          results/llava_bench_results.jsonl \
      --output \
          results/llava_bench_results_review.jsonl
    python -u eval/llava_bench/summarize_gpt_review.py -f results/llava_bench_results_review.jsonl
fi

if [ ${DATASET} == "pope" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/pope/evaluate_pope.py --checkpoint ${CHECKPOINT} --datasets pope "${ARGS[@]:2}"
fi

if [ ${DATASET} == "pope-2experts-cluster0" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/pope/evaluate_pope.py --checkpoint ${CHECKPOINT} --datasets pope-2experts-cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "pope-2experts-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/pope/evaluate_pope.py --checkpoint ${CHECKPOINT} --datasets pope-2experts-cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "pope-balanced-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/pope/evaluate_pope.py --checkpoint ${CHECKPOINT} --datasets pope-balanced-2experts-cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "pope-balanced-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/pope/evaluate_pope.py --checkpoint ${CHECKPOINT} --datasets pope-balanced-2experts-cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "pope-balanced-vitl14-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/pope/evaluate_pope.py --checkpoint ${CHECKPOINT} --datasets pope-balanced-vitl14-2experts-cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "pope-balanced-vitl14-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/pope/evaluate_pope.py --checkpoint ${CHECKPOINT} --datasets pope-balanced-vitl14-2experts-cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "tiny_lvlm" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/tiny_lvlm/evaluate_lvlm.py --checkpoint ${CHECKPOINT} --datasets updated_datasets "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmvet" ]; then
    python eval/mmvet/evaluate_mmvet.py --checkpoint ${CHECKPOINT} --datasets mmvet "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmvetv2" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmvetv2/evaluate_mmvet_v2.py --checkpoint ${CHECKPOINT} --datasets mmvet-v2 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmbench-dev-en" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmbench/evaluate_mmbench.py --checkpoint ${CHECKPOINT} --datasets mmbench_dev_20230712 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmbench-dev-cn" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmbench/evaluate_mmbench.py --checkpoint ${CHECKPOINT} --datasets mmbench_dev_cn_20231003 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmbench-test-en" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmbench/evaluate_mmbench.py --checkpoint ${CHECKPOINT} --datasets mmbench_test_en_20231003 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmbench-test-cn" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmbench/evaluate_mmbench.py --checkpoint ${CHECKPOINT} --datasets mmbench_test_cn_20231003 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "ccbench-dev" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmbench/evaluate_mmbench.py --checkpoint ${CHECKPOINT} --datasets ccbench_dev_cn "${ARGS[@]:2}"
fi

if [ ${DATASET} == "scienceqa" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/scienceqa/evaluate_scienceqa.py --checkpoint ${CHECKPOINT} --datasets sqa_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "scienceqa-cluster0" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/scienceqa/evaluate_scienceqa.py --checkpoint ${CHECKPOINT} --datasets sqa_test_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "scienceqa-cluster1" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/scienceqa/evaluate_scienceqa.py --checkpoint ${CHECKPOINT} --datasets sqa_test_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "scienceqa-2experts-cluster0" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/scienceqa/evaluate_scienceqa.py --checkpoint ${CHECKPOINT} --datasets sqa_test_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "scienceqa-2experts-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/scienceqa/evaluate_scienceqa.py --checkpoint ${CHECKPOINT} --datasets sqa_test_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "scienceqa-balanced-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/scienceqa/evaluate_scienceqa.py --checkpoint ${CHECKPOINT} --datasets sqa_test_balanced_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "scienceqa-balanced-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/scienceqa/evaluate_scienceqa.py --checkpoint ${CHECKPOINT} --datasets sqa_test_balanced_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "scienceqa-balanced-vitl14-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/scienceqa/evaluate_scienceqa.py --checkpoint ${CHECKPOINT} --datasets sqa_test_balanced_vitl14_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "scienceqa-balanced-vitl14-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/scienceqa/evaluate_scienceqa.py --checkpoint ${CHECKPOINT} --datasets sqa_test_balanced_vitl14_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "scienceqa-cluster2" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/scienceqa/evaluate_scienceqa.py --checkpoint ${CHECKPOINT} --datasets sqa_test_cluster2 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "scienceqa-cluster3" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/scienceqa/evaluate_scienceqa.py --checkpoint ${CHECKPOINT} --datasets sqa_test_cluster3 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mantis" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mantis_eval/evaluate_mantis.py --checkpoint ${CHECKPOINT} --datasets Mantis-Eval "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mirb" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mirb/evaluate_mirb.py --checkpoint ${CHECKPOINT} --datasets MIRB "${ARGS[@]:2}"
fi

if [ ${DATASET} == "m3cot" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/scienceqa/evaluate_scienceqa.py --checkpoint ${CHECKPOINT} --datasets m3cot_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmmu-dev" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmmu/evaluate_mmmu.py --checkpoint ${CHECKPOINT} --datasets MMMU_dev "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmmu-val" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmmu/evaluate_mmmu.py --checkpoint ${CHECKPOINT} --datasets MMMU_validation "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmmu-test" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmmu/evaluate_mmmu.py --checkpoint ${CHECKPOINT} --datasets MMMU_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmmu-dev-cot" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmmu/evaluate_mmmu_cot.py --checkpoint ${CHECKPOINT} --datasets MMMU_dev "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmmu-val-cot" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmmu/evaluate_mmmu_cot.py --checkpoint ${CHECKPOINT} --datasets MMMU_validation "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmmu-test-cot" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmmu/evaluate_mmmu_cot.py --checkpoint ${CHECKPOINT} --datasets MMMU_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmvp" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmvp/evaluate_mmvp.py --checkpoint ${CHECKPOINT} --datasets MMVP "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mathvista-testmini" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mathvista/evaluate_mathvista.py --checkpoint ${CHECKPOINT} --datasets MathVista_testmini "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mathvista-test" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mathvista/evaluate_mathvista.py --checkpoint ${CHECKPOINT} --datasets MathVista_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "seed" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/seed/evaluate_seed.py --checkpoint ${CHECKPOINT} --datasets SEEDv1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "seed-cluster0" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/seed/evaluate_seed.py --checkpoint ${CHECKPOINT} --datasets SEEDv1_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "seed-cluster1" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/seed/evaluate_seed.py --checkpoint ${CHECKPOINT} --datasets SEEDv1_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mvbench" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mvbench/evaluate_mvbench.py --checkpoint ${CHECKPOINT} --num_segments 16 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmiu" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmiu/evaluate_mmiu.py --checkpoint ${CHECKPOINT} "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmhal" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmhal/evaluate_mmhal.py --checkpoint ${CHECKPOINT} "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mathvista-testmini" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mathvista/evaluate_mathvista.py --checkpoint ${CHECKPOINT} --datasets MathVista_testmini "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mathvista-test" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mathvista/evaluate_mathvista.py --checkpoint ${CHECKPOINT} --datasets MathVista_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "seed" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/seed/evaluate_seed.py --checkpoint ${CHECKPOINT} --datasets SEEDv1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "seed-cluster0" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/seed/evaluate_seed.py --checkpoint ${CHECKPOINT} --datasets SEEDv1_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "seed-cluster1" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/seed/evaluate_seed.py --checkpoint ${CHECKPOINT} --datasets SEEDv1_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mvbench" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mvbench/evaluate_mvbench.py --checkpoint ${CHECKPOINT} --num_segments 16 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmiu" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmiu/evaluate_mmiu.py --checkpoint ${CHECKPOINT} "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmhal" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmhal/evaluate_mmhal.py --checkpoint ${CHECKPOINT} "${ARGS[@]:2}"
fi

if [ ${DATASET} == "scienceqa-balanced-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/scienceqa/evaluate_scienceqa.py --checkpoint ${CHECKPOINT} --datasets sqa_test_balanced_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "scienceqa-balanced-vitl14-cluster0" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/scienceqa/evaluate_scienceqa.py --checkpoint ${CHECKPOINT} --datasets sqa_test_balanced_vitl14_2experts_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "scienceqa-balanced-vitl14-cluster1" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate internvl && torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/scienceqa/evaluate_scienceqa.py --checkpoint ${CHECKPOINT} --datasets sqa_test_balanced_vitl14_2experts_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "scienceqa-cluster2" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/scienceqa/evaluate_scienceqa.py --checkpoint ${CHECKPOINT} --datasets sqa_test_cluster2 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "scienceqa-cluster3" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/scienceqa/evaluate_scienceqa.py --checkpoint ${CHECKPOINT} --datasets sqa_test_cluster3 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mantis" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mantis_eval/evaluate_mantis.py --checkpoint ${CHECKPOINT} --datasets Mantis-Eval "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mirb" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mirb/evaluate_mirb.py --checkpoint ${CHECKPOINT} --datasets MIRB "${ARGS[@]:2}"
fi

if [ ${DATASET} == "m3cot" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/scienceqa/evaluate_scienceqa.py --checkpoint ${CHECKPOINT} --datasets m3cot_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmmu-dev" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmmu/evaluate_mmmu.py --checkpoint ${CHECKPOINT} --datasets MMMU_dev "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmmu-val" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmmu/evaluate_mmmu.py --checkpoint ${CHECKPOINT} --datasets MMMU_validation "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmmu-test" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmmu/evaluate_mmmu.py --checkpoint ${CHECKPOINT} --datasets MMMU_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmmu-dev-cot" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmmu/evaluate_mmmu_cot.py --checkpoint ${CHECKPOINT} --datasets MMMU_dev "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmmu-val-cot" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmmu/evaluate_mmmu_cot.py --checkpoint ${CHECKPOINT} --datasets MMMU_validation "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmmu-test-cot" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmmu/evaluate_mmmu_cot.py --checkpoint ${CHECKPOINT} --datasets MMMU_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmvp" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmvp/evaluate_mmvp.py --checkpoint ${CHECKPOINT} --datasets MMVP "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mathvista-testmini" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mathvista/evaluate_mathvista.py --checkpoint ${CHECKPOINT} --datasets MathVista_testmini "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mathvista-test" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mathvista/evaluate_mathvista.py --checkpoint ${CHECKPOINT} --datasets MathVista_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "seed" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/seed/evaluate_seed.py --checkpoint ${CHECKPOINT} --datasets SEEDv1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "seed-cluster0" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/seed/evaluate_seed.py --checkpoint ${CHECKPOINT} --datasets SEEDv1_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "seed-cluster1" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/seed/evaluate_seed.py --checkpoint ${CHECKPOINT} --datasets SEEDv1_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mvbench" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mvbench/evaluate_mvbench.py --checkpoint ${CHECKPOINT} --num_segments 16 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmiu" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmiu/evaluate_mmiu.py --checkpoint ${CHECKPOINT} "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmhal" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmhal/evaluate_mmhal.py --checkpoint ${CHECKPOINT} "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mathvista-testmini" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mathvista/evaluate_mathvista.py --checkpoint ${CHECKPOINT} --datasets MathVista_testmini "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mathvista-test" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mathvista/evaluate_mathvista.py --checkpoint ${CHECKPOINT} --datasets MathVista_test "${ARGS[@]:2}"
fi

if [ ${DATASET} == "seed" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/seed/evaluate_seed.py --checkpoint ${CHECKPOINT} --datasets SEEDv1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "seed-cluster0" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/seed/evaluate_seed.py --checkpoint ${CHECKPOINT} --datasets SEEDv1_cluster0 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "seed-cluster1" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/seed/evaluate_seed.py --checkpoint ${CHECKPOINT} --datasets SEEDv1_cluster1 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mvbench" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mvbench/evaluate_mvbench.py --checkpoint ${CHECKPOINT} --num_segments 16 "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmiu" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmiu/evaluate_mmiu.py --checkpoint ${CHECKPOINT} "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmhal" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmhal/evaluate_mmhal.py --checkpoint ${CHECKPOINT} "${ARGS[@]:2}"
fi