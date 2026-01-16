



#!/bin/bash

# ============ Configuration ============
BASE_MODEL="llama3-llava-next-8b-hf"
STAGE1_CKPT_PATH="llavanext/checkpoints/stage1_ckpt"
DATASET_PATH="data/sample-mix-512.json"
TOTAL_NUM=512
BATCH_SIZE=1
SHUFFLE=True
LOW_RANK=32     # Low rank dimension: 32/64/128 (see Appendix for details)                           # Low rank dimension
SEED=42

# Output path will include the LOW_RANK value
OUTPUT_PATH="llavanext/analysis_svd/results/rope32_low_rank${LOW_RANK}.pt"
# =======================================

CUDA_VISIBLE_DEVICES=0 python src/mha2mla-vlm/llavanext/run_svd_init.py \
  --base_model "${BASE_MODEL}" \
  --stage1_ckpt_path "${STAGE1_CKPT_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --output_path ${OUTPUT_PATH} \
  --total_num ${TOTAL_NUM} \
  --batch_size ${BATCH_SIZE} \
  --shuffle ${SHUFFLE} \
  --low_rank ${LOW_RANK} \
  --seed ${SEED}

