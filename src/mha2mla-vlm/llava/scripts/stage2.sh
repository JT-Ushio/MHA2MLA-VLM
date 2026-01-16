#!/bin/bash

# LLaVA MHA2MLA Stage2 Training Script with YAML Configuration
# This script demonstrates how to run stage2 training using YAML config file

# ========== Environment Setup ==========

source /xxx/envs/mha2mla-vlm/bin/activate 
conda activate /xxx/envs/mha2mla-vlm

export WANDB_MODE=offline

# ========== Training with YAML Config ==========

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun \
    --nproc_per_node 1 \
    train_stage2.py \
    --cfg_file cfgs/stage2.yaml

