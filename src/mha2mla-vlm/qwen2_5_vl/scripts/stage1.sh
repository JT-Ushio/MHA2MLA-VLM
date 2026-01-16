#!/bin/bash

# Qwen2.5-VL MHA2MLA Training Script with YAML Configuration
# This script demonstrates how to run training using YAML config file

# ========== Environment Setup ==========

source /xxx/envs/mha2mla-vlm/bin/activate 
conda activate /xxx/envs/mha2mla-vlm

export WANDB_MODE=offline

# ========== Training with YAML Config ==========

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
torchrun --nproc_per_node 2 \
    train_stage1.py \
    --cfg_file cfgs/stage1.yaml
