export PYTHONPATH=$PYTHONPATH:src/mha2mla-vlm/qwen2_5_vl

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_processes=8

model_path="qwen2_5_vl/checkpoints/stage2_ckpt"



accelerate launch \
    --num_processes=${num_processes} \
    -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=${model_path},max_pixels=12845056 \
    --tasks ai2d,gqa,pope,seedbench,realworldqa,mmbench_en,chartqa,docvqa_val \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix reproduce \
    --output_path eval/qwen2_5_vl/logs 




