

export PYTHONPATH=$PYTHONPATH:src/mha2mla-vlm/llavanext
export PYTHONPATH=$PYTHONPATH:lmms-eval


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_processes=8

model_path="llavanext/checkpoints/stage2_ckpt"
cache_config="HQQ_4" # HQQ_4/Quanto_4

accelerate launch \
    --num_processes=${num_processes} \
    -m lmms_eval \
    --model llava_hf \
    --model_args pretrained="${model_path}",cache_config="${cache_config}" \
    --tasks ai2d,gqa,pope,seedbench,realworldqa,mmbench_en,chartqa,docvqa_val \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix reproduce \
    --output_path eval/llavanext_quant/logs 
