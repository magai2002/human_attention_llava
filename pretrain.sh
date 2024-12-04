#!/bin/bash

#SBATCH -t 48:00:00
#SBATCH -o setup_llava_env_output.txt   # Output log file
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:4


# Hugging Face Transformers cache directory
export TRANSFORMERS_CACHE=/scratch/am11351/hf_cache

# PyTorch cache directory
export TORCH_HOME=/scratch/am11351/torch_cache

# CUDA cache directory
export CUDA_CACHE_PATH=/scratch/am11351/cuda_cache

# Hugging Face Datasets cache directory
export HF_DATASETS_CACHE=/scratch/am11351/datasets_cache

# Set temporary files directory
export TMPDIR=/scratch/am11351/tmp

# Create directories if they don't exist
mkdir -p $TRANSFORMERS_CACHE $TORCH_HOME $CUDA_CACHE_PATH $HF_DATASETS_CACHE $TMPDIR

export PYTHONPATH=$PYTHONPATH:/home/am11351/ondemand/data/sys/myjobs/project/default/2

#Add the lines for running your code/application
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate

conda activate /home/am11351/.conda/envs/llava_env

echo "Current conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --version plain \
    --data_path /scratch/am11351/data/blip_laion_cc_sbu_558k.json \
    --image_folder /scratch/am11351/data/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /scratch/am11351/llava/LLaVA/checkpoints/llava-v1.5-13b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb