#!/bin/bash

# Configuration
NUM_GPUS=8
PROMPT_FILE=/mnt/bn/wxd-video-understanding/wangxd/ULM-R1/project/WISE/data/wise.json
OUTPUT_DIR=/mnt/bn/wxd-video-understanding/wangxd/ULM-R1/project/WISE/data/janus-pro-1B-RFT22k-CycleMatchAccFormat-UniReward-G8-beta004-bs16
MODEL_PATH="/mnt/bn/wxd-video-understanding/wangxd/ULM-R1/experiments/JanusPro-1B-CoRL-Unified/RFT22k-CycleMatchAccFormat-UniReward-G8-beta004-bs16"

# Create output directory
mkdir -p $OUTPUT_DIR

# Launch inference on all GPUs
for ((i=0; i<$NUM_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i python generate_image.py \
        --num_splits $NUM_GPUS \
        --split_idx $i \
        --prompt_file $PROMPT_FILE \
        --output_dir $OUTPUT_DIR \
        --model_path $MODEL_PATH &
done

# Wait for all background processes to finish
wait

echo "All inference jobs completed!"