#!/bin/bash

# required: trl>=0.18.1

# ******************************************************************** #
CKPT_PATH=XXX/checkpoint/Janus-Pro-1B
DATA_PATH=ttrl/data/mmstar
SAVE_DIR=XXX/experiment/JanusPro-1B-TTRL-MM2T

reward_funcs="mcq_ttrl"
beta=0.04
learning_rate=2e-6
num_train_epochs=1
max_steps=60

max_prompt_length=1024
max_completion_length=512

num_generation=8
gradient_accumulation_steps=2
per_device_train_batch_size=8

SAVE_PATH=${SAVE_DIR}/mmstar_G8-bs16-ebs128-lr2e6
mkdir -p $SAVE_PATH
cp $0 $SAVE_PATH/run.sh
# --deepspeed scripts/zero3.json
# --max_steps $max_steps \
torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    ttrl/grpo_janus_mm2t.py \
    --reward_funcs ${reward_funcs} \
    --output_dir ${SAVE_PATH}  \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --report_to "none" \
    --logging_steps 1 \
    --beta $beta \
    --max_prompt_length $max_prompt_length \
    --max_completion_length $max_completion_length \
    --num_generations $num_generation \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --num_train_epochs $num_train_epochs \
    --learning_rate $learning_rate \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing false \
    --save_steps 50 \
    --save_total_limit 5 \
    --save_only_model true
# ******************************************************************** #
# inference
N_CHUNKS=8

for i in $(seq 0 $(($N_CHUNKS - 1))); do
    echo "Launching process for GPU $i (chunk index $i of $N_CHUNKS)"
    CUDA_VISIBLE_DEVICES=$i python eval/mm2t_infer_gpus.py \
    --model_path ${SAVE_PATH} \
    --eval_data eval/mm2t/mmstar/test.json \
    --img_dir eval/mm2t/mmstar \
    --index $i \
    --n_chunks $N_CHUNKS &
done

wait
echo "All processes finished."
# ******************************************************************** #

