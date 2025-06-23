#!/bin/bash

# ******************************************************************** #
CKPT_PATH=XXX/checkpoint/Janus-Pro-1B
DATA_PATH=ttrl/data/geneval
SAVE_DIR=XXX/experiment/JanusPro-1B-TTRL-T2I

reward_funcs="ttrl_cycle_cs"
beta=0.04
learning_rate=3e-6
num_train_epochs=1
max_steps=100
max_prompt_length=1024
max_completion_length=576

num_generation=8
gradient_accumulation_steps=1
per_device_train_batch_size=16

# JaccardBertscoreMse
caption_cs_metrics="jaccard bertscore"  # jaccard bertscore
image_cs_metrics="mse"  # mse
using_image_cs=True
using_simcse=False
using_external_caption_model=False

SAVE_PATH=${SAVE_DIR}/geneval-selfCS_G8-bs16-ebs128-lr3e6
mkdir -p $SAVE_PATH
cp $0 $SAVE_PATH/run.sh
torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    ttrl/grpo_janus_t2i.py \
    --reward_funcs ${reward_funcs} \
    --using_external_caption_model $using_external_caption_model \
    --caption_cs_metrics ${caption_cs_metrics} \
    --image_cs_metrics ${image_cs_metrics} \
    --using_image_cs $using_image_cs \
    --using_simcse $using_simcse \
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
    --max_steps $max_steps \
    --learning_rate $learning_rate \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing false \
    --save_steps 20 \
    --save_total_limit 5 \
    --save_only_model true
# ******************************************************************** #
# inference
N_CHUNKS=8

for i in $(seq 0 $(($N_CHUNKS - 1))); do
    echo "Launching process for GPU $i (chunk index $i of $N_CHUNKS)"
    CUDA_VISIBLE_DEVICES=$i python eval/t2i_infer_gpus.py \
    --model_path ${SAVE_PATH} \
    --eval_data eval/t2i/geneval/prompts/geneval_prompt.jsonl \
    --index $i \
    --n_chunks $N_CHUNKS &
done

wait
echo "All processes finished."
# ******************************************************************** #


