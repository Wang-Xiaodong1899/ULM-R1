IMAGE_DIR=/mnt/bn/wxd-video-understanding/wangxd/ULM-R1/project/WISE/data/janus-pro-1B-RFT22k-CycleMatchAccFormat-UniReward-G8-beta004-bs16

python gpt_eval.py \
    --json_path data/cultural_common_sense.json \
    --output_dir ${IMAGE_DIR}/Results/cultural_common_sense \
    --image_dir ${IMAGE_DIR} \
    --api_key "" \
    --model "gpt-4o-2024-05-13" \
    --result_full ${IMAGE_DIR}/Results/cultural_common_sense_full_results.json \
    --result_scores ${IMAGE_DIR}/Results/cultural_common_sense_scores_results.jsonl \
    --max_workers 96

python gpt_eval.py \
    --json_path data/spatio-temporal_reasoning.json \
    --output_dir ${IMAGE_DIR}/Results/spatio-temporal_reasoning \
    --image_dir ${IMAGE_DIR} \
    --api_key "" \
    --model "gpt-4o-2024-05-13" \
    --result_full ${IMAGE_DIR}/Results/spatio-temporal_reasoning_results.json \
    --result_scores ${IMAGE_DIR}/Results/spatio-temporal_reasoning_results.jsonl \
    --max_workers 96

python gpt_eval.py \
    --json_path data/natural_science.json \
    --output_dir ${IMAGE_DIR}/Results/natural_science \
    --image_dir ${IMAGE_DIR} \
    --api_key "" \
    --model "gpt-4o-2024-05-13" \
    --result_full ${IMAGE_DIR}/Results/natural_science_full_results.json \
    --result_scores ${IMAGE_DIR}/Results/natural_science_scores_results.jsonl \
    --max_workers 96

python Calculate.py \
    "${IMAGE_DIR}/Results/cultural_common_sense_scores_results.jsonl" \
    "${IMAGE_DIR}/Results/natural_science_scores_results.jsonl" \
    "${IMAGE_DIR}/Results/spatio-temporal_reasoning_results.jsonl" \
    --category all