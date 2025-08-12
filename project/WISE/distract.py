import json

json_path = "/mnt/bn/wxd-video-understanding/wangxd/ULM-R1/project/WISE/data/cultural_common_sense.json"
with open(json_path, "r") as f:
    data = json.load(f)

prompt_id_common = []
for item in data:
    prompt_id = item["prompt_id"]
    prompt_id_common.append(prompt_id)

json_path = "/mnt/bn/wxd-video-understanding/wangxd/ULM-R1/project/WISE/data/natural_science.json"
with open(json_path, "r") as f:
    data = json.load(f)

prompt_id_natural = []
for item in data:
    prompt_id = item["prompt_id"]
    prompt_id_natural.append(prompt_id)

json_path = "/mnt/bn/wxd-video-understanding/wangxd/ULM-R1/project/WISE/data/spatio-temporal_reasoning.json"
with open(json_path, "r") as f:
    data = json.load(f)

prompt_id_spatio_temporal = []
for item in data:
    prompt_id = item["prompt_id"]
    prompt_id_spatio_temporal.append(prompt_id)

image_dir = "/mnt/bn/wxd-video-understanding/wangxd/ULM-R1/project/WISE/janus-pro-1B-wise"

# if number.png in prompt_id_common, move the file number.png to Results/cultural_common_sense dir
import os
import shutil

for prompt_id in prompt_id_common:
    prompt_id = str(prompt_id)
    if prompt_id + ".png" in os.listdir(image_dir):
        shutil.move(f"{image_dir}/" + prompt_id + ".png", f"{image_dir}/Results/cultural_common_sense/" + prompt_id + ".png")

for prompt_id in prompt_id_natural:
    prompt_id = str(prompt_id)
    if prompt_id + ".png" in os.listdir(image_dir):
        shutil.move(f"{image_dir}/" + prompt_id + ".png", f"{image_dir}/Results/natural_science/" + prompt_id + ".png")

for prompt_id in prompt_id_spatio_temporal:
    prompt_id = str(prompt_id)
    if prompt_id + ".png" in os.listdir(image_dir):
        shutil.move(f"{image_dir}/" + prompt_id + ".png", f"{image_dir}/Results/spatio-temporal_reasoning/" + prompt_id + ".png")