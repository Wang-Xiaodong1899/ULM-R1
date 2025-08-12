import os
import PIL.Image
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
import threading
from queue import Queue
import json
from tqdm import tqdm
import argparse

# 全局变量，用于存储模型和处理器（避免重复加载）
global_models = {}
model_lock = threading.Lock()
SAVE_DIR = "/mnt/bn/wxd-video-understanding/wangxd/ULM-R1/project/WISE/janus-pro-1B-wise/"

def init_model(gpu_id, args):
    model_path = args.model_path
    with model_lock:
        if gpu_id not in global_models:
            torch.cuda.set_device(gpu_id)
            print(f"Initializing model on GPU {gpu_id}")
            
            # 加载处理器和tokenizer
            vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
            
            # 加载模型
            vl_gpt = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True
            )
            vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
            
            global_models[gpu_id] = {
                'model': vl_gpt,
                'processor': vl_chat_processor
            }
    return global_models[gpu_id]

def worker(gpu_id, prompt_queue, args):
    # 初始化模型
    model_info = init_model(gpu_id, args)
    vl_gpt = model_info['model']
    vl_chat_processor = model_info['processor']
    
    while True:
        item = prompt_queue.get()
        if item is None:  # 结束信号
            break
        
        prompt_data = item['prompt_data']
        worker_id = item['worker_id']
        
        try:
            # 准备对话输入
            conversation = [
                {
                    "role": "<|User|>",
                    "content": prompt_data["Prompt"],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            
            # 应用模板
            sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=vl_chat_processor.sft_format,
                system_prompt="",
            )
            prompt = sft_format + vl_chat_processor.image_start_tag
            
            # 生成图像
            generate(vl_gpt, vl_chat_processor, prompt, prompt_data["prompt_id"], gpu_id, args=args)
            
            print(f"Worker {worker_id} (GPU {gpu_id}) processed prompt_id {prompt_data['prompt_id']}")
        except Exception as e:
            print(f"Worker {worker_id} (GPU {gpu_id}) failed to process prompt_id {prompt_data['prompt_id']}: {str(e)}")
        
        prompt_queue.task_done()

@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    prompt_id: int,
    gpu_id: int = 0,
    temperature: float = 1,
    parallel_size: int = 1,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    args=None,
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda(gpu_id)
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda(gpu_id)

    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state
        
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)


    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"{prompt_id}.png")
    PIL.Image.fromarray(visual_img[0]).save(save_path)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--model_path", type=str, default=None)
    args.add_argument("--save_dir", type=str, default=None)
    args = args.parse_args()

    # 指定模型路径
    # model_path = "/mnt/bn/wxd-video-understanding/wangxd/ULM-R1/experiments/JanusPro-1B-CoRL-Unified/RFT22k-CycleMatchAccFormat-UniReward-G8-beta004-bs16"
    # model_path = "/mnt/bn/wxd-video-understanding/wangxd/models/Janus-Pro-1B"
    
    # 加载prompt列表
    with open('/mnt/bn/wxd-video-understanding/wangxd/ULM-R1/project/WISE/data/cultural_common_sense.json', 'r') as f:
        prompt_list = json.load(f)
    
    with open('/mnt/bn/wxd-video-understanding/wangxd/ULM-R1/project/WISE/data/natural_science.json', 'r') as f:
        prompt_list1 = json.load(f)
    
    with open('/mnt/bn/wxd-video-understanding/wangxd/ULM-R1/project/WISE/data/spatio-temporal_reasoning.json', 'r') as f:
        prompt_list2 = json.load(f)
    
    prompt_list.extend(prompt_list1)
    prompt_list.extend(prompt_list2)
    
    # 配置GPU分配 [0,0,1,1,2,2,3] 表示:
    # GPU0上运行2个worker, GPU1上运行2个worker, GPU2上运行2个worker, GPU3上运行1个worker
    gpu_assignments = [0, 1, 2, 3, 4, 5, 6, 7]
    
    # 创建任务队列
    prompt_queue = Queue()
    
    # 添加任务到队列
    for i, prompt_data in enumerate(tqdm(prompt_list, desc="Processing prompts")):
        prompt_queue.put({
            'prompt_data': prompt_data,
            'worker_id': i % len(gpu_assignments)
        })
    
    # 创建并启动worker线程
    threads = []
    for worker_id, gpu_id in enumerate(gpu_assignments):
        if worker_id >= len(prompt_list):  # 不需要超过任务数量的worker
            break
            
        t = threading.Thread(
            target=worker,
            args=(gpu_id, prompt_queue, args)
        )
        t.start()
        threads.append(t)
    
    # 等待所有任务完成
    prompt_queue.join()
    
    # 发送结束信号给worker
    for _ in range(len(threads)):
        prompt_queue.put(None)
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    print("All workers finished")