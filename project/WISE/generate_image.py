import os
import PIL.Image
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
import argparse
import json
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_splits", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--split_idx", type=int, required=True, 
                       help="Index of this split (0 to num_splits-1)")
    parser.add_argument("--prompt_file", type=str, required=True,
                       help="Path to JSON file containing prompts")
    parser.add_argument("--output_dir", type=str, default="generated_samples",
                       help="Output directory for generated images")
    parser.add_argument("--model_path", type=str, 
                       default="/mnt/bn/wxd-video-understanding/wangxd/ULM-R1/experiments/JanusPro-1B-CoRL-Unified/RFT22k-CycleMatchAccFormat-UniReward-G8-beta004-bs16",
                       help="Path to pretrained model")
    return parser.parse_args()

def load_model(model_path, gpu_id):
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    return vl_gpt, vl_chat_processor

def prepare_prompt(prompt_text, vl_chat_processor):
    conversation = [
        {"role": "<|User|>", "content": prompt_text},
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    return sft_format + vl_chat_processor.image_start_tag

@torch.inference_mode()
def generate_single_image(mmgpt, vl_chat_processor, prompt, output_path, 
                         temperature=1, cfg_weight=5, 
                         image_token_num_per_image=576, img_size=384, patch_size=16):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids).cuda()
    
    parallel_size = 1
    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id
    
    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((1, image_token_num_per_image), dtype=torch.int).cuda()

    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds, 
            use_cache=True, 
            past_key_values=outputs.past_key_values if i != 0 else None
        )
        hidden_states = outputs.last_hidden_state
        
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0:1, :]  # conditioned
        logit_uncond = logits[1:2, :]  # unconditioned
        
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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    PIL.Image.fromarray(visual_img[0]).save(output_path)

def main():
    args = parse_args()
    
    # Load prompts
    with open(args.prompt_file) as f:
        prompts = json.load(f)
    
    # Split data across GPUs
    total = len(prompts)
    print(f'inference {total} samples')
    chunk_size = (total + args.num_splits - 1) // args.num_splits
    start = args.split_idx * chunk_size
    end = min(start + chunk_size, total)
    chunk = prompts[start:end]
    
    # Load model on the specified GPU
    model, processor = load_model(args.model_path, args.split_idx)
    
    # Process each prompt in the chunk
    for item in tqdm(chunk, desc=f"GPU {args.split_idx}"):
        prompt_text = item["Prompt"]
        output_path = os.path.join(args.output_dir, f"{item['prompt_id']}.png")
        
        if not os.path.exists(output_path):  # Skip if already generated
            prompt = prepare_prompt(prompt_text, processor)
            generate_single_image(
                model, 
                processor, 
                prompt, 
                output_path
            )

if __name__ == "__main__":
    main()