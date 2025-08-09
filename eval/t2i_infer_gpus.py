import os
import math
import json
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor
from eval.utils import load_jsonl, read_txt, load_json


@torch.inference_mode()
def generate(
        mmgpt,
        vl_chat_processor,
        prompt: str,
        parallel_size: int = 16,
        temperature: float = 1,
        cfg_weight: float = 5,
        image_token_num_per_image: int = 576,  # 24x24
        img_size: int = 384,
        patch_size: int = 16,
):
    # tokenizer = vl_chat_processor.tokenizer
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)  # len_seq

    # attention_mask = torch.ones(
    #     (1, len(input_ids) + image_token_num_per_image), device=input_ids.device)
    # attention_mask = attention_mask.repeat_interleave(2 * parallel_size, dim=0)

    # [parallel_size * 2, len_seq]
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:  #
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros(
        (parallel_size, image_token_num_per_image), dtype=torch.int).cuda()
    for i in range(image_token_num_per_image):  # Autoregressive
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            # attention_mask=attention_mask,
            use_cache=True,
            past_key_values=outputs.past_key_values if i != 0 else None
        )
        hidden_states = outputs.last_hidden_state

        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]

        # CFG
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat(
            [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)],
            dim=1
        ).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size]
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

    return dec


def get_chunk(lst, n, k):
    def split_list(lst, n):
        """Split a list into n (roughly) equal-sized chunks"""
        chunk_size = math.ceil(len(lst) / n)  # integer division
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

    chunks = split_list(lst, n)
    return chunks[k]


def main4_gen_eval(
        vl_gpt,
        vl_chat_processor,
        args
):
    eval_data = load_jsonl(args.eval_data)
    eval_data_current = get_chunk(eval_data, args.n_chunks, args.index)

    for meta_data in tqdm(eval_data_current, ncols=80):
        index = meta_data['index']
        out_path = os.path.join(
            args.model_path, f"generated_samples_gen_eval/{index:0>5}"
        )
        os.makedirs(out_path, exist_ok=True)
        # for saving samples
        os.makedirs(f'{out_path}/samples', exist_ok=True)
        # save metadata.jsonl
        with open(f"{out_path}/metadata.jsonl", "w") as fp:
            json.dump(meta_data, fp)

        conversation = [
            {
                "role": "<|User|>",
                "content": f"{meta_data['prompt']}",
            },
            {
                "role": "<|Assistant|>",
                "content": ""
            },
        ]

        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + vl_chat_processor.image_start_tag

        generation_img = generate(
            vl_gpt,
            vl_chat_processor,
            prompt,
            parallel_size=args.parallel_size,
            cfg_weight=args.cfg_weight,
        )

        for i in range(args.parallel_size):
            save_path = f'{out_path}/samples/{i:05}.png'
            # save_path = os.path.join('generated_samples', "img_{}.jpg".format(i))
            img = Image.fromarray(generation_img[i])

            # plt.imshow(img)
            # plt.show()

            img.save(save_path)


def main4_dpg_bench(
        vl_gpt,
        vl_chat_processor,
        args
):
    out_path = os.path.join(args.model_path, f"generated_samples_dpg_bench")
    os.makedirs(out_path, exist_ok=True)

    # loading evaluation prompts
    prompt_files = os.listdir(args.eval_data)
    prompt_files_current = get_chunk(prompt_files, args.n_chunks, args.index)

    for index, prompt_file in enumerate(tqdm(prompt_files_current, ncols=80)):
        prompt = read_txt(os.path.join(args.eval_data, prompt_file))

        conversation = [
            {
                "role": "<|User|>",
                "content": f"{prompt.strip()}",
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + vl_chat_processor.image_start_tag

        generation_img = generate(
            vl_gpt,
            vl_chat_processor,
            prompt,
            parallel_size=args.parallel_size,
            cfg_weight=args.cfg_weight,
        )

        # for i in range(args.parallel_size):
        #     Image.fromarray(generation_img[i]).save(save_path)

        # generate 4 images per prompt and grid them to 2x2 format
        save_path = f'{out_path}/{prompt_file[:-4]}.png'
        h, w = generation_img[0].shape[0], generation_img[0].shape[1]
        grid = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

        grid[0:h, 0:w] = generation_img[0]
        grid[0:h, w:2 * w] = generation_img[1]
        grid[h:2 * h, 0:w] = generation_img[2]
        grid[h:2 * h, w:2 * w] = generation_img[3]

        Image.fromarray(grid).save(save_path)


def main4_wise(
        vl_gpt,
        vl_chat_processor,
        args
):
    out_path = os.path.join(args.model_path, f"generated_samples_wise")
    os.makedirs(out_path, exist_ok=True)

    eval_data = load_json(args.eval_data)
    eval_data_current = get_chunk(eval_data, args.n_chunks, args.index)

    for index, meta_data in enumerate(tqdm(eval_data_current, ncols=80)):
        conversation = [
            {
                "role": "<|User|>",
                "content": f"{meta_data['Prompt']}",
            },
            {
                "role": "<|Assistant|>",
                "content": ""
            },
        ]

        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + vl_chat_processor.image_start_tag

        generation_img = generate(
            vl_gpt,
            vl_chat_processor,
            prompt,
            parallel_size=1,
            cfg_weight=args.cfg_weight,
        )

        save_path =f"{out_path}/{meta_data['prompt_id']}.png"
        img = Image.fromarray(generation_img[0])
        # plt.imshow(img)
        # plt.show()
        img.save(save_path)


def main(args):
    vl_chat_processor = VLChatProcessor.from_pretrained(args.model_path)
    vl_gpt = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    if "geneval" in args.eval_data:
        main4_gen_eval(
            vl_gpt,
            vl_chat_processor,
            args
        )
    elif "dpg_bench" in args.eval_data:
        main4_dpg_bench(
            vl_gpt,
            vl_chat_processor,
            args
        )
    elif "wise" in args.eval_data:
        main4_wise(
            vl_gpt,
            vl_chat_processor,
            args
        )
    else:
        raise ValueError("Invalid eval data name.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str,
        default="/mnt/bn/wxd-video-understanding/wangxd/models/Janus-Pro-1B",
    )
    parser.add_argument(
        "--eval_data", type=str,
        default="eval/t2i/geneval/prompts/geneval_prompt.jsonl",
    )
    parser.add_argument(
        "--parallel_size", type=int,
        # default=16,
        default=4,
    )
    parser.add_argument(
        "--cfg_weight", type=float,
        default=5,
    )
    parser.add_argument(
        "--index", type=int, default=0, help="Chunk index to process (0-indexed)")
    parser.add_argument(
        "--n_chunks", type=int, default=1, help="Total number of chunks")

    args_main = parser.parse_args()

    main(args_main)
