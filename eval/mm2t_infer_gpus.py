import argparse
import os
import math
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor
from janus.utils.io import load_pil_images
from eval.utils import load_json, save_json

OPTIONS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
MCQ_PROMPT = (
    "{Question}\n\n"
    "Answer the question based on the given image and your knowledge.\n\n"
    "Please write your thinking process inside <think> </think> tags, and provide your final answer (option letter, e.g., A/B/C/D) inside <answer> </answer> tags.\n"
    "Your response MUST strictly follow this format: <think> ... </think><answer>option letter</answer>"
)

def get_choice_text(choices):
    choice_list = []
    for i, c in enumerate(choices):
        c = c if c != "" else "None"
        choice_list.append(f"{OPTIONS[i]}. {c}")
    choice_txt = "\n".join(choice_list)
    return choice_txt

def get_chunk(lst, n, k):
    def split_list(lst, n):
        """Split a list into n (roughly) equal-sized chunks"""
        chunk_size = math.ceil(len(lst) / n)  # integer division
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

    chunks = split_list(lst, n)
    return chunks[k]


def main(args):
    # load model
    vl_chat_processor = VLChatProcessor.from_pretrained(args.model_path)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    data = load_json(args.eval_data)

    data_current = get_chunk(data, args.n_chunks, args.index)

    os.makedirs(f"{args.model_path}/local_result", exist_ok=True)

    # inference
    for datum in tqdm(data_current, ncols=80):
        local_result_file = f"{args.model_path}/local_result/{datum['question_id']}.json"
        if os.path.exists(local_result_file):
            continue

        img_path = f"{args.img_dir}/{datum['img_dir']}"
        choice_text = get_choice_text(datum["choice"])
        problem = f"Question: {datum['question'].strip()}\nOptions:\n{choice_text}"
        problem = MCQ_PROMPT.format(Question=problem)

        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{problem}",
                "images": [img_path],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(vl_gpt.device)

        with torch.inference_mode():
            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

            outputs = vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
                temperature=1,
            )
        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

        local_result = {
            "question_id": datum["question_id"],
            "response": answer,
            "problem": problem,
            "correct_answer": OPTIONS[datum['answer']],
        }
        save_json(local_result, local_result_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str,
        default="XXX/experiment/JanusPro-1B-TTRL-MM2T",
    )
    parser.add_argument(
        "--eval_data", type=str,
        default="eval/mm2t/mmmu/val.json",
        # default="eval/mm2t/mmstar/test.json",
    )
    parser.add_argument(
        "--img_dir", type=str,
        default="eval/mm2t/mmmu",
        # default="eval/mm2t/mmstar",
    )
    parser.add_argument(
        "--index", type=int, default=0, help="Chunk index to process (0-indexed)")
    parser.add_argument(
        "--n_chunks", type=int, default=1, help="Total number of chunks")

    args = parser.parse_args()
    main(args)
