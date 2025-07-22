import os
import math
import random
import argparse
import numpy as np
from PIL import Image
import torch

from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor
from janus.utils.io import load_pil_images
from eval.t2i.UniEval.uni_eval import uni_eval, statistics
from eval.utils import load_json, save_json
from ttrl.rewards.mm2t import MCQAnswerExtractor


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_chunk(lst, n, k):
    def split_list(lst, n):
        """Split a list into n (roughly) equal-sized chunks"""
        chunk_size = math.ceil(len(lst) / n)  # integer division
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

    chunks = split_list(lst, n)
    return chunks[k]


class JanusUniInfer:
    def __init__(
            self, model_path="Janus-Pro-1B",
    ):
        self.model_path = model_path

        # load model
        self.vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = self.model.to(torch.bfloat16).cuda().eval()
        print('Model loaded')

        self.mcq_extractor = MCQAnswerExtractor()
        self.OPTIONS = ["A", "B", "C", "D", "E"]
        self.MCQ_PROMPT = (
            "{Question}\n\n"
            "Answer the question based on the given image and your knowledge.\n\n"
            "Please write your thinking process inside <think> </think> tags, and provide your final answer (option letter, e.g., A/B/C/D) inside <answer> </answer> tags.\n"
            "Your response MUST strictly follow this format: <think> ... </think><answer>option letter</answer>"
        )

    @torch.no_grad()
    def generate(
            self,
            prompt: str,
            temperature: float = 1,
            cfg_weight: float = 5,
            image_token_num_per_image: int = 576,  # 24x24
            img_size: int = 384,
            patch_size: int = 16,
    ):
        input_ids = self.vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)  # len_seq

        tokens = torch.zeros((2, len(input_ids)), dtype=torch.int).cuda()
        for i in range(2):
            tokens[i, :] = input_ids
            if i % 2 != 0:  # 无条件输入
                tokens[i, 1:-1] = self.vl_chat_processor.pad_id

        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)

        generated_tokens = torch.zeros((1, image_token_num_per_image), dtype=torch.int).cuda()
        for i in range(image_token_num_per_image):  # Autoregressive
            outputs = self.model.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=outputs.past_key_values if i != 0 else None
            )
            hidden_states = outputs.last_hidden_state

            logits = self.model.gen_head(hidden_states[:, -1, :])
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
            img_embeds = self.model.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        dec = self.model.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[1, 8, img_size // patch_size, img_size // patch_size]
        )
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

        return Image.fromarray(dec[0])

    @torch.no_grad()
    def t2i_generation(self, prompt, img_num, save_dir):
        gen_img_paths = []
        os.makedirs(save_dir, exist_ok=True)
        for i in range(img_num):
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"{prompt}",
                },
                {
                    "role": "<|Assistant|>",
                    "content": ""
                },
            ]
            sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=self.vl_chat_processor.sft_format,
                system_prompt="",
            )
            prompt = sft_format + self.vl_chat_processor.image_start_tag

            image = self.generate(prompt)

            img_path = os.path.join(save_dir, f"img_{i}.png")
            image.save(img_path)
            gen_img_paths.append(img_path)

        return gen_img_paths

    @torch.no_grad()
    def understand(self, img_dir, prompt):
        prepare_inputs = self.wrap_mm_prompt(img_dir, prompt)

        with torch.inference_mode():
            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

            outputs = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
                temperature=1,
            )
        response = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        extract_answer = self.mcq_extractor.extract_answer(response)
        answer = extract_answer if extract_answer else response
        return answer

    def wrap_mm_prompt(self, img_dir, prompt):
        orig_task_prompt = "Based on the image, answer with the option letter directly in the format (A), (B), (C), (D), or (E)."

        prompt = prompt.replace(orig_task_prompt, '').strip()
        problem = self.MCQ_PROMPT.format(Question=prompt)

        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{problem}",
                "images": [img_dir],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.model.device)
        return prepare_inputs


def main(args):
    set_seed(seed=0)

    eval_model = JanusUniInfer(model_path=args.model_path)
    args.save_img_path = os.path.join(args.model_path, f"generated_samples_unieval")
    os.makedirs(args.save_img_path, exist_ok=True)

    uni_bench = load_json(args.eval_data)
    uni_bench_current = get_chunk(uni_bench, args.n_chunks, args.index)

    records = uni_eval(
        eval_model.t2i_generation, eval_model.understand, uni_bench_current, args.save_img_path,
    )
    save_json(records, os.path.join(args.model_path, f'records_rank_{args.index}.json'))
    print(f"Saved records for rank {args.index}")


def compute_scores(args):
    uni_bench = load_json(args.eval_data)
    print(f"len(uni_bench): {len(uni_bench)}")

    records = []
    for idx in range(args.n_chunks):
        _records = load_json(f"{args.model_path}/records_rank_{idx}.json")
        print(f"records_rank_{idx}: {len(_records)}")
        records.extend(_records)
    print(f"len(records): {len(records)}")

    uniScores = statistics(records, uni_bench)
    save_json(uniScores, os.path.join(args.model_path, "results.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str,
        default="XXX/experiment/JanusPro-1B-TTRL-T2I",
    )
    parser.add_argument(
        "--eval_data", type=str,
        default="eval/t2i/UniEval/prompts/uni_bench.json",
    )
    parser.add_argument(
        "--cfg_weight", type=float,
        default=5,
        # default=6,
    )
    parser.add_argument(
        "--index", type=int, default=0, help="Chunk index to process (0-indexed)")
    parser.add_argument(
        "--n_chunks", type=int, default=1, help="Total number of chunks")
    parser.add_argument("--merge_only", action="store_true", help="Only merge existing results")

    args_main = parser.parse_args()

    if args_main.merge_only:
        compute_scores(args_main)
    else:
        main(args_main)
