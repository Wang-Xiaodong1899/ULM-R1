import os
import datasets
from PIL import Image
from tqdm import tqdm
from eval.utils import load_jsonl, load_json


OPTIONS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]
MCQ_PROMPT = (
    "{Question}\n\n"
    "Answer the question based on the given image and your knowledge.\n\n"
    "Please write your thinking process inside <think> </think> tags, and provide your final answer (option letter, e.g., A/B/C/D) inside <answer> </answer> tags.\n"
    "Your response MUST strictly follow this format: <think> ... </think><answer>option letter</answer>"
)

def main_t2i(data_path, output_path, is_uni_bench=True):
    os.makedirs(output_path, exist_ok=True)

    if is_uni_bench:
        data = load_json(data_path)
        features = datasets.Features(
            {
                "prompt": datasets.Value("string"),
                "problems": datasets.Sequence(datasets.Value("string")),
                "answers": datasets.Sequence(datasets.Value("string")),
            }
        )
        orig_task_prompt = "Based on the image, answer with the option letter directly in the format (A), (B), (C), (D), or (E)."
        data_items = []
        for datum in tqdm(data):
            QAs = datum["QAs"]
            problems = []
            answers = []
            for QA in QAs:
                question = QA["question"]
                question = question.replace(orig_task_prompt, '').strip()
                question = MCQ_PROMPT.format(Question=question)
                problems.append(question)

                answers.append(QA["answer"])

            data_items.append({
                "prompt": datum["prompt"],
                "problems": problems,
                "answers": answers,
            })

        shard_dict = {
            "prompt": [item["prompt"] for item in data_items],
            "problems": [item["problems"] for item in data_items],
            "answers": [item["answers"] for item in data_items],
        }
    else:
        data = load_jsonl(data_path)

        features = datasets.Features(
            {
                "prompt": datasets.Value("string"),
            }
        )
        shard_dict = {
            "prompt": [item["prompt"] for item in data],
        }

    shard_ds = datasets.Dataset.from_dict(shard_dict, features=features)
    shard_ds.to_parquet(
        f"{output_path}/train.parquet",
        batch_size=8,
        compression="snappy"
    )


def main_mm2t(data_path, output_path, img_dir):
    data = load_json(data_path)
    os.makedirs(output_path, exist_ok=True)

    def get_choice_text(choices):
        choice_list = []
        for i, c in enumerate(choices):
            c = c if c != "" else "None"
            choice_list.append(f"{OPTIONS[i]}. {c}")
        choice_txt = "\n".join(choice_list)
        return choice_txt

    features = datasets.Features(
        {
            "image": datasets.Image(),
            "problem": datasets.Value("string"),
            "answer": datasets.Value("string"),
        }
    )
    data_items = []
    for datum in tqdm(data):
        image = Image.open(f"{img_dir}/{datum['img_dir']}").convert('RGB')
        choice_text = get_choice_text(datum["choice"])

        problem = f"Question: {datum['question'].strip()}\nOptions:\n{choice_text}"
        problem = MCQ_PROMPT.format(Question=problem)
        answer = OPTIONS[datum["answer"]]
        answer = f"<answer>{answer}</answer>"

        data_items.append({
            "image": image,
            "problem": problem,
            "answer": answer,
        })
    shard_dict = {
        "image": [item["image"] for item in data_items],
        "problem": [item["problem"] for item in data_items],
        "answer": [item["answer"] for item in data_items],
    }

    shard_ds = datasets.Dataset.from_dict(shard_dict, features=features)
    shard_ds.to_parquet(
        f"{output_path}/train.parquet",
        batch_size=8,
        compression="snappy"
    )


if __name__ == "__main__":
    # data_path = "eval/t2i/geneval/prompts/evaluation_metadata.jsonl"
    # save_dir = "ttrl/data/geneval/"

    data_path = "eval/t2i/UniEval/prompts/uni_bench.json"
    save_dir = "ttrl/data/unieval/"

    main_t2i(data_path, save_dir)

    # **********************************************************************

    # img_dir = "eval/mm2t/mmmu/"
    # # data_path = "eval/mm2t/mmmu/val.json"
    # # save_dir = "ttrl/data/mmmu/"

    # img_dir = "eval/mm2t/mmstar/"
    # # data_path = "eval/mm2t/mmstar/test.json"
    # # save_dir = "ttrl/data/mmstar/"
    #
    # main_mm2t(data_path, save_dir, img_dir)

