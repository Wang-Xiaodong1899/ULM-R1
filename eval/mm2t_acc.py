import argparse
import glob
from tqdm import tqdm
from eval.utils import load_json, save_json
from ttrl.rewards.mm2t import MCQAnswerExtractor


def main(args):
    # compute acc
    mcq_extractor = MCQAnswerExtractor()

    # load local results
    local_result_files = glob.glob(f"{args.model_path}/local_result/*.json")
    results = [load_json(f) for f in local_result_files]

    eval_data = load_json(args.eval_data)
    assert len(local_result_files) == len(eval_data)

    results_true, results_false = [], []
    for result in tqdm(results, ncols=80):
        response = result["response"]
        pred_answer = mcq_extractor.extract_answer(response)

        result['parsed_answer'] = pred_answer

        if pred_answer == result["correct_answer"]:
            results_true.append(result)
        else:
            results_false.append(result)

    overall_acc = 100 * len(results_true) / len(results)
    scores = {
        "overall_acc": f"{overall_acc:.1f}"
    }
    print(scores)
    print(f"\033[40;32m{'#' * 120}\033[0m")
    output = {
        "scores": scores,
        "results": {
            "correct": results_true,
            "incorrect": results_false
        }
    }

    output_file = f"{args.model_path}/{args.model_path.split('/')[-1]}_result.jsonl"
    save_json(output, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str,
        default="XXX/experiment/JanusPro-1B-TTRL-MM2T",
    )
    parser.add_argument(
        "--eval_data", type=str,
        # default="eval/mm2t/mmmu/val.json",
        default="eval/mm2t/mmstar/test.json",
    )

    args = parser.parse_args()
    main(args)
