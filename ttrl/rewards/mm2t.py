import re
from collections import Counter
from ttrl.verifier.ans_extractor import MCQAnswerExtractor


def format_reward(completions, **kwargs):
    # pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    pattern = re.compile(r"<think>.*?</think>[.\s]*<answer>.*?</answer>", re.DOTALL)

    matches = [pattern.fullmatch(content) for content in completions]
    return [1.0 if match else 0.0 for match in matches]


def init_reward_components(completions):
    # answer extraction
    extractor = MCQAnswerExtractor()
    extracted_answers = []
    for content in completions:
        content = str(content).strip()
        # student_answer = extract_mcq_answer(content)
        student_answer = extractor.extract_answer(content)
        extracted_answers.append(student_answer)

    return {
        "extracted_answers": extracted_answers,
        "format_rewards": format_reward(completions)
    }


def single_prompt_answers_voting(answers, num_generations):
    single_prompt_rewards = []
    valid_answers = [a for a in answers if a is not None]
    if not valid_answers:
        # return single_prompt_rewards.extend([-1] * num_generations)
        return single_prompt_rewards.extend([0] * num_generations)

    # majority voting
    counter = Counter(valid_answers)
    majority_answer, majority_count = counter.most_common(1)[0]

    for answer in answers:
        if answer is None:
            single_prompt_rewards.append(0)
            # single_prompt_rewards.append(-1)
        else:
            single_prompt_rewards.append(1 if answer == majority_answer else 0)

    return single_prompt_rewards, majority_answer


def mcq_ttrl_reward(
        extracted_answers,
        gt_labels,
        num_generations,
        **kwargs
):
    # compute rewards
    batch_size = len(extracted_answers) // num_generations
    if batch_size < 1:
        batch_size = 1

    rewards_hits = 0
    rewards = []
    for bs in range(batch_size):
        gt_labels_batch = gt_labels[bs * num_generations:(bs + 1) * num_generations]
        assert len(set(gt_labels_batch)) == 1

        extracted_answers_batch = extracted_answers[bs * num_generations:(bs + 1) * num_generations]
        single_voted_rewards, voted_answer = single_prompt_answers_voting(
            extracted_answers_batch, num_generations
        )
        rewards.extend(single_voted_rewards)

        if voted_answer == gt_labels_batch[0]:
            rewards_hits += 1

    rewards_hit_rate = 100 * rewards_hits / batch_size
    print(f'\033[40;32mreward_accuracy: {rewards_hit_rate:.2f}%\033[0m')

    return rewards


def mcq_accuracy_reward(completions, **kwargs):
    # answer extraction
    extractor = MCQAnswerExtractor()
    extracted_answers = []
    for content in completions:
        content = str(content).strip()
        # student_answer = extract_mcq_answer(content)
        student_answer = extractor.extract_answer(content)
        extracted_answers.append(student_answer)

    assert len(extracted_answers) == len(completions)

    # compute rewards
    num_generations = kwargs['num_generations']
    batch_size = len(completions) // num_generations
    if batch_size < 1:
        batch_size = 1
    # prompts = kwargs['prompts']

    rewards_hits = []
    rewards = []
    for bs in range(batch_size):
        extracted_answers_batch = extracted_answers[bs * num_generations:(bs + 1) * num_generations]
        gt_labels_batch = kwargs['gt_labels'][bs * num_generations:(bs + 1) * num_generations]

        valid_answers = [a for a in extracted_answers_batch if a is not None]

        if not valid_answers:
            # return rewards.extend([-1] * num_generations)
            return rewards.extend([0] * num_generations)

        # majority voting
        counter = Counter(valid_answers)
        majority_answer, majority_count = counter.most_common(1)[0]

        for idx, answer in enumerate(extracted_answers_batch):
            if majority_answer == gt_labels_batch[idx]:
                rewards_hits.append(1)
            else:
                rewards_hits.append(0)

            if answer is None:
                rewards.append(0)
                # rewards.append(-1)
            else:
                rewards.append(1 if answer == majority_answer else 0)

    rewards_hit_rate = sum(rewards_hits) / len(rewards_hits)
    print(f'\033[40;32mreward_accuracy: {100*rewards_hit_rate:.2f}%\033[0m')

    return rewards
