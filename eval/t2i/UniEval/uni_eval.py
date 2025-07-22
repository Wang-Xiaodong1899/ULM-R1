import os
import re
import random
import numpy as np
from tqdm import tqdm

from eval.utils import *


def check_answer(reply, answer, options):
    """
    Parse the reply and return true (1) or false (0).

    Parameters:
    - reply: The text reply from model.
    - answer: The correct option letter, A/B/C/D.
    - options: If letter is not matched check the last option contents

    Return:
    - model true of false: correct -> 1, false -> 0, invalid response -> -1.
    - model response: A/B/C/D/E/None
    """

    # models fail to generate images reply ''
    if reply == '':
        return -1, 'None'

    pred = re.findall(r'\(([ABCDE])|([ABCDE])\)', reply)

    if len(pred) == 0:
        # for answer without (), match the last letter.
        if reply[-1] in ['A', 'B', 'C', 'D', 'E']:
            pred = reply[-1]

        # for answer without option letter, use the last substr in options.
        else:
            reply_ = reply.lower()
            pos = [reply_.rfind(_.lower()) for _ in options]
            last_idx = pos.index(max(pos))

            # last option as the answer
            if pos[last_idx] != -1:
                pred = ['A', 'B', 'C', 'D', 'E'][last_idx]

            # not finded option and answer, invalid response
            else:
                pred = ''

    # if match, use the last letter
    else:
        pred = pred[-1]

    # assign prediction label
    if isinstance(pred, tuple):
        for _ in pred:
            if _ == answer:
                return 1, _
        return 0, next((_ for _ in pred if _ != ''), None)
    else:
        if pred == '':
            return -1, 'None'
        elif pred == answer:
            return 1, pred
        else:
            return 0, pred


def statistics(records, uni_bench):
    """
    Statistics and Formatting for results ranked tag, complexity, co-variance, etc.

    Parameters:
    - records: model prediction records collected by tags: [[QA_id, model_true_false, model_response]].
    - uni_bench: list of cases of UniBench

    Return:
    - results: A result dict including:
        0) the response statistics
        1) tag-level accuracy for three hierarchical levels (L0, L1, L2),
           first get the mean accuracy across QAs of L2 tag then mean for the upper levels progressively.
        2) case-level accuracy for the whole benchmark, varied word sizes and QA sizes,
           first get the mean accuracy across QAs of a text (the case Uniscore), then operate across texts.
        3) the overall UniScore L0-L2, UniScoreA (all correct), Case-UniScore
    """

    # tag set structure, used to print results in order
    tags = {
        "Textual": {
            "Numerals": [
                "Cardinal Numbers",
                "Ordinal Numbers",
                "Multiplicative Numbers",
                "Fractional Numbers",
                "Approximate Numbers",
                "Percentages",
                "Time Expression Numbers",
                "Range Numbers"
            ],
            "Adjectives": [
                "Qualitative Adjectives",
                "State Adjectives",
                "Sensory Adjectives",
                "Emotional Adjectives",
                "Evaluative Adjectives",
                "Temporal and Spatial Adjectives",
                "Color Adjectives",
                "Pure Adjectives",
                "Compound Adjectives",
                "Limiting Adjectives",
                "Verb-derived Adjectives",
                "Degree Adjectives"
            ],
            "Nouns": [
                "Natural Objects",
                "Man-made Objects",
                "Qualities",
                "Phenomena",
                "Psychological Activities",
                "Personal Names",
                "Place Names",
                "Organization Names",
                "Event Names",
                "Job Titles",
                "Technical Terms",
                "Culturally Specific Names",
                "Groups",
                "Categories",
                "Collections",
                "Directions",
                "Time",
                "Compound Nouns",
                "Gerunds"
            ],
            "Verbs": [
                "Transitive Verbs",
                "Intransitive Verbs",
                "Double Object Verbs",
                "Linking Verbs",
                "Compound Verbs",
                "Continuous Verbs",
                "Momentary Verbs",
                "Perception Verbs",
                "Cognitive Verbs",
                "Directional Verbs",
                "Existential Verbs",
                "Psychological Verbs",
                "Change Verbs",
                "Interacting Verbs"
            ],
            "Adverbs": [
                "Locative Adverbs",
                "Manner Adverbs",
                "Degree Adverbs",
                "Negative Adverbs"
            ]
        },
        "Visual": {
            "Text Content Images": [
                "Language Texts",
                "Programming Language",
                "Symbol"
            ],
            "Chart and Documents": [
                "Table",
                "Figures",
                "Documents"
            ],
            "Image Styles": [
                "Art Genre Styles",
                "Photography Styles",
                "Digital Art and Illustration Styles",
                "Design Styles",
                "Architectural Style",
                "Other Special Styles"
            ],
            "Image Modalities": [
                "Signal Modalities",
                "Computational Modalities"
            ],
            "Image Quality and Distortion": [
                "Noise Types",
                "Distortion Artifacts"
            ],
            "Color and Light Effects": [
                "Color Schemes",
                "Light and Shadow Directions and Types"
            ],
            "Composition and Visual Focus": [
                "Composition Types",
                "Focus Positions"
            ],
            "UI": [
                "App",
                "Web",
                "Game Interface",
                "Operating System"
            ]
        }
    }

    # load information for each case and QA
    case_records, tag_records, QA_info, results = {}, {}, {}, {}
    word_sizes, QA_sizes, option_ratio = [], [], [0] * 6

    for idx, item in enumerate(uni_bench):
        for QA in item['QAs']:
            QA_info[QA['QA_id']] = {'tag': QA['tag'], 'prompt_id': idx}

        word_size = len(item['prompt'].split())
        word_sizes.append(word_size)
        QA_sizes.append(len(item['QAs']))
        case_records[idx] = {'word_size': word_size, 'QA_size': len(item['QAs']), 'preds': []}

    # put model predictions into case_record and tag_records
    for QA_id, pred, response in records:
        l0, l1, l2 = QA_info[QA_id]['tag'].split(', ')
        prompt_id = QA_info[QA_id]['prompt_id']

        # for invalid response, predict as 0
        if pred < 0:
            pred = 0

        if l0 not in tag_records:
            tag_records[l0] = {}
        if l1 not in tag_records[l0]:
            tag_records[l0][l1] = {}
        if l2 not in tag_records[l0][l1]:
            tag_records[l0][l1][l2] = []

        tag_records[l0][l1][l2].append(pred)
        case_records[prompt_id]['preds'].append(pred)

        # count distribution of selected option
        if response in ['A', 'B', 'C', 'D', 'E']:
            option_ratio[ord(response) - ord('A')] += 1
        else:
            option_ratio[-1] += 1

    option_sum = sum(option_ratio)
    option_ratio = [_ / option_sum for _ in option_ratio]
    results['response-statistics'] = {'A': round(option_ratio[0], 3),
                                      'B': round(option_ratio[1], 3),
                                      'C': round(option_ratio[2], 3),
                                      'D': round(option_ratio[3], 3),
                                      'E (N/A)': round(option_ratio[4], 3),
                                      'Invalid': round(option_ratio[5], 3)}

    # count tag-level results
    results['tag-L0'] = {}
    results['tag-L1'] = {}
    results['tag-L2'] = {}

    for l0 in tag_records.keys():
        l0_score = []

        for l1 in tag_records[l0].keys():
            l1_score = []

            for l2 in tag_records[l0][l1].keys():
                l2_score = sum(tag_records[l0][l1][l2]) / len(tag_records[l0][l1][l2])
                l1_score.append(l2_score)
                results['tag-L2'][l2] = round(l2_score, 3)

            l1_score = sum(l1_score) / len(l1_score)
            l0_score.append(l1_score)
            results['tag-L1'][l1] = round(l1_score, 3)

        l0_score = sum(l0_score) / len(l0_score)
        results['tag-L0'][l0] = round(l0_score, 3)

    # calculate the divide size and count case-level results
    # sort word_sizes, QA_sizes and get the boundary size at 1/3 2/3 to decide few, middle, and many
    word_sizes.sort()
    QA_sizes.sort()
    word_b1, word_b2 = word_sizes[len(word_sizes) // 3], word_sizes[len(word_sizes) * 2 // 3]
    QA_b1, QA_b2 = QA_sizes[len(QA_sizes) // 3], QA_sizes[len(QA_sizes) * 2 // 3]
    few_words, mid_words, many_words = [], [], []
    few_QAs, mid_QAs, many_QAs = [], [], []
    few_words_ac, mid_words_ac, many_words_ac = [], [], []  # all correct
    few_QAs_ac, mid_QAs_ac, many_QAs_ac = [], [], []
    whole, whole_ac = [], []

    for k, v in case_records.items():
        if len(v['preds']) == 0:
            continue
        case_uniScore = sum(v['preds']) / len(v['preds'])
        all_correct = 0 if 0 in v['preds'] else 1
        whole.append(case_uniScore)
        whole_ac.append(all_correct)

        if v['word_size'] < word_b1:
            few_words.append(case_uniScore)
            few_words_ac.append(all_correct)
        elif v['word_size'] <= word_b2:
            mid_words.append(case_uniScore)
            mid_words_ac.append(all_correct)
        else:
            many_words.append(case_uniScore)
            many_words_ac.append(all_correct)

        # middle use [], takes major cases beyond 1/3
        if v['QA_size'] < QA_b1:
            few_QAs.append(case_uniScore)
            few_QAs_ac.append(all_correct)
        elif v['QA_size'] <= QA_b2:
            mid_QAs.append(case_uniScore)
            mid_QAs_ac.append(all_correct)
        else:
            many_QAs.append(case_uniScore)
            many_QAs_ac.append(all_correct)

    results['case-word-size'] = \
        {'few [%d-%d]' % (word_sizes[0], word_b1 - 1): round(sum(few_words) / len(few_words),
                                                             3) if len(few_words) > 0 else 'N/A',
         'middle [%d-%d]' % (word_b1, word_b2): round(sum(mid_words) / len(mid_words), 3) if len(
             mid_words) > 0 else 'N/A',
         'many [%d-%d]' % (word_b2 + 1, word_sizes[-1]): round(sum(many_words) / len(many_words),
                                                               3) if len(many_words) > 0 else 'N/A',
         'few  (all correct) [%d-%d]' % (word_sizes[0], word_b1 - 1): round(
             sum(few_words_ac) / len(few_words_ac), 3) if len(few_words_ac) > 0 else 'N/A',
         'middle (all correct) [%d-%d]' % (word_b1, word_b2): round(
             sum(mid_words_ac) / len(mid_words_ac), 3) if len(mid_words_ac) > 0 else 'N/A',
         'many (all correct) [%d-%d]' % (word_b2 + 1, word_sizes[-1]): round(
             sum(many_words_ac) / len(many_words_ac), 3) if len(many_words_ac) > 0 else 'N/A'}

    results['case-QA-size'] = \
        {'few [%d-%d]' % (QA_sizes[0], QA_b1 - 1): round(sum(few_QAs) / len(few_QAs), 3) if len(
            few_QAs) > 0 else 'N/A',
         'middle [%d-%d]' % (QA_b1, QA_b2): round(sum(mid_QAs) / len(mid_QAs), 3) if len(
             mid_QAs) > 0 else 'N/A',
         'many [%d-%d]' % (QA_b2 + 1, QA_sizes[-1]): round(sum(many_QAs) / len(many_QAs), 3) if len(
             many_QAs) > 0 else 'N/A',
         'few (all correct) [%d-%d]' % (QA_sizes[0], QA_b1 - 1): round(
             sum(few_QAs_ac) / len(few_QAs_ac), 3) if len(few_QAs_ac) > 0 else 'N/A',
         'middle (all correct) [%d-%d]' % (QA_b1, QA_b2): round(sum(mid_QAs_ac) / len(mid_QAs_ac),
                                                                3) if len(
             mid_QAs_ac) > 0 else 'N/A',
         'many (all correct) [%d-%d]' % (QA_b2 + 1, QA_sizes[-1]): round(
             sum(many_QAs_ac) / len(many_QAs_ac), 3) if len(many_QAs_ac) > 0 else 'N/A'}

    # overall case-UniScore, macro accuracy for cases, UniScoreA indicates QAs of a text are all correct
    results['case-UniScore'] = round(sum(whole) / len(whole), 3)
    results['UniScoreA'] = round(sum(whole_ac) / len(whole_ac), 3)

    # the final UniScore to report, micro for L1 tags. It's easier to report with more counted QAs than L2.
    results['tag-L0-UniScore'] = round(
        sum(list(results['tag-L0'].values())) / len(results['tag-L0']), 3)
    results['tag-L1-UniScore'] = round(
        sum(list(results['tag-L1'].values())) / len(results['tag-L1']), 3)
    results['tag-L2-UniScore'] = round(
        sum(list(results['tag-L2'].values())) / len(results['tag-L2']), 3)

    # print results
    print('\n\n==================== response statistics ====================')
    for k, v in results['response-statistics'].items():
        print('%s%s%s' % (k, ' ' * (56 - len(k)), str(v)))

    print('\n\n====================  tag-level results  ====================')
    print('\n--------------------   Level 0 results   --------------------')
    for k0 in tags:
        if k0 in results['tag-L0']:
            print('%s%s%s' % (k0, ' ' * (56 - len(k0)), str(results['tag-L0'][k0])))

    print('\n--------------------   Level 1 results   --------------------')
    for k0 in tags:
        for k1 in tags[k0]:
            if k1 in results['tag-L1']:
                print('%s%s%s' % (k1, ' ' * (56 - len(k1)), str(results['tag-L1'][k1])))

    print('\n--------------------   Level 2 results   --------------------')
    for k0 in tags:
        for k1 in tags[k0]:
            for k2 in tags[k0][k1]:
                if k2 in results['tag-L2']:
                    print('%s%s%s' % (k2, ' ' * (56 - len(k2)), str(results['tag-L2'][k2])))

    print('\n\n====================  case-level results ====================')
    print('\n--------------------      word size      --------------------')
    for k, v in results['case-word-size'].items():
        print('%s%s%s' % (k, ' ' * (56 - len(k)), str(v)))

    print('\n--------------------       QA size       --------------------')
    for k, v in results['case-QA-size'].items():
        print('%s%s%s' % (k, ' ' * (56 - len(k)), str(v)))

    print('\n\n====================    overall results  ====================')
    print('\n--------------------    macro-UniScore   --------------------')
    print('%s%s%s' % ('case-UniScore', ' ' * 43, str(results['case-UniScore'])))
    print('%s%s%s' % ('case-all-UniScore', ' ' * 39, str(results['UniScoreA'])))

    print('\n--------------------    macro-UniScore    --------------------')
    print('%s%s%s' % ('tag-L0-UniScore', ' ' * 41, str(results['tag-L0-UniScore'])))
    print('%s%s%s' % ('tag-L1-UniScore', ' ' * 41, str(results['tag-L1-UniScore'])))
    print('%s%s%s' % ('tag-L2-UniScore', ' ' * 41, str(results['tag-L2-UniScore'])))

    print('\n\n====================  Reported UniScore  ====================\n')

    abbr = {'Numerals': 'Num', 'Adjectives': 'Adj', 'Nouns': 'Noun', 'Verbs': 'verb',
            'Adverbs': 'Adv',
            'Text Content Images': 'Text', 'Chart and Documents': 'Doc', 'Image Styles': 'Sty',
            'Image Modalities': 'Moda',
            'Image Quality and Distortion': 'Qual', 'Color and Light Effects': 'Effe',
            'Composition and Visual Focus': 'Comp',
            'UI': 'UI'}
    label_s, score_s = '', ''
    for k in abbr:
        if k in results['tag-L1']:
            label_s += abbr[k] + '\t'
            score_s += str(results['tag-L1'][k]) + '\t'
    label_s += 'UniScore'
    score_s += str(results['tag-L1-UniScore'])
    print(label_s)
    print(score_s, '\n')

    return results


def uni_eval(generate, understand, uni_bench, save_path='', img_num=4):
    """
    Evaluate unfied Multimodal Understanding and Generation.

    Parameters:
    - generate: A callback function that generates N image (text, num, temp_save_path).
    - understand: A callback function that understans images with text outputs (img, prompt).
    - uni_bench: A list of loaded UniBench.
    - save_path: Path to save all generated images and answers for vis and analysis, defualt no saving.
    - img_num: generate N=4 images for each prompt

    Returns:
    - records: Return the records for batch evaluation.
    """

    # record model prediction for each QA [[QA_id, model_pred], ...]
    records = []
    for i, item in tqdm(enumerate(uni_bench), total=len(uni_bench), desc="Evaluating Cases"):
        prompt_id = item['prompt_id']
        input_prompt = item['prompt']
        QAs = item['QAs']

        save_dir = os.path.join(save_path, str(prompt_id))

        if os.path.exists(f"{save_dir}/text_records.json"):
            # imgs = [f"{save_dir}/img_{i}.png" for i in range(img_num)]
            results = load_json(f"{save_dir}/text_records.json")
            for result in results:
                records.append([result['QA_id'], int(result['isCorrect']), result['response']])
        else:
            # Call the image generation function to generate N imgs
            os.makedirs(save_dir, exist_ok=True)
            imgs = generate(input_prompt, img_num, save_dir)

            # save understanding outputs via txt files
            text_record = []
            # Call the image understanding function for each generated image
            for img in imgs:
                for QA in QAs:
                    if img != '':
                        reply = understand(img, QA['question'])
                    else:
                        reply = ''

                    answer = QA['answer']
                    options = [_.split(',')[0] for _ in QA['question'].split('\n')[1].split(') ')[1:]]

                    isCorrect, response = check_answer(reply, answer, options)

                    records.append([QA['QA_id'], isCorrect, response])
                    text_record.append({
                        "pid": f"{img.split('/')[-1][:-4]}_{QA['QA_id']}",
                        "img_name": img,
                        "input_prompt": input_prompt,
                        "question": QA['question'],
                        "answer": QA['answer'],

                        "response": response,
                        "isCorrect": isCorrect,
                        "QA_id": QA['QA_id'],
                    })

            save_json(text_record, os.path.join(save_dir, 'text_records.json'))

    return records
    # uniScores = statistics(records, uni_bench)
    #
    # if save_path != '':
    #     result_save_path = save_path.replace(save_path.split('/')[-1], '')
    #     open(os.path.join(result_save_path, 'results.json'), 'w').write(json.dumps(uniScores, indent=4))
    #
    # return records


# Setting seeds helps with reproducibility.
# Note that different devices,  CUDA, and dependency may lead to different results.
# Without seed, the difference of overall UniScore is generally within 1%.
def seed_everything(seed=1024):
    if seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# def main(model_name, gpus='0', save_path='', uni_bench='uni_bench.json', extra_model=''):
#     """
#     Conduct uni_eval.
#     Support Unified models and Gen only models.
#     Support multi-GPU evaluation.
#
#     Parameters:
#     - model_name: The name of the unified model, or the name of the gen only model.
#     - gpus: GPU ids for multi processors. (0_1 indicates 2 workers on GPUs 0,1; 0,1_2,3 runs two works and each 2 GPUs)
#     - save_path: The dump path to save records and results, '' indicats no saving.
#     - uni_bench: The path of uni_bench.json.
#     - extra_model: A extra model providing to evaluate the understanding part of generation-only model.
#
#     Returns:
#     - print results anf save records
#     """
#
#     uni_bench = json.load(open(uni_bench))
#
#     model = load_model(model_name)
#     seed_everything()
#
#     uni_eval(model.generate, model.understand, uni_bench, save_path)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='UniEval.')
#
#     parser.add_argument('model_name', type=str, help='Model name of Unified model')
#     parser.add_argument('--gpus', type=str, default='0',
#                         help="Which GPUs are assigned, e.g., '0' for single eval ('0,1' for multi-gpus), '0_1' for batch eval.")
#     parser.add_argument('--save_path', type=str, default='',
#                         help="The folder to save records and results, default don't save, just print results.")
#     parser.add_argument('--uni_bench', type=str, default='uni_bench.json',
#                         help="Path to uni_bench.json")
#     parser.add_argument('--extra_model', type=str, default='',
#                         help='The model name of extra understand model to eval Gen only model.')
#
#     args = parser.parse_args()
#
#     main(args.model_name, args.gpus, args.save_path, args.uni_bench, args.extra_model)
