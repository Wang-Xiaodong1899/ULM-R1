import torch
import torchmetrics.functional as metric_F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms
from transformers import AutoConfig
from transformers import BlipProcessor, BlipForConditionalGeneration
from corl.open_r1.rewards.bert_score.bert_score_wrapper import BertScoreWrapper, BertSimCSEWrapper
from corl.open_r1.rewards.r_utils import soft_jaccard, word_jaccard
from ttrl.rewards.mm2t import mcq_accuracy_reward, format_reward


@torch.inference_mode()
def t2i_mcq_reward(
        completions, prompts=None, problems=None,
        mmgpt=None, processing_class=None,
        **kwargs
):
    """Applies to UniEval bench"""

    device = mmgpt.device
    qa_prompts, images = [], []
    gt_labels = []
    for idx, (img, problem) in enumerate(zip(completions, problems)):
        # only using one QA
        problem = problem[0]
        gt_labels.append(kwargs['answers'][idx][0])
        qa_prompts.append(
            [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{problem}",
                    # "images": [example["image"]],
                },
                {"role": "<|Assistant|>", "content": ""},
            ],
        )
        images.append([img])

    prepare_inputs = processing_class(
        conversations=qa_prompts, images=images, force_batchify=True,
    ).to(device)

    inputs_embeds = mmgpt.prepare_inputs_embeds(**prepare_inputs)

    outputs = mmgpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        max_new_tokens=512,
        do_sample=True,
        temperature=1,
        pad_token_id=processing_class.tokenizer.eos_token_id,
        bos_token_id=processing_class.tokenizer.bos_token_id,
        eos_token_id=processing_class.tokenizer.eos_token_id,
    )
    responses = processing_class.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    acc_rewards = mcq_accuracy_reward(
        completions=responses,
        num_generations=kwargs['num_generations'],
        gt_labels=gt_labels,
    )
    format_rewards = format_reward(responses)
    rewards = [a + b for a, b in zip(acc_rewards, format_rewards)]

    return rewards


class TTRLT2ICycleCSReward:
    def __init__(self, task_args):
        self.args = task_args

        self.cap_cs_metrics = task_args.caption_cs_metrics
        self.using_simcse = task_args.using_simcse
        self.img_cs_metrics = task_args.image_cs_metrics
        self.using_img_cs = task_args.using_image_cs
        self.using_external_caption_model = task_args.using_external_caption_model

        if self.using_img_cs:
            if "lpips" in self.img_cs_metrics:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor()
                ])

        self.blip_model = None
        self.blip_processor = None
        self.bert_scorer = None
        self.lpips_metric = None

    @torch.inference_mode()
    def generate_caption_with_policy_mmgpt(
            self, images, cap_mmgpt=None, processing_class=None,
    ):
        device = cap_mmgpt.device

        # generate captions
        # task_instruct = "Generate an accurate visual description of the image in a single sentence."
        # task_instruct = "Generate a concise and accurate description of the image in one sentence."
        task_instruct = "Describe the main content of the image in one sentence."
        _prompts, _images = [], []
        for img in images:
            _prompts.append(
                [
                    {
                        "role": "<|User|>",
                        "content": f"<image_placeholder>\n{task_instruct}",
                        # "images": [example["image"]],
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ],
            )
            _images.append([img])

        prepare_inputs = processing_class(
            conversations=_prompts, images=_images, force_batchify=True,
        ).to(device)

        inputs_embeds = cap_mmgpt.prepare_inputs_embeds(**prepare_inputs)
        outputs = cap_mmgpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            max_new_tokens=256,
            do_sample=True,
            temperature=1,
            pad_token_id=processing_class.tokenizer.eos_token_id,
            bos_token_id=processing_class.tokenizer.bos_token_id,
            eos_token_id=processing_class.tokenizer.eos_token_id,
        )
        gen_captions = processing_class.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return gen_captions

    @torch.inference_mode()
    def generate_caption_with_external_model(self, images):
        device = list(self.blip_model.parameters())[0].device

        inputs = self.blip_processor(images=images, return_tensors="pt").to(device)
        out = self.blip_model.generate(**inputs, max_new_tokens=50)
        gen_captions = self.blip_processor.batch_decode(out, skip_special_tokens=True)

        return gen_captions

    @torch.inference_mode()
    def compute_caption_consistency(self, gen_captions, prompts):
        if "bertscore" in self.cap_cs_metrics and "jaccard" in self.cap_cs_metrics:
            if self.using_simcse:
                bert_score_f1 = self.bert_scorer.compute_simcse(gen_captions, prompts)
            else:
                bert_score_f1 = self.bert_scorer.compute_f1(gen_captions, prompts)
            jaccards = [soft_jaccard(pp, gg) for pp, gg in zip(prompts, gen_captions)]
            cap_cs_scores = [(a + b) / 2. for a, b in zip(bert_score_f1, jaccards)]

        elif "jaccard" in self.cap_cs_metrics:
            cap_cs_scores = [soft_jaccard(pp, gg) for pp, gg in zip(prompts, gen_captions)]

        elif "bertscore" in self.cap_cs_metrics:
            if self.using_simcse:
                cap_cs_scores = self.bert_scorer.compute_simcse(gen_captions, prompts)
                cap_cs_scores = [a for a in cap_cs_scores]
            else:
                cap_cs_scores = self.bert_scorer.compute_f1(gen_captions, prompts)
        else:
            raise NotImplementedError("No valid caption consistency computation.")

        return cap_cs_scores

    @torch.inference_mode()
    def compute_image_consistency(self, gen_images, gt_images, device):
        img_cs_scores = []
        for idx, (gen_img, real_img) in enumerate(zip(gen_images, gt_images)):
            if "lpips" in self.img_cs_metrics:
                img_cs_score = 1. - self.lpips_metric(
                    self.transform(gen_img).to(device).unsqueeze(0),
                    self.transform(real_img).to(device).unsqueeze(0)
                )
            else:  # mse
                img_cs_score = 1. - metric_F.image.root_mean_squared_error_using_sliding_window(
                    self.transform(gen_img).to(device).unsqueeze(0),
                    self.transform(real_img).to(device).unsqueeze(0)
                )  # near to 1, the better
            img_cs_scores.append(img_cs_score)

        return img_cs_scores

    def __call__(
            self, completions, prompts, mmgpt=None, processing_class=None, **kwargs
    ):
        device = mmgpt.device

        if self.args.using_external_caption_model:
            gen_captions = self.generate_caption_with_external_model(completions)
        else:
            gen_captions = self.generate_caption_with_policy_mmgpt(
                completions, cap_mmgpt=mmgpt, processing_class=processing_class
            )
        cap_cs_scores = self.compute_caption_consistency(gen_captions, prompts)

        if self.using_img_cs:
            # **** best-of-N **** #
            # 1. using cap_cs_scores
            max_value = max(cap_cs_scores)
            max_index = cap_cs_scores.index(max_value)
            selected_image = completions[max_index]
            # selected_image.save(f"./selected_image.png")
            selected_images = [selected_image] * len(completions)
            # 2. using image cs

            img_cs_scores = self.compute_image_consistency(completions, selected_images, device)

            rewards = [a + b for a, b in zip(cap_cs_scores, img_cs_scores)]
        else:
            rewards = cap_cs_scores

        return rewards

    def load_external_model(self, load_device):
        if self.args.using_external_caption_model:
            print(f"loading external captioning model: blip")
            self.blip_processor = BlipProcessor.from_pretrained(f"{self.args.blip_model_ckpt}")
            config = AutoConfig.from_pretrained(f"{self.args.blip_model_ckpt}")
            self.blip_model = BlipForConditionalGeneration(config)
            checkpoint = torch.load(
                f"{self.args.blip_model_ckpt}/pytorch_model.bin", map_location='cpu')
            self.blip_model.load_state_dict(checkpoint, strict=False)
            for param in self.blip_model.parameters():
                param.requires_grad = False

            self.blip_model = self.blip_model.to(load_device)
            self.blip_model = self.blip_model.eval()

        if "bertscore" in self.cap_cs_metrics:
            if self.using_simcse:
                self.bert_scorer = BertSimCSEWrapper(
                    f"{self.args.model_ckpt_dir}/sup-simcse-bert-base-uncased")
                print(f"loaded: sup-simcse-bert-base-uncased")
            else:
                self.bert_scorer = BertScoreWrapper(f"{self.args.model_ckpt_dir}/all-mpnet-base-v2")
                print(f"loaded: all-mpnet-base-v2")

        if self.using_img_cs and "lpips" in self.img_cs_metrics:
            self.lpips_metric = LearnedPerceptualImagePatchSimilarity(
                net_type='vgg', normalize=True
            ).to(load_device)
            self.lpips_metric = self.lpips_metric.eval()
            print(f"loaded: lpips")

    @property
    def __name__(self):
        return 'TTRL_CycleCS'
