# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Sized

import datasets
import torch
import torch.utils.data
import transformers
from packaging import version
from datasets import Dataset, IterableDataset
from torch.utils.data import Sampler, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_peft_available

from accelerate.utils import gather, is_peft_model, set_seed
from trl import ScriptArguments
from trl.import_utils import is_liger_kernel_available
from trl.trainer.callbacks import SyncRefModelCallback
from trl.models import (
    create_reference_model,
    prepare_deepspeed,
    unwrap_model_for_generation
)
from trl.models.utils import _ForwardRedirection
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import selective_log_softmax, disable_dropout_in_model

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

if is_peft_available():
    from peft import PeftConfig, get_peft_model

from janus.models import VLChatProcessor
from ttrl.rewards.mm2t import init_reward_components, mcq_ttrl_reward

# What we call a reward function is a callable that takes
# a list of prompts and completions and returns a list of rewards.
# When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the dataset.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,

     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            self.generator = torch.Generator()  # Create a local random generator
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
            indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        else:
            indexes = list(range(self.num_samples))

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)


def split_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int
) -> list[dict[str, Optional[torch.Tensor]]]:
    """
    Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts.

    Example:
        >>> x = torch.arange(12).reshape(6, 2)
        >>> y = torch.arange(6).reshape(6, 1)
        >>> tensor_dict = {"x": x, "y": y}
        >>> split_tensor_dict(tensor_dict, 3)
        [
            {"x": tensor([[0, 1], [2, 3]]), "y": tensor([[0], [1]])},
            {"x": tensor([[4, 5], [6, 7]]), "y": tensor([[2], [3]])},
            {"x": tensor([[ 8,  9], [10, 11]]), "y": tensor([[4], [5]])}
        ]
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    chunk_size = first_tensor.shape[0] // num_chunks
    return [
        {
            key: tensor[i * chunk_size : (i + 1) * chunk_size] if tensor is not None else None
            for key, tensor in tensor_dict.items()
        }
        for i in range(num_chunks)
    ]


def shuffle_tensor_dict(tensor_dict: dict[str, Optional[torch.Tensor]]) -> dict[str, Optional[torch.Tensor]]:
    """
    Shuffles a dictionary of tensors along the first dimension in unison.

    Example:
        >>> x = torch.arange(6).reshape(3, 2)
        >>> y = torch.arange(3).reshape(3, 1)
        >>> tensor_dict = {"x": x, "y": y}
        >>> shuffle_tensor_dict(tensor_dict)
        {'x': tensor([[2, 3],
                      [0, 1],
                      [4, 5]]),
         'y': tensor([[1],
                      [0],
                      [2]])}
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    batch_size = first_tensor.shape[0]
    permutation = torch.randperm(batch_size)
    return {key: tensor[permutation] if tensor is not None else None for key, tensor in tensor_dict.items()}


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


class JanusProTTGRPOTrainer(Trainer):
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            reward_funcs: Union[RewardFunc, list[RewardFunc]],
            args: GRPOConfig = None,  # inherit from TrainingArguments

            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            reward_processing_classes: Optional[
                Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
            ] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[
                Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
            ] = (None, None),

            attn_implementation: str = "flash_attention_2",
            peft_config: Optional["PeftConfig"] = None,
            task_args: ScriptArguments = None,
    ):
        self.task_args = task_args

        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # ******************* Models *******************
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            if "Janus" in model_id:
                model = AutoModelForCausalLM.from_pretrained(
                    model, trust_remote_code=True, torch_dtype=torch.bfloat16
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already "
                    "instantiated. This argument can only be used when the `model` argument is a string."
                )
        # ********************************************
        model = self.init_trainable_parameters(model)
        # ********************************************

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT is required to use `peft_config`. Run `pip install peft`.")
            model = get_peft_model(model, peft_config)

        # Processing class
        if processing_class is None:
            if "Janus" in model_id:
                processing_class = VLChatProcessor.from_pretrained(model_id)
                # processing_class.system_prompt = SYSTEM_PROMPT
                # processing_class.system_prompt = ""
            else:
                processing_class = AutoTokenizer.from_pretrained(
                    model.config._name_or_path, padding_side="left")

        # ******************* Reward functions *******************
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of "
                    f"reward functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError(
                    "The number of reward processing classes must "
                    "match the number of reward functions."
                )

        for i, (reward_processing_class, reward_func) in enumerate(
                zip(reward_processing_classes, reward_funcs)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(
                        reward_func.config._name_or_path
                    )
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token
                # in the input sequence. So it's important to set the pad token ID to
                # the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # ******************* Training arguments *******************
        self.max_prompt_length = args.max_prompt_length  # prompt+image
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        # param for .generate()
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty

        self.use_liger_loss = args.use_liger_loss
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.mask_truncated_completions = args.mask_truncated_completions

        # Datasets
        self.shuffle_dataset = args.shuffle_dataset

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of
        # elements in the input tensor associated with the key "input_ids". However, in GRPO,
        # the sampled data does not include the "input_ids" key. Instead, the available keys is
        # "prompt". As a result, the trainer issues the warning: "Could not estimate the number of
        # tokens of the input, floating-point operations will not be computed." To suppress this
        # warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to
        # True. This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            if "Janus" in model_id:
                self.ref_model = AutoModelForCausalLM.from_pretrained(
                    model_id, trust_remote_code=True)
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(
                    model_id, **model_init_kwargs
                )
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter
            # can be disabled to revert to the initial model.
            self.ref_model = None
        else:
            # If PEFT config is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Liger loss
        if self.use_liger_loss:
            if not is_liger_kernel_available():
                raise ImportError(
                    "Liger is required to use `liger_loss` as the GRPO loss. Run `pip install liger-kernel`."
                )
            # redirect the model.module forward to the model forward to ensure pre-forward hooks are called
            self._forward_redirection = _ForwardRedirection()

            self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
                beta=self.beta,
                epsilon_low=self.epsilon_low,
                epsilon_high=self.epsilon_high,
                temperature=self.temperature,
                use_ref_model=self.beta != 0.0,
                loss_type=self.loss_type,
                max_completion_length=self.max_completion_length,
            )

        # Initialize the metrics
        self._metrics = defaultdict(list)
        self._total_train_tokens = 0
        # self.log_completions = args.log_completions
        # self.num_completions_to_print = args.num_completions_to_print

        # Ensure each process receives a unique seed to prevent duplicate completions when
        # generating with transformers if num_generations exceeds per_device_train_batch_size.
        # We could skip it if we use vLLM, but it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        # original param of janus
        # self.generation_config = GenerationConfig(
        #     max_new_tokens=self.max_completion_length,
        #     do_sample=True,
        #     temperature=1, # HACK
        #     num_return_sequences=self.num_generations,
        #     pad_token_id=processing_class.tokenizer.eos_token_id,
        #     bos_token_id=processing_class.tokenizer.bos_token_id,
        #     eos_token_id=processing_class.tokenizer.eos_token_id,
        # )
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            # temperature=self.temperature,
            temperature=1,  # HACK
            pad_token_id=processing_class.tokenizer.eos_token_id,
            bos_token_id=processing_class.tokenizer.bos_token_id,
            eos_token_id=processing_class.tokenizer.eos_token_id,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            cache_implementation=args.cache_implementation,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class
        # depends on whether the model accepts loss-related kwargs.
        # Since we compute our own loss, this check is irrelevant.
        # We set self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            # FIXME
            del self.ref_model.gen_vision_model
            del self.ref_model.gen_aligner
            del self.ref_model.gen_head
            del self.ref_model.gen_embed

            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(
                ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                else:
                    # set device placement to True to make `prepare_model` move `reward_func` to device when using fsdp
                    self.reward_funcs[i] = self.accelerator.prepare_model(
                        reward_func, evaluation_mode=True, device_placement=True
                    )

    # @staticmethod
    def init_trainable_parameters(self, model):
        # fix und
        for param in model.vision_model.parameters():
            param.requires_grad = False
        for param in model.aligner.parameters():
            param.requires_grad = False

        # fix gen
        for param in model.gen_vision_model.parameters():
            param.requires_grad = False
        # fix gen_aligner
        for param in model.gen_aligner.parameters():
            param.requires_grad = False
        # fix gen_head and gen_embed ???
        for param in model.gen_head.parameters():
            param.requires_grad = False
        for param in model.gen_embed.parameters():
            param.requires_grad = False

        # trainable: llm, (gen_embed, gen_head)

        return model

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method,
        # hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # This method overrides `Trainer.get_train_dataloader` to support our custom batching strategy.
    # Instead of returning a standard per-step batch (i.e., `per_device_batch_size), our dataloader loads an
    # *generation* batch (i.e., `per_device_batch_size × steps_per_generation`). This allows us to generate completions
    # once every steps_per_generation step—rather than once per accumulation step—which is significantly more
    # efficient. The only change from the original implementation is multiplying the batch size by
    # `steps_per_generation`. Thus, `_prepare_inputs` is called with this *generation* batch, and it handles the
    # splitting internally.
    # Maintenance note: This method is a copy-paste of the original `Trainer.get_train_dataloader` with only one line
    # modification. As a result, some parts of the method aren't relevant to GRPO, but we keep them to stay one line
    # apart from the super method, ensuring easier maintenance in the future.
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.steps_per_generation,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "drop_last": True,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            dataloader_params["drop_last"] = True

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self, dataset: Optional[Dataset] = None) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first `steps_per_generation` (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second `steps_per_generation` (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    def _prepare_inputs(
        self, generation_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                generation_batch = self._generate_and_score_completions(generation_batch)
                generation_batch = shuffle_tensor_dict(generation_batch)
                self._buffered_inputs = split_tensor_dict(
                    generation_batch, self.args.steps_per_generation)
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple
            # iterations, hence local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    # Get the per-token log probabilities for the completions for the model and reference model
    def _get_per_token_logps(
            self, model,
            input_ids, images_seq_mask, pixel_values, images_emb_mask, attention_mask,
            logits_to_keep, batch_size=None
    ):
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            images_seq_mask_batch = images_seq_mask[i : i + batch_size]
            pixel_values_batch = pixel_values[i : i + batch_size]
            images_emb_mask_batch = images_emb_mask[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]

            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                mm2t_input_ids=input_ids_batch,
                mm2t_images_seq_mask=images_seq_mask_batch,
                mm2t_pixel_values=pixel_values_batch,
                mm2t_images_emb_mask=images_emb_mask_batch,
                mm2t_attention_mask=attention_mask_batch,
                mm2t_logits_to_keep=logits_to_keep + 1
            ).logits  # (B, L, V)
            # exclude the last logit: it corresponds to the next token pred
            logits = logits[:, :-1, :]  # (B, L-1, V)

            # exclude the first input ID since we don't have logits for it
            input_ids_batch = input_ids_batch[:, -logits_to_keep:]  # (B, L-1)
            # For transformers<=4.48, logits_to_keep argument isn't supported
            # logits = logits[:, -logits_to_keep:]
            # Divide logits by sampling temperature.
            logits = logits / self.temperature

            logps = selective_log_softmax(logits, input_ids_batch)
            all_logps.append(logps)
        # Compute the log probabilities for the input tokens.
        return torch.cat(all_logps, dim=0)

    def wrap_mm2t_prompt(self, inputs, device):
        prompts = [x["prompt"] for x in inputs]
        images = [[x["image"]] for x in inputs]  # PIL.Image

        prompt_inputs = self.processing_class(
            conversations=prompts, images=images, force_batchify=True,
        ).to(device)
        return prompt_inputs

    def _generate_and_score_completions(
            self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompt_inputs = self.wrap_mm2t_prompt(inputs, device)

        prompt_ids = prompt_inputs["input_ids"]  # [bs, n_prompt+n_img]
        prompt_mask = prompt_inputs["attention_mask"]  # [bs, n_prompt+n_img]
        images_in_prompt_mask = prompt_inputs["images_seq_mask"]  # [bs, n_prompt+n_img]
        pixel_values = prompt_inputs["pixel_values"]  # [bs, 1, 3, 384, 384]
        images_emb_mask = prompt_inputs["images_emb_mask"]  # [bs, 1, 576]
        prompts = [
            pp.strip('<|User|>: <image_placeholder>').strip('<|Assistant|>:').strip()
            for pp in prompt_inputs["sft_format"]
        ]
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]
            images_in_prompt_mask = images_in_prompt_mask[:, -self.max_prompt_length:]

        # === Generate completions ===
        with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            inputs_embeds = unwrapped_model.prepare_inputs_embeds(
                input_ids=prompt_ids,
                images_seq_mask=images_in_prompt_mask,
                pixel_values=pixel_values,
                images_emb_mask=images_emb_mask,
            )  # [bs, n_prompt+n_img, dim]

            completion_ids = unwrapped_model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prompt_mask,
                generation_config=self.generation_config
            )  # [bs * num_generations, n_completion]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function,
        # avoiding the need to re-tokenize completions if the reward is computed from tokens.
        # completion_ids_list = [
        #     [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
        # ]
        # Sum along sequence dimension (dim=1) to get completion length per sequence, used for logging
        # completion_lengths = completion_mask.sum(1)

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        completion_images_seq_mask = torch.zeros_like(completion_mask).to(images_in_prompt_mask)
        images_seq_mask = torch.cat([
            images_in_prompt_mask, completion_images_seq_mask
        ], dim=1)

        # we only need to compute the logits for the completion tokens
        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.inference_mode():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can
            # skip it's computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps:
                old_per_token_logps = self._get_per_token_logps(
                    self.model,
                    input_ids=prompt_completion_ids,
                    images_seq_mask=images_seq_mask,
                    pixel_values=pixel_values,
                    images_emb_mask=images_emb_mask,
                    attention_mask=attention_mask,
                    logits_to_keep=logits_to_keep,
                    batch_size=batch_size,
                )
            else:
                old_per_token_logps = None

        # === Decode completions for reward model ===
        completions = self.processing_class.tokenizer.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        # only used to compute reward accuracy
        gt_labels = [a['answer'].replace('<answer>', '').replace('</answer>', '').strip()
                     for a in inputs]
        # === Compute rewards ===
        reward_components = init_reward_components(completions=completions)
        # Gather final answers to compute acc rewards by majority voting
        format_rewards = self.accelerator.gather_for_metrics(
            reward_components['format_rewards'], use_gather_object=True
        )
        extracted_answers = self.accelerator.gather_for_metrics(
            reward_components['extracted_answers'], use_gather_object=True
        )
        gt_labels = self.accelerator.gather_for_metrics(
            gt_labels, use_gather_object=True
        )
        acc_rewards = self.reward_funcs[0](
            extracted_answers,
            gt_labels,
            num_generations=self.num_generations
        )

        # Convert None values to NaN
        format_rewards = [reward if reward is not None else torch.nan for reward in format_rewards]
        acc_rewards = [reward if reward is not None else torch.nan for reward in acc_rewards]
        format_rewards = torch.tensor(format_rewards, dtype=torch.float32, device=device)
        acc_rewards = torch.tensor(acc_rewards, dtype=torch.float32, device=device)

        rewards_per_func = torch.stack([acc_rewards, format_rewards], dim=1)

        # rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        # # Gather the reward per function: this part is crucial, because the rewards are normalized
        # # per group and the completions may be distributed across processes
        # rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # === Compute grouped-wise rewards ===
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        # all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(
            completion_mask.sum(1)
        ).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_func_names = [reward_func.__name__ for reward_func in self.reward_funcs]
        for i, reward_func_name in enumerate(reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            # std_rewards = nanstd(rewards_per_func[:, i]).item()
            # self._metrics[f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        return {
            "prompt_completion_ids": prompt_completion_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "images_seq_mask": images_seq_mask,
            "images_emb_mask": images_emb_mask,

            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
        }

    def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None
    ):
        logits_to_keep = inputs["completion_mask"].size(1)
        per_token_logps = self._get_per_token_logps(
            model,
            input_ids=inputs["prompt_completion_ids"],
            attention_mask=inputs["attention_mask"],
            images_seq_mask=inputs["images_seq_mask"],
            pixel_values=inputs["pixel_values"],
            images_emb_mask=inputs["images_emb_mask"],
            logits_to_keep=logits_to_keep,
        )

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            with torch.inference_mode():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model,
                        input_ids=inputs["prompt_completion_ids"],
                        attention_mask=inputs["attention_mask"],
                        images_seq_mask=inputs["images_seq_mask"],
                        pixel_values=inputs["pixel_values"],
                        images_emb_mask=inputs["images_emb_mask"],
                        logits_to_keep=logits_to_keep,
                    )
                else:  # for peft
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model,
                            input_ids=inputs["prompt_completion_ids"],
                            attention_mask=inputs["attention_mask"],
                            images_seq_mask=inputs["images_seq_mask"],
                            pixel_values=inputs["pixel_values"],
                            images_emb_mask=inputs["images_emb_mask"],
                            logits_to_keep=logits_to_keep,
                        )
            per_token_kl = (
                    torch.exp(ref_per_token_logps - per_token_logps) - (
                        ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        completion_mask = inputs["completion_mask"]

        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip
        # it's computation (see _generate_and_score_completions) and use per_token_logps.detach()
        old_per_token_logps = (
            per_token_logps.detach() if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"]
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # Two-sided clipping
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics["kl"].append(
                self.accelerator.gather_for_metrics(mean_kl).mean().item())

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(
                min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (
                    per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics["clip_ratio"].append(gathered_clip_ratio.nanmean().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)

        self._metrics.clear()
