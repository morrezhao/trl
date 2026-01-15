# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

import random
import textwrap
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_liger_kernel_available, is_peft_available

from ...models import prepare_deepspeed
from ...models.utils import unwrap_model_for_generation
from ...trainer.sft_trainer import SFTTrainer
from ...trainer.utils import DataCollatorForChatML, disable_dropout_in_model, empty_cache
from .gkd_config import GKDConfig


if is_peft_available():
    from peft import PeftConfig

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss


def extract_final_answer(answer_text: str) -> str:
    """Extract the final numerical answer from GSM8K format (after ####)."""
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    return answer_text.strip()


class GKDTrainer(SFTTrainer):
    """Trainer for Generalized Knowledge Distillation (GKD) of language models.

    For details on GKD, see the paper: [On-Policy Distillation of Language Models: Learning from Self-Generated
    Mistakes](https://huggingface.co/papers/2306.13649).

    Args:
        model ([`~transformers.PreTrainedModel`] or `torch.nn.Module` or `str`, *optional*):
            Model to be trained, or the string identifier of the model to be instantiated from a pretrained model.
        teacher_model ([`~transformers.PreTrainedModel`] or `torch.nn.Module` or `str`, *optional*):
            Teacher model for knowledge distillation, or the string identifier of the model to be instantiated from a
            pretrained model.
        args ([`experimental.gkd.GKDConfig`], *optional*):
            Training arguments.
        data_collator ([`~transformers.DataCollator`], *optional*):
            Data collator to batch samples from the dataset. It defaults to a [`DataCollatorForChatML`] using the
            `processing_class`.
        train_dataset ([`~datasets.Dataset`], *optional*):
            Dataset for training.
        eval_dataset ([`~datasets.Dataset`] or `dict` of [`~datasets.Dataset`], *optional*):
            Dataset for evaluation.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.BaseImageProcessor`], [`~transformers.FeatureExtractionMixin`] or [`~transformers.ProcessorMixin`], *optional*):
           Class to process the data.
        compute_metrics (`Callable`, *optional*):
            Function to compute metrics at evaluation. Must take in an [`~transformers.EvalPrediction`] and return a
            dictionary string to float.
        callbacks (`list` of [`~transformers.TrainerCallback`], *optional*):
            Callbacks to use during training.
        optimizers (`tuple` of `torch.optim.Optimizer` and `torch.optim.lr_scheduler.LambdaLR`, *optional*, defaults to `(None, None)`):
            Tuple containing the optimizer and the learning rate scheduler to use for training.
        preprocess_logits_for_metrics (`Callable`, *optional*):
            Function to preprocess the logits before computing the metrics. Must take in the `logits` and `labels` and
            return the logits to be used for metrics computation.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration to use PEFT for training. If `None`, PEFT is not used. If provided, the `model` will be
            wrapped with the specified PEFT adapter.
        formatting_func (`Callable`, *optional*):
            Function to format the dataset. Must take in an example and return an example.
    """

    _tag_names = ["trl", "gkd"]
    _name = "GKD"
    _paper = {
        "title": "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes",
        "id": "2306.13649",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @inproceedings{agarwal2024on-policy,
                title        = {{On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes}},
                author       = {Rishabh Agarwal and Nino Vieillard and Yongchao Zhou and Piotr Stanczyk and Sabela Ramos Garea and Matthieu Geist and Olivier Bachem},
                year         = 2024,
                booktitle    = {The Twelfth International Conference on Learning Representations, {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
                publisher    = {OpenReview.net},
                url          = {https://openreview.net/forum?id=3zKtaqxLhW},
            }"""),
    }

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str | None = None,
        teacher_model: PreTrainedModel | nn.Module | str = None,
        args: GKDConfig | None = None,
        data_collator: DataCollator | None = None,  # type: ignore
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: "PeftConfig | None" = None,
        formatting_func: Callable | None = None,
    ):
        # Ensure Trainer does not drop non-signature columns used by the collator (e.g., "prompts")
        args.remove_unused_columns = False
        # Respect a user-provided data_collator; otherwise, provide a ChatML collator that
        if data_collator is None:
            data_collator = DataCollatorForChatML(tokenizer=processing_class, max_length=args.max_length)

        # Ensure SFTTrainer does not pre-process the dataset when using a ChatML collator,
        # so that raw conversational fields (e.g., "messages") remain available to the collator.
        if args.dataset_kwargs is None:
            args.dataset_kwargs = {"skip_prepare_dataset": True}
        else:
            args.dataset_kwargs["skip_prepare_dataset"] = True

        # Liger fused GKD loss (JSD)
        self.use_liger_gkd_loss = False
        if args.use_liger_kernel:
            self.liger_jsd_loss = LigerFusedLinearJSDLoss(
                beta=args.beta,
                ignore_index=-100,
                temperature=args.temperature,
                compiled=False,
            )
            self.use_liger_gkd_loss = True

        super().__init__(
            model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            formatting_func=formatting_func,
        )

        # Self-distillation mode: student and teacher share the same model
        self.self_distillation = args.self_distillation

        if self.self_distillation:
            # In self-distillation mode, the teacher is the same as the student model
            self.teacher_model = None
        else:
            if args.teacher_model_init_kwargs is None:
                teacher_model_init_kwargs = {}
            elif not isinstance(teacher_model, str):
                raise ValueError(
                    "You passed teacher_model_init_kwargs to the GKDConfig, but your teacher_model is already instantiated."
                )
            else:
                teacher_model_init_kwargs = args.teacher_model_init_kwargs
                teacher_model_init_kwargs["dtype"] = (
                    teacher_model_init_kwargs["dtype"]
                    if teacher_model_init_kwargs["dtype"] in ["auto", None]
                    else getattr(torch, teacher_model_init_kwargs["dtype"])
                )

            if isinstance(teacher_model, str):
                teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model, **teacher_model_init_kwargs)

            if self.is_deepspeed_enabled:
                self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
            else:
                self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)

        # Disable dropout in the model
        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        self.lmbda = args.lmbda
        self.beta = args.beta
        self.temperature = args.temperature
        self.seq_kd = args.seq_kd
        self.debug_alignment = args.debug_alignment
        self._debug_step_count = 0  # Counter for debug prints

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=True,
            top_k=0,
            use_cache=False if args.gradient_checkpointing else True,
            pad_token_id=self.processing_class.pad_token_id,
        )
        # Set custom EOS tokens if they are specified by the model's generation
        # config. This is important for models with the Llama 3 chat template,
        # which use special tokens <|eot_id|> and <|eom_id|> to mark the end of
        # turns or messages.
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.generation_config.eos_token_id = self.model.generation_config.eos_token_id

    @staticmethod
    def generalized_jsd_loss(
        student_logits, teacher_logits, labels=None, beta=0.5, temperature=1.0, reduction="batchmean"
    ):
        """
        Compute the generalized Jensen-Shannon Divergence loss for knowledge distillation using F.kl_div. See Eq. (1)
        of https://huggingface.co/papers/2306.13649 for the definition.

        Args:
            student_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            teacher_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            labels:
                Tensor of shape (batch_size, sequence_length) with -100 for padding tokens to ignore when computing
                loss
            beta:
                Interpolation coefficient between 0 and 1 (default: 0.5)
            temperature:
                Softmax temperature (default: 1.0)
            reduction:
                Specifies the reduction to apply to the output (default: 'batchmean')

        Returns:
            loss: Scalar tensor with the generalized JSD loss
        """

        # Apply temperature scaling
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        # Compute log probabilities for student and probabilities for teacher
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if beta == 0:
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            # Compute the log of the mixture distribution
            # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
            beta = torch.tensor(beta, dtype=student_log_probs.dtype)
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_log_probs + torch.log(1 - beta), teacher_log_probs + torch.log(beta)]),
                dim=0,
            )

            # Compute KL divergences using F.kl_div
            # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is swapped compared to that defined in the paper.
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)

            # Compute the Generalized Jensen-Shannon Divergence
            jsd = beta * kl_teacher + (1 - beta) * kl_student

        # Masking
        if labels is not None:
            mask = labels != -100
            jsd = jsd[mask]

        # Apply reduction
        if reduction == "batchmean":
            return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / jsd.size(0)
        elif reduction == "sum":
            return jsd.sum()
        elif reduction == "mean":
            return jsd.mean()
        else:
            return jsd

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.self_distillation:
            # Self-distillation mode: student and teacher share the same model
            # Student: forward on (prompt + generated response)
            # Teacher: forward on (prompt + answer + generated response)
            return self._compute_self_distillation_loss(model, inputs, return_outputs)

        if self.use_liger_gkd_loss:
            # Forward only through the base models (avoid lm_head to save memory)
            unwrapped_student = self.accelerator.unwrap_model(model)
            if hasattr(unwrapped_student, "get_decoder") and unwrapped_student.get_decoder() is not None:
                base_student = unwrapped_student.get_decoder()
            else:
                base_student = getattr(
                    unwrapped_student, getattr(unwrapped_student, "base_model_prefix", "model"), unwrapped_student
                )

            student_outputs = base_student(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
            )

            self.teacher_model.eval()
            unwrapped_teacher = self.accelerator.unwrap_model(self.teacher_model)
            if hasattr(unwrapped_teacher, "get_decoder") and unwrapped_teacher.get_decoder() is not None:
                base_teacher = unwrapped_teacher.get_decoder()
            else:
                base_teacher = getattr(
                    unwrapped_teacher, getattr(unwrapped_teacher, "base_model_prefix", "model"), unwrapped_teacher
                )
            with torch.no_grad():
                teacher_outputs = base_teacher(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=False,
                )

            # hidden states (shifted)
            student_hidden = student_outputs.last_hidden_state[:, :-1]
            teacher_hidden = teacher_outputs.last_hidden_state[:, :-1]

            # Release full outputs to free memory
            del student_outputs, teacher_outputs

            # labels mask and labels (shifted)
            labels_mask = inputs["labels"] != -100
            masked_input_ids = torch.where(
                labels_mask, inputs["input_ids"], torch.full_like(inputs["input_ids"], -100)
            )
            true_labels = masked_input_ids[:, 1:].contiguous()

            # Release intermediate tensors
            del labels_mask, masked_input_ids

            # heads
            student_head = unwrapped_student.get_output_embeddings()
            teacher_head = unwrapped_teacher.get_output_embeddings()

            # liger fused jsd loss
            loss = self.liger_jsd_loss(
                student_input=student_hidden,
                student_weight=student_head.weight,
                teacher_input=teacher_hidden,
                teacher_weight=teacher_head.weight,
                true_labels=true_labels,
                student_bias=getattr(student_head, "bias", None),
                teacher_bias=getattr(teacher_head, "bias", None),
            )

            # Release hidden states after loss computation
            del student_hidden, teacher_hidden, true_labels
        else:
            # compute student output
            student_outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

            # compute teacher output in eval mode
            self.teacher_model.eval()
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )

            # Use labels-based slicing instead of prompts.shape[1]
            # For next-token prediction: logits[:-1] predict labels[1:]
            # Response tokens are where labels != -100
            shifted_student_logits = student_outputs.logits[:, :-1, :]
            shifted_teacher_logits = teacher_outputs.logits[:, :-1, :]
            shifted_labels = inputs["labels"][:, 1:]  # Shifted labels for next-token prediction

            # compute loss (labels mask is applied inside generalized_jsd_loss)
            loss = self.generalized_jsd_loss(
                student_logits=shifted_student_logits,
                teacher_logits=shifted_teacher_logits,
                labels=shifted_labels,
                beta=self.beta,
                temperature=self.temperature,
            )

        # empty cache
        empty_cache()

        # Return loss
        return (loss, student_outputs) if return_outputs else loss

    def _print_alignment_debug(
        self,
        student_input_ids,
        student_attention_mask,
        teacher_input_ids,
        teacher_attention_mask,
        teacher_response_starts,
        original_input_ids,
        original_attention_mask,
        original_labels,
        response_masks,
        batch_size,
        on_policy_generated=False,
    ):
        """Print human-readable debug information to verify token alignment for loss computation."""
        tokenizer = self.processing_class

        print("\n" + "=" * 80)
        print(f"[DEBUG] Self-Distillation Alignment Check - Step {self._debug_step_count}")
        print("=" * 80)

        # Show whether on-policy generation happened
        if on_policy_generated:
            print("[INFO] ON-POLICY GENERATION: YES - y is model-generated")
        else:
            print("[INFO] ON-POLICY GENERATION: NO - y = y* (dataset answer used as response)")
            print("[WARNING] When y = y*, teacher sees (x + y* + y*) and student sees (x + y*)")
        print(f"[INFO] Temperature: {self.temperature}, Beta: {self.beta}")

        for i in range(min(batch_size, 2)):  # Print first 2 samples max
            print(f"\n{'─' * 40}")
            print(f"Sample {i + 1}/{batch_size}")
            print(f"{'─' * 40}")

            # 1. Decode student input (x + y): prompt + generated response
            student_ids = student_input_ids[i]
            student_attn = student_attention_mask[i]
            # Remove left-padding for cleaner display
            student_valid_start = (student_attn == 0).sum().item()
            student_valid_ids = student_ids[student_valid_start:]
            student_decoded = tokenizer.decode(student_valid_ids, skip_special_tokens=False)

            print(f"\n[1] STUDENT INPUT (x + y) - prompt + generated response:")
            print(f"    Length: {len(student_valid_ids)} tokens")
            print(f"    Decoded:\n    {repr(student_decoded)}")

            # 2. Decode original input (x + y*): prompt + dataset ground-truth
            orig_ids = original_input_ids[i]
            orig_attn = original_attention_mask[i]
            orig_valid_start = (orig_attn == 0).sum().item()
            orig_valid_ids = orig_ids[orig_valid_start:]
            orig_decoded = tokenizer.decode(orig_valid_ids, skip_special_tokens=False)

            print(f"\n[2] ORIGINAL INPUT (x + y*) - prompt + DATASET ground-truth:")
            print(f"    Length: {len(orig_valid_ids)} tokens")
            print(f"    Decoded:\n    {repr(orig_decoded)}")

            # 2b. Show y* tokens specifically (dataset ground-truth answer)
            if original_labels is not None:
                orig_labels_i = original_labels[i]
                y_star_mask = orig_labels_i != -100
                y_star_tokens = orig_ids[y_star_mask]
                y_star_decoded = tokenizer.decode(y_star_tokens, skip_special_tokens=False)
                print(f"\n[2b] y* TOKENS (dataset ground-truth answer only):")
                print(f"    Count: {y_star_mask.sum().item()} tokens")
                print(f"    Decoded:\n    {repr(y_star_decoded)}")

            # 3. Decode teacher input (x + y* + y): original + generated response
            teacher_ids = teacher_input_ids[i]
            teacher_attn = teacher_attention_mask[i]
            teacher_valid_len = teacher_attn.sum().item()
            teacher_valid_ids = teacher_ids[:teacher_valid_len]
            teacher_decoded = tokenizer.decode(teacher_valid_ids, skip_special_tokens=False)

            print(f"\n[3] TEACHER INPUT (x + y* + y) - original + generated response:")
            print(f"    Length: {teacher_valid_len} tokens")
            print(f"    Response starts at position: {teacher_response_starts[i]}")
            print(f"    Decoded:\n    {repr(teacher_decoded)}")

            # 4. Decode loss tokens (labels != -100): only generated response tokens
            response_mask = response_masks[i]
            loss_token_ids = student_ids[response_mask]
            loss_decoded = tokenizer.decode(loss_token_ids, skip_special_tokens=False)
            num_loss_tokens = response_mask.sum().item()

            print(f"\n[4] LOSS TOKENS (labels != -100) - tokens contributing to KL/JSD loss:")
            print(f"    Count: {num_loss_tokens} tokens")
            print(f"    Decoded:\n    {repr(loss_decoded)}")

            # 5. Show the breakdown of teacher input structure
            teacher_response_start = teacher_response_starts[i]
            orig_part = teacher_valid_ids[:teacher_response_start]
            response_part = teacher_valid_ids[teacher_response_start:]

            print(f"\n[5] TEACHER INPUT BREAKDOWN:")
            print(f"    Original part (x + y*): {teacher_response_start} tokens")
            print(f"    Decoded: {repr(tokenizer.decode(orig_part, skip_special_tokens=False))}")
            print(f"    Response part (y): {len(response_part)} tokens")
            print(f"    Decoded: {repr(tokenizer.decode(response_part, skip_special_tokens=False))}")

            # 6. Verify alignment
            print(f"\n[6] ALIGNMENT VERIFICATION:")
            loss_tokens_match = torch.equal(loss_token_ids, response_part) if len(loss_token_ids) == len(response_part) else False
            print(f"    Loss tokens (y) == Teacher response part: {loss_tokens_match}")
            if not loss_tokens_match and len(loss_token_ids) > 0 and len(response_part) > 0:
                print(f"    Loss tokens length: {len(loss_token_ids)}, Response part length: {len(response_part)}")
                print(f"    Loss tokens (first 10): {loss_token_ids[:10].tolist()}")
                print(f"    Response part (first 10): {response_part[:10].tolist()}")

            # 6b. Compare y vs y* to verify on-policy generation worked
            if original_labels is not None:
                orig_labels_i = original_labels[i]
                y_star_mask = orig_labels_i != -100
                y_star_tokens = orig_ids[y_star_mask]
                y_equals_y_star = torch.equal(loss_token_ids, y_star_tokens) if len(loss_token_ids) == len(y_star_tokens) else False
                print(f"    y (loss tokens) == y* (dataset answer): {y_equals_y_star}")
                if y_equals_y_star and on_policy_generated:
                    print("    [WARNING] y == y* but on_policy_generated=True - generation might have reproduced ground-truth")
                elif not y_equals_y_star and not on_policy_generated:
                    print("    [ERROR] y != y* but on_policy_generated=False - this should not happen!")

        print("\n" + "=" * 80 + "\n")

    def _compute_self_distillation_loss(self, model, inputs, return_outputs=False):
        """
        Compute the conditional self-distillation loss: min KL( P(y|x) || P(y|x,y*) )

        Data flow:
            - original_input_ids: prompt (x) + dataset ground-truth answer (y*)
              This comes from DataCollatorForChatML and is saved BEFORE generation in training_step.
            - input_ids: prompt (x) + student-generated response (y)
              This is produced by generate_on_policy_outputs.

        Teacher input construction:
            teacher_input = original_input_ids (x + y*) + response_tokens (y)
            This ensures the teacher sees: P(y | x, y*) - conditioned on the TRUE dataset answer.

        Student input:
            student_input = input_ids (x + y)
            The student sees: P(y | x) - without the ground-truth.

        Loss computation:
            - Response tokens (y) are identified using labels != -100
            - Both student and teacher logits are extracted for these response positions
            - JSD/KL loss is computed only on the response tokens

        IMPORTANT: We do NOT use inputs["prompts"] for slicing because ChatML inserts special tokens.
        All token identification is based on labels.
        """
        batch_size = inputs["input_ids"].shape[0]
        device = inputs["input_ids"].device

        # Student input: x + y (prompt + student-generated response)
        student_input_ids = inputs["input_ids"]
        student_attention_mask = inputs["attention_mask"]
        student_labels = inputs["labels"]

        # Original input from dataset: x + y* (prompt + DATASET ground-truth answer)
        # This is saved in training_step BEFORE generation, ensuring y* is the true reference
        # Per-sample alignment: original_input_ids[i] corresponds to student_input_ids[i]
        original_input_ids = inputs["original_input_ids"]
        original_attention_mask = inputs["original_attention_mask"]

        # Build teacher inputs: (x + y*) + y for each sample
        # Teacher sees the dataset's ground-truth y*, then the generated response y
        teacher_input_ids_list = []
        teacher_attention_mask_list = []

        # Identify response tokens (y) using labels != -100
        response_masks = student_labels != -100  # [batch_size, seq_len]

        for i in range(batch_size):
            # Extract generated response tokens (y) from student input using labels mask
            sample_response_mask = response_masks[i]
            response_tokens = student_input_ids[i][sample_response_mask]  # y tokens

            # Get original (x + y*) without left-padding
            orig_attn = original_attention_mask[i]
            orig_valid_start = (orig_attn == 0).sum().item()  # skip padding
            orig_valid_ids = original_input_ids[i, orig_valid_start:]  # x + y* (no padding)
            orig_valid_mask = original_attention_mask[i, orig_valid_start:]

            # Teacher input: (x + y*) + y
            # This ensures teacher computes P(y | x, y*) conditioned on dataset ground-truth
            teacher_ids = torch.cat([orig_valid_ids, response_tokens], dim=0)
            teacher_mask = torch.cat([orig_valid_mask, torch.ones_like(response_tokens)], dim=0)

            teacher_input_ids_list.append(teacher_ids)
            teacher_attention_mask_list.append(teacher_mask)

        # Pad teacher inputs (right-padding for batch processing)
        max_teacher_len = max(t.shape[0] for t in teacher_input_ids_list)
        pad_token_id = self.processing_class.pad_token_id

        teacher_input_ids = torch.full(
            (batch_size, max_teacher_len), pad_token_id, dtype=torch.long, device=device
        )
        teacher_attention_mask = torch.zeros(
            (batch_size, max_teacher_len), dtype=torch.long, device=device
        )

        # Track where response tokens start in each teacher sequence
        teacher_response_starts = []
        for i in range(batch_size):
            seq_len = teacher_input_ids_list[i].shape[0]
            teacher_input_ids[i, :seq_len] = teacher_input_ids_list[i]
            teacher_attention_mask[i, :seq_len] = teacher_attention_mask_list[i]

            # Response starts after original valid content
            orig_attn = original_attention_mask[i]
            orig_valid_len = (orig_attn == 1).sum().item()
            teacher_response_starts.append(orig_valid_len)

        # Debug: print decoded inputs for alignment verification
        if self.debug_alignment:
            self._debug_step_count += 1
            on_policy_generated = inputs.get("_on_policy_generated", False)
            original_labels = inputs.get("original_labels", None)
            self._print_alignment_debug(
                student_input_ids=student_input_ids,
                student_attention_mask=student_attention_mask,
                teacher_input_ids=teacher_input_ids,
                teacher_attention_mask=teacher_attention_mask,
                teacher_response_starts=teacher_response_starts,
                original_input_ids=original_input_ids,
                original_attention_mask=original_attention_mask,
                original_labels=original_labels,
                response_masks=response_masks,
                batch_size=batch_size,
                on_policy_generated=on_policy_generated,
            )

        # Compute student output
        student_outputs = model(
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
        )

        # Compute teacher output (same model, but with different input)
        with torch.no_grad():
            teacher_outputs = model(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
            )

        # Compute loss per-sample and aggregate
        # For next-token prediction: logit[t] predicts token[t+1]
        # Use shifted approach: shifted_logits = logits[:-1], shifted_labels = labels[1:]
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_tokens = 0

        for i in range(batch_size):
            num_response_tokens = response_masks[i].sum().item()
            if num_response_tokens == 0:
                continue

            # Student: shift logits and labels for next-token prediction
            student_logits_i = student_outputs.logits[i]  # [student_seq_len, vocab_size]
            shifted_student_logits = student_logits_i[:-1]  # [seq_len-1, vocab]
            shifted_student_labels = student_labels[i, 1:]  # [seq_len-1]

            # Extract student logits for response tokens (where shifted_labels != -100)
            response_mask_shifted = shifted_student_labels != -100
            shifted_student_logits_response = shifted_student_logits[response_mask_shifted]

            # Teacher: extract logits that predict the response tokens
            # Response tokens in teacher seq start at teacher_response_start
            # To predict tokens at positions [start, start+1, ..., start+n-1],
            # we need logits at positions [start-1, start, ..., start+n-2]
            teacher_logits_i = teacher_outputs.logits[i]
            teacher_response_start = teacher_response_starts[i]
            teacher_logit_start = max(0, teacher_response_start - 1)
            teacher_logit_end = teacher_response_start + num_response_tokens - 1
            shifted_teacher_logits_response = teacher_logits_i[teacher_logit_start:teacher_logit_end]

            # Handle potential shape mismatch due to edge cases
            if shifted_student_logits_response.shape[0] != shifted_teacher_logits_response.shape[0]:
                min_len = min(shifted_student_logits_response.shape[0], shifted_teacher_logits_response.shape[0])
                shifted_student_logits_response = shifted_student_logits_response[:min_len]
                shifted_teacher_logits_response = shifted_teacher_logits_response[:min_len]

            if shifted_student_logits_response.shape[0] == 0:
                continue

            # Compute JSD loss for this sample
            sample_loss = self.generalized_jsd_loss(
                student_logits=shifted_student_logits_response.unsqueeze(0),
                teacher_logits=shifted_teacher_logits_response.unsqueeze(0),
                labels=None,  # Already filtered to response tokens only
                beta=self.beta,
                temperature=self.temperature,
                reduction="sum",
            )

            total_loss = total_loss + sample_loss
            total_tokens += shifted_student_logits_response.shape[0]

        # Average over all response tokens (batchmean)
        if total_tokens > 0:
            loss = total_loss / total_tokens
        else:
            loss = total_loss

        # Empty cache
        empty_cache()

        return (loss, student_outputs) if return_outputs else loss

    @staticmethod
    def generate_on_policy_outputs(model, inputs, generation_config, pad_token_id=None):
        """
        Generate on-policy outputs from the model.

        Returns:
            generated_tokens: [batch_size, seq_len] - prompt + generated response
            new_attention_mask: [batch_size, seq_len] - attention mask
            new_labels: [batch_size, seq_len] - labels with -100 for prompt and padding tokens
        """
        # Get prompt information (accounting for left-padding)
        prompts = inputs["prompts"]
        prompt_attention_mask = inputs.get("prompt_attention_mask", None)

        # Generate output with respect to the prompt-only
        generated_outputs = model.generate(
            input_ids=prompts,
            attention_mask=prompt_attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
        )

        # Get the generated token IDs (includes prompt + generated)
        generated_tokens = generated_outputs.sequences  # [batch_size, total_seq_len]

        # Calculate new attention mask
        new_attention_mask = torch.ones_like(generated_tokens)
        new_labels = generated_tokens.clone()

        # Mask prompt tokens with -100 in labels
        # model.generate preserves the input structure, so prompt occupies first prompt_seq_len positions
        prompt_seq_len = prompts.shape[1]
        new_labels[:, :prompt_seq_len] = -100

        # Set pad tokens to -100 in labels and 0 in attention mask
        if pad_token_id is not None:
            # Only mask padding AFTER the prompt portion (generated padding)
            pad_mask = generated_tokens == pad_token_id
            # Don't modify prompt region labels again (already -100)
            pad_mask[:, :prompt_seq_len] = False
            new_labels[pad_mask] = -100
            new_attention_mask[generated_tokens == pad_token_id] = 0

        return generated_tokens, new_attention_mask, new_labels

    def _build_original_from_answers(self, inputs):
        """
        Build teacher input with different system prompt that reveals the answer.

        For teacher (conditional distribution):
        - System prompt: tells teacher it will evaluate student's reasoning process
        - User prompt: Q + "The answer to the above question is [extracted answer]"
        - y* = just the final answer number (after ####)

        Returns left-padded tensors for: input_ids, attention_mask, labels, and final_answers list.
        """
        answers = inputs["answers"]  # List of answer strings (full GSM8K format)
        device = inputs["prompts"].device
        batch_size = inputs["prompts"].shape[0]

        # Extract final answers (just the number after ####)
        final_answers = [extract_final_answer(ans) for ans in answers]

        # Tokenize each teacher prompt + answer
        all_input_ids = []
        all_attention_mask = []
        all_labels = []

        for i in range(batch_size):
            final_ans = final_answers[i]

            # Build teacher system prompt - explains teacher's role
            teacher_system_prompt = (
                "You are a helpful assistant that evaluates mathematical reasoning. "
                "You will be given a math problem along with its correct answer, "
                "and you need to evaluate the student's reasoning process."
            )

            # Get user message from original prompt by decoding
            # We need to extract the user's question from the student prompt
            prompt_attn = inputs["prompt_attention_mask"][i]
            prompt_valid_start = (prompt_attn == 0).sum().item()
            prompt_ids = inputs["prompts"][i, prompt_valid_start:]
            prompt_text = self.processing_class.decode(prompt_ids, skip_special_tokens=False)

            # Extract user content from the decoded prompt
            # The prompt follows chat template: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
            user_content = ""
            if "<|im_start|>user" in prompt_text:
                user_part = prompt_text.split("<|im_start|>user")[-1]
                if "<|im_end|>" in user_part:
                    user_content = user_part.split("<|im_end|>")[0].strip()

            # Build teacher user prompt: Q + answer hint
            teacher_user_content = (
                f"{user_content}\n\n"
                f"The answer to the above question is {final_ans}."
            )

            # Build teacher messages with modified prompts
            teacher_messages = [
                {"role": "system", "content": teacher_system_prompt},
                {"role": "user", "content": teacher_user_content},
            ]

            # Apply chat template to get teacher prompt
            teacher_prompt = self.processing_class.apply_chat_template(
                teacher_messages, tokenize=False, add_generation_prompt=True
            )

            # Tokenize teacher prompt
            teacher_prompt_tokens = self.processing_class(
                teacher_prompt,
                truncation=True,
                max_length=self.args.max_length,
                padding=False,
                return_tensors=None,
                add_special_tokens=False,
            )["input_ids"]
            teacher_prompt_ids = torch.tensor(teacher_prompt_tokens, dtype=torch.long, device=device)

            # Tokenize the final answer only (y* = just the number)
            answer_tokens = self.processing_class(
                final_ans,
                truncation=True,
                max_length=self.args.max_new_tokens,
                padding=False,
                return_tensors=None,
                add_special_tokens=False,
            )["input_ids"]
            answer_ids = torch.tensor(answer_tokens, dtype=torch.long, device=device)

            # Concatenate: teacher_prompt + final_answer
            input_ids = torch.cat([teacher_prompt_ids, answer_ids], dim=0)
            attention_mask = torch.ones_like(input_ids)

            # Labels: -100 for prompt, actual tokens for answer
            labels = torch.full_like(input_ids, -100)
            labels[len(teacher_prompt_ids):] = answer_ids

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)

        # Left-pad to same length
        max_len = max(ids.shape[0] for ids in all_input_ids)
        pad_token_id = self.processing_class.pad_token_id

        padded_input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
        padded_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        padded_labels = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)

        for i, (ids, attn, lbl) in enumerate(zip(all_input_ids, all_attention_mask, all_labels)):
            pad_len = max_len - ids.shape[0]
            padded_input_ids[i, pad_len:] = ids
            padded_attention_mask[i, pad_len:] = attn
            padded_labels[i, pad_len:] = lbl

        return padded_input_ids, padded_attention_mask, padded_labels, final_answers

    def training_step(
        self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items_in_batch: int | None = None
    ) -> torch.Tensor:
        """
        Perform a training step for the Generalized Knowledge Distillation (GKD) model.

        This method implements the on-policy learning approach described in the GKD paper. With probability
        `self.lmbda`, it generates new responses using the student model, which are then used for training instead of
        the original inputs.

        For conditional self-distillation, the teacher input is (x, y*, y) where:
            - x  = prompt from the dataset
            - y* = ground-truth answer from the dataset (NOT model prediction)
            - y  = student-generated response

        This implements: min KL( P(y|x) || P(y|x, y*) )
        """
        # For self-distillation: build teacher input with different system prompt
        # Teacher prompt includes the answer in the user message
        if self.self_distillation:
            if "answers" in inputs:
                # Build teacher input with hint about the answer
                original_input_ids, original_attention_mask, original_labels, _ = self._build_original_from_answers(inputs)
            else:
                # Fallback: use DataCollatorForChatML output
                original_input_ids = inputs["input_ids"].clone()  # prompt + y* (from dataset)
                original_attention_mask = inputs["attention_mask"].clone()
                original_labels = inputs["labels"].clone()  # -100 for prompt, actual tokens for y*

        # seq_kd: generate from teacher model (NOT compatible with self_distillation where teacher_model=None)
        if self.seq_kd:
            if self.teacher_model is None:
                raise ValueError(
                    "seq_kd=True requires a teacher model, but teacher_model is None. "
                    "seq_kd is not compatible with self_distillation mode."
                )
            with unwrap_model_for_generation(self.teacher_model, self.accelerator) as unwrapped_model:
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels
            inputs["_on_policy_generated"] = True  # Flag for debug

        # On-policy generation with probability lmbda
        on_policy_generated = False
        if random.random() <= self.lmbda:
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels
            on_policy_generated = True

        # For self-distillation: pass the original dataset input (prompt + ground-truth y*)
        # Per-sample alignment is preserved because generate() maintains batch order
        if self.self_distillation:
            inputs["original_input_ids"] = original_input_ids  # prompt + y* (dataset ground-truth)
            inputs["original_attention_mask"] = original_attention_mask
            inputs["original_labels"] = original_labels
            inputs["_on_policy_generated"] = on_policy_generated  # For debug printer

        loss = super().training_step(model, inputs, num_items_in_batch)
        return loss
