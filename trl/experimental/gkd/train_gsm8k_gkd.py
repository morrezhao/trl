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

"""
GKD (Generalized Knowledge Distillation) training script for GSM8K.

This script trains the student model using on-policy distillation on the
GSM8K math reasoning dataset and tracks Pass Rate during training.

Usage:
    # Single GPU
    python train_gsm8k_gkd.py

    # Multi-GPU with DeepSpeed
    accelerate launch --config_file configs/accelerate_deepspeed.yaml train_gsm8k_gkd.py
"""

import os

os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"  # To avoid potential NCCL issues]
os.environ["NCCL_IB_DISABLE"] = "1"

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState

from trl import ModelConfig, get_peft_config
from trl.experimental.gkd import GKDConfig, GKDTrainer
from gsm8k_eval import verify_gsm8k_response
from tqdm import tqdm



def is_main_process() -> bool:
    # torchrun/accelerate/deepspeed 都会设 RANK；单卡时默认当作主进程
    return int(os.environ.get("RANK", "0")) == 0

def main_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)

SYSTEM_PROMPT = (
    "You are a helpful assistant that solves math problems step by step. "
    "Think through the problem carefully and show your reasoning."
)

USER_PROMPT_TEMPLATE = (
    "{question}\n\n"
    "Show your work in <think> </think> tags. "
    "And return the final numerical answer in <answer> </answer> tags, "
    "for example <answer> 42 </answer>."
)

def format_gsm8k_to_messages(example: dict) -> dict:
    """Convert GSM8K example to chat message format."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(question=example["question"]),
        },
    ]
    return {"messages": messages, "answer": example["answer"], "question": example["question"]}



class PassRateEvaluationCallback(TrainerCallback):
    """
    Callback to evaluate Pass Rate during training.

    Pass Rate = (Number of correct solutions) / (Total number of problems)
    """

    def __init__(
        self,
        eval_dataset,
        tokenizer,
        eval_steps: int = 100,
        num_eval_samples: int = 100,
        max_new_tokens: int = 512,
    ):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.num_eval_samples = min(num_eval_samples, len(eval_dataset))
        self.max_new_tokens = max_new_tokens
        self.pass_rates = []

    def on_step_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,  # noqa: ARG002
        model=None,
        **kwargs,  # noqa: ARG002
    ):
        if state.global_step % self.eval_steps != 0 or state.global_step == 0:
            return

        if model is None:
            return

        # Only evaluate on main process to avoid distributed deadlock
        if not is_main_process():
            return

        pass_rate = self._evaluate_pass_rate(model, state.global_step)
        self.pass_rates.append((state.global_step, pass_rate))

        # Log to console
        main_print(f"\n[Step {state.global_step}] Pass Rate: {pass_rate:.2%}\n")

        # Log to trainer's logger if available
        if hasattr(args, "report_to") and args.report_to:
            # Will be picked up by wandb/tensorboard if configured
            state.log_history.append(
                {"step": state.global_step, "eval/pass_rate": pass_rate}
            )

    def _evaluate_pass_rate(self, model, step: int = 0, batch_size: int = 8) -> float:
        """Evaluate pass rate on a subset of the evaluation dataset using batched inference."""
        model.eval()

        base = getattr(model, "base_model", model)

        device = torch.device('cuda:0')
        model = model.to(device)

        orig_use_cache = getattr(model.config, "use_cache", None)
        orig_gc_enabled = None
        if hasattr(base, "is_gradient_checkpointing") and callable(getattr(base, "is_gradient_checkpointing")):
            try:
                orig_gc_enabled = base.is_gradient_checkpointing
            except Exception:
                orig_gc_enabled = None

        # Turn off gradient checkpointing if possible
        if hasattr(base, "gradient_checkpointing_disable"):
            try:
                base.gradient_checkpointing_disable()
            except Exception:
                pass

        # Select samples for evaluation
        indices = list(range(self.num_eval_samples))
        eval_samples = self.eval_dataset.select(indices)

        correct = 0
        total = 0

        is_main = int(os.environ.get("RANK", "0")) == 0

        # Prepare all prompts and metadata
        all_prompts = []
        all_answers = []
        for sample in eval_samples:
            messages = sample["messages"][:2]  # System + User
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            all_prompts.append(prompt)
            all_answers.append(sample["answer"])

        # Process in batches
        num_batches = (len(all_prompts) + batch_size - 1) // batch_size

        iterator = range(num_batches)
        if is_main:
            iterator = tqdm(
                iterator,
                desc=f"Eval pass rate (step={step})",
                leave=False,
            )

        with torch.no_grad():
            for batch_idx in iterator:
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(all_prompts))

                batch_prompts = all_prompts[start_idx:end_idx]
                batch_answers = all_answers[start_idx:end_idx]

                # Tokenize batch with left padding for generation
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                ).to(device)

                # Generate responses for the batch
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=1.0,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                )

                # Decode and verify each response in the batch
                # Use full input length (including padding) since output includes padding tokens
                prompt_length = inputs["input_ids"].shape[1]
                for i, (output, answer) in enumerate(zip(outputs, batch_answers)):
                    response = self.tokenizer.decode(
                        output[prompt_length:],
                        skip_special_tokens=True,
                    )

                    is_correct = verify_gsm8k_response(response, answer)
                    if is_correct:
                        correct += 1
                    total += 1


        # ---- restore original states ----
        if orig_use_cache is not None:
            model.config.use_cache = orig_use_cache

        # restore gradient checkpointing if it was enabled before
        if orig_gc_enabled:
            if hasattr(base, "gradient_checkpointing_enable"):
                try:
                    base.gradient_checkpointing_enable()
                except Exception:
                    pass

        model.train()
        return correct / total if total > 0 else 0.0

    def get_pass_rate_history(self) -> list[tuple[int, float]]:
        """Return the history of pass rates."""
        return self.pass_rates


def main():
    # Model configuration
    student_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    teacher_model_name = "Qwen/Qwen3-4B-Instruct-2507"

    # LoRA configuration for memory efficiency
    model_config = ModelConfig(
        model_name_or_path=student_model_name,
        dtype="bfloat16",
        attn_implementation="flash_attention_2",  # Use "eager" if flash attention unavailable
        use_peft=True,
        lora_r=32,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # GKD training configuration
    training_args = GKDConfig(
        output_dir="./qwen-1.5b-gsm8k-gkd",
        # GKD specific parameters
        beta=0.0,
        lmbda=1,
        # Generation parameters
        temperature=1,
        max_new_tokens=512,  # GSM8K needs longer responses
        max_length=1024,
        # Teacher model
        teacher_model_name_or_path=teacher_model_name,
        # Training hyperparameters
        learning_rate=2e-4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        # Memory optimization
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Logging and saving
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        eval_strategy="no",  # We use custom callback for evaluation
        # Other settings
        disable_dropout=True,
        report_to=["tensorboard"],
        run_name="qwen-1.5b-gsm8k-gkd",
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        student_model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load GSM8K dataset (only on main process, then broadcast)
    main_print("Loading GSM8K dataset...")
    if is_main_process():
        dataset = load_dataset("openai/gsm8k", "main")

        # Convert to ChatML format and keep metadata for evaluation
        main_print("Formatting dataset...")
        formatted_train = dataset["train"].map(
            format_gsm8k_to_messages,
            desc="Formatting train dataset",
        )
        formatted_test = dataset["test"].map(
            format_gsm8k_to_messages,
            desc="Formatting test dataset",
        )

        # Use full train set and subset of test for evaluation
        train_dataset = formatted_train.shuffle(seed=42)
        eval_dataset = formatted_test.shuffle(seed=43).select(range(min(100, len(formatted_test))))

        # Remove metadata columns that aren't needed for training
        train_dataset_for_trainer = train_dataset.remove_columns(["answer", "question"])

        # Save to disk for other processes to load
        train_dataset_for_trainer.save_to_disk("/data/zhaoenhan/on-policy-distill/trl/trl/experimental/gkd/tmp/gkd_gsm8k_train_dataset")
        eval_dataset.save_to_disk("/data/zhaoenhan/on-policy-distill/trl/trl/experimental/gkd/tmp/gkd_gsm8k_eval_dataset")

    # Synchronize all processes
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Non-main processes load from disk
    if not is_main_process():
        from datasets import load_from_disk
        train_dataset_for_trainer = load_from_disk("/data/zhaoenhan/on-policy-distill/trl/trl/experimental/gkd/tmp/gkd_gsm8k_train_dataset")
        eval_dataset = load_from_disk("/data/zhaoenhan/on-policy-distill/trl/trl/experimental/gkd/tmp/gkd_gsm8k_eval_dataset")

    main_print(f"Train dataset size: {len(train_dataset_for_trainer)}")
    main_print(f"Eval dataset size: {len(eval_dataset)}")

    # Create Pass Rate evaluation callback
    pass_rate_callback = PassRateEvaluationCallback(
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        eval_steps=50,  # Evaluate every 50 steps
        num_eval_samples=100,  # Use 100 samples for evaluation
        max_new_tokens=512,
    )

    # Initialize trainer
    main_print("Initializing GKD Trainer...")
    trainer = GKDTrainer(
        model=student_model_name,
        teacher_model=teacher_model_name,
        args=training_args,
        train_dataset=train_dataset_for_trainer,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_config),
        callbacks=[pass_rate_callback],
    )

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    main_print("trainable/total:", trainable, "/", total)

    # Initial evaluation before training (only on main process to avoid distributed deadlock)
    if is_main_process():
        print("\nInitial Pass Rate evaluation...")
        initial_pass_rate = pass_rate_callback._evaluate_pass_rate(trainer.model, step=0)
        print(f"Initial Pass Rate of Student Model: {initial_pass_rate:.2%}")
        initial_teacher_pass_rate = pass_rate_callback._evaluate_pass_rate(
            trainer.teacher_model, step=0
        )
        print(f"Pass Rate of Teacher Model: {initial_teacher_pass_rate:.2%}")
        pass_rate_callback.pass_rates.append((0, initial_pass_rate))

    # Synchronize all processes before training
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Train    
    main_print("\nStarting training...")
    trainer.train()

    # Final evaluation (only on main process)
    final_pass_rate = 0.0
    if is_main_process():
        main_print("\nFinal Pass Rate evaluation...")
        final_pass_rate = pass_rate_callback._evaluate_pass_rate(
            trainer.model, step=trainer.state.global_step
        )
        main_print(f"Final Pass Rate: {final_pass_rate:.2%}")

        # Print pass rate history
        main_print("\n" + "=" * 50)
        main_print("Pass Rate History:")
        main_print("=" * 50)
        for step, rate in pass_rate_callback.get_pass_rate_history():
            main_print(f"Step {step:5d}: {rate:.2%}")
        main_print(f"Final    : {final_pass_rate:.2%}")
        main_print("=" * 50)

    # Synchronize before saving
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Save the final model
    trainer.save_model(training_args.output_dir)
    main_print(f"\nModel saved to {training_args.output_dir}")

    # Save pass rate history (only on main process)
    if is_main_process():
        import json

        history_path = os.path.join(training_args.output_dir, "pass_rate_history.json")
        with open(history_path, "w") as f:
            json.dump(
                {
                    "history": pass_rate_callback.get_pass_rate_history(),
                    "final_pass_rate": final_pass_rate,
                },
                f,
                indent=2,
            )
        main_print(f"Pass rate history saved to {history_path}")


if __name__ == "__main__":
    main()
