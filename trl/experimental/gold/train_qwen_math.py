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
GOLD distillation training script for Qwen2.5-0.5B-Instruct with Qwen2.5-7B-Instruct as teacher.
Uses Reverse KL divergence loss on MATH-lighteval dataset with LoRA.

Reference: https://github.com/thinking-machines-lab/tinker-cookbook

Usage (with DeepSpeed ZeRO-2):
    accelerate launch --config_file configs/accelerate_deepspeed.yaml train_qwen_math.py

Single GPU (smaller models only):
    python train_qwen_math.py
"""

import os

# Disable P2P and IB for RTX 4000 series compatibility
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"

from datasets import load_dataset
from transformers import AutoTokenizer

from trl import ModelConfig, get_peft_config
from trl.experimental.gold import GOLDConfig, GOLDTrainer


def format_math_to_messages(example):
    """Convert MATH dataset format to ChatML messages format."""
    messages = [
        {
            "role": "user",
            "content": f"Solve the following math problem step by step.\n\nProblem: {example['problem']}"
        },
        {
            "role": "assistant",
            "content": example["solution"]
        }
    ]
    return {"messages": messages}


def main():
    # Model configuration
    student_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    teacher_model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    # LoRA configuration
    model_config = ModelConfig(
        model_name_or_path=student_model_name,
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",  # Use "eager" if flash attention is not available
        use_peft=True,
        lora_r=32,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # GOLD training configuration
    # beta=1.0 means using Reverse KL divergence (as mentioned in tinker-cookbook)
    training_args = GOLDConfig(
        output_dir="./qwen-0.5b-math-gold-rkl",

        # Reverse KL divergence: beta=1.0
        # When beta=0.0, it's forward KL; when beta=1.0, it's reverse KL
        beta=1.0,

        # Lambda controls the student data fraction (on-policy ratio)
        # 0.5 means 50% on-policy (student-generated) and 50% off-policy (teacher-generated)
        lmbda=0.5,

        # Generation parameters - reduced for memory
        temperature=0.7,
        max_completion_length=256,
        max_length=512,

        # Teacher model (no quantization, use bf16)
        teacher_model_name_or_path=teacher_model_name,

        # Training hyperparameters
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=3,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,

        # Memory optimization
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # DDP settings
        ddp_find_unused_parameters=False,

        # Reduce memory fragmentation
        dataloader_pin_memory=False,
        dataloader_num_workers=0,

        # Logging
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="no",

        # Other settings
        disable_dropout=True,
        report_to=None,  
        run_name="qwen-0.5b-math-gold-rkl",

        # Disable completion logging to save memory
        log_completions=False,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        student_model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and preprocess dataset
    dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval")

    # Convert to ChatML format
    train_dataset = dataset["train"].map(
        format_math_to_messages,
        remove_columns=dataset["train"].column_names,
        desc="Formatting train dataset",
    )


    # Initialize trainer
    trainer = GOLDTrainer(
        model=student_model_name,
        teacher_model=teacher_model_name,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_config),
    )

    # Train
    trainer.train()

    # Save the final model
    trainer.save_model(training_args.output_dir)



if __name__ == "__main__":
    main()
