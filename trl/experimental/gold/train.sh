export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1" 
export TRL_EXPERIMENTAL_SILENCE="1"

accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero2.yaml trl/experimental/gold/gold.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --teacher_model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name DigitalLearningGmbH/MATH-lighteval \
    --learning_rate 1e-3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --output_dir gold-model \
    --num_train_epochs 3 \
    --gradient_checkpointing \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --report_to tensorboard \
    --log_completions True \
    --logging_strategy steps \
    --logging_steps 2 \
    --log_completions_steps 100 \
    --num_completions_to_print 5 \
    --max_length 2048 \
    --max_completion_length 1024 \
    --lmbda 1.0 \
    --beta 1.0 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \

## debug
# python trl/experimental/gold/gold.py \
#     --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
#     --teacher_model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
#     --dataset_name DigitalLearningGmbH/MATH-lighteval \
#     --learning_rate 2e-5 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --output_dir gold-model \
#     --num_train_epochs 1 \
#     --gradient_checkpointing \
#     --use_peft \
#     --lora_r 16 \
#     --lora_alpha 32 \
#     --report_to tensorboard \
#     --log_completions True \
#     --log_completions_steps 100 \
#     --num_completions_to_print 5 \
#     --max_length 256 \
#     --max_completion_length 32 \
#     --lmbda 1.0 \
#     --beta 1.0 \