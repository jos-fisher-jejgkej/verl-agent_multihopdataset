#!/bin/bash

# 1. 指定 Ray 临时目录
# 建议选择空间充足的分区，比如 /mnt/project 下新建 ray_tmp 目录
export RAY_TMPDIR="/tmp/ray"
# 确保目录存在，不存在则创建
mkdir -p $RAY_TMPDIR

# export RAY_DEBUG_POST_MORTEM=1
export WANDB_API_KEY=fb66753c54f510557f918cff15492604850941ee
export SWANLAB_API_KEY=GkK8zRDsIytg2wAr7Wm6d
export RAY_DEBUG=0

set -x

ENGINE=${1:-vllm}

train_data_size=256
val_data_size=512
group_size=5

# 从脚本中处理后的文件不包含env_kwargs字段
TRAIN_DATA="/home/expand_disk/data_repository/zxw2/Search-R1/_multihot_dataset_v2/nq_hotpotqa/v1_0_10000/train_nq_hotpotqa_qwen3max_em_correcttrajectory_rewrittenqueries_documentsllmjudge_effectivedocnum3.parquet"
VAL_DATA="/home/expand_disk/data_repository/fsh2/verl-agent_multihopdataset/_data/test_nq_hotpotqa_musique_bamboogle.parquet"

# ===================== 核心修改1：定义模型列表 =====================
# 在这里添加/修改你想要评估的模型路径
MODELS=(
    "/home/expand_disk/data_repository/zxw2/LlamaFactory/saves/Qwen2.5-0.5B-Instruct/lora/2_qwen2.5_0.5b-instruct_nq_hotpotqa_qwen3max_correct_only_multi_step_all_0_50000_28194/checkpoint-100_merged"
    "/home/expand_disk/data_repository/zxw2/LlamaFactory/saves/Qwen2.5-0.5B-Instruct/lora/2_qwen2.5_0.5b-instruct_nq_hotpotqa_qwen3max_correct_only_multi_step_all_0_50000_28194/checkpoint-200_merged"
    "/home/expand_disk/data_repository/zxw2/LlamaFactory/saves/Qwen2.5-0.5B-Instruct/lora/2_qwen2.5_0.5b-instruct_nq_hotpotqa_qwen3max_correct_only_multi_step_all_0_50000_28194/checkpoint-300_merged"
    "/home/expand_disk/data_repository/zxw2/LlamaFactory/saves/Qwen2.5-0.5B-Instruct/lora/2_qwen2.5_0.5b-instruct_nq_hotpotqa_qwen3max_correct_only_multi_step_all_0_50000_28194/checkpoint-400_merged"
    "/home/expand_disk/data_repository/zxw2/LlamaFactory/saves/Qwen2.5-0.5B-Instruct/lora/2_qwen2.5_0.5b-instruct_nq_hotpotqa_qwen3max_correct_only_multi_step_all_0_50000_28194/checkpoint-500_merged"
    "/home/expand_disk/data_repository/zxw2/LlamaFactory/saves/Qwen2.5-0.5B-Instruct/lora/2_qwen2.5_0.5b-instruct_nq_hotpotqa_qwen3max_correct_only_multi_step_all_0_50000_28194/checkpoint-600_merged"
)



# ===================== 核心修改2：日志配置 =====================
# 日志保存根目录（可根据需要修改）
LOG_ROOT="/home/expand_disk/data_repository/fsh2/verl-agent_multihopdataset/_log/sft_model_evaluation_logs_all_0_50000_28194"
# 创建日志根目录
mkdir -p $LOG_ROOT
# 获取当前时间戳，用于日志文件命名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 获取当前 shell 脚本的 basename（不含路径）
BASE_EXPERIMENT_NAME=$(basename "$0" .sh)

total_training_steps=300

# ===================== 核心修改3：循环评估每个模型并保存日志 =====================
for MODEL in "${MODELS[@]}"; do
    # 为每个模型生成唯一的实验名（避免覆盖）
    # 提取模型路径中的关键部分作为后缀
    MODEL_SUFFIX=$(basename "$MODEL" | sed 's/[^a-zA-Z0-9]/_/g')
    EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_${MODEL_SUFFIX}"
    
    # 为每个模型创建独立的日志文件
    LOG_FILE="${LOG_ROOT}/${EXPERIMENT_NAME}_${TIMESTAMP}.log"
    # 错误日志单独保存（可选）
    ERROR_LOG_FILE="${LOG_ROOT}/${EXPERIMENT_NAME}_${TIMESTAMP}_error.log"

    echo "========================================"
    echo "开始评估模型: $MODEL"
    echo "实验名称: $EXPERIMENT_NAME"
    echo "日志文件: $LOG_FILE"
    echo "错误日志: $ERROR_LOG_FILE"
    echo "========================================"

    # 执行评估命令，并将所有输出（标准输出+标准错误）保存到日志文件
    # tee -a 同时输出到终端和日志文件，方便实时查看
    CUDA_VISIBLE_DEVICES=2 python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=$TRAIN_DATA \
        data.val_files=$VAL_DATA \
        data.train_batch_size=$train_data_size \
        data.val_batch_size=$val_data_size \
        data.max_prompt_length=4096 \
        data.max_response_length=512 \
        data.filter_overlong_prompts=False \
        data.truncation='left' \
        data.return_raw_chat=True \
        actor_rollout_ref.model.path=$MODEL \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=8 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=$ENGINE \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=False \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.use_invalid_action_penalty=True \
        +actor_rollout_ref.actor.use_invalid_action_penalty_type=2 \
        actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
        algorithm.use_kl_in_reward=False \
        env.env_name=search \
        env.seed=0 \
        env.max_steps=4 \
        env.rollout.n=$group_size \
        env.history_length=4 \
        env.search.search_url='http://192.168.10.3:8123/retrieve' \
        trainer.critic_warmup=0 \
        trainer.logger=['console'] \
        trainer.project_name='verl_agent_search_multihopdataset_0.5b' \
        trainer.experiment_name=$EXPERIMENT_NAME \
        trainer.n_gpus_per_node=1 \
        trainer.nnodes=1 \
        trainer.total_training_steps=$total_training_steps \
        trainer.test_freq=30 \
        trainer.save_freq=$total_training_steps \
        trainer.max_actor_ckpt_to_keep=3 \
        +algorithm.use_multihop_dataset=False \
        +algorithm.retrieval_reward_type=3 \
        +algorithm.retrieval_reward_coef=1.0 \
        +algorithm.use_Rollback=False \
        +algorithm.Max_Rollback_Step=2 \
        +algorithm.use_RollBacked_Step=False \
        trainer.val_before_train=True \
        trainer.val_only=True \
    1>> "$LOG_FILE" 2> "$ERROR_LOG_FILE" | tee -a "$LOG_FILE"

    # 检查上一个模型评估是否成功
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "模型 $MODEL 评估失败，错误码: $EXIT_CODE" | tee -a "$LOG_FILE"
        echo "错误详情请查看: $ERROR_LOG_FILE" | tee -a "$LOG_FILE"
        # 取消注释下面一行可以在某个模型失败时终止脚本
        # exit $EXIT_CODE
    fi

    echo "========================================"
    echo "模型 $MODEL 评估完成"
    echo "所有输出已保存到: $LOG_FILE"
    echo "========================================"
    echo "" | tee -a "$LOG_FILE"
done

# 生成评估汇总文件
SUMMARY_FILE="${LOG_ROOT}/evaluation_summary_${TIMESTAMP}.txt"
echo "模型评估汇总 - $(date)" > "$SUMMARY_FILE"
echo "========================================" >> "$SUMMARY_FILE"
echo "评估时间: $(date)" >> "$SUMMARY_FILE"
echo "日志根目录: $LOG_ROOT" >> "$SUMMARY_FILE"
echo "评估的模型列表:" >> "$SUMMARY_FILE"
for MODEL in "${MODELS[@]}"; do
    MODEL_SUFFIX=$(basename "$MODEL" | sed 's/[^a-zA-Z0-9]/_/g')
    LOG_FILE="${LOG_ROOT}/${BASE_EXPERIMENT_NAME}_${MODEL_SUFFIX}_${TIMESTAMP}.log"
    echo "- $MODEL" >> "$SUMMARY_FILE"
    echo "  日志文件: $LOG_FILE" >> "$SUMMARY_FILE"
done
echo "========================================" >> "$SUMMARY_FILE"

echo "所有模型评估完成！"
echo "评估汇总文件: $SUMMARY_FILE"
echo "所有日志文件保存在: $LOG_ROOT"