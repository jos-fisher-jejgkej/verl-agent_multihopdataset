# 1. 指定 Ray 临时目录
# 建议选择空间充足的分区，比如 /mnt/project 下新建 ray_tmp 目录
export RAY_TMPDIR="/mnt/project/fsh/ray_tmp"
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

# GiGPO config
mode="mean_std_norm" # "mean_norm" or "mean_std_norm"
enable_similarity=True # enable similarity-based GiGPO
similarity_thresh=0.9 # similarity threshold for GiGPO

TRAIN_DATA="/mnt/project/fsh/verl-agent/_data/searchR1_processed_direct/train.parquet"
VAL_DATA="/mnt/project/fsh/verl-agent/_data/searchR1_processed_direct/test.parquet"

# MODEL="/mnt/project/fsh/verl-agent_multihopdataset/checkpoints/verl_agent_search_multihopdataset/0_grpo_run_search-qwen2.5-3b-instruct_train_data_size_64_ppo_mini_batch_size_128/global_step_100/actor_hf"
# MODEL="/mnt/project/fsh/verl-agent_multihopdataset/checkpoints/verl_agent_search_multihopdataset/1_grpo_run_search-qwen2.5-3b-instruct_train_data_size_64_ppo_mini_batch_size_128_multihopdataset/global_step_100/actor_hf"
# MODEL="/mnt/project/fsh/verl-agent_multihopdataset/checkpoints/verl_agent_search_multihopdataset/2_grpo_run_search-qwen2.5-3b-instruct_train_data_size_64_ppo_mini_batch_size_128_multihopdataset_only_process_incorrect_sample/global_step_100/actor_hf"
# MODEL="/mnt/project/fsh/verl-agent_multihopdataset/checkpoints/verl_agent_search_multihopdataset/3_grpo_run_search-qwen2.5-3b-instruct_train_data_size_64_ppo_mini_batch_size_128_multihopdataset_without_retrieval_reward/global_step_100/actor_hf"
MODEL="/mnt/project/fsh/verl-agent_multihopdataset/checkpoints/verl_agent_search_multihopdataset/4_grpo_run_search-qwen2.5-3b-instruct_train_data_size_64_ppo_mini_batch_size_128_multihopdataset_retrieval_reward_coef05/global_step_100/actor_hf"
# MODEL="/mnt/project/fsh/verl-agent_multihopdataset/checkpoints/verl_agent_search_multihopdataset/5_grpo_run_search-qwen2.5-3b-instruct_train_data_size_64_ppo_mini_batch_size_128_multihopdataset_retrieval_reward2/global_step_100/actor_hf"

# 获取当前 shell 脚本的 basename（不含路径）
EXPERIMENT_NAME=$(basename "$0" .sh)  # 如果脚本是 train_grpo_search.sh，则 SCRIPT_NAME=train_grpo_search

echo "Experiment Name: $EXPERIMENT_NAME"
echo "MODEL: $MODEL"

total_training_steps=200


CUDA_VISIBLE_DEVICES=0,1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    algorithm.gigpo.enable_similarity=$enable_similarity \
    algorithm.gigpo.similarity_thresh=$similarity_thresh \
    env.env_name=search \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n=$group_size \
    env.history_length=4 \
    env.search.search_url='http://192.168.10.7:8000/retrieve' \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_search' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_training_steps=$total_training_steps \
    trainer.test_freq=10 \
    trainer.save_freq=$total_training_steps \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.balance_batch=False \
    trainer.val_before_train=True \
    trainer.val_only=True \
