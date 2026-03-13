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

# train_data_size=256
train_data_size=64
val_data_size=512
group_size=5

# 从脚本中处理后的文件不包含env_kwargs字段
TRAIN_DATA="/mnt/project/fsh/verl-agent_multihopdataset/_data/multihopdataset/musique/train_musique_multihop_addllmjudge_truesample_searchresults_subquestionllmjudge_filt0.5_3.parquet"
VAL_DATA="/mnt/project/fsh/verl-agent/_data/searchR1_processed_direct/test_300samples_sampled_by_each_source.parquet"

# 获取当前 shell 脚本的 basename（不含路径）
EXPERIMENT_NAME=$(basename "$0" .sh)  # 如果脚本是 train_grpo_search.sh，则 SCRIPT_NAME=train_grpo_search
echo "Experiment Name: $EXPERIMENT_NAME"

total_training_steps=100

CUDA_VISIBLE_DEVICES=2,3 python3 -m verl.trainer.main_ppo \
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
    actor_rollout_ref.model.path=/mnt/project/fsh/verl-agent/_model/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
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
    env.search.search_url='http://192.168.10.7:8000/retrieve' \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='verl_agent_search_multihopdataset' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_training_steps=$total_training_steps \
    trainer.test_freq=10 \
    trainer.save_freq=$total_training_steps \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.val_before_train=False \
    trainer.rollout_data_dir=./_log/$EXPERIMENT_NAME \
    +algorithm.use_multihop_dataset=True \
