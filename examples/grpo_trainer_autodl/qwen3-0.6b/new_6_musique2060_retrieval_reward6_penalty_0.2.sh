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
TRAIN_DATA="./_data/train_musique_qwen3max_em_correcttrajectory_rewrittenqueries_documentsllmjudge_effectivedocnum3_2060.parquet"
VAL_DATA="./_data/test_nq_hotpotqa_musique_bamboogle.parquet"

# 获取当前 shell 脚本的 basename（不含路径）
EXPERIMENT_NAME=$(basename "$0" .sh)  # 如果脚本是 train_grpo_search.sh，则 SCRIPT_NAME=train_grpo_search
echo "Experiment Name: $EXPERIMENT_NAME"

#####################################################################################
# # 核心修改1：动态检测当前路径是否包含autodl，并设置对应的search_url
# current_path=$(pwd)
# if [[ $current_path == *autodl* ]]; then
#     SEARCH_URL='http://127.0.0.1:8889/retrieve'  # 本机8889端口
#     # 1. 指定 Ray 临时目录
#     # 建议选择空间充足的分区，比如 /mnt/project 下新建 ray_tmp 目录
#     export RAY_TMPDIR="/root/autodl-tmp/tmp_ray"
# else
#     SEARCH_URL='http://192.168.10.3:8123/retrieve'  # 原地址
#     export RAY_TMPDIR="/tmp/ray"
# fi

SEARCH_URL='http://127.0.0.1:8123/retrieve'
export RAY_TMPDIR="/root/autodl-tmp/tmp_ray"

# 确保目录存在，不存在则创建
mkdir -p $RAY_TMPDIR

# 核心修改2：curl检测URL服务状态（匹配实际业务的JSON请求体格式）
echo "=== 检测搜索服务可用性: $SEARCH_URL ==="
# 定义与业务一致的测试JSON请求体
# 字段说明：
# - query: 单字符串（非列表），使用测试用查询词
# - topk: 数字类型（示例值5，可根据实际TOP_K调整）
# - return_scores: 布尔值True
TEST_JSON='{
    "query": "test search query",
    "topk": 5,
    "return_scores": true
}'

# 使用curl发送POST请求，携带标准JSON请求体
# 参数严格匹配业务调用格式，确保检测结果准确
HTTP_STATUS=$(curl -X POST \
    -H "Content-Type: application/json" \
    -d "$TEST_JSON" \
    -s -o /dev/null -w "%{http_code}" \
    --connect-timeout 10 --max-time 15 \
    "$SEARCH_URL")

# 判断服务状态
if [[ "$HTTP_STATUS" -ge 200 && "$HTTP_STATUS" -lt 300 ]]; then
    echo "✅ 搜索服务正常 (HTTP状态码: $HTTP_STATUS)"
else
    echo "❌ 搜索服务异常！"
    echo "  - 检测URL: $SEARCH_URL"
    echo "  - HTTP状态码: $HTTP_STATUS (预期2xx)"
    echo "  - 测试请求体: $TEST_JSON"
    echo "  - 请检查服务是否启动、端口是否正确、请求格式是否匹配"
    exit 1  # 非0退出码终止脚本
fi
#####################################################################################

total_training_steps=300

CUDA_VISIBLE_DEVICES=0,1 python3 -m verl.trainer.main_ppo ray_init.num_cpus=32 \
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
    actor_rollout_ref.model.path=./_model/Qwen3-0.6B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
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
    env.search.search_url=$SEARCH_URL \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='verl_agent_search_multihopdataset_qwen3-0.6b-autodl' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_training_steps=$total_training_steps \
    trainer.test_freq=30 \
    trainer.save_freq=$total_training_steps \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.val_before_train=False \
    trainer.rollout_data_dir=./_log/$EXPERIMENT_NAME \
    +algorithm.use_multihop_dataset=True \
    +algorithm.retrieval_reward_type=6 \
    +algorithm.retrieval_reward_coef=1.0 \
    +algorithm.invalid_search_action_penalty=-0.2 \
    +algorithm.use_Rollback=False \
    +algorithm.Max_Rollback_Step=2 \
    +algorithm.use_RollBacked_Step=False \
    
