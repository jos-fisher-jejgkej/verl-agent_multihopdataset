# save_path=$HOME/data/searchR1

# index_file=$save_path/e5_Flat.index
# corpus_file=$save_path/wiki-18.jsonl
# retriever_name=e5
# retriever_path=intfloat/e5-base-v2

# python examples/search/retriever/retrieval_server.py \
#   --index_path $index_file \
#   --corpus_path $corpus_file \
#   --topk 3 \
#   --retriever_name $retriever_name \
#   --retriever_model $retriever_path \
#   --faiss_gpu \
#   --port 8000 \


export HF_ENDPOINT='https://hf-mirror.com'
export HF_HOME=./.cache
export HF_DATASETS_CACHE=$HF_HOME/datasets_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers_cache
save_path=./_data/searchR1_corpus
index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=./_model/e5-base-v2

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/search/retriever/retrieval_server.py \
  --index_path $index_file \
  --corpus_path $corpus_file \
  --topk 3 \
  --retriever_name $retriever_name \
  --retriever_model $retriever_path \
  --port 8123 \
  --faiss_gpu \


# python3 -c "
# import faiss
# path = '/home/lhyq-4090/han/Search-R1/corpus/e5_Flat.index'
# index = faiss.read_index(path)
# print(f'Vectors: {index.ntotal}, Dim: {index.d}')
# print(f'Theoretical memory: {index.ntotal * index.d * 4 / (1024**3):.2f} GB')
# "
# Vectors: 21,015,324, Dim: 768
# Theoretical memory: 60.13 GB

# /home/lhyq-4090/han/Search-R1/corpus/wiki-18.jsonl 中是文档
# /home/lhyq-4090/han/Search-R1/corpus/e5_Flat.index 中对应向量


# (verl-agent) lhyq-4090@lhyq-4090-MS-7D94:~$ sudo ufw status
# 状态： 激活

# 至                          动作          来自
# -                          --          --
# 22                         ALLOW       Anywhere                  
# 22 (v6)                    ALLOW       Anywhere (v6)       

# 防火墙处于激活（开启） 状态；
# 仅放行（ALLOW）了 22 端口（SSH 远程登录）的 IPv4 和 IPv6 流量；
# 8000 端口未被放行 —— 这是导致 111.0.130.56:8000 连接被拒绝的核心原因之一。

# # 放行 8000 端口的 TCP 流量（永久生效）
# sudo ufw allow 8000/tcp

# # 重新加载防火墙规则（确保生效）
# sudo ufw reload

# (verl-agent) lhyq-4090@lhyq-4090-MS-7D94:~$ sudo ufw status
# 状态： 激活

# 至                          动作          来自
# -                          --          --
# 22                         ALLOW       Anywhere                  
# 8000/tcp                   ALLOW       Anywhere                  
# 22 (v6)                    ALLOW       Anywhere (v6)             
# 8000/tcp (v6)              ALLOW       Anywhere (v6)    