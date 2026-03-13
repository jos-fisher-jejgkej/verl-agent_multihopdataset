# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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

import argparse
import logging
import os
import tempfile

import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
import json

# 注释掉 hdfs_io 相关导入（如果不需要 HDFS 功能，可删除）
# from verl.utils.hdfs_io import copy, makedirs

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_SYSTEM_CONTENT = "You are a helpful and harmless assistant."
DEFAULT_USER_CONTENT_PREFIX = ""


def process_single_row(row, current_split_name, row_index):
    """
    Process a single row of data for SearchR1-like format.

    Args:
        row: DataFrame row containing the original data
        current_split_name: Name of the current split (train/test)
        row_index: Index of the row in the DataFrame

    Returns:
        pd.Series: Processed row data in the required format
    """
    question = row.get("question", "")
    
    # 获取新增的steps字段
    steps = row.get("steps", "")

    # Build prompt structure
    user_content = user_content_prefix.rstrip("\n") + question
    prompt = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]

    # Extract ground truth from reward_model or fallback to golden_answers
    reward_model_data = row.get("reward_model")
    if isinstance(reward_model_data, dict) and "ground_truth" in reward_model_data:
        ground_truth = reward_model_data.get("ground_truth")
    else:
        ground_truth = row.get("golden_answers", [])

    # Process data source
    data_source_tagged = str(row.get("data_source", ""))

    # Build tools kwargs structure
    tools_kwargs = {
        "search": {
            "create_kwargs": {"ground_truth": ground_truth, "question": question, "data_source": data_source_tagged}
        }
    }

    # Build complete extra_info structure
    extra_info = {
        "index": row_index,
        "need_tools_kwargs": True,
        "question": question,
        "split": current_split_name,
        "tools_kwargs": tools_kwargs,
        "steps": steps,  # 将steps添加到extra_info中
    }

    return pd.Series(
        {
            "data_source": data_source_tagged,
            "prompt": prompt,
            "ability": row.get("ability"),
            "reward_model": reward_model_data,
            "extra_info": extra_info,
            "metadata": row.get("metadata"),
            "env_kwargs": {
                "ground_truth": ground_truth, 
                "question": question, 
                "data_source": data_source_tagged,
            },
        }
    )


def main():
    # 加载参考的 JSON 数据（用于过滤）
    # file_path = "/mnt/project/fsh/verl-agent_multihopdataset/_data/multihopdataset/musique/test_musique_multihop_addllmjudge_truesample_searchresults_subquestionllmjudge_filt0.5_3.json"
    # output_file_path = "/mnt/project/fsh/verl-agent_multihopdataset/_data/multihopdataset/musique/test_musique_multihop_addllmjudge_truesample_searchresults_subquestionllmjudge_filt0.5_3.parquet"
    
    file_path = "/mnt/project/fsh/verl-agent_multihopdataset/_data/multihopdataset/musique/train_musique_multihop_addllmjudge_truesample_searchresults_subquestionllmjudge_filt0.5_3.json"
    output_file_path = "/mnt/project/fsh/verl-agent_multihopdataset/_data/multihopdataset/musique/train_musique_multihop_addllmjudge_truesample_searchresults_subquestionllmjudge_filt0.5_3.parquet"
    # 以只读模式打开文件，指定编码为 utf-8 避免中文乱码
    with open(file_path, 'r', encoding='utf-8') as f:
        # 加载 JSON 数据
        json_data = json.load(f)

    multihop_data = {}
    for data in json_data:
        question = data["question"]
        multihop_data[question] = data
    
    # 提取需要保留的问题列表（用于过滤）
    keep_questions = set(multihop_data.keys())
    logger.info(f"Loaded {len(keep_questions)} questions to keep from reference JSON")

    local_save_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    processed_files = []

    # Download and process files using temporary directory
    with tempfile.TemporaryDirectory() as tmp_download_dir:
        for split in ["train"]:
            local_parquet_filepath = f"/mnt/project/fsh/verl-agent_multihopdataset/_data/multihopdataset/musique/{split}.parquet"
            # Load and process Parquet file
            df_raw = pd.read_parquet(local_parquet_filepath)
            logger.info(f"Loaded {len(df_raw)} rows from {local_parquet_filepath}")

            # ========== 核心修改：过滤数据并添加step字段 ==========
            # 只保留 question 存在于 keep_questions 中的行
            # 先检查 df_raw 是否有 question 列
            if "question" not in df_raw.columns:
                logger.error("DataFrame does not contain 'question' column, skipping filtering")
                continue

            # 过滤数据
            df_filtered = df_raw[df_raw["question"].isin(keep_questions)]
            
            # 核心修改：添加step字段到df_filtered
            def get_step(question):
                steps = multihop_data[question]["steps"]
                _steps = []
                for step in steps:
                    if "sub_question_index" not in step:
                        print(step)
                    else:
                        step.pop("sub_question_index")
                        step.pop("target_documents")
                    _steps.append(step)
                steps = _steps
                return steps
            

            # 新增step列
            df_filtered["steps"] = df_filtered["question"].apply(get_step)

            logger.info(f"Filtered to {len(df_filtered)} rows (removed {len(df_raw)-len(df_filtered)} rows)")
            logger.info(f"Added 'step' column to {len(df_filtered)} rows")
            # ========== 过滤和添加字段结束 ==========

            def apply_process_row(row, split_name=split):
                return process_single_row(row, current_split_name=split_name, row_index=row.name)

            df_processed = df_filtered.apply(apply_process_row, axis=1)

            # Save processed DataFrame
            df_processed.to_parquet(output_file_path, index=False)
            logger.info(f"Saved {len(df_processed)} processed rows to {output_file_path}")
            processed_files.append(output_file_path)

    if not processed_files:
        logger.warning("No data was processed or saved")
        return

    logger.info(f"Successfully processed {len(processed_files)} files to {local_save_dir}")

    # 注释掉 HDFS 相关逻辑（如果需要可取消注释）
    # if args.hdfs_dir:
    #     try:
    #         makedirs(args.hdfs_dir)
    #         copy(src=local_save_dir, dst=args.hdfs_dir)
    #         logger.info(f"Successfully copied files to HDFS: {args.hdfs_dir}")
    #     except Exception as e:
    #         logger.error(f"Error copying files to HDFS: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON files to Search-R1 formatted Parquet files.")
    # 移除 HF 仓库相关参数，新增 JSON 输入目录参数
    parser.add_argument(
        "--json_input_dir",
        default="/mnt/project/fsh/verl-agent_multihopdataset/_data/multihopdataset",
        help="Directory containing train.json and test.json files.",
    )
    parser.add_argument(
        "--local_dir",
        default="/mnt/project/fsh/verl-agent_multihopdataset/_data/multihopdataset",
        help="Local directory to save the processed Parquet files.",
    )
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to copy the Parquet files to.")

    args = parser.parse_args()

    # System and user content configuration
    system_content = DEFAULT_SYSTEM_CONTENT
    user_content_prefix = DEFAULT_USER_CONTENT_PREFIX

    main()
    