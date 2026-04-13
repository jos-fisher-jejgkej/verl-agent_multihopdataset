#!/usr/bin/env python3
"""
test_reranker_vllm.py – Quick smoke-test for the vLLM Qwen3-Reranker server.

Usage:
    python scripts/test_reranker_vllm.py \
        --url http://localhost:8000 \
        --model Qwen/Qwen3-Reranker-0.6B

The script calls both /score and /rerank endpoints and prints the results.
"""

import argparse
import json
import sys

import requests


def test_score_endpoint(base_url: str, model: str):
    """POST /score – returns a relevance score for each (query, doc) pair."""
    url = base_url.rstrip("/") + "/score"
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital and most populous city of France.",
        "The Eiffel Tower is located in Paris.",
        "Berlin is the capital of Germany.",
        "Python is a popular programming language.",
    ]

    payload = {
        "model": model,
        "text_1": query,
        "text_2": documents,
    }

    print(f"\n[/score] POST {url}")
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    scores = [item["score"] for item in data["data"]]
    print(f"  Query   : {query}")
    for doc, score in zip(documents, scores):
        print(f"  [{score:.4f}] {doc[:80]}")
    return scores


def test_rerank_endpoint(base_url: str, model: str):
    """POST /rerank – cohere-compatible rerank API."""
    url = base_url.rstrip("/") + "/rerank"
    query = "How to train a neural network?"
    documents = [
        "Gradient descent is an optimization algorithm.",
        "I enjoy hiking in the mountains.",
        "Backpropagation computes gradients for training neural networks.",
        "Python is widely used for machine learning.",
    ]

    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": 3,
    }

    print(f"\n[/rerank] POST {url}")
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    print(f"  Query: {query}")
    for item in data["results"]:
        idx = item["index"]
        score = item["relevance_score"]
        print(f"  rank {item.get('rank', '?')} [{score:.4f}] idx={idx}: {documents[idx][:80]}")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000", help="vLLM server base URL")
    parser.add_argument(
        "--model", default="Qwen/Qwen3-Reranker-0.6B", help="Model name as known to vLLM"
    )
    args = parser.parse_args()

    print(f"Testing vLLM reranker at {args.url}  (model={args.model})")

    ok = True
    try:
        test_score_endpoint(args.url, args.model)
    except Exception as e:
        print(f"[ERROR] /score failed: {e}", file=sys.stderr)
        ok = False

    try:
        test_rerank_endpoint(args.url, args.model)
    except Exception as e:
        print(f"[ERROR] /rerank failed: {e}", file=sys.stderr)
        ok = False

    if ok:
        print("\n✓ All endpoints working correctly.")
    else:
        print("\n✗ Some endpoints failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

# python scripts/test_reranker_vllm.py --url http://localhost:8000 \
#     --model Qwen/Qwen3-Reranker-0.6B