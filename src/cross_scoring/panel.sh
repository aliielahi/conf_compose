#!/usr/bin/env bash
# Score every ProcessBench split with every panel model (run inside the container).
set -euo pipefail
cd "$(dirname "$0")/../.."

MODELS=(
  Qwen/Qwen2.5-0.5B-Instruct
  Qwen/Qwen2.5-1.5B-Instruct
  Qwen/Qwen2.5-7B-Instruct
  deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
  meta-llama/Llama-3.2-3B-Instruct
  meta-llama/Llama-3.1-8B-Instruct
  google/gemma-2-2b-it
  google/gemma-2-9b-it
  mistralai/Mistral-7B-Instruct-v0.3
  microsoft/Phi-3.5-mini-instruct
)

for model in "${MODELS[@]}"; do
  python -m cross_scoring.score --model "$model"
done
python -m cross_scoring.evaluate | tee runs/cross_scoring/evaluation.txt
