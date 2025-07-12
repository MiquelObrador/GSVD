#!/bin/bash

# python evaluate.py \
#     --model_base "huggyllama/llama-7b" \
#     --model_path "results/svd_llm/SVD_LLM_llama-7b_r0.6_cs256_s42.pt" \
#     --batch_size 8 \
#     --eval_metric "ppl" \
#     --eval_dataset "c4" \

# #!/usr/bin/env bash
set -euo pipefail

MODEL_BASE="huggyllama/llama-7b"
MODEL_PATH="results/gsvd_hmsecos/gsvd_llama-7b_r0.6_g50_cb10.pt"
EVAL_METRIC="accuracy"

DATASETS=(
  "arc_easy"
  "arc_challenge"
  "openbookqa"
  "winogrande"
  "hellaswag"
  "piqa"
  "mathqa"
)

for DS in "${DATASETS[@]}"; do
  echo "=== Evaluating on ${DS} ==="
  python evaluate.py \
    --model_base "$MODEL_BASE" \
    --model_path "$MODEL_PATH" \
    --eval_metric "$EVAL_METRIC" \
    --eval_dataset "$DS"
  echo
done

echo "All evaluations complete!"
