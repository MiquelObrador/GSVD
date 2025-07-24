#!/bin/bash
set -euo pipefail

MODEL_BASE="huggyllama/llama-7b"
MODEL_PATH="results/gsvd_hmsecos/gsvd_llama-7b_r0.2_g50_cb10.pt"

# Accuracy evaluations (no batch size needed)
ACC_DATASETS=(
  "arc_easy"
  "arc_challenge"
  "openbookqa"
  "winogrande"
  "hellaswag"
  "piqa"
  "mathqa"
)

for DS in "${ACC_DATASETS[@]}"; do
  echo "=== Evaluating accuracy on ${DS} ==="
  python evaluate.py \
    --model_base "$MODEL_BASE" \
    --model_path "$MODEL_PATH" \
    --eval_metric "accuracy" \
    --eval_dataset "$DS"
  echo
done

# Perplexity evaluations (with batch_size=8)
PPL_DATASETS=("c4" "ptb")
for DS in "${PPL_DATASETS[@]}"; do
  echo "=== Evaluating perplexity on ${DS} ==="
  python evaluate.py \
    --model_base "$MODEL_BASE" \
    --model_path "$MODEL_PATH" \
    --eval_metric "ppl" \
    --eval_dataset "$DS" \
    --batch_size 8
  echo
done

echo "All evaluations complete!"