#!/bin/bash
set -euo pipefail

MODEL_BASE="huggyllama/llama-7b"

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
  python evaluate_uncompressed.py \
    --model_base "$MODEL_BASE" \
    --eval_metric "accuracy" \
    --eval_dataset "$DS"
  echo
done

# Perplexity evaluations (with batch_size=8)
PPL_DATASETS=("c4" "ptb")
for DS in "${PPL_DATASETS[@]}"; do
  echo "=== Evaluating perplexity on ${DS} ==="
  python evaluate_uncompressed.py \
    --model_base "$MODEL_BASE" \
    --eval_metric "ppl" \
    --eval_dataset "$DS" \
    --batch_size 4
  echo
done

echo "All evaluations complete!"