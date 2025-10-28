#!/bin/bash

# Example script to evaluate a LoRA fine-tuned SVD model
# Usage: ./evaluate_lora.sh

# Set default values - modify these according to your setup
BASE_MODEL="meta-llama/Llama-3.1-8B"  # Base model name or path
SVD_MODEL_PATH="results/llama3/gsvd_Llama-3.1-8B_r0.4_g50_cb10.pt"
LORA_ADAPTER_PATH="finetuned_models/llama3_0.4"  # Path to your LoRA adapter directory
OUTPUT_DIR="benchmark_results"

# Evaluation settings
SEQ_LEN=2048
BATCH_SIZE_ACC=4  # Batch size for accuracy evaluations (if needed)
BATCH_SIZE_PPL=4  # Batch size for perplexity evaluations
SEED=42

# Optional arguments
QUANTIZATION_BITS=""  # Leave empty if not using quantization
LOW_RANK_DICT_PATH=""  # Leave empty if not using dynamic ranks

# Check if required paths are set
if [ -z "$SVD_MODEL_PATH" ]; then
    echo "Error: Please set SVD_MODEL_PATH in the script"
    echo "Example: SVD_MODEL_PATH='path/to/your/svd_model_r0.6.pt'"
    exit 1
fi

if [ -z "$LORA_ADAPTER_PATH" ]; then
    echo "Error: Please set LORA_ADAPTER_PATH in the script"
    echo "Example: LORA_ADAPTER_PATH='path/to/your/lora_output_dir'"
    exit 1
fi

# Accuracy evaluations (no batch size needed, but using BATCH_SIZE_ACC if specified)
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
  CMD="python evaluate_lora.py \
    --model_base $BASE_MODEL \
    --svd_model_path $SVD_MODEL_PATH \
    --lora_adapter_path $LORA_ADAPTER_PATH \
    --eval_metric accuracy \
    --eval_dataset $DS \
    --seq_len $SEQ_LEN \
    --batch_size $BATCH_SIZE_ACC \
    --seed $SEED \
    --output_dir $OUTPUT_DIR"

  # Add optional arguments if they are set
  if [ -n "$QUANTIZATION_BITS" ]; then
    CMD="$CMD --quantization_bits $QUANTIZATION_BITS"
  fi

  if [ -n "$LOW_RANK_DICT_PATH" ]; then
    CMD="$CMD --low_rank_dict_path $LOW_RANK_DICT_PATH"
  fi

  echo "Running evaluation with the following command:"
  echo "$CMD"
  echo ""

  # Execute the command
  eval $CMD
  echo
done

# Perplexity evaluations (with batch_size=BATCH_SIZE_PPL)
PPL_DATASETS=("c4" "ptb" "wikitext2")
for DS in "${PPL_DATASETS[@]}"; do
  echo "=== Evaluating perplexity on ${DS} ==="
  CMD="python evaluate_lora.py \
    --model_base $BASE_MODEL \
    --svd_model_path $SVD_MODEL_PATH \
    --lora_adapter_path $LORA_ADAPTER_PATH \
    --eval_metric ppl \
    --eval_dataset $DS \
    --seq_len $SEQ_LEN \
    --batch_size $BATCH_SIZE_PPL \
    --seed $SEED \
    --output_dir $OUTPUT_DIR"

  # Add optional arguments if they are set
  if [ -n "$QUANTIZATION_BITS" ]; then
    CMD="$CMD --quantization_bits $QUANTIZATION_BITS"
  fi

  if [ -n "$LOW_RANK_DICT_PATH" ]; then
    CMD="$CMD --low_rank_dict_path $LOW_RANK_DICT_PATH"
  fi

  echo "Running evaluation with the following command:"
  echo "$CMD"
  echo ""

  # Execute the command
  eval $CMD
  echo
done

echo "All evaluations complete!"