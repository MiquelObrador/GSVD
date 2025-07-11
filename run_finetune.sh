MODEL_PATH="results/llama-7b/gsvd_llama-7b_r0.6_g500_c256_m1_w_2.pt"
# Directory where the final fine-tuned model will be saved.
OUTPUT_DIR="finetuned_models/"
# This is the batch size per device. The effective batch size is BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS.
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=32 # Set to 1 for no accumulation.
CHECKPOINT_STEPS=100 # Save a checkpoint every N optimization steps.
LEARNING_RATE=1e-3
EPOCHS=1
MAX_SEQ_LEN=2048
BASE_MODEL="huggyllama/llama-7b"
echo "Base Model: $BASE_MODEL"
echo "Compressed Model Path: $MODEL_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Effective Batch Size: $(($BATCH_SIZE * $GRADIENT_ACCUMULATION_STEPS))"

python alpaca_finetune.py \
  --base-model "$BASE_MODEL" \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size $BATCH_SIZE \
  --gradient-accumulation-steps $GRADIENT_ACCUMULATION_STEPS \
  --checkpoint-steps $CHECKPOINT_STEPS \
  --learning-rate $LEARNING_RATE \
  --epochs $EPOCHS \
  --max-seq-len $MAX_SEQ_LEN \
