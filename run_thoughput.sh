#!/bin/bash
# ===================================================================
# Enhanced Script to Evaluate LLaMA Models Across Various Parameters
# Features: Progress tracking, resumption capability, better error handling
# ===================================================================

set -euo pipefail  # Exit on any error, undefined variable, or pipe failure

# Set environment variables for better GPU memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0  # Keep async for better performance

# ==== CONFIGURATION ====
MODEL_BASE="huggyllama/llama-7b"
UPDATED_MODEL_PATH_TEMPLATE="results/llama7b/gsvd_hmsecos/gsvd_llama-7b_r{RATIO}_g50_cb10.pt"
DATASET="wikitext2"
SEED=0

# Output configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="throughput_results_${TIMESTAMP}"
RESULTS_FILE="${RESULTS_DIR}/evaluation_results.jsonl"
LOG_FILE="${RESULTS_DIR}/evaluation.log"
PROGRESS_FILE="${RESULTS_DIR}/progress.txt"

# Create results directory
mkdir -p "$RESULTS_DIR"
echo "Results will be saved to: $RESULTS_DIR"

# Define parameter ranges - Adjusted for better compatibility
RATIOS=(1.0 0.8 0.6 0.4 0.2)  # Added 0.2 for more comprehensive testing
BATCH_SIZES=(64 128 256 512)  # Reduced starting batch sizes to avoid OOM
GENERATED_LENGTHS=(32 16 128 256)  # Reduced max length to avoid memory issues

# Initialize progress tracking
TOTAL_EXPERIMENTS=0
COMPLETED_EXPERIMENTS=0

# Function to log messages
log_message() {
    local message="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $message" | tee -a "$LOG_FILE"
}

# Function to check if experiment was already completed
experiment_completed() {
    local ratio=$1
    local batch_size=$2
    local gen_len=$3
    local experiment_name=$4
    
    if [ -f "$PROGRESS_FILE" ]; then
        grep -q "${experiment_name}_r${ratio}_b${batch_size}_g${gen_len}" "$PROGRESS_FILE" 2>/dev/null
    else
        return 1
    fi
}

# Function to mark experiment as completed
mark_experiment_completed() {
    local ratio=$1
    local batch_size=$2
    local gen_len=$3
    local experiment_name=$4
    
    echo "${experiment_name}_r${ratio}_b${batch_size}_g${gen_len}" >> "$PROGRESS_FILE"
}

# Function to validate model path
validate_model_path() {
    local model_path="$1"
    local ratio="$2"
    
    if (( $(echo "$ratio == 1.0" | bc -l) )); then
        return 0  # Original model, no file check needed
    fi
    
    if [ ! -f "$model_path" ]; then
        log_message "ERROR: Model file not found: $model_path"
        return 1
    fi
    return 0
}

# Function to run a single experiment
run_experiment() {
    local ratio=$1
    local batch_size=$2
    local gen_len=$3
    local experiment_name=$4
    
    log_message "--- Experiment: $experiment_name, Ratio=$ratio, Batch=$batch_size, GenLen=$gen_len ---"
    
    # Check if already completed
    if experiment_completed "$ratio" "$batch_size" "$gen_len" "$experiment_name"; then
        log_message "⏭️  Skipping completed experiment"
        ((COMPLETED_EXPERIMENTS++))
        return 0
    fi
    
    # Determine model path
    if (( $(echo "$ratio == 1.0" | bc -l) )); then
        MODEL_PATH="$MODEL_BASE"
        log_message "🧠 Using ORIGINAL model: $MODEL_PATH"
    else
        MODEL_PATH=$(echo "$UPDATED_MODEL_PATH_TEMPLATE" | sed "s/{RATIO}/$ratio/g")
        log_message "⚙️  Using COMPRESSED model: $MODEL_PATH"
        
        # Validate model exists
        if ! validate_model_path "$MODEL_PATH" "$ratio"; then
            log_message "❌ Skipping due to missing model file"
            return 1
        fi
    fi
    
    # Run the evaluation
    local start_time=$(date +%s)
    log_message "🚀 Starting evaluation..."
    
    if python3 evaluate_speed.py \
        --model_base "$MODEL_BASE" \
        --updated_model_path "$MODEL_PATH" \
        --eval_dataset "$DATASET" \
        --ratio "$ratio" \
        --batch_size "$batch_size" \
        --generated_len "$gen_len" \
        --seed "$SEED" \
        --output_file "$RESULTS_FILE" 2>&1 | tee -a "$LOG_FILE"; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        mark_experiment_completed "$ratio" "$batch_size" "$gen_len" "$experiment_name"
        ((COMPLETED_EXPERIMENTS++))
        
        log_message "✅ Completed in ${duration}s (${COMPLETED_EXPERIMENTS}/${TOTAL_EXPERIMENTS})"
        
        # Clear GPU memory between experiments
        python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        
        return 0
    else
        log_message "❌ Experiment failed"
        return 1
    fi
}

# Calculate total experiments
calculate_total_experiments() {
    local count=0
    
    # Part 1: Batch size sweep
    for ratio in "${RATIOS[@]}"; do
        for batch_size in "${BATCH_SIZES[@]}"; do
            ((count++))
        done
    done
    
    # Part 2: Generated length sweep
    for ratio in "${RATIOS[@]}"; do
        for gen_len in "${GENERATED_LENGTHS[@]}"; do
            ((count++))
        done
    done
    
    echo $count
}

# Initialize counters
TOTAL_EXPERIMENTS=$(calculate_total_experiments)
log_message "Total experiments to run: $TOTAL_EXPERIMENTS"

# Check if we're resuming
if [ -f "$PROGRESS_FILE" ]; then
    COMPLETED_EXPERIMENTS=$(wc -l < "$PROGRESS_FILE" 2>/dev/null || echo 0)
    log_message "Resuming evaluation. Already completed: $COMPLETED_EXPERIMENTS experiments"
fi

# ===================================================================
# PART 1: SWEEP BATCH SIZES (for Figure 4a)
# Fixed generated length of 32
# ===================================================================
FIXED_GEN_LEN=32
log_message "============================================================"
log_message "PART 1: Sweeping Batch Sizes (Generated Length = $FIXED_GEN_LEN)"
log_message "============================================================"

FAILED_EXPERIMENTS=0

for RATIO in "${RATIOS[@]}"; do
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        if ! run_experiment "$RATIO" "$BATCH_SIZE" "$FIXED_GEN_LEN" "batch_sweep"; then
            ((FAILED_EXPERIMENTS++))
        fi
        
        # Progress update (removed local keyword)
        progress_pct=$((COMPLETED_EXPERIMENTS * 100 / TOTAL_EXPERIMENTS))
        log_message "Progress: ${COMPLETED_EXPERIMENTS}/${TOTAL_EXPERIMENTS} (${progress_pct}%)"
    done
done

# ===================================================================
# PART 2: SWEEP GENERATED LENGTHS (for Figure 4b)
# Fixed batch size of 16 (reduced from 64 to avoid OOM)
# ===================================================================
FIXED_BATCH_SIZE=16
log_message "============================================================"
log_message "PART 2: Sweeping Generated Lengths (Batch Size = $FIXED_BATCH_SIZE)"
log_message "============================================================"

for RATIO in "${RATIOS[@]}"; do
    for GEN_LEN in "${GENERATED_LENGTHS[@]}"; do
        if ! run_experiment "$RATIO" "$FIXED_BATCH_SIZE" "$GEN_LEN" "length_sweep"; then
            ((FAILED_EXPERIMENTS++))
        fi
        
        # Progress update (removed local keyword)
        progress_pct=$((COMPLETED_EXPERIMENTS * 100 / TOTAL_EXPERIMENTS))
        log_message "Progress: ${COMPLETED_EXPERIMENTS}/${TOTAL_EXPERIMENTS} (${progress_pct}%)"
    done
done

# ===================================================================
# FINAL SUMMARY
# ===================================================================
log_message "============================================================"
log_message "✅ Evaluation Complete!"
log_message "Completed experiments: $COMPLETED_EXPERIMENTS"
log_message "Failed experiments: $FAILED_EXPERIMENTS"
log_message "Results saved to: $RESULTS_FILE"
log_message "Logs saved to: $LOG_FILE"
log_message "============================================================"

# Generate summary statistics
if [ -f "$RESULTS_FILE" ]; then
    total_records=$(wc -l < "$RESULTS_FILE")
    log_message "Total records in results file: $total_records"
    
    # Check for any errors in results
    error_records=$(grep -c '"error"' "$RESULTS_FILE" 2>/dev/null || echo 0)
    if [ "$error_records" -gt 0 ]; then
        log_message "⚠️  Warning: $error_records records contain errors"
    fi
fi

# Provide next steps
log_message ""
log_message "Next steps:"
log_message "1. Review results: cat $RESULTS_FILE | jq ."
log_message "2. Generate plots with your plotting script"
log_message "3. Check logs for any issues: cat $LOG_FILE"