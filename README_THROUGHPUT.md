# Enhanced Throughput Evaluation System

This directory contains an improved throughput evaluation system for SVD-compressed language models with better error handling, progress tracking, and comprehensive analysis capabilities.

## 🚀 Quick Start

1. **Make the evaluation script executable:**
   ```bash
   chmod +x run_thoughput.sh
   ```

2. **Update model paths in `run_thoughput.sh`** (if needed):
   - Edit the `UPDATED_MODEL_PATH_TEMPLATE` variable to match your model storage structure
   - Verify the `MODEL_BASE` path is correct

3. **Run the throughput evaluation:**
   ```bash
   ./run_thoughput.sh
   ```

4. **Analyze results and generate plots:**
   ```bash
   # Install analysis dependencies (optional, for plotting)
   pip install -r analysis_requirements.txt
   
   # Analyze results
   python analyze_throughput_results.py throughput_results_*/evaluation_results.jsonl
   ```

## 📁 File Overview

### Core Scripts

- **`evaluate_speed.py`** - Enhanced evaluation script with:
  - ✅ Robust error handling and logging
  - ✅ GPU memory management
  - ✅ Comprehensive metrics collection
  - ✅ Progress tracking and debugging info
  - ✅ Automatic directory creation

- **`run_thoughput.sh`** - Improved batch evaluation script with:
  - ✅ Progress tracking and resumption capability
  - ✅ Model path validation
  - ✅ Organized output structure
  - ✅ Comprehensive logging
  - ✅ Error recovery

- **`analyze_throughput_results.py`** - Analysis and visualization script with:
  - ✅ Publication-ready plots
  - ✅ Comprehensive statistics
  - ✅ Memory efficiency analysis
  - ✅ Graceful dependency handling

### Utility Files

- **`analysis_requirements.txt`** - Python dependencies for plotting
- **`validate_improvements.py`** - Validation script for testing improvements
- **`README_THROUGHPUT.md`** - This documentation file

## 🔧 Key Improvements Made

### 1. Enhanced Error Handling
- **Graceful model loading failures** with detailed error messages
- **GPU memory overflow protection** with automatic cache clearing
- **Invalid generation output detection** and handling
- **Comprehensive exception logging** with stack traces

### 2. Better Directory and Results Management
- **Automatic directory creation** for output files
- **Timestamped result directories** to prevent overwrites
- **Progress tracking files** for resumption capability
- **Comprehensive metadata** in results (timestamps, configurations, system info)

### 3. Improved Metrics and Debugging
- **Warmup batches** to exclude from timing measurements
- **Per-batch timing and memory tracking**
- **Success/failure rate monitoring**
- **Detailed GPU memory information**
- **Experiment identification and tagging**

### 4. Robust Shell Script
- **Model file validation** before running experiments
- **Resumption capability** - can continue from where it left off
- **Progress indicators** with completion percentages
- **Comprehensive logging** with timestamps
- **Failed experiment tracking** and recovery

### 5. Analysis and Visualization
- **Publication-ready plots** matching your reference figure style
- **Memory efficiency analysis** 
- **Summary statistics tables**
- **Graceful handling of missing dependencies**

## 📊 Output Structure

When you run the evaluation, it creates a timestamped directory structure:

```
throughput_results_YYYYMMDD_HHMMSS/
├── evaluation_results.jsonl     # Raw results data
├── evaluation.log              # Detailed execution logs
└── progress.txt                # Progress tracking (for resumption)
```

Analysis outputs go to:
```
analysis_output/
├── throughput_vs_batch_size.png
├── throughput_vs_sequence_length.png
├── efficiency_analysis.png
└── summary_statistics.csv
```

## 🎯 Configuration Options

### evaluate_speed.py Parameters
```bash
python evaluate_speed.py \
    --model_base "huggyllama/llama-7b" \
    --updated_model_path "path/to/compressed/model.pt" \
    --ratio 0.6 \
    --batch_size 64 \
    --generated_len 32 \
    --eval_dataset "wikitext2" \
    --output_file "results.jsonl"
```

### Key Parameters to Adjust in run_thoughput.sh
- `RATIOS` - Compression ratios to test (default: 1.0, 0.8, 0.6, 0.4, 0.2)
- `BATCH_SIZES` - Batch sizes for Figure 4a (default: 32, 64, 128, 256, 512)
- `GENERATED_LENGTHS` - Sequence lengths for Figure 4b (default: 16, 32, 64, 128, 256)
- `MODEL_BASE` - Base model path
- `UPDATED_MODEL_PATH_TEMPLATE` - Template for compressed model paths

## 🐛 Troubleshooting

### Common Issues

1. **Model file not found:**
   - Check the `UPDATED_MODEL_PATH_TEMPLATE` in `run_thoughput.sh`
   - Ensure compressed model files exist for non-1.0 ratios

2. **GPU out of memory:**
   - Reduce batch sizes in the script
   - The script automatically handles OOM and continues with other experiments

3. **Dependencies missing:**
   - For basic evaluation: `torch`, `transformers`, `datasets`
   - For plotting: `pip install -r analysis_requirements.txt`

4. **Resuming interrupted runs:**
   - The script automatically resumes from where it left off
   - Check `progress.txt` to see completed experiments

### Checking Results

```bash
# View results summary
python analyze_throughput_results.py <results_file> --output_dir plots

# Check for errors in results
grep -c '"error"' evaluation_results.jsonl

# View detailed logs
tail -f throughput_results_*/evaluation.log
```

## 📈 Expected Performance

The improved system provides:
- **Better reliability** - handles failures gracefully
- **Progress tracking** - shows completion status
- **Resumption** - can continue interrupted runs
- **Comprehensive data** - detailed metrics for analysis
- **Publication-ready plots** - professional visualization

## 🔄 Next Steps After Running

1. **Review the summary statistics** in the generated CSV file
2. **Examine the plots** to understand performance patterns
3. **Check logs** for any issues or warnings
4. **Customize analysis** by modifying the analysis script for your specific needs

The system is now ready for production use with comprehensive error handling and professional-quality output generation.