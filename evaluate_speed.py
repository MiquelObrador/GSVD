#!/usr/bin/env python3
"""
Enhanced speed evaluation script for SVD-compressed language models.
Provides comprehensive throughput and memory usage benchmarking.
"""

import argparse
import os
import sys
from pathlib import Path
from torch.utils.data.dataset import Dataset
from datasets import load_dataset
import torch
import numpy as np
import random
import itertools
import time
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer
from svdmodels import SVDModel
import json
import datetime
import logging
from typing import Dict, Any, Optional, Tuple

def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_file = output_dir / "evaluation.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def ensure_directories(output_file: str) -> Path:
    """Ensure output directories exist and return the output directory path."""
    output_path = Path(output_file)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def validate_model_path(model_path: str, ratio: float) -> bool:
    """Validate that the model path exists for non-original models."""
    if ratio == 1.0:
        return True  # Original model from HuggingFace
    
    if not os.path.exists(model_path):
        logging.error(f"Model path does not exist: {model_path}")
        return False
    return True


def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory information."""
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "free_gb": 0.0}
    
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free = total - reserved
    
    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "free_gb": free,
        "total_gb": total
    }


# (The get_test_data function remains exactly the same as in your original file)
def get_test_data(name, tokenizer, seq_len=2048, batch_size=4):
    class IndexDataset(Dataset):
        def __init__(self, tensors):
            self.tensors = tensors
        def __getitem__(self, index):
            return self.tensors[index]
        def __len__(self):
            return len(self.tensors)

    def process_data(samples, tokenizer, seq_len, field_name):
        # Concatenate all text and tokenize
        text = "\n\n".join(samples[field_name])
        test_ids = tokenizer(text, return_tensors='pt').input_ids[0]
        
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len
        
        # Ensure we have at least some samples
        if nsamples == 0:
            # If input is too short, pad or truncate to seq_len
            if test_ids.numel() < seq_len:
                # Pad with EOS tokens
                pad_length = seq_len - test_ids.numel()
                pad_tokens = torch.full((pad_length,), tokenizer.eos_token_id, dtype=test_ids.dtype)
                test_ids = torch.cat([test_ids, pad_tokens])
            else:
                # Truncate to seq_len
                test_ids = test_ids[:seq_len]
            nsamples = 1
        
        # Create batches
        for i in range(min(nsamples, 20)):  # Limit to 20 samples max
            start_idx = i * seq_len
            end_idx = start_idx + seq_len
            if end_idx <= test_ids.numel():
                batch = test_ids[start_idx:end_idx]
                test_ids_batch.append(batch)
        
        if len(test_ids_batch) == 0:
            # Fallback: create at least one batch
            if test_ids.numel() >= seq_len:
                test_ids_batch.append(test_ids[:seq_len])
            else:
                # Pad the sequence
                pad_length = seq_len - test_ids.numel()
                pad_tokens = torch.full((pad_length,), tokenizer.eos_token_id, dtype=test_ids.dtype)
                padded_tokens = torch.cat([test_ids, pad_tokens])
                test_ids_batch.append(padded_tokens)
        
        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)

    if 'wikitext2' in name:
        test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    elif 'ptb' in name:
        test_data = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')
    elif 'c4' in name:
        test_data = load_dataset("json", data_files="utils/c4-validation.json")['train']
        test_dataset = process_data(test_data[0:2000], tokenizer, seq_len, 'text')
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def eff_eval(model, tokenizer, dataset='wikitext2', original_len=4, generated_len=32,
             batch_size=64, device="cuda", warmup_batches=0) -> Dict[str, Any]:
    """
    Enhanced evaluation function with better error handling and metrics.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer to use
        dataset: Dataset name
        original_len: Input sequence length
        generated_len: Number of tokens to generate
        batch_size: Batch size for evaluation
        device: Device to run on
        warmup_batches: Number of warmup batches to skip in timing
    
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    logger = logging.getLogger(__name__)
    
    # Initialize metrics
    throughput_times = []
    token_counts = []
    memory_peaks = []
    num_batches_to_fetch = 10 + warmup_batches
    successful_batches = 0
    failed_batches = 0
    consecutive_oom_failures = 0

    try:
        test_loader = get_test_data(dataset, tokenizer, seq_len=original_len, batch_size=batch_size)
        initial_memory = torch.cuda.memory_allocated()
        
        logger.info(f"Starting evaluation with {num_batches_to_fetch} batches "
                   f"(including {warmup_batches} warmup batches)")
        
        for batch_idx, batch_data in enumerate(itertools.islice(test_loader, num_batches_to_fetch)):
            batch = batch_data.long().to(device)
            expected_tokens = batch.shape[0] * generated_len
            attention_mask = torch.ones_like(batch, dtype=torch.long, device=device)

            # Clear cache and reset memory stats
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            start_time = time.time()
            batch_success = False

            try:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    generation_output = model.generate(
                        input_ids=batch,
                        attention_mask=attention_mask,
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=False,  # Use greedy decoding for consistent timing
                        use_cache=True,
                        max_length=original_len + generated_len,
                    )
                
                torch.cuda.synchronize()
                end_time = time.time()
                batch_time = end_time - start_time
                batch_peak = torch.cuda.max_memory_allocated()
                
                # Validate generation output
                if torch.isfinite(generation_output.float()).all():
                    batch_success = True
                    successful_batches += 1
                    consecutive_oom_failures = 0  # Reset OOM counter
                    
                    # Only count non-warmup batches for metrics
                    if batch_idx >= warmup_batches:
                        throughput_times.append(batch_time)
                        token_counts.append(expected_tokens)
                        memory_peaks.append(batch_peak)
                    
                    logger.info(f"Batch {batch_idx+1}/{num_batches_to_fetch}: "
                              f"Time {batch_time:.3f}s, Peak {batch_peak / (1024**3):.2f}GB, "
                              f"Tokens/sec {expected_tokens/batch_time:.1f}")
                else:
                    logger.warning(f"Batch {batch_idx+1}: Invalid generation output detected")
                    
            except RuntimeError as e:
                failed_batches += 1
                error_str = str(e).lower()
                
                if "out of memory" in error_str:
                    consecutive_oom_failures += 1
                    logger.error(f"Batch {batch_idx+1} failed with OOM (consecutive: {consecutive_oom_failures})")
                    torch.cuda.empty_cache()
                    
                    # If we have too many consecutive OOM failures, stop trying
                    if consecutive_oom_failures >= 5:
                        logger.error("Too many consecutive OOM failures, stopping evaluation")
                        break
                else:
                    logger.error(f"Batch {batch_idx+1} failed: {e}")

            if not batch_success:
                failed_batches += 1

        # Calculate final metrics
        if len(throughput_times) == 0:
            logger.error("No successful batches for timing calculation")
            return {
                "throughput_tokens_per_sec": 0.0,
                "total_peak_memory_gb": 0.0,
                "activation_memory_gb": 0.0,
                "successful_batches": successful_batches,
                "failed_batches": failed_batches,
                "avg_batch_time_sec": 0.0,
                "total_tokens_generated": 0,
                "error": "No successful batches",
                "consecutive_oom_failures": consecutive_oom_failures
            }

        total_time = sum(throughput_times)
        total_tokens = sum(token_counts)
        peak_memory = max(memory_peaks)
        
        throughput = total_tokens / total_time if total_time > 0 else 0.0
        total_memory_gb = peak_memory / (1024 ** 3)
        initial_memory_gb = initial_memory / (1024 ** 3)
        activation_memory_gb = max(0.0, total_memory_gb - initial_memory_gb)
        avg_batch_time = total_time / len(throughput_times)
        
        logger.info(f"Final Results:")
        logger.info(f"  Throughput: {throughput:.2f} tokens/sec")
        logger.info(f"  Total peak GPU memory: {total_memory_gb:.2f} GB")
        logger.info(f"  Activation memory: {activation_memory_gb:.2f} GB")
        logger.info(f"  Successful/Failed batches: {successful_batches}/{failed_batches}")
        
        return {
            "throughput_tokens_per_sec": throughput,
            "total_peak_memory_gb": total_memory_gb,
            "activation_memory_gb": activation_memory_gb,
            "successful_batches": successful_batches,
            "failed_batches": failed_batches,
            "avg_batch_time_sec": avg_batch_time,
            "total_tokens_generated": total_tokens,
            "evaluation_batches": len(throughput_times),
            "consecutive_oom_failures": consecutive_oom_failures
        }
        
    except Exception as e:
        logger.error(f"Critical error in evaluation: {e}")
        logger.error(traceback.format_exc())
        return {
            "throughput_tokens_per_sec": 0.0,
            "total_peak_memory_gb": 0.0,
            "activation_memory_gb": 0.0,
            "successful_batches": 0,
            "failed_batches": num_batches_to_fetch,
            "avg_batch_time_sec": 0.0,
            "total_tokens_generated": 0,
            "error": str(e),
            "consecutive_oom_failures": consecutive_oom_failures
        }


def load_model_safely(args) -> Tuple[Any, AutoTokenizer]:
    """
    Safely load model and tokenizer with error handling.
    
    Returns:
        Tuple of (model, tokenizer)
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load tokenizer first
        logger.info(f"Loading tokenizer from {args.model_base}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_base)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info("Set pad_token_id to eos_token_id")

        # Load low rank dictionary if provided
        low_rank_dict = None
        if args.low_rank_dict_path is not None:
            logger.info(f"Loading low rank dictionary from {args.low_rank_dict_path}")
            with open(args.low_rank_dict_path, 'r') as f:
                low_rank_dict = json.load(f)

        # Load model
        if args.ratio == 1.0 and args.quantization_bits is None:
            logger.info(f"Loading original model from {args.updated_model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                args.updated_model_path, 
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            logger.info(f"Loading SVD model with ratio {args.ratio}")
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_base, 
                torch_dtype=torch.float16
            )
            model = SVDModel.load_model(
                model=base_model,
                ratio=args.ratio,
                model_path=args.updated_model_path,
                quantization_bits=args.quantization_bits,
                low_rank_dict=low_rank_dict,
                SEQLEN=args.seq_len
            )

        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error(traceback.format_exc())
        raise


def save_results(args, results: Dict[str, Any], output_file: str) -> None:
    """
    Save evaluation results to file with comprehensive metadata.
    """
    logger = logging.getLogger(__name__)
    
    # Get system information
    gpu_info = get_gpu_memory_info()
    
    record = {
        # Metadata
        'timestamp': datetime.datetime.now().isoformat(),
        'experiment_id': f"{Path(args.model_base).name}_r{args.ratio}_b{args.batch_size}_g{args.generated_len}",
        
        # Model configuration
        'model_base': args.model_base,
        'updated_model_path': args.updated_model_path,
        'ratio': args.ratio,
        'quantization_bits': args.quantization_bits,
        'low_rank_dict_path': args.low_rank_dict_path,
        
        # Evaluation configuration
        'eval_dataset': args.eval_dataset,
        'batch_size': args.batch_size,
        'generated_len': args.generated_len,
        'seq_len': args.seq_len,
        'seed': args.seed,
        
        # System information
        'gpu_total_memory_gb': gpu_info.get('total_gb', 0.0),
        'pytorch_version': torch.__version__,
        
        # Results
        **results
    }
    
    try:
        with open(output_file, 'a') as f:
            f.write(json.dumps(record, indent=None) + '\n')
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        # Also try to save to a backup location
        backup_file = f"backup_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        try:
            with open(backup_file, 'w') as f:
                f.write(json.dumps(record, indent=2))
            logger.info(f"Results saved to backup file: {backup_file}")
        except:
            logger.error("Failed to save to backup file as well")


def main(args):
    """Main evaluation function with enhanced error handling."""
    # Setup directories and logging
    output_dir = ensure_directories(args.output_file) if args.output_file else Path("./results")
    logger = setup_logging(output_dir)
    
    logger.info("=== Starting Speed Evaluation ===")
    logger.info(f"Configuration: {vars(args)}")
    
    # Validate paths
    if not validate_model_path(args.updated_model_path, args.ratio):
        logger.error("Model validation failed, exiting")
        sys.exit(1)
    
    # Set seeds for reproducibility
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    logger.info(f"Set random seed to {SEED}")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.error("CUDA is not available")
        sys.exit(1)
    
    device = torch.device('cuda:0')
    logger.info(f"Using device: {device}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Initial GPU memory: {get_gpu_memory_info()}")
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_safely(args)
        model.to(device=device, dtype=torch.float16)
        
        logger.info(f"Model loaded successfully")
        logger.info(f"GPU memory after model loading: {get_gpu_memory_info()}")
        
        # Run evaluation
        logger.info("Starting throughput evaluation...")
        results = eff_eval(
            model=model,
            tokenizer=tokenizer,
            dataset=args.eval_dataset,
            original_len=args.seq_len,  # Fixed: was using wrong parameter name
            generated_len=args.generated_len,
            batch_size=args.batch_size,
            device=device
        )
        
        # Save results
        if args.output_file:
            save_results(args, results, args.output_file)
        
        logger.info("=== Evaluation Complete ===")
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.error(traceback.format_exc())
        
        # Save error information
        if args.output_file:
            error_results = {
                "throughput_tokens_per_sec": 0.0,
                "total_peak_memory_gb": 0.0,
                "activation_memory_gb": 0.0,
                "error": str(e),
                "error_type": type(e).__name__
            }
            save_results(args, error_results, args.output_file)
        
        sys.exit(1)
    
    finally:
        # Cleanup
        torch.cuda.empty_cache()
        logger.info(f"Final GPU memory: {get_gpu_memory_info()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate throughput and memory usage of SVD-compressed language models"
    )
    
    # Model settings
    parser.add_argument("--updated_model_path", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Path to the model to evaluate")
    parser.add_argument("--model_base", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Base model path for loading tokenizer and original model")
    parser.add_argument("--low_rank_dict_path", type=str, default=None,
                        help="Path to low rank dictionary JSON file")
    
    # Device settings
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")
    
    # Dataset and evaluation settings
    parser.add_argument("--eval_dataset", type=str, default="wikitext2", 
                        choices=["wikitext2", "c4", "ptb"],
                        help="Dataset to use for evaluation")
    parser.add_argument("--generated_len", type=int, default=32,
                        help="Number of tokens to generate per sequence")
    parser.add_argument("--seq_len", type=int, default=4,
                        help="Input sequence length (reduced default to avoid OOM)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for evaluation (reduced default for stability)")
    
    # Compression settings
    parser.add_argument("--ratio", type=float, default=0.6,
                        help="Compression ratio (1.0 = original model)")
    parser.add_argument("--quantization_bits", type=int, default=None,
                        help="Number of bits for quantization")

    # Output settings
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to JSONL file to save evaluation results")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.batch_size <= 0:
        print("Error: batch_size must be positive")
        sys.exit(1)
    
    if args.generated_len <= 0:
        print("Error: generated_len must be positive")
        sys.exit(1)
        
    if args.ratio <= 0 or args.ratio > 1.0:
        print("Error: ratio must be between 0 and 1.0")
        sys.exit(1)
    
    main(args)