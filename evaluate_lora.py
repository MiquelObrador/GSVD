import argparse
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np
from tqdm import tqdm
import json
import re

from svdmodels import SVDModel
from utils import load_eval_tokenized_dataset


def evaluate_perplexity(model, data_loader, batch_size, seq_len):
    """Evaluate perplexity on a given dataset"""
    model.eval()
    model.half()
    
    DEVICE = model.device
    
    nlls = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating perplexity", total=len(data_loader)):
            batch = batch.to(DEVICE)
            logits = model(input_ids=batch, use_cache=False).logits
            if torch.isfinite(logits).all():
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                nlls.append(loss.cpu())
            else:
                print("Bad logits detected, skipping batch.")
                continue
        mean_loss = torch.cat(nlls).mean()
        ppl = torch.exp(mean_loss).item()
        if ppl > 1000:
            ppl = int(ppl)
            
        return ppl


def evaluate_commonsense(model, tokenizer, eval_dataset_name):
    """Evaluate accuracy on commonsense reasoning tasks"""
    import lm_eval
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    
    model.eval()
    model.half()
    
    hflm = HFLM(pretrained=model, tokenizer=tokenizer)
    
    results = evaluator.simple_evaluate(
        model=hflm,
        tasks=[eval_dataset_name]
    )
    
    return results['results']


def load_lora_model(base_model_name, svd_model_path, lora_adapter_path, 
                   quantization_bits=None, low_rank_dict=None, seq_len=2048):
    """Load SVD model with LoRA adapters"""
    
    # Infer ratio from SVD model path
    match = re.search(r"_r([0-9]+(?:\.[0-9]+)?)", svd_model_path)
    if not match:
        raise ValueError("Could not infer ratio from SVD model path. Ensure it contains '_r<ratio>' (e.g., '_r0.8').")
    ratio = float(match.group(1))
    
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        torch_dtype=torch.float16
    )
    
    print(f"Applying SVD compression with ratio {ratio}")
    svd_model = SVDModel.load_model(
        model=base_model,
        ratio=ratio,
        model_path=svd_model_path,
        quantization_bits=quantization_bits,
        low_rank_dict=low_rank_dict,
        SEQLEN=seq_len
    )
    
    print(f"Loading LoRA adapters from: {lora_adapter_path}")
    # Load the PEFT model (LoRA adapters) on top of the SVD model
    model = PeftModel.from_pretrained(svd_model, lora_adapter_path)
    
    # Merge adapters into the base model for inference
    model = model.merge_and_unload()
    
    return model


def main(args):
    # Set random seeds
    SEED = args.seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    
    DEV_GPU = torch.device('cuda:0')
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Create model name for output files
    adapter_name = os.path.basename(args.lora_adapter_path.rstrip('/'))
    svd_name = os.path.basename(args.svd_model_path)
    if svd_name.endswith('.pt'):
        svd_name = svd_name[:-3]
    
    model_name = f"{svd_name}_lora_{adapter_name}"
    
    # Create model-specific output directory
    model_output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Load low rank dictionary if provided
    low_rank_dict = None
    if args.low_rank_dict_path is not None:
        with open(args.low_rank_dict_path, 'r') as f:
            low_rank_dict = json.load(f)
        print(f"Loaded low rank dictionary from {args.low_rank_dict_path}")
    
    # Load the fine-tuned model
    model = load_lora_model(
        base_model_name=args.model_base,
        svd_model_path=args.svd_model_path,
        lora_adapter_path=args.lora_adapter_path,
        quantization_bits=args.quantization_bits,
        low_rank_dict=low_rank_dict,
        seq_len=args.seq_len
    )
    
    model.half()  # Convert model to half precision
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    
    model.to(DEV_GPU)
    
    if args.eval_metric == "ppl":
        valid_datasets = {"wikitext2", "c4", "ptb"}
        if args.eval_dataset not in valid_datasets:
            raise ValueError(f"Invalid dataset {args.eval_dataset}. Choose from {valid_datasets}")
        
        DATASET_NAME = args.eval_dataset
        tokenized_valdata = load_eval_tokenized_dataset(
            tokenizer=tokenizer,
            dataset_name=DATASET_NAME,
            seq_len=args.seq_len,
            batch_size=args.batch_size
        )
        
        ppl = evaluate_perplexity(
            model=model,
            data_loader=tokenized_valdata,
            batch_size=args.batch_size,
            seq_len=args.seq_len
        )
        
        print(f"Perplexity on {DATASET_NAME}: {ppl}")
        
        # Save the results
        results = {
            "model_name": model_name,
            "base_model": args.model_base,
            "svd_model_path": args.svd_model_path,
            "lora_adapter_path": args.lora_adapter_path,
            "dataset_name": DATASET_NAME,
            "perplexity": ppl,
            "eval_metric": "ppl"
        }
        results_file = os.path.join(model_output_dir, f"{model_name}_{DATASET_NAME}_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved in {results_file}")
        
    elif args.eval_metric == "accuracy":
        valid_datasets = {"arc_easy", "arc_challenge", "openbookqa", "winogrande", "hellaswag", "piqa", "mathqa"}
        if args.eval_dataset not in valid_datasets:
            raise ValueError(f"Invalid dataset {args.eval_dataset}. Choose from {valid_datasets}")
        
        accuracy = evaluate_commonsense(model, tokenizer, args.eval_dataset)
        print(f"Accuracy on {args.eval_dataset}: {accuracy}")
        
        results = {
            "model_name": model_name,
            "base_model": args.model_base,
            "svd_model_path": args.svd_model_path,
            "lora_adapter_path": args.lora_adapter_path,
            "dataset_name": args.eval_dataset,
            "accuracy": accuracy,
            "eval_metric": "accuracy"
        }

        results_file = os.path.join(model_output_dir, f"{model_name}_{args.eval_dataset}_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved in {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a LoRA fine-tuned SVD model")
    
    parser.add_argument("--model_base", type=str, 
                        required=True, help="Base model name (e.g., huggyllama/llama-7b)")
    
    parser.add_argument("--svd_model_path", type=str, 
                        required=True, help="Path to the SVD-compressed model .pt file")
    
    parser.add_argument("--lora_adapter_path", type=str, 
                        required=True, help="Path to the directory containing LoRA adapters")
    
    parser.add_argument("--quantization_bits", type=int,
                        default=None, help="Number of bits for quantization")
    
    parser.add_argument("--low_rank_dict_path", type=str,
                        default=None, help="Path to low rank dictionary JSON file")
    
    parser.add_argument("--seq_len", type=int, 
                        default=2048, help="Sequence length")
    
    parser.add_argument("--batch_size", type=int, 
                        default=4, help="Batch size")
    
    parser.add_argument("--eval_metric", type=str, 
                        default="ppl", choices=["ppl", "accuracy"], 
                        help="Evaluation metric")
    
    parser.add_argument("--seed", type=int, 
                        default=42, help="Random seed")
    
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "ptb", "arc_easy", "arc_challenge", "openbookqa", "winogrande", "hellaswag", "piqa", "mathqa"],
        help="Evaluation dataset",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results",
        help="Directory to save the evaluation results",
    )
    
    args = parser.parse_args()
    main(args)