import argparse
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm
import json    

from utils import load_eval_tokenized_dataset


def evaluate_perplexity(model, data_loader, batch_size, seq_len):
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
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval import evaluator
    
    model.eval()
    model.half()
    
    hflm = HFLM(pretrained=model, tokenizer=tokenizer)
    
    results = evaluator.simple_evaluate(
        model=hflm,
        tasks=[eval_dataset_name]
    )
    
    return results['results']


def main(args):
    # Random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_load_dtype = torch.float16
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    model_name = os.path.basename(args.model_base)
    model_output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Load uncompressed model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_base,
        torch_dtype=model_load_dtype,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    
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
        results = {
            "model_name": model_name,
            "dataset_name": DATASET_NAME,
            "perplexity": ppl
        }
        results_file = os.path.join(model_output_dir, f"{model_name}_{DATASET_NAME}_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f)
        print(f"Results saved in {results_file}")
        
    elif args.eval_metric == "accuracy":
        valid_datasets = {"arc_easy", "arc_challenge", "openbookqa", "winogrande", "hellaswag", "piqa", "mathqa"}
        if args.eval_dataset not in valid_datasets:
            raise ValueError(f"Invalid dataset {args.eval_dataset}. Choose from {valid_datasets}")
        
        accuracy = evaluate_commonsense(model, tokenizer, args.eval_dataset)
        print(f"Accuracy on {args.eval_dataset}: {accuracy}")
        
        results = {
            "model_name": model_name,
            "dataset_name": args.eval_dataset,
            "accuracy": accuracy
        }
        results_file = os.path.join(model_output_dir, f"{model_name}_{args.eval_dataset}_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f)
        print(f"Results saved in {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an uncompressed model")
    
    parser.add_argument("--model_base", type=str, required=True, help="HuggingFace model name or path")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--eval_metric", type=str, default="ppl", choices=["ppl", "accuracy"], help="Evaluation metric")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "ptb", "arc_easy", "arc_challenge", "openbookqa", "winogrande", "hellaswag", "piqa", "mathqa"],
        help="Evaluation dataset"
    )
    parser.add_argument("--output_dir", type=str, default="benchmark_results", help="Directory to save results")
    
    args = parser.parse_args()
    main(args)
