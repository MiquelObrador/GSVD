import argparse
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm
import json    


from svdmodels import SVDModel
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
    from lm_eval import tasks
    from lm_eval import utils as lm_eval_utils
    from lm_eval.api.registry import ALL_TASKS
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
    # setting random seed of numpy and torch
    SEED = args.seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    
    DEV_GPU = torch.device('cuda:0')
    DEV_CPU= torch.device('cpu')
    model_load_dtype = torch.float16
    computeSVD_dtype = torch.float32
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    model_name = os.path.basename(args.model_path)
    if model_name.endswith('.pt'):
        model_name = model_name[:-3]  # Remove .pt extension

    # Create model-specific output directory
    model_output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Infer ratio from model_path
    if args.ratios_path is not None:
        # Ratios dict is a csv file with the name of the layer and the ratio
        # Load the ratios dict
        ratios_dict = {}
        with open(args.ratios_path, "r") as f:
            for line in f:
                line = line.strip().split(",")
                # This first line is the header
                if line[1] == "Ratio" or line[0] == "MODULE":
                    continue
                layer_name = line[0]
                ratio = float(line[1])
                ratios_dict[layer_name] = ratio
    else:
        ratios_dict = None
        
    # Infer ratio from model_path, "_r0.x_" is the ratio
    
    ratio = float(args.model_path.split("_r")[1].split("_")[0])
    
    model = SVDModel.load_model(
        model=AutoModelForCausalLM.from_pretrained(args.model_base, 
                                                  torch_dtype=model_load_dtype),
        ratio=ratio,
        model_path=args.model_path,
        quantization_bits=args.quantization_bits,
        ratios_dict=ratios_dict
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
            "dataset_name": DATASET_NAME,
            "perplexity": ppl
        }
        results_file = os.path.join(model_output_dir, f"{model_name}_{DATASET_NAME}_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f)
        print(f"Results saved in {results_file}")
        
    if args.eval_metric == "accuracy":
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
    parser = argparse.ArgumentParser(description="Evaluate a model")
    
    parser.add_argument("--model_base", type=str, 
                        required=True, help="Base model name")
    
    parser.add_argument("--model_path", type=str, 
                        required=True, help="Path to the compressed model")
    
    parser.add_argument("--quantization_bits", type=int,
                        default=None, help="Number of bits for quantization")
    
    parser.add_argument("--ratios_path", type=str,
                        default=None, help="Path to the ratios dictionary")
    
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
        help="finetuning dataset",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results",
        help="Directory to save the evaluation results",
    )
    
    args = parser.parse_args()
    main(args)