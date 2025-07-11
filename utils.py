import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset
import math
import os
from collections import OrderedDict
import pandas as pd
import numpy as np
import random
import torch.nn as nn


def replace_module_by_name(root_module, module_full_name, new_module):
    """
    Replaces a module inside root_module given its full name (e.g. "layer1.conv").
    """
    name_parts = module_full_name.split('.')
    parent = root_module
    # Traverse to the parent of the target module.
    for part in name_parts[:-1]:
        parent = getattr(parent, part)
    # Replace the target module.
    setattr(parent, name_parts[-1], new_module)
    
class CustomDictDataset(Dataset):
    """
    A custom PyTorch Dataset that returns a dictionary with specified keys.

    Args:
        data_dict (dict): A dictionary where keys are strings (e.g., 'input_ids')
                          and values are lists or tensors of the data.
    """
    def __init__(self, data_dict):
        self.data = {key: torch.as_tensor(value) for key, value in data_dict.items()}
        self.keys = list(data_dict.keys())
        self._len = len(self.data[self.keys[0]])

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return {key: self.data[key][idx] for key in self.keys}

def get_calib_train_data(name, tokenizer, nsamples, seqlen=2048, seed=3, batch_size=1, dataset_cache_dir=None):
    """
    Returns a DataLoader for calibration training data where each item is a dictionary.

    Args:
        name (str): The name of the dataset to use ('c4', 'ptb', 'wikitext2').
        tokenizer: The tokenizer to use for encoding the text.
        nsamples (int): The number of samples to generate.
        seqlen (int, optional): The sequence length of each sample. Defaults to 2048.
        seed (int, optional): The random seed for reproducibility. Defaults to 3.
        batch_size (int, optional): The batch size for the DataLoader. Defaults to 1.
        dataset_cache_dir (str, optional): The directory to cache the dataset. Defaults to None.

    Returns:
        torch.utils.data.DataLoader: A DataLoader containing the calibration data as dictionaries.
    """
    random.seed(seed)
    
    cache_dir = "cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    cache_file = os.path.join(cache_dir, f"{name}_{nsamples}_{seqlen}_{seed}_dict.pt")

    if os.path.exists(cache_file):
        print(f"Loading cached dataset from {cache_file}")
        data_dict = torch.load(cache_file)
    else:
        print("Generating and caching dataset...")
        if name == "c4":
            traindata = load_dataset("json", data_files="utils/c4-train.json")['train']
            text_data = "\n\n".join(traindata["text"])
        elif name == "ptb":
            traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', cache_dir=dataset_cache_dir)
            text_data = "\n\n".join(traindata["sentence"])
        elif name == "wikitext2":
            traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir=dataset_cache_dir)
            text_data = "\n\n".join(traindata["text"])
        else:
            raise NotImplementedError

        all_input_ids = []
        all_attention_masks = []
        
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(text_data) - seqlen * 10 - 1)
                j = i + seqlen * 10
                encoded_text = tokenizer(text_data[i:j], return_tensors="pt")
                
                if encoded_text.input_ids.shape[1] >= seqlen:
                    inp = encoded_text.input_ids[:, :seqlen]
                    attention_mask = torch.ones_like(inp)
                    all_input_ids.append(inp)
                    all_attention_masks.append(attention_mask)
                    break
        
        data_dict = {
            'input_ids': torch.cat(all_input_ids, dim=0),
            'attention_mask': torch.cat(all_attention_masks, dim=0)
        }
        
        torch.save(data_dict, cache_file)

    dataset = CustomDictDataset(data_dict)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def load_eval_tokenized_dataset(tokenizer,
                           dataset_name: str = 'wikitext-2-raw-v1',
                           seq_len: int = 2048,
                           batch_size: int = 4,
                           samples: int = None,
                           seed: int = 42):
    """
    Load and tokenize a variety of NLP datasets with reproducible sampling.

    Args:
        tokenizer: a HuggingFace tokenizer instance
        dataset_name: config name or subset identifier (e.g. 'wikitext-2-raw-v1', 'penn_treebank')
        seq_len: sequence length for each example
        batch_size: batch size for DataLoader
        samples: optional limit on number of examples to sample
        seed: optional random seed for reproducible sampling

    Returns:
        DataLoader yielding batches of token IDs of shape [batch_size, seq_len]
    """
    class IndexDataset(Dataset):
        def __init__(self, tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            return self.tensors[index]

        def __len__(self):
            return len(self.tensors)

    def process_data(records, field_name: str):
        # concatenate texts, tokenize, and split into fixed subsequences
        raw_text = "\n\n".join([r[field_name] for r in records])
        token_ids = tokenizer(raw_text, return_tensors='pt').input_ids[0]

        # truncate to a multiple of seq_len
        total_len = (token_ids.size(0) // seq_len) * seq_len
        token_ids = token_ids[:total_len]

        # reshape into chunks
        chunks = token_ids.view(-1, seq_len)
        return IndexDataset(tensors=chunks)

    # Load the requested dataset split
    if "wikitext" in dataset_name.lower():
        ds = load_dataset('wikitext', "wikitext-2-raw-v1", split="test")
        field = 'text'
    elif "ptb" in dataset_name.lower():
        ds = load_dataset('ptb_text_only', "penn_treebank", split="test")
        field = 'sentence'
    elif "c4" in dataset_name.lower():
        # expects a local JSON file with a 'text' field
        ds = load_dataset('json', data_files="datasets/c4-validation.00000-of-00008.json.gz")["train"]
        field = 'text'
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_name}. Supported types are: wikitext, penn_treebank, c4.")
    
    import random

    # Optionally sample a subset of examples reproducibly
    if samples is not None and samples < len(ds):
        if seed is not None:
            random.seed(seed)
        indices = random.sample(range(len(ds)), samples)
        ds = ds.select(indices)

    # Process and tokenize
    token_dataset = process_data(ds, field_name=field)

    # Build DataLoader
    loader = DataLoader(token_dataset, batch_size=batch_size, shuffle=False)
    return loader

def get_truncate(in_features, out_features, ratio):
    return int(in_features * out_features * ratio / (in_features + out_features))

def get_unique_path(path: str) -> str:
    base, ext = os.path.splitext(path)
    counter = 1
    new_path = path
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base}_{counter}{ext}"
    return new_path

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

import re
from collections import defaultdict, OrderedDict

def calculate_truncation_ranks(model, importance_dict, compression_ratio, smoothing_alpha):
    """
    Calculates the number of singular values (k) to keep for each layer,
    applying the compression budget independently to each group of layers (e.g.,
    all 'q_proj' layers, all 'down_proj' layers, etc.).

    Args:
        model (nn.Module): The transformer model to be compressed.
        importance_dict (OrderedDict): A dictionary with layer names as keys and
                                       their importance scores as values.
        compression_ratio (float): The target compression ratio (e.g., 0.6 for 60%).
                                   This means the final size should be 40% of the original.
        smoothing_alpha (float): A factor to smooth the importance distribution.

    Returns:
        dict: A dictionary containing the calculated ranks for each layer (with clean keys).
        dict: A dictionary containing detailed stats for each layer.
    """
    print("Starting rank calculation with grouped compression...")

    # --- 1. Group Layers by Type ---
    grouped_layer_info = defaultdict(dict)
    # Regex to extract the layer type (e.g., 'self_attn.q_proj') from the full name
    layer_type_re = re.compile(r'.*\.layers\.\d+\.([a-zA-Z_]+\.[a-zA-Z_]+proj)')

    for name, param in model.named_parameters():
        module_name = name.rsplit('.', 1)[0]
        is_in_importance_dict = name in importance_dict or module_name in importance_dict
        
        try:
            is_linear_layer = isinstance(model.get_submodule(module_name), nn.Linear)
        except AttributeError:
            is_linear_layer = False

        if is_linear_layer and is_in_importance_dict and param.dim() == 2:
            match = layer_type_re.match(name)
            if not match:
                print(f"Warning: Could not determine group for layer {name}. Skipping.")
                continue
            group_key = match.group(1)

            importance_key = name if name in importance_dict else module_name
            importance_score = importance_dict.get(importance_key, 0.0)
            if isinstance(importance_score, torch.Tensor):
                importance_score = torch.mean(torch.abs(importance_score)).item()

            rows, cols = param.shape
            max_rank = (min(rows, cols) // 2) - 1
            if max_rank <= 0:
                continue

            grouped_layer_info[group_key][name] = {
                'shape': (rows, cols),
                'original_params': rows * cols,
                'cost_per_rank': rows + cols,
                'max_rank': max_rank,
                'importance': importance_score
            }

    if not grouped_layer_info:
        print("No compressible layers found or matched with importance_dict. Aborting.")
        return {}, {}

    # --- 2. Process Each Group Independently ---
    final_ranks = {}
    detailed_stats = OrderedDict()
    overall_original_params = 0
    overall_final_params = 0

    for group_name, layer_info in grouped_layer_info.items():
        print(f"\n--- Processing Group: {group_name} ({len(layer_info)} layers) ---")
        
        # --- 2a. Normalize and Smooth Importance (within group) ---
        total_importance_group = sum(info['importance'] for info in layer_info.values())
        if total_importance_group == 0:
            for name in layer_info:
                layer_info[name]['final_weight'] = 1.0 / len(layer_info)
        else:
            for name in layer_info:
                normalized_importance = layer_info[name]['importance'] / total_importance_group
                layer_info[name]['smoothed_importance'] = normalized_importance ** smoothing_alpha
            total_smoothed_importance = sum(info['smoothed_importance'] for info in layer_info.values())
            for name in layer_info:
                layer_info[name]['final_weight'] = layer_info[name]['smoothed_importance'] / total_smoothed_importance

        # --- 2b. Calculate Group Budget ---
        total_original_params_group = sum(info['original_params'] for info in layer_info.values())
        target_total_params_group = total_original_params_group * compression_ratio
        overall_original_params += total_original_params_group
        
        print(f"Group Original Params: {total_original_params_group:,}")
        print(f"Group Target Params:   {int(target_total_params_group):,}")

        # --- 2c. Iterative Rank Allocation (for this group) ---
        group_final_ranks = {}
        remaining_budget = target_total_params_group
        layers_to_process = list(layer_info.keys())
        
        is_stable = False
        while not is_stable and layers_to_process:
            is_stable = True
            weighted_cost_sum = sum(layer_info[name]['final_weight'] * layer_info[name]['cost_per_rank'] for name in layers_to_process)
            if weighted_cost_sum == 0: break
            
            alloc_const = remaining_budget / weighted_cost_sum
            next_layers_to_process = []
            
            for name in layers_to_process:
                info = layer_info[name]
                tentative_rank = alloc_const * info['final_weight']
                if tentative_rank >= info['max_rank']:
                    is_stable = False
                    group_final_ranks[name] = info['max_rank']
                    remaining_budget -= info['max_rank'] * info['cost_per_rank']
                else:
                    next_layers_to_process.append(name)
            layers_to_process = next_layers_to_process

        if layers_to_process:
            weighted_cost_sum = sum(layer_info[name]['final_weight'] * layer_info[name]['cost_per_rank'] for name in layers_to_process)
            if weighted_cost_sum > 0:
                alloc_const = remaining_budget / weighted_cost_sum
                for name in layers_to_process:
                    group_final_ranks[name] = int(max(1, np.floor(alloc_const * layer_info[name]['final_weight'])))
        
        final_ranks.update(group_final_ranks)

        # --- 2d. Update Stats for Reporting ---
        for name, info in layer_info.items():
            rank = group_final_ranks.get(name, 0)
            new_params = rank * info['cost_per_rank']
            overall_final_params += new_params
            individual_compression = 1.0 - (new_params / info['original_params']) if info['original_params'] > 0 else 0
            detailed_stats[name] = {
                "group": group_name,
                "shape": info['shape'],
                "importance": info['importance'],
                "original_params": info['original_params'],
                "final_rank_k": rank,
                "new_params": new_params,
                "compression": f"{individual_compression:.2%}"
            }

    # --- 3. Final Report Generation ---
    actual_compression_ratio = 1.0 - (overall_final_params / overall_original_params) if overall_original_params > 0 else 0
    print("\n--- Overall Compression Results ---")
    print(f"Target Compression Ratio:   {compression_ratio:.2%}")
    print(f"Achieved Compression Ratio: {actual_compression_ratio:.2%}")
    print(f"Original Parameters: {overall_original_params:,}")
    print(f"Final Parameters:    {int(overall_final_params):,}")
    print("-----------------------------------\n")

    # --- 4. Clean up keys for return ---
    final_ranks_clean = {k.replace('.weight', ''): v for k, v in final_ranks.items()}

    return final_ranks_clean, detailed_stats