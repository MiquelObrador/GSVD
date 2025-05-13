import random
import torch
from datasets import load_dataset
from torch.utils.data.dataset import Dataset
import math
import os
from collections import OrderedDict
from typing import Tuple

random.seed(42)

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
    
    
def load_wikitext(tokenizer, seq_len=2048, batch_size = 4, samples=None):
    class IndexDataset(Dataset):
        def __init__(self, tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            return self.tensors[index]

        def __len__(self):
            return len(self.tensors)
    ####
    def process_data(samples, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)

    test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    if samples is not None:
        test_data = test_data.select(list(random.sample(range(len(test_data)), samples)))
    test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def get_ratios(final_ratio, matrix_iters):
    ratios = []
    for i in range(1, matrix_iters + 1):
        r = 1 - i * (1 - final_ratio) / matrix_iters
        ratios.append(r)
    return ratios

def get_truncate(in_features, out_features, ratio):
    if in_features == out_features:
        return math.ceil(ratio * in_features / 2)
    else:
        return math.ceil((ratio * in_features * out_features) / (in_features + out_features))
    
def get_unique_path(path: str) -> str:
    base, ext = os.path.splitext(path)
    counter = 1
    new_path = path
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base}_{counter}{ext}"
    return new_path

def allocate_compression(
    g: OrderedDict[str, float],
    p: OrderedDict[str, int],
    keep_frac: float = 0.6,
    delta: float = 0.1,
) -> OrderedDict[str, float]:
    """
    Distribute a global keep-budget among layers, but force each layer's
    keep-ratio r_i = keep_i / p_i to lie in [lower, upper].

    Args:
        g: per-layer mean gradient magnitudes (OrderedDict[layer_name -> float])
        p: per-layer parameter counts      (OrderedDict[layer_name -> int])
        keep_frac: overall fraction of params to keep (0 < keep_frac <= 1)
        delta: fraction of lower/upper bounds to use for clamping
        (0 < delta <= 1)

    Returns:
        keep_ratios: an OrderedDict[layer_name -> final keep-ratio in [lower, upper]]
    """
    # total and target budget
    P_total = sum(p.values())
    P_keep  = keep_frac * P_total
    
    # lower and upper bounds
    lower = keep_frac - delta
    upper = keep_frac + delta

    # active set and allocation placeholders
    S = set(g.keys())
    keep_counts = OrderedDict((k, 0.0) for k in g)

    # iterative clamp loop
    while True:
        # 1) importance weights (pure g*p here — you could soften if desired)
        w = {k: g[k] * p[k] for k in S}
        W = sum(w.values())

        # 2) tentative counts for active layers
        k_tent = {k: (w[k] / W) * P_keep for k in S}

        # 3) detect layers violating bounds
        to_clamp_low  = [k for k, v in k_tent.items() if v < lower * p[k]]
        to_clamp_high = [k for k, v in k_tent.items() if v > upper * p[k]]

        # if no clamping needed, accept all
        if not to_clamp_low and not to_clamp_high:
            for k, v in k_tent.items():
                keep_counts[k] = v
            break

        # otherwise clamp low-bound layers
        for k in to_clamp_low:
            clamp_val = lower * p[k]
            keep_counts[k] = clamp_val
            P_keep -= clamp_val        # subtract exactly what we're committing
            S.remove(k)

        # clamp high-bound layers
        for k in to_clamp_high:
            clamp_val = upper * p[k]
            keep_counts[k] = clamp_val
            P_keep -= clamp_val
            S.remove(k)

    # build final ratios
    keep_ratios = OrderedDict(
        (k, keep_counts[k] / p[k]) for k in g.keys()
    )
    return keep_ratios