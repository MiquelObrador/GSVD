import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset, DataLoader
import math
import os
from collections import OrderedDict
import pandas as pd
import numpy as np

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
    
def load_wikitext(tokenizer, seq_len=2048, batch_size = 4, samples=None, seed=42):
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
    
    import random
    if seed is not None:
        random.seed(seed)

    test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    if samples is not None:
        test_data = test_data.select(list(random.sample(range(len(test_data)), samples)))
    test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def load_train_data(
    tokenizer,
    seq_len: int = 2048,
    batch_size: int = 4,
    samples: int = None,
    seed: int = 42,
    c4_path: str = "datasets/c4-train.00000-of-01024.json.gz",
):
    class IndexDataset(Dataset):
        def __init__(self, tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            return self.tensors[index]

        def __len__(self):
            return len(self.tensors)

    def process_data(dataset, tokenizer, seq_len, field_name="text"):
        # tokenize the concatenated text into one long tensor
        all_ids = tokenizer(
            "\n\n".join(dataset[field_name]), return_tensors="pt"
        ).input_ids[0]
        # split into non-overlapping seq_len chunks
        n_chunks = all_ids.numel() // seq_len
        chunks = [
            all_ids[i * seq_len : (i + 1) * seq_len]
            for i in range(n_chunks)
        ]
        return IndexDataset(tensors=torch.stack(chunks))
    
    import random

    # set RNG
    if seed is not None:
        random.seed(seed)

    # 1) Load raw datasets
    ds_wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ds_c4  = load_dataset("json", data_files=c4_path)["train"]
    ds_ptb = load_dataset("ptb_text_only", "penn_treebank", split="train")

    # 2) If sampling, pick roughly equal parts from each
    if samples is not None:
        # base count per dataset, distribute remainder
        base = samples // 3
        extras = samples % 3
        counts = [base + (1 if i < extras else 0) for i in range(3)]
        ds_wiki = ds_wiki.select(random.sample(range(len(ds_wiki)), counts[0]))
        ds_c4   = ds_c4.select(  random.sample(range(len(ds_c4)),   counts[1]))
        ds_ptb  = ds_ptb.select( random.sample(range(len(ds_ptb)),  counts[2]))

    # 3) Normalize field names to "text"
    #    WikiText already has 'text', C4 typically has 'text', PTB has 'sentence'
    ds_ptb = ds_ptb.rename_column("sentence", "text")
    # (if C4 field is not named "text", rename similarly here)

    # 4) Concatenate and optionally shuffle
    ds_all = concatenate_datasets([ds_wiki, ds_c4, ds_ptb])
    # you can shuffle here if you want mixed order:
    # ds_all = ds_all.shuffle(seed=seed)  

    # 5) Tokenize & chunk
    dataset = process_data(ds_all, tokenizer, seq_len, field_name="text")

    # 6) Return a DataLoader
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
        ds = load_dataset('ptb_text_only', "penn_treebank", split="validation")
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

def calculate_loss_slope(df: pd.DataFrame, module_name: str, target_ratio: float) -> float:
    """
    Calculates the slope of the interpolated final loss for a given module and ratio.
    """
    
    df_module = df[df["MODULE"] == module_name].copy()

    if df_module.empty:
        raise ValueError(f"Module '{module_name}' not found in the DataFrame.")

    ref_loss_series = df_module[np.isclose(df_module["ratio"], 0.8)]["FINAL LOSS"]
    if ref_loss_series.empty:
        raise ValueError(f"Reference ratio 0.8 not found for module '{module_name}' for normalization.")
    ref_loss_value = ref_loss_series.values[0]
    if np.isclose(ref_loss_value, 0):
        raise ValueError(f"Reference loss at ratio 0.8 is zero for module '{module_name}', cannot normalize.")

    df_module.loc[:, "FINAL LOSS"] = df_module["FINAL LOSS"] / ref_loss_value
    ratios = df_module["ratio"].values
    final_loss_normalized = df_module["FINAL LOSS"].values

    sort_indices = np.argsort(ratios)
    ratios_sorted = ratios[sort_indices]
    final_loss_sorted = final_loss_normalized[sort_indices]

    num_points = int(round((0.8 - 0.2) / 0.1)) + 1
    ratios_interp = np.linspace(0.2, 0.8, num_points)
    ratios_interp = np.round(ratios_interp, decimals=10)

    if not (ratios_sorted[0] <= ratios_interp[0] and ratios_sorted[-1] >= ratios_interp[-1]):
        print(f"Warning for module '{module_name}': Interpolation range "
              f"[{ratios_interp[0]:.2f}, {ratios_interp[-1]:.2f}] may extend beyond "
              f"actual data range [{ratios_sorted[0]:.2f}, {ratios_sorted[-1]:.2f}]. "
              "np.interp will extrapolate.")

    final_loss_interp = np.interp(ratios_interp, ratios_sorted, final_loss_sorted)

    def _get_slope_from_interpolated(ratio_point: float,
                                     interp_ratios_arr: np.ndarray,
                                     interp_final_loss_arr: np.ndarray) -> float:
        """
        Internal helper to get the slope from interpolated data.
        Uses forward/backward difference at endpoints, central elsewhere.
        """
        idxs = np.where(np.isclose(interp_ratios_arr, ratio_point, atol=1e-8))[0]
        if len(idxs) == 0:
            raise ValueError(
                f"Target ratio {ratio_point:.2f} not found in the set of interpolated ratios: "
                f"{np.round(interp_ratios_arr, 2)}. Please choose a value from this set."
            )
        idx = idxs[0]
        n = len(interp_ratios_arr)
        # Forward difference for first point
        if idx == 0:
            delta_loss = interp_final_loss_arr[1] - interp_final_loss_arr[0]
            delta_ratio = interp_ratios_arr[1] - interp_ratios_arr[0]
        # Backward difference for last point
        elif idx == n - 1:
            delta_loss = interp_final_loss_arr[n - 1] - interp_final_loss_arr[n - 2]
            delta_ratio = interp_ratios_arr[n - 1] - interp_ratios_arr[n - 2]
        # Central difference for interior points
        else:
            delta_loss = interp_final_loss_arr[idx + 1] - interp_final_loss_arr[idx - 1]
            delta_ratio = interp_ratios_arr[idx + 1] - interp_ratios_arr[idx - 1]

        if np.isclose(delta_ratio, 0):
            raise ValueError("Ratio difference between neighbors is zero, cannot calculate slope.")

        calculated_slope = delta_loss / delta_ratio
        return -calculated_slope

    slope = _get_slope_from_interpolated(target_ratio, ratios_interp, final_loss_interp)
    return slope


def allocate_compression_slope(
    g: OrderedDict[str, float],
    p: OrderedDict[str, int],
    s: OrderedDict[str, float],
    keep_frac: float = 0.6,
    delta: float = 0.1,
    alpha: float = 1.0,
) -> OrderedDict[str, float]:
    """
    Distribute a global keep‐budget among layers, using a tunable fusion
    between a multiplicative importance signal and uniform allocation.

    Args:
        g: per‐layer mean gradient magnitudes
        p: per‐layer parameter counts
        s: per‐layer slope of loss‐vs‐ratio curves
        keep_frac: overall fraction of params to keep
        delta: clamp window around keep_frac
        alpha: blending exponent in [0,1];
               0 => pure uniform (all layers at keep_frac),
               1 => pure multiplicative fusion

    Returns:
        keep_ratios: layer_name→final keep‐ratio in [keep_frac−delta, keep_frac+delta]
    """
    # total budget and clamp bounds
    P_total = sum(p.values())
    P_keep = keep_frac * P_total
    lower, upper = keep_frac - delta, keep_frac + delta

    # normalize signals
    max_g = max(g.values()) or 1.0
    max_s = max(s.values()) or 1.0
    g_norm = {k: g[k] / max_g for k in g}
    s_norm = {k: s[k] / max_s for k in s}

    # build tunable importance:
    #   imp_signal = (g_norm * s_norm)^alpha
    # when alpha=0, imp_signal==1 => uniform; when alpha=1, full product
    imp = {k: (g_norm[k] * s_norm[k])**alpha * p[k] for k in g.keys()}

    # iterative clamping and allocation
    S = set(imp)
    keep_counts = OrderedDict((k, 0.0) for k in g)

    while True:
        W = sum(imp[k] for k in S)
        # tentative allocation
        k_tent = {k: imp[k] / W * P_keep for k in S}

        # find any that violate bounds
        to_low  = [k for k, v in k_tent.items() if v < lower * p[k]]
        to_high = [k for k, v in k_tent.items() if v > upper * p[k]]

        if not to_low and not to_high:
            # accept all
            for k, v in k_tent.items():
                keep_counts[k] = v
            break

        # clamp low
        for k in to_low:
            clamp_val = lower * p[k]
            keep_counts[k] = clamp_val
            P_keep -= clamp_val
            S.remove(k)
        # clamp high
        for k in to_high:
            clamp_val = upper * p[k]
            keep_counts[k] = clamp_val
            P_keep -= clamp_val
            S.remove(k)

    # compute final ratios
    return OrderedDict((k, keep_counts[k] / p[k]) for k in keep_counts)