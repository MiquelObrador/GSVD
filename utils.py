import random
import torch
from datasets import load_dataset
from torch.utils.data.dataset import Dataset
import math
import os
from collections import OrderedDict
from typing import Tuple
import pandas as pd
import numpy as np

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

def calculate_loss_slope(df: pd.DataFrame, module_name: str, target_ratio: float) -> float:
    """
    Calculates the slope of the interpolated final loss for a given module and ratio.
    """
    
    # Graph the data for the selected module being the x axis the ratio and the y axis the final loss
    # Use .copy() to avoid SettingWithCopyWarning on the original DataFrame slice
    df_module = df[df["MODULE"] == module_name].copy()

    if df_module.empty:
        raise ValueError(f"Module '{module_name}' not found in the DataFrame.")

    # Divide the final loss by the value on the final loss ratio of 0.8
    # Ensure the reference ratio exists and its loss is not zero
    ref_loss_series = df_module[np.isclose(df_module["ratio"], 0.8)]["FINAL LOSS"]
    if ref_loss_series.empty:
        raise ValueError(f"Reference ratio 0.8 not found for module '{module_name}' for normalization.")
    
    ref_loss_value = ref_loss_series.values[0]
    if np.isclose(ref_loss_value, 0):
        raise ValueError(f"Reference loss at ratio 0.8 is zero for module '{module_name}', cannot normalize.")

    # Apply normalization. Use .loc to ensure modification on the copy
    df_module.loc[:, "FINAL LOSS"] = df_module["FINAL LOSS"] / ref_loss_value

    # Get the values of the ratio and the final loss
    ratios = df_module["ratio"].values
    final_loss_normalized = df_module["FINAL LOSS"].values

    # np.interp expects x-coordinates (ratios) to be monotonically increasing.
    # Sort ratios and corresponding final_loss_normalized values.
    sort_indices = np.argsort(ratios)
    ratios_sorted = ratios[sort_indices]
    final_loss_sorted = final_loss_normalized[sort_indices]
    
    # Interpolate the values
    # The original code's np.arange(0.2, 0.9, 0.1) results in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # Using np.linspace is often more robust for float steps.
    num_points = int(round((0.8 - 0.2) / 0.1)) + 1
    ratios_interp = np.linspace(0.2, 0.8, num_points)
    # Round to avoid floating point inaccuracies if exact matches like 0.3 are critical
    ratios_interp = np.round(ratios_interp, decimals=10)


    # Check if interpolation range is within data range, otherwise extrapolation will occur
    if not (ratios_sorted[0] <= ratios_interp[0] and ratios_sorted[-1] >= ratios_interp[-1]):
        print(f"Warning for module '{module_name}': Interpolation range "
              f"[{ratios_interp[0]:.2f}, {ratios_interp[-1]:.2f}] may extend beyond "
              f"actual data range [{ratios_sorted[0]:.2f}, {ratios_sorted[-1]:.2f}]. "
              "np.interp will extrapolate.")

    final_loss_interp = np.interp(ratios_interp, ratios_sorted, final_loss_sorted)

    # Now given a ratio we want to work on calculate the slope of the line
    # between the next ratio and the previous ratio

    # This nested function encapsulates the original get_slope logic
    def _get_slope_from_interpolated(ratio_point: float, interp_ratios_arr: np.ndarray, interp_final_loss_arr: np.ndarray) -> float:
        """
        Internal helper to get the slope from interpolated data.
        Uses central difference method.
        """
        # Get the index of the ratio_point in the interpolated ratios
        # Use np.isclose for float comparison with a tolerance
        idxs = np.where(np.isclose(interp_ratios_arr, ratio_point, atol=1e-8))[0]
        
        if len(idxs) == 0:
            raise ValueError(
                f"Target ratio {ratio_point:.2f} not found in the set of interpolated ratios: "
                f"{np.round(interp_ratios_arr, 2)}. Please choose a value from this set."
            )
        idx = idxs[0]
        
        # For central difference, we need points on both sides.
        # So, the point cannot be the first or the last in the interpolated array.
        if idx == 0 or idx == len(interp_ratios_arr) - 1:
            raise ValueError(
                f"Cannot calculate central difference slope for target ratio {ratio_point:.2f} "
                f"as it is an endpoint of the interpolated ratios: {np.round(interp_ratios_arr, 2)}. "
                "Choose an interior point."
            )
            
        # Now get neighbors for central difference
        prev_i, next_i = idx - 1, idx + 1
        
        delta_loss = interp_final_loss_arr[next_i] - interp_final_loss_arr[prev_i]
        delta_ratio = interp_ratios_arr[next_i] - interp_ratios_arr[prev_i]

        if np.isclose(delta_ratio, 0): # Should not happen with distinct interp_ratios_arr
            raise ValueError("Ratio difference between neighbors is zero, cannot calculate slope.")
            
        calculated_slope = delta_loss / delta_ratio
        return -calculated_slope # Original code returns the negative slope

    # Get the slope for the target_ratio using the interpolated values
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