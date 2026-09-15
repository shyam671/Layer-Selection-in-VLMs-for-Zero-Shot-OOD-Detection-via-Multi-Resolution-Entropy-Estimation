import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn.metrics as sk
import torch
import torch.nn.functional as F
from itertools import combinations
from joblib import Parallel, delayed
from tqdm import tqdm


def compute_mcm_score(output, temperature=1.0):
    """
    Compute the MCM score (Maximum Concept Matching) for layer selection.
    Formula: -max(softmax(logits))
    """
    if isinstance(output, torch.Tensor):
        output = output.detach()
        to_np = lambda x: x.cpu().numpy()
        smax = to_np(F.softmax(output / temperature, dim=-1))
    else:
        # Assume numpy array
        smax = np.exp(output / temperature) / np.sum(np.exp(output / temperature), axis=-1, keepdims=True)
        
    # Returns negative max confidence. 
    return -np.max(smax, axis=-1)


def entropy_distribution_multi(data):
    # Can we unflaten and bin every thing at once or make prompt wise
    """Compute the entropy of the score distribution (histogram based)."""
    min_entropy = 0.0
    num_bins = [4, 8, 16, 32, 64]
    for b in num_bins:
        hist, _ = np.histogram(data, bins=b, density=True)
        hist = hist[hist > 0]
        min_entropy = min_entropy + stats.entropy(hist)
    return min_entropy / len(num_bins)

def entropy_distribution(data):
    # Can we unflaten and bin every thing at once or make prompt wise
    """Compute the entropy of the score distribution (histogram based)."""
    min_entropy = 0.0
    num_bins = [16]
    for b in num_bins:
        hist, _ = np.histogram(data, bins=b, density=True)
        hist = hist[hist > 0]
        min_entropy = min_entropy + stats.entropy(hist)
    return min_entropy / len(num_bins)

def _evaluate_combination(args, id_scores_group, combination):
    """
    Internal helper to calculate entropy for a specific layer combination.
    Since OOD data is excluded from the pipeline, this only computes entropy.
    """
    stats_result = {}
    if args.text_prompt == 'single':
        stats_result['entropy'] = entropy_distribution(id_scores_group)
    else:
        stats_result['entropy'] = entropy_distribution_multi(id_scores_group)

    
    # Store metadata
    stats_result['Combination'] = combination
    stats_result['length'] = len(combination)
    
    return stats_result

def select_best_layer_combination(args, id_logits, temperature=1.0, 
                                  max_layer_idx=None, reference_layer=-1, 
                                  max_length=11, sample_size=100000, n_jobs=-1):
    """
    Selects the best layer combination using ONLY ID data.
    
    The selection strategy is based on minimizing the entropy of the 
    MCM score distribution for the ID data.

    Args:
        id_logits: Tensor or Array of ID data logits (N_samples, N_layers, N_classes).
        temperature: Temperature scaling for Softmax.
        max_layer_idx: Limit evaluation to first N layers (optional).
        reference_layer: Index of the layer that must be included (usually the last one, -1).
        max_length: Maximum size of the layer combination.
        sample_size: Number of combinations to sample if exhaustive search is too expensive.
        n_jobs: Number of parallel jobs for evaluation.

    Returns:
        best_combination (list): The list of layer indices representing the best combination.
        layer_optimization_summary (pd.DataFrame): Detailed report of all evaluated combinations, ranked by entropy.
    """
    # --- 1. Compute MCM Scores for ID data ---
    id_scores = compute_mcm_score(id_logits, temperature=temperature)

    num_layers = id_scores.shape[1]
    if max_layer_idx is not None:
        num_layers = min(num_layers, max_layer_idx)
    
    # Handle negative indexing for reference layer
    if reference_layer < 0:
        reference_layer = num_layers + reference_layer
        
    # --- 2. Generate Layer Combinations ---
    # We pool from all layers except the reference layer (which is always included)
    pool_layers = [i for i in range(num_layers) if i != reference_layer]

    all_groups = []

    # Base case: just the reference layer
    all_groups.append([reference_layer])

    # Combinations of size 2 up to max_length
    for r in range(1, max_length):
        for comb in combinations(pool_layers, r):
            all_groups.append(list(comb) + [reference_layer])

    # Sampling strategy if search space is too large
    if 0 < sample_size < len(all_groups):
        sampled_groups = []
        groups_by_len = {}
        for g in all_groups:
            l = len(g)
            if l not in groups_by_len: groups_by_len[l] = []
            groups_by_len[l].append(g)
            
        per_len = max(1, sample_size // len(groups_by_len))
        for l in groups_by_len:
            indices = np.random.choice(len(groups_by_len[l]), min(len(groups_by_len[l]), per_len), replace=False)
            for idx in indices:
                sampled_groups.append(groups_by_len[l][idx])
        all_groups = sampled_groups

    # --- 3. Evaluate Combinations (Parallelized) ---
    def process_group(group):
        # Average the MCM scores across the specific layers in this group
        id_mean = id_scores[:, group].mean(axis=1)
        return _evaluate_combination(args, id_mean, group)

    # Use 'loky' backend with prefer="threads" to avoid process spawning issues with CUDA
    # If n_jobs is -1 or > 1, use parallel processing, otherwise sequential
    if n_jobs == 1:
        # Sequential processing (no parallelization)
        results_list = [process_group(group) for group in tqdm(all_groups, desc="Optimizing Layer Selection")]
    else:
        # Parallel processing with threading backend to avoid CUDA conflicts
        results_list = Parallel(n_jobs=n_jobs, backend='threading')(delayed(process_group)(group) for group in tqdm(all_groups, desc="Optimizing Layer Selection"))
    
    # --- 4. Process and Rank Results ---
    layer_optimization_summary = pd.DataFrame(results_list)
    
    # Sort by Entropy Ascending (Lower entropy is better for ID data selection)
    # Use stable sort and add secondary sort by combination to ensure determinism
    if not layer_optimization_summary.empty:
        # Add a string representation for stable secondary sorting
        layer_optimization_summary['_comb_str'] = layer_optimization_summary['Combination'].apply(lambda x: str(sorted(x)))
        layer_optimization_summary = layer_optimization_summary.sort_values(
            by=['entropy', '_comb_str'], 
            ascending=[True, True],
            kind='stable'  # Ensure stable sort
        ).reset_index(drop=True)
        layer_optimization_summary = layer_optimization_summary.drop('_comb_str', axis=1)
        best_combination = layer_optimization_summary.iloc[0]['Combination']
    else:
        best_combination = [reference_layer]

    return best_combination, layer_optimization_summary