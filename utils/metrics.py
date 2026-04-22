"""
Metrics utilities for evaluating pruned networks.
"""

import torch
import numpy as np


def calculate_sparsity(model, threshold=1e-2):
    """
    Calculate overall sparsity across all prunable layers.
    
    Args:
        model: PyTorch model with PrunableLinear layers
        threshold: Gate values below this threshold are considered pruned
    
    Returns:
        sparsity_percent: Percentage of pruned connections
    """
    total_gates = 0
    pruned_gates = 0
    
    for name, param in model.named_parameters():
        if 'gate_scores' in name:
            gates = torch.sigmoid(param).detach().cpu().numpy()
            total_gates += gates.size
            pruned_gates += (gates < threshold).sum()
    
    if total_gates == 0:
        return 0.0
    
    return 100.0 * pruned_gates / total_gates


def get_layer_sparsity(model, threshold=1e-2):
    """
    Get sparsity for each prunable layer separately.
    
    Args:
        model: PyTorch model with PrunableLinear layers
        threshold: Gate threshold for pruning
    
    Returns:
        dict: Layer names mapped to sparsity percentages
    """
    layer_sparsity = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'gate_scores'):
            gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy()
            pruned = (gates < threshold).sum()
            total = gates.size
            sparsity = 100.0 * pruned / total if total > 0 else 0.0
            layer_sparsity[name] = sparsity
    
    return layer_sparsity


def count_parameters(model, count_gates=True):
    """
    Count total learnable parameters in model.
    
    Args:
        model: PyTorch model
        count_gates: Whether to include gate parameters
    
    Returns:
        int: Total parameter count
    """
    total = 0
    for name, param in model.named_parameters():
        if not count_gates and 'gate_scores' in name:
            continue
        total += param.numel()
    return total


def get_compression_ratio(model, threshold=1e-2):
    """
    Estimate compression ratio based on pruned weights.
    
    Args:
        model: PyTorch model with PrunableLinear layers
        threshold: Gate threshold for pruning
    
    Returns:
        float: Compression ratio (original_size / pruned_size)
    """
    sparsity = calculate_sparsity(model, threshold)
    # Sparsity of X% means we keep (100-X)% of weights
    active_percentage = 100 - sparsity
    
    if active_percentage == 0:
        return float('inf')
    
    return 100.0 / active_percentage
