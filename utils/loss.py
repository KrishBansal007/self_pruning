"""
Loss functions for self-pruning neural networks.
"""

import torch
import torch.nn as nn


class SparsityLoss(nn.Module):
    """
    L1 regularization loss on gate values to encourage sparsity.
    
    This loss penalizes the L1 norm of all gate values across all prunable layers.
    When gates are driven to 0, the corresponding weights are effectively pruned.
    """
    
    def __init__(self, lambda_param=1e-3, gate_temperature=0.2):
        """
        Args:
            lambda_param: Strength of the sparsity penalty (λ in the paper)
        """
        super(SparsityLoss, self).__init__()
        self.lambda_param = lambda_param
        self.gate_temperature = gate_temperature
    
    def forward(self, model):
        """
        Compute L1 norm of all gate values in the model.
        
        Args:
            model: PyTorch model containing PrunableLinear layers
        
        Returns:
            sparsity_loss: Weighted L1 norm of gate values
        """
        sparsity_loss = 0
        
        # Iterate through all parameters and find gate_scores
        for name, param in model.named_parameters():
            if 'gate_scores' in name:
                # Apply sigmoid to get gates in (0, 1)
                gates = torch.sigmoid(param / self.gate_temperature)
                # Add L1 norm (sum of absolute values)
                sparsity_loss += gates.abs().sum()
        
        # Scale by lambda parameter
        return self.lambda_param * sparsity_loss


def compute_total_loss(model, classification_loss, lambda_param=1e-3, gate_temperature=0.2):
    """
    Compute total loss = classification loss + lambda * sparsity loss.
    
    Args:
        model: PyTorch model (must contain PrunableLinear layers)
        classification_loss: The classification loss value
        lambda_param: Sparsity regularization strength
    
    Returns:
        total_loss: classification_loss + lambda * sparsity_loss
    """
    sparsity_loss = 0
    
    # Compute sparsity loss
    for name, param in model.named_parameters():
        if 'gate_scores' in name:
            gates = torch.sigmoid(param / gate_temperature)
            sparsity_loss += gates.abs().sum()
    
    sparsity_loss = lambda_param * sparsity_loss
    total_loss = classification_loss + sparsity_loss
    
    return total_loss, sparsity_loss
