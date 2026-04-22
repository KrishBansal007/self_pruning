"""
PrunableLinear: A custom linear layer with learnable pruning gates.
Each weight is associated with a learnable gate parameter that determines
whether that weight is active or pruned during the forward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    """
    A linear layer that learns to prune its own weights during training.
    
    Each weight w_ij is scaled by a gate g_ij ∈ (0, 1) before use.
    The gates are learned via specialized regularization to encourage sparsity.
    """
    
    def __init__(self, in_features, out_features, bias=True, gate_temperature=0.2):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gate_temperature = gate_temperature
        
        # Standard weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Gate scores: will be transformed to gates via sigmoid in forward pass
        # Same shape as weight, initialized to 0 (corresponds to gate = 0.5)
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize weights and biases."""
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        # Initialize gate_scores to 0 for symmetric initialization (gate ≈ 0.5)
        nn.init.zeros_(self.gate_scores)
    
    def forward(self, input):
        """
        Forward pass with learnable weight pruning.
        
        Args:
            input: Tensor of shape (batch_size, in_features)
        
        Returns:
            output: Tensor of shape (batch_size, out_features)
        """
        # Lower temperature sharpens gate values and makes near-zero gates easier to reach.
        gates = torch.sigmoid(self.gate_scores / self.gate_temperature)
        
        # Compute pruned weights: element-wise multiplication
        pruned_weight = self.weight * gates
        
        # Standard linear layer operation with pruned weights
        output = F.linear(input, pruned_weight, self.bias)
        
        return output
    
    def get_gates(self):
        """Return the current gate values (sigmoid applied to gate_scores)."""
        return torch.sigmoid(self.gate_scores / self.gate_temperature)
    
    def get_sparsity(self, threshold=1e-2):
        """
        Calculate sparsity level: percentage of gates below threshold.
        
        Args:
            threshold: Gate values below this are considered "pruned"
        
        Returns:
            sparsity_percent: Percentage of pruned connections (0-100)
        """
        gates = self.get_gates()
        num_pruned = (gates < threshold).sum().item()
        total = gates.numel()
        return 100 * num_pruned / total
    
    def extra_repr(self):
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'bias={self.bias is not None}, gate_temperature={self.gate_temperature}'
        )
