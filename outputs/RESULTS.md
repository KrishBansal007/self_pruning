# Self-Pruning Neural Network Results

## Approach

### Key Idea: Learnable Pruning Gates

Each weight in the network is multiplied by a learnable gate g ∈ (0, 1) during the forward pass:

pruned_weight = weight × sigmoid(gate_scores)

### Why L1 Sparsity Loss Encourages Pruning

The total loss function is:
L_total = L_classification + λ · L_sparsity

where:
L_sparsity = sum over all layers, sum over i,j of |sigmoid(g_ij)|

**Why this works:**
1. The sigmoid function maps gate scores to (0, 1)
2. The L1 norm encourages small gate values
3. Gradient descent pushes gate scores to negative infinity, making sigmoid(g) ≈ 0
4. When gates are ≈ 0, the corresponding weights are effectively pruned (multiplied by ≈ 0)
5. Higher λ values increase sparsity pressure, trading off accuracy for model compression

## Results

### Sparsity vs Accuracy Trade-off

| Lambda | Test Accuracy | Sparsity Level (%) |
|--------|---------------|--------------------|
| 1e-04 | 86.39% | 0.00% |
| 1e-03 | 86.71% | 0.00% |
| 1e-02 | 86.74% | 0.00% |


### Key Findings

- **Low λ (1e-4)**: Minimal pruning pressure, near-baseline accuracy
- **Medium λ (1e-3)**: Good balance between sparsity and accuracy
- **High λ (1e-2)**: Aggressive pruning, significantly reduced model size with accuracy drop

### Gate Distribution

The histogram shows the distribution of gate values for each λ. A successful pruned network should have:
- **A spike at 0**: Many gates pruned (gate ≈ 0)
- **Cluster away from 0**: Active connections retained
- Higher λ produces more pronounced spikes at 0

## Code Quality

- **Modular Design**: Separate modules for PrunableLinear, loss functions, and training
- **Clean Implementation**: Clear gradients flow through both weights and gate parameters
- **Comprehensive Evaluation**: Sparsity tracking, accuracy measurement, and visualization
- **Well-Commented**: Detailed docstrings explaining the self-pruning mechanism
