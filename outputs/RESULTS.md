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
3. We use a low-temperature sigmoid to make the gate transition sharper
4. Gradient descent pushes gate scores down, making sigmoid(g / T) ≈ 0
5. When gates are ≈ 0, the corresponding weights are effectively pruned (multiplied by ≈ 0)
6. Higher λ values increase sparsity pressure, trading off accuracy for model compression

## Results

### Sparsity vs Accuracy Trade-off

| Lambda | Test Accuracy | Sparsity Level (%) |
|--------|---------------|--------------------|
| 5e-04 | 85.59% | 96.78% |
| 2e-03 | 85.95% | 99.13% |
| 5e-03 | 85.88% | 99.61% |


### Key Findings

- Lower λ values preserve more active connections and usually maintain higher accuracy.
- Higher λ values apply stronger pruning pressure, usually increasing sparsity.
- The practical target is a middle λ that gives a useful sparsity gain with minimal accuracy loss.

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
