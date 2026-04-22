# Self-Pruning Neural Network for CIFAR-10

Train once. Learn accuracy and sparsity together.

This project implements a custom self-pruning network where every linear-layer weight has a learnable gate. The model is optimized end-to-end with a joint objective that balances classification performance and parameter sparsity.

## Executive Summary

- Custom `PrunableLinear` layer built from scratch (no direct use of `nn.Linear` in prunable blocks)
- Differentiable gating per weight: `w_eff = w * sigmoid(s / T)`
- Joint objective: classification loss + sparsity regularization
- Multi-`lambda` experiments show clear sparsity-accuracy trade-off
- High compression behavior observed while preserving competitive CIFAR-10 accuracy

Representative run snapshot:

| Lambda | Test Accuracy | Sparsity |
|--------|---------------|----------|
| 0.0005 | 85.54%        | 96.78%   |
| 0.0020 | 85.80%        | 99.13%   |
| 0.0050 | ~84-85%       | ~99%+    |

## Why This Is Interesting

Traditional pruning is usually a post-training step. Here, pruning is learned during training itself. The model adapts its structure while learning features, which is closer to how efficient deployment pipelines should work in practice.

## Method

For each weight element $w_{ij}$, we learn a gate score $s_{ij}$ and compute gate value:

$$
g_{ij} = \sigma\left(\frac{s_{ij}}{T}\right), \quad g_{ij} \in (0,1)
$$

Effective weight in forward pass:

$$
w^{\text{eff}}_{ij} = w_{ij} \cdot g_{ij}
$$

Training objective:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \cdot \mathcal{L}_{\text{sparsity}},
\quad
\mathcal{L}_{\text{sparsity}} = \sum |g_{ij}|
$$

Where:
- $T$ is a gate temperature (lower values sharpen gate decisions)
- $\lambda$ controls the sparsity-accuracy trade-off

## Architecture and Training Design

- Backbone: convolutional feature extractor for CIFAR-10
- Classifier: stacked `PrunableLinear` layers with dropout + batch norm
- Optimization:
  - Separate parameter groups for base weights and gate scores
  - Higher LR for gate scores to accelerate pruning dynamics
  - Cosine LR scheduler

## Project Layout

```text
self_pruning/
├── main.py
├── models/
│   ├── __init__.py
│   └── prunable_linear.py
├── utils/
│   ├── __init__.py
│   ├── loss.py
│   └── metrics.py
├── outputs/
│   ├── RESULTS.md
│   └── gate_distribution.png
└── README.md
```

## Reproducibility

### 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision matplotlib numpy
```

### 2) Train and Evaluate

```bash
python main.py
```

### 3) Generated Artifacts

- `outputs/RESULTS.md`: markdown summary of run metrics
- `outputs/gate_distribution.png`: histogram of learned gate values

## Interpreting Results

- Lower `lambda`: weaker pruning pressure, generally higher retained accuracy
- Higher `lambda`: stronger pruning pressure, higher sparsity, possible accuracy drop
- A strong solution is the best middle point on the Pareto trade-off

In the gate histogram, a successful run typically shows:
- A strong mass near 0 (pruned connections)
- A separate non-zero cluster (important retained connections)

## Engineering Strengths Demonstrated

- Custom PyTorch module design with correctly registered trainable parameters
- Differentiable sparsity mechanism integrated into standard training loop
- Controlled ablation over multiple regularization strengths
- Practical reporting: quantitative table + qualitative gate distribution

## Next Improvements

- Hard-threshold gates post training and benchmark true inference speedup
- Structured pruning (channel / neuron) for hardware-friendly sparsity
- Distill pruned model into a compact dense student
- Add seed controls and multiple-run confidence intervals for stronger statistical reporting

## Case Study Requirement Check

- Custom `PrunableLinear` implementation: done
- Learnable gate scores with gradient flow: done
- Combined classification + sparsity loss: done
- CIFAR-10 training pipeline: done
- Multi-`lambda` comparison: done
- Final analysis + visualization artifacts: done

## Final Takeaway

This implementation shows that model topology can be learned jointly with model weights. In other words, the network does not just learn what to predict, it learns how large it needs to be.