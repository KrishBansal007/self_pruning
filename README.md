# Self-Pruning Neural Network (CIFAR-10)

This project implements a self-pruning neural network that learns which connections to remove during training.

Instead of pruning after training, each weight in custom linear layers has a learnable gate. During forward pass, the effective weight is:

$w_{ij}^{\text{effective}} = w_{ij} \cdot \sigma\left(\frac{s_{ij}}{T}\right)$

where:
- $w_{ij}$ is the original weight
- $s_{ij}$ is a learnable gate score
- $\sigma$ is sigmoid
- $T$ is gate temperature (lower $T$ makes gates sharper)

## Problem Objective

Train a CIFAR-10 classifier that:
- Maintains good test accuracy
- Pushes many gates toward zero
- Demonstrates the sparsity vs. accuracy trade-off across multiple $\lambda$ values

Total loss used during training:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{classification}} + \lambda \cdot \mathcal{L}_{\text{sparsity}}
$$

$$
\mathcal{L}_{\text{sparsity}} = \sum_{\text{all gates}} \left|\sigma\left(\frac{s}{T}\right)\right|
$$

## Project Structure

- `main.py`: End-to-end training, evaluation, comparison across different $\lambda$, and artifact generation
- `models/prunable_linear.py`: Custom `PrunableLinear` layer with learnable gates
- `utils/loss.py`: Sparsity regularization and total-loss utility
- `utils/metrics.py`: Sparsity and compression utility functions
- `outputs/`: Generated artifacts (`RESULTS.md`, gate histogram image)

## Implementation Highlights

1. Custom Prunable Layer
- `PrunableLinear(in_features, out_features)` creates:
  - `weight`
  - `bias`
  - `gate_scores` (same shape as `weight`)
- Forward pass computes gates with sigmoid and multiplies element-wise with weights.

2. Sparsity Regularization
- Uses L1 penalty on gate values to encourage many gates to collapse near 0.
- Higher $\lambda$ increases pruning pressure.

3. Training Strategy
- Separate optimizer groups:
  - Base network parameters
  - Gate parameters (higher learning rate)
- Evaluates accuracy each epoch and final sparsity per run.

4. Evaluation Outputs
- Final test accuracy for each $\lambda$
- Sparsity percentage based on threshold (default: `1e-2`)
- Gate value distribution plot in `outputs/gate_distribution.png`

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision matplotlib numpy
```

## Run Training

```bash
python main.py
```

The script will:
- Download CIFAR-10 via `torchvision.datasets`
- Train multiple models with different $\lambda$ values
- Print a comparison table
- Save:
  - `outputs/gate_distribution.png`
  - `outputs/RESULTS.md`

## How to Read Results

- Low $\lambda$: usually higher accuracy, lower sparsity
- High $\lambda$: usually higher sparsity, possible accuracy drop
- Good operating point: a middle $\lambda$ with strong sparsity and acceptable accuracy

## Notes for Interview Discussion

If you present this project in an interview, emphasize:
- Why post-training pruning was avoided (adaptive pruning during optimization)
- Why L1 on gates encourages sparsity
- Why temperature-scaled sigmoid helps produce sharper gate decisions
- How you tuned the sparsity-accuracy trade-off with $\lambda$
- Potential production extensions:
  - Hard-thresholding gates after training for deployment
  - Structured pruning (channel/neuron pruning)
  - Knowledge distillation on the pruned model

## Quick Checklist Against Case Study

- Custom `PrunableLinear` implemented from scratch
- Learnable `gate_scores` registered as trainable parameters
- Total loss includes classification + sparsity term
- Trained on CIFAR-10
- Compared at least 3 different $\lambda$ values
- Report and gate distribution plot generated
