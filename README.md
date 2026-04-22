# Self-Pruning Neural Network (CIFAR-10)

## 🚀 Overview

This project implements a **self-pruning neural network** that learns which connections to remove *during training* using learnable gates on each weight. The model balances **accuracy** and **sparsity** via a combined loss.

---

## 🧠 Core Idea

Each weight has a learnable gate. The effective weight used in forward pass is:

**w_effective = w * sigmoid(s / T)**

Where:
- `w` = original weight
- `s` = learnable gate score
- `sigmoid` squashes values to (0, 1)
- `T` = temperature (controls sharpness of gating)

This lets the network **learn which weights matter** during optimization.

---

## 🎯 Objective

Train a CIFAR-10 classifier that:
- Maintains strong accuracy
- Prunes unnecessary weights
- Demonstrates the **sparsity–accuracy trade-off** across different lambda values

---

## ⚙️ Loss Function

**L_total = L_classification + lambda * L_sparsity**

**L_sparsity = sum(sigmoid(s / T))**

- L1-style penalty on gates encourages many gates → 0
- Higher `lambda` ⇒ stronger pruning pressure

---

## 🏗️ Project Structure

```
self_pruning/
├── main.py
├── models/
│   └── prunable_linear.py
├── utils/
│   ├── loss.py
│   └── metrics.py
├── outputs/
│   ├── RESULTS.md
│   └── gate_distribution.png
└── README.md
```

---

## 🔍 Implementation Highlights

### 1) Custom Prunable Layer
- Built from scratch (no `nn.Linear`)
- Parameters: `weight`, `bias`, `gate_scores`
- Forward pass:
  - `gates = sigmoid(gate_scores)`
  - `pruned_weights = weight * gates`

### 2) Sparsity Mechanism
- L1 penalty on gate activations
- Drives many gates near zero → effective pruning

### 3) Training Strategy
- Joint optimization of weights and gate scores
- (Optional) higher LR for gate parameters
- Evaluate across multiple `lambda` values

### 4) Evaluation Metrics
- **Test Accuracy**
- **Sparsity (%)** using a practical threshold (e.g., gate < 0.05)
- **Gate Distribution** (histogram)

---

## 📊 Results

| Lambda | Accuracy | Sparsity |
|--------|---------|----------|
| 0.0005 | ~85.6%  | ~96.7%   |
| 0.002  | ~85.9%  | ~99.1%   |
| 0.005  | ~84–85% | ~99%+    |

---

## 📈 Key Observations

- The model **learns to prune itself during training**
- Increasing `lambda` increases sparsity
- Even with **>95% pruning**, accuracy remains high
- Indicates strong **redundancy in dense networks**

---

## 📉 Gate Distribution

See `outputs/gate_distribution.png`:
- Large spike near 0 → pruned weights
- Cluster away from 0 → important connections

---

## ⚡ Why This Works

- L1 penalty promotes sparsity
- Sigmoid gates make pruning differentiable
- Joint optimization avoids post-processing
- Temperature sharpens gate decisions

---

## 🛠️ Setup & Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision matplotlib numpy
```

```bash
python main.py
```

---

## 📦 Outputs

- `outputs/gate_distribution.png` (histogram)
- `outputs/RESULTS.md` (summary)

---

## 🧠 Engineering Insights

- Custom layer design + loss engineering
- Clear sparsity–accuracy trade-off
- End-to-end reproducible pipeline

---

## 🚀 Possible Extensions

- Hard pruning (threshold gates and zero weights)
- Structured pruning (neurons/channels)
- Distillation on pruned model

---

## ✅ Case Study Coverage

- Custom `PrunableLinear` ✔
- Learnable gates ✔
- Combined loss ✔
- CIFAR-10 training ✔
- Multi-`lambda` comparison ✔
- Quantitative + visual analysis ✔

---

## 🏁 Takeaway

**Neural networks can optimize both performance and structure simultaneously—removing redundant parameters while maintaining strong accuracy.**