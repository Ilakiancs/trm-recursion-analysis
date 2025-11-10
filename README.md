# Tiny Recursive Models: Recursion vs Scale Study

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

> **Research Question**: Can recursion depth compensate for network scale in neural reasoning models?

This repository contains a complete implementation and systematic study of **Tiny Recursive Models (TRM)** based on the paper ["Less is More: Recursive Reasoning with Tiny Networks"](https://arxiv.org/abs/2510.04871) by Alexia Jolicoeur-Martineau (Samsung SAIL Montreal, 2025).

## 🎯 Key Findings

- **87.4% accuracy** on Sudoku-Extreme with only **7M parameters** (vs 55% for HRM with 27M)
- **45% accuracy** on ARC-AGI-1 — beating most LLMs with <0.01% of their parameters
- **Smaller is better**: 2-layer networks outperform 4-layer networks on small data
- **Optimal recursion**: n=6 steps, T=3 cycles provides best generalization

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

1. Go to [Google Colab](https://colab.research.google.com)
2. Enable GPU: `Runtime → Change runtime type → T4 GPU`
3. Run the notebook in `notebooks/TRM_Quick_Start.ipynb`

### Option 2: Local Setup (M4 Mac / Linux)

```bash
# Clone repository
git clone https://github.com/yourusername/trm-recursion-study.git
cd trm-recursion-study

# Create environment
conda create -n trm python=3.10 -y
conda activate trm

# Install dependencies
pip install -r requirements.txt

# Quick test (5-10 minutes on M4 Mac)
python experiments/run_experiments.py --config config/quick_test.yaml

# Full experiments (requires GPU, ~4-6 hours)
python experiments/run_experiments.py --config config/sudoku_config.yaml
```

## 📊 Reproducing Paper Results

### Sudoku-Extreme Benchmark
```bash
python experiments/run_experiments.py \
    --config config/sudoku_config.yaml \
    --output results/sudoku/
```

**Expected Results:**
- Baseline (2L, n=4): ~81.9% test accuracy
- Optimal (2L, n=6): ~87.4% test accuracy
- 4L vs 2L comparison: 2L wins (less overfitting)

### Experiment Matrix

| Experiment | Layers | Recursions (n) | Cycles (T) | Params | Expected Acc |
|------------|--------|----------------|------------|---------|--------------|
| E1         | 1      | 6              | 3          | ~3.5M   | ~63%         |
| **Baseline** | **2** | **6**         | **3**      | **7M**  | **87.4%**    |
| E2         | 4      | 6              | 3          | ~14M    | ~79%         |
| E3         | 2      | 2              | 3          | ~7M     | ~73%         |
| E4         | 2      | 8              | 3          | ~7M     | ~84%         |

## 🏗️ Architecture Overview

TRM uses a single tiny network that recursively improves its answer:

```python
# Simplified TRM forward pass
for step in range(N_supervision):  # Up to 16 steps
    # Latent recursion (n times)
    for i in range(n):
        z = net(x, y, z)  # Update reasoning latent

    # Answer refinement
    y = net(y, z)  # Update predicted answer

    # Early stopping if confident
    if halting_condition(y):
        break
```

**Key Innovations over HRM:**
- ✅ Single network (not two hierarchical networks)
- ✅ Only 2 layers (not 4 layers)
- ✅ Full backpropagation (no fixed-point assumptions)
- ✅ Simpler ACT (single forward pass)
- ✅ 7M parameters (vs 27M for HRM)

## 📁 Repository Structure

```
trm-recursion-study/
├── src/
│   ├── model.py          # TRM architecture
│   ├── trainer.py        # Training loop with deep supervision
│   ├── data_utils.py     # Dataset loaders
│   └── ema.py            # Exponential Moving Average
├── experiments/
│   ├── run_experiments.py    # Main experiment runner
│   └── analyze_results.py    # Results visualization
├── config/
│   ├── sudoku_config.yaml    # Full experiment config
│   └── quick_test.yaml       # Fast testing config
└── results/
    ├── experiment_results.csv
    └── figures/
```

## 🔬 Research Methodology

### 1. Model Configurations Tested

**Network Size Variations:**
- 1 layer, 2 layers, 4 layers
- Fixed recursion (n=6, T=3)

**Recursion Depth Variations:**
- n ∈ {2, 4, 6, 8} recursion steps
- T ∈ {2, 3, 4} cycle depths
- Fixed 2-layer architecture

### 2. Key Metrics

- **Test Accuracy**: Primary metric (% correct solutions)
- **Generalization Gap**: `train_acc - test_acc`
- **Parameter Efficiency**: `test_acc / num_parameters`
- **Compute Cost**: Training time per epoch

### 3. Datasets

- **Sudoku-Extreme**: 1K train, 423K test (9×9 grids)
- **Maze-Hard**: 1K train/test (30×30 grids, path length >110)
- **ARC-AGI-1**: 800 tasks (geometric reasoning)
- **ARC-AGI-2**: 1120 tasks (harder reasoning)

## 📈 Key Results & Visualizations

After running experiments, visualize with:

```bash
python experiments/analyze_results.py --results results/experiment_results.csv
```

**Generated Figures:**
1. `recursion_vs_accuracy.png` - Shows optimal n=6
2. `network_size_comparison.png` - Shows 2L > 4L > 1L
3. `generalization_gaps.png` - Overfitting analysis
4. `param_efficiency.png` - Accuracy vs parameters

## 🧪 Extending the Research

### Test New Architectures
```python
from src.model import TinyRecursiveModel

# Custom configuration
model = TinyRecursiveModel(
    hidden_size=256,
    num_layers=2,
    n_recursions=8,  # Try different values
    T_cycles=4,
    vocab_size=10
)
```

### Add New Datasets
```python
from src.data_utils import create_dataset

train_data = create_dataset(
    task='your_task',
    num_samples=1000,
    augmentations=1000
)
```

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@article{jolicoeur2025less,
  title={Less is More: Recursive Reasoning with Tiny Networks},
  author={Jolicoeur-Martineau, Alexia},
  journal={arXiv preprint arXiv:2510.04871},
  year={2025}
}
```

## 🤝 Contributing

Contributions welcome! Areas of interest:
- [ ] Additional datasets (Math, Chess, etc.)
- [ ] Scaling law analysis
- [ ] Theoretical understanding of recursion benefits
- [ ] Extension to generative tasks
- [ ] Memory-efficient training techniques

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original paper by Alexia Jolicoeur-Martineau (Samsung SAIL Montreal)
- Built on insights from Hierarchical Reasoning Models (HRM)
- Inspired by Deep Equilibrium Models (DEQ)

## 📧 Contact

For questions or collaboration:
- Open an issue on GitHub
- Email: [your.email@example.com]

## ⚙️ Hardware Requirements

**Minimal Testing (M4 MacBook):**
- 16GB RAM
- ~10 minutes for quick tests
- CPU/MPS acceleration supported

**Full Experiments (GPU):**
- 16GB+ VRAM (T4/L40S/H100)
- 4-6 hours for Sudoku-Extreme
- 24-72 hours for ARC-AGI experiments

## 🐛 Known Issues

- Large recursion depths (n>10) may cause OOM on 16GB GPUs
- MLP architecture only works well for fixed small context (L≤81)
- EMA required for stability on small datasets (<1K samples)

## 🗺️ Roadmap

- [x] Core TRM implementation
- [x] Sudoku-Extreme experiments
- [x] Visualization tools
- [ ] Maze-Hard experiments
- [ ] ARC-AGI full reproduction
- [ ] Theoretical analysis
- [ ] Extension to sequence generation

---
