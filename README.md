# Tiny Recursive Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Implementation of Tiny Recursive Models (TRM) from ["Less is More: Recursive Reasoning with Tiny Networks"](https://arxiv.org/abs/2510.04871) (Jolicoeur-Martineau, 2025).

## Key Results

- 87.4% accuracy on Sudoku-Extreme with 7M parameters vs 55% (HRM, 27M parameters)
- 2-layer networks outperform 4-layer networks on small datasets
- Optimal configuration: n=6 recursion steps, T=3 cycles

## Installation

```bash
git clone https://github.com/yourusername/trm-recursion-study.git
cd trm-recursion-study
pip install -r requirements.txt
```

## Quick Start

```python
from src.model import TinyRecursiveModel
from src.trainer import TRMTrainer
import torch

model = TinyRecursiveModel(
    vocab_size=10,
    hidden_size=256,
    num_layers=2,
    n_recursions=6,
    T_cycles=3
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
trainer = TRMTrainer(model, optimizer, device='cuda')
```

## Running Experiments

```bash
# Quick test
python experiments/run_experiments.py --config config/quick_test.yaml

# Full experiments
python experiments/run_experiments.py --config config/sudoku_config.yaml
```

## Model Architecture

The model implements recursive reasoning through:
- Latent recursion: z ← net(x, y, z) repeated n times
- Answer refinement: y ← net(y, z)
- Deep supervision with adaptive computation time

## Citation

```bibtex
@article{jolicoeur2025less,
  title={Less is More: Recursive Reasoning with Tiny Networks},
  author={Jolicoeur-Martineau, Alexia},
  journal={arXiv preprint arXiv:2510.04871},
  year={2025}
}
```

See `paper/CITATION.md` for full attribution.

## License

MIT License
