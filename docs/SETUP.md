# Setup Guide for TRM Recursion Study

Complete setup instructions for running Tiny Recursive Model experiments.

## Prerequisites

- **Python**: 3.10 or higher
- **GPU**: Optional but recommended (16GB+ VRAM)
- **RAM**: 16GB minimum
- **Storage**: 10GB free space

## Quick Setup (5 minutes)

### Option 1: Automated Setup (Linux/Mac)

```bash
# Clone repository
git clone https://github.com/yourusername/trm-recursion-study.git
cd trm-recursion-study

# Run setup script
chmod +x setup.sh
./setup.sh

# Activate environment
source venv/bin/activate

# Run quick test
python experiments/run_experiments.py --config config/quick_test.yaml
```

### Option 2: Manual Setup

```bash
# Create environment
conda create -n trm python=3.10 -y
conda activate trm

# Install PyTorch (choose one)
# For CUDA (Linux/Windows):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For Mac (M1/M2/M3/M4):
pip install torch torchvision torchaudio

# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt

# Create directories
mkdir -p results/figures checkpoints logs data
```

## Verify Installation

```python
# Test script
python -c "
import torch
from src.model import TinyRecursiveModel

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

model = TinyRecursiveModel()
print(f'Model created: {model.count_parameters()/1e6:.2f}M parameters')
print('✓ Installation successful!')
"
```

## Running Experiments

### 1. Quick Test (5-10 minutes)

Perfect for testing on M4 MacBook or verifying setup:

```bash
python experiments/run_experiments.py --config config/quick_test.yaml
```

**What it does:**
- Runs 3 small experiments
- Uses toy data (100 train, 50 test samples)
- Trains for 50 epochs
- Takes ~5-10 minutes on M4 Mac

### 2. Full Sudoku Experiments (4-6 hours with GPU)

```bash
# Single GPU
python experiments/run_experiments.py --config config/sudoku_config.yaml

# Specify device
python experiments/run_experiments.py --config config/sudoku_config.yaml --device cuda

# CPU (slower)
python experiments/run_experiments.py --config config/sudoku_config.yaml --device cpu
```

**What it does:**
- Runs 6 experiments testing recursion vs scale
- Uses 500 train, 200 test samples
- Trains for 150 epochs each
- Saves results to `results/sudoku/`

## Expected Results

### Quick Test
```
Experiment Results:
name                  params_M  test_acc
baseline_2L_n4       1.83      0.18-0.25
1L_n4                0.92      0.15-0.22
2L_n2                1.83      0.12-0.20
```

### Full Experiments (Sudoku-Extreme)
```
Experiment Results:
name                  layers  n  params_M  test_acc
1_layer_n6           1       6  3.5       0.60-0.65
baseline_2L_n6       2       6  7.0       0.82-0.87
4_layer_n6           4       6  14.0      0.75-0.80
2L_n2                2       2  7.0       0.70-0.75
2L_n4                2       4  7.0       0.78-0.83
2L_n8                2       8  7.0       0.82-0.86
```

**Key findings:**
- ✅ 2 layers outperforms 1 or 4 layers
- ✅ n=6 recursions is optimal
- ✅ Small networks + deep recursion > large networks

## Configuration

### Modifying Hyperparameters

Edit `config/sudoku_config.yaml`:

```yaml
# Adjust for your hardware
training:
  batch_size: 32      # Reduce if OOM (16, 8)
  num_epochs: 150     # Increase for better results (300, 500)
  learning_rate: 1.0e-3  # Tune if needed

# Adjust model size
model:
  hidden_size: 256    # Reduce if OOM (128, 64)
  num_layers: 2       # Test different sizes
  n_recursions: 6     # Test recursion depth
```

### Adding New Experiments

Add to `experiments` list in config:

```yaml
experiments:
  - name: "my_experiment"
    num_layers: 2
    n_recursions: 10  # Try deeper recursion
    T_cycles: 4       # More cycles
```

## Troubleshooting

### Out of Memory (OOM)

**Solution 1**: Reduce batch size
```yaml
training:
  batch_size: 16  # or 8
```

**Solution 2**: Reduce model size
```yaml
model:
  hidden_size: 128  # or 64
  n_recursions: 4   # or 2
```

**Solution 3**: Enable gradient checkpointing (advanced)
```python
# In src/model.py, add to forward():
from torch.utils.checkpoint import checkpoint
z = checkpoint(self.net, combined, use_reentrant=False)
```

### Slow Training

**For M4 Mac:**
- Use `config/quick_test.yaml` for fast testing
- Reduce `num_epochs` and `data.train_samples`
- Consider Google Colab for full experiments

**For GPU:**
- Check `nvidia-smi` to verify GPU usage
- Increase `batch_size` if VRAM allows
- Enable mixed precision (advanced)

### NaN Loss

**Solutions:**
- Reduce learning rate: `1.0e-4` instead of `1.0e-3`
- Increase gradient clipping: `grad_clip: 2.0`
- Check data preprocessing

### ImportError

```bash
# Ensure you're in the right directory
cd trm-recursion-study

# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall in development mode
pip install -e .
```

## Next Steps

After successful experiments:

1. **Analyze Results**
   ```bash
   python experiments/analyze_results.py --results results/sudoku/experiment_results.csv
   ```

2. **Visualize**
   - Check `results/figures/` for generated plots
   - Import results into notebooks for custom analysis

3. **Extend**
   - Add new datasets (Maze, ARC-AGI)
   - Try different architectures
   - Explore theoretical understanding

## Hardware-Specific Notes

### M4 MacBook Pro
- ✅ Perfect for development and testing
- ✅ Use MPS acceleration (`device: mps`)
- ⚠️ Full experiments take 10-20x longer
- 💡 Use Colab for full experiments

### Google Colab
- ✅ Free GPU (T4, 15GB VRAM)
- ✅ Can run full experiments
- ⚠️ Session timeout after ~12 hours
- 💡 Use notebook in `notebooks/` folder

### NVIDIA GPU (16GB+)
- ✅ Optimal setup
- ✅ Can run all experiments
- 💡 Increase batch size to utilize VRAM

### CPU Only
- ⚠️ Very slow (50-100x slower)
- ✅ Use `quick_test.yaml` only
- 💡 Consider cloud GPU services

## File Structure Reference

```
trm-recursion-study/
├── src/                    # Source code
│   ├── model.py           # TRM architecture
│   ├── trainer.py         # Training loop
│   └── ...
├── experiments/           # Experiment scripts
│   └── run_experiments.py
├── config/                # Configuration files
│   ├── sudoku_config.yaml
│   └── quick_test.yaml
├── results/               # Results (auto-generated)
│   ├── experiment_results.csv
│   └── figures/
├── checkpoints/           # Model checkpoints
├── requirements.txt       # Dependencies
└── README.md             # Main documentation
```

## Getting Help

- **Issues**: Open a GitHub issue
- **Questions**: Check README.md first
- **Paper**: See uploaded PDF for theoretical details
- **Code**: Comments in `src/` explain key components

## Citation

```bibtex
@article{jolicoeur2025less,
  title={Less is More: Recursive Reasoning with Tiny Networks},
  author={Jolicoeur-Martineau, Alexia},
  journal={arXiv preprint arXiv:2510.04871},
  year={2025}
}
```
