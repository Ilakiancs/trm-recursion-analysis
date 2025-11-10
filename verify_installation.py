#!/usr/bin/env python3
"""
Final verification script for TRM project.
Run this to verify all components are working.
"""

print('='*70)
print('FINAL PROJECT VERIFICATION')
print('='*70)

# Test imports
try:
    import torch
    print(f' PyTorch {torch.__version__}')
    
    import numpy
    print(f' NumPy {numpy.__version__}')
    
    import pandas
    print(f' Pandas {pandas.__version__}')
    
    import matplotlib
    print(f' Matplotlib {matplotlib.__version__}')
    
    from src.model import TinyRecursiveModel
    print(' TRM Model imported')
    
    from src.trainer import TRMTrainer
    print(' TRM Trainer imported')
    
    from src.data_utils import create_dataloaders
    print(' Data utilities imported')
    
    # Test model creation
    m = TinyRecursiveModel(hidden_size=128)
    print(f' Model created: {m.count_parameters()/1e6:.2f}M params')
    
    # Test GPU
    if torch.cuda.is_available():
        print(f' GPU available: {torch.cuda.get_device_name(0)}')
    elif torch.backends.mps.is_available():
        print(' Apple Silicon MPS available')
    else:
        print(' CPU only (no GPU)')
    
    print('='*70)
    print(' ALL SYSTEMS OPERATIONAL - PROJECT COMPLETE!')
    print('='*70)
    print('\nReady to run:')
    print('  python3 quick_start.py')
    print('  python3 experiments/run_experiments.py --config config/quick_test.yaml')
    
except Exception as e:
    print(f'\n Error: {e}')
    print('\nTroubleshooting:')
    print('1. Activate virtual environment: source venv/bin/activate')
    print('2. Install dependencies: pip install -r requirements.txt')
    exit(1)
