#!/bin/bash
# Quick setup script for TRM Recursion Study
# Run with: bash setup.sh

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  TRM Recursion Study - Quick Setup                        ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
echo "1️⃣  Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "   ✓ Python $PYTHON_VERSION detected"
echo ""

# Create virtual environment (optional but recommended)
read -p "2️⃣  Create virtual environment? (recommended) [y/N]: " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "   Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "   ✓ Virtual environment created and activated"
else
    echo "   ⚠️  Skipping virtual environment"
fi
echo ""

# Install dependencies
echo "3️⃣  Installing dependencies..."
if [ -f "requirements.txt" ] && [ -s "requirements.txt" ]; then
    pip install -r requirements.txt --quiet
    echo "   ✓ Dependencies installed"
else
    echo "   ❌ requirements.txt is missing or empty!"
    echo "   Creating requirements.txt..."
    cat > requirements.txt << 'EOF'
# Core dependencies
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
pyyaml>=6.0

# Optional
wandb>=0.15.0
jupyter>=1.0.0
EOF
    echo "   ✓ Created requirements.txt"
    pip install -r requirements.txt --quiet
    echo "   ✓ Dependencies installed"
fi
echo ""

# Create necessary directories
echo "4️⃣  Creating directory structure..."
mkdir -p results/figures
mkdir -p checkpoints
mkdir -p data
touch results/.gitkeep
touch results/figures/.gitkeep
echo "   ✓ Directories created"
echo ""

# Test imports
echo "5️⃣  Testing imports..."
python3 -c "
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
print('   ✓ All imports successful')
"
echo ""

# Check for CUDA/MPS
echo "6️⃣  Checking GPU support..."
python3 -c "
import torch
if torch.cuda.is_available():
    print('   ✓ CUDA available: ' + torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available():
    print('   ✓ MPS (Apple Silicon) available')
else:
    print('   ⚠️  No GPU detected (CPU only)')
"
echo ""

# Test quick_start.py
echo "7️⃣  Testing quick_start.py..."
if [ -f "quick_start.py" ]; then
    python3 -m py_compile quick_start.py
    if [ $? -eq 0 ]; then
        echo "   ✓ quick_start.py compiles successfully"
    else
        echo "   ❌ quick_start.py has syntax errors"
    fi
else
    echo "   ⚠️  quick_start.py not found"
fi
echo ""

# Summary
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  SETUP COMPLETE!                                          ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Quick test:    python3 quick_start.py"
echo "  2. Full test:     python3 experiments/run_experiments.py"
echo "  3. Analyze:       python3 analyze_results.py"
echo ""
echo "Documentation:"
echo "  • README.md       - Project overview"
echo "  • start_here.md   - Getting started guide"
echo "  • checklist.md    - Setup checklist"
echo ""
echo "If you created a venv, activate with: source venv/bin/activate"
echo ""