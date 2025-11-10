"""
Quick Start Test for TRM

Minimal script to verify installation and run a fast test.
Perfect for M4 MacBook or quick testing.

Usage:
    python quick_start.py
"""

import torch
from torch.utils.data import TensorDataset, DataLoader
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.model import TinyRecursiveModel
from src.trainer import TRMTrainer


def main():
    print("\n" + "=" * 70)
    print("TRM Quick Start Test")
    print("=" * 70 + "\n")

    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f" Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(" Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print(" Using CPU (will be slow)")

    # Create tiny dataset
    print("\nCreating toy dataset...")
    train_x = torch.randint(0, 10, (50, 81))
    train_y = torch.randint(1, 10, (50, 81))
    test_x = torch.randint(0, 10, (20, 81))
    test_y = torch.randint(1, 10, (20, 81))

    train_data = TensorDataset(train_x, train_y)
    test_data = TensorDataset(test_x, test_y)

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16)

    print(f"  Train: {len(train_data)} samples")
    print(f"  Test: {len(test_data)} samples")

    # Create model
    print("\nCreating model...")
    model = TinyRecursiveModel(
        vocab_size=10,
        hidden_size=128,  # Small for speed
        num_layers=2,
        n_recursions=4,  # Moderate recursion
        T_cycles=2,
        use_attention=False,
    )

    print(f"  Parameters: {model.count_parameters() / 1e6:.2f}M")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

    # Create trainer
    print("\nCreating trainer...")
    trainer = TRMTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        use_ema=True,
        n_supervision=4,  # Fewer steps for speed
    )

    # Train
    print("\nTraining (20 epochs)...")
    print("This should take ~2-5 minutes...\n")

    results = trainer.train(
        train_loader=train_loader, test_loader=test_loader, num_epochs=20, eval_freq=5
    )

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Best Test Accuracy: {results['best_test_acc']:.4f}")
    print(f"Final Train Accuracy: {results['final_train_acc']:.4f}")
    print(
        f"Generalization Gap: {results['final_train_acc'] - results['best_test_acc']:.4f}"
    )
    print("=" * 70 + "\n")

    if results["best_test_acc"] > 0.10:
        print(" Installation verified! Model is learning.")
        print("\nNext steps:")
        print(
            "1. Run full tests: python experiments/run_experiments.py --config config/quick_test.yaml"
        )
        print("2. See docs/SETUP.md for detailed instructions")
        print("3. Check README.md for experiment details")
    else:
        print(" Model accuracy very low - this is expected on random data")
        print("Run with real Sudoku data for meaningful results")

    return results


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Interrupted by user")
    except Exception as e:
        print(f"\n\n Error: {e}")
        print("\nTroubleshooting:")
        print(
            "1. Check that all dependencies are installed: pip install -r requirements.txt"
        )
        print(
            "2. Verify PyTorch installation: python -c 'import torch; print(torch.__version__)'"
        )
        print("3. See docs/SETUP.md for detailed setup instructions")
