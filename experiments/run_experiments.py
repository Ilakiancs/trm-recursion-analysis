"""
Main Experiment Runner for TRM
"""

import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import yaml
import argparse
from pathlib import Path
import pandas as pd
import json
import sys
import random
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model import TinyRecursiveModel
from src.trainer import TRMTrainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str):
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def create_toy_sudoku_data(n_samples, augmentations=10):
    puzzles = torch.randint(0, 10, (n_samples, 81))
    solutions = torch.randint(1, 10, (n_samples, 81))

    if augmentations > 1:
        puzzles_aug = puzzles.repeat_interleave(augmentations, dim=0)
        solutions_aug = solutions.repeat_interleave(augmentations, dim=0)
        noise_mask = torch.rand_like(puzzles_aug.float()) < 0.1
        puzzles_aug = puzzles_aug.masked_fill(noise_mask, 0)
        return TensorDataset(puzzles_aug, solutions_aug)

    return TensorDataset(puzzles, solutions)


def run_single_experiment(config, exp_config, device):
    print(f"\n{'=' * 70}")
    print(f"Experiment: {exp_config['name']}")
    print(f"  Layers: {exp_config['num_layers']}")
    print(f"  Recursions (n): {exp_config['n_recursions']}")
    print(f"  Cycles (T): {exp_config['T_cycles']}")
    print(f"{'=' * 70}\n")

    # Create datasets
    print("Creating datasets...")
    train_data = create_toy_sudoku_data(
        config["data"]["train_samples"], config["data"]["augmentations"]
    )
    test_data = create_toy_sudoku_data(
        config["data"]["test_samples"],
        augmentations=1,  # No augmentation for test
    )

    train_loader = DataLoader(
        train_data, batch_size=config["training"]["batch_size"], shuffle=True
    )
    test_loader = DataLoader(test_data, batch_size=config["training"]["batch_size"] * 2)

    print(f"  Train: {len(train_data)} samples")
    print(f"  Test: {len(test_data)} samples")

    # Create model
    model = TinyRecursiveModel(
        vocab_size=config["model"]["vocab_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=exp_config["num_layers"],
        n_recursions=exp_config["n_recursions"],
        T_cycles=exp_config["T_cycles"],
        seq_len=config["model"]["seq_len"],
        use_attention=config["model"]["use_attention"],
    )

    n_params = model.count_parameters()
    print(f"\nModel parameters: {n_params / 1e6:.2f}M")

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        betas=tuple(config["training"]["betas"]),
        eps=config["training"]["eps"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Create trainer
    trainer = TRMTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        use_ema=config["training"]["use_ema"],
        ema_decay=config["training"]["ema_decay"],
        grad_clip=config["training"]["grad_clip"],
        n_supervision=config["training"]["n_supervision"],
    )

    # Train
    save_path = Path(config["checkpoint_dir"]) / f"{exp_config['name']}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    results = trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=config["training"]["num_epochs"],
        eval_freq=config["eval_freq"],
        save_path=save_path if config["save_best"] else None,
    )

    # Compile results
    return {
        "name": exp_config["name"],
        "num_layers": exp_config["num_layers"],
        "n_recursions": exp_config["n_recursions"],
        "T_cycles": exp_config["T_cycles"],
        "params_M": n_params / 1e6,
        "effective_depth": exp_config["T_cycles"]
        * (exp_config["n_recursions"] + 1)
        * exp_config["num_layers"],
        "best_test_acc": results["best_test_acc"],
        "final_train_acc": results["final_train_acc"],
        "gen_gap": results["final_train_acc"] - results["best_test_acc"],
        "history": results["history"],
    }


def main(args):
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set seed
    set_seed(config.get("seed", 42))

    # Get device
    device = get_device(config["device"])
    print(f"\n{'=' * 70}")
    print(f"TRM Experiment Runner")
    print(f"{'=' * 70}")
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Experiment: {config['experiment_name']}")
    print(f"{'=' * 70}\n")

    # Create output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiments
    all_results = []

    for exp_config in config["experiments"]:
        try:
            result = run_single_experiment(config, exp_config, device)
            all_results.append(result)

            print(f"\n {exp_config['name']} complete!")
            print(f"  Best Test Acc: {result['best_test_acc']:.4f}")
            print(f"  Gen Gap: {result['gen_gap']:.4f}")

        except Exception as e:
            print(f"\n {exp_config['name']} failed: {str(e)}")
            continue

    # Save results
    print(f"\n{'=' * 70}")
    print("Saving results...")

    # CSV summary (without history)
    df = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "history"} for r in all_results]
    )
    csv_path = output_dir / "experiment_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"   CSV saved: {csv_path}")

    # JSON details (with history)
    json_path = output_dir / "detailed_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"   JSON saved: {json_path}")

    # Print summary table
    print(f"\n{'=' * 70}")
    print("EXPERIMENT RESULTS")
    print(f"{'=' * 70}")
    print(df.to_string(index=False))
    print(f"{'=' * 70}\n")

    print(" All experiments complete!")
    print(f"Results saved to: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TRM experiments")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device from config (cuda/cpu/mps/auto)",
    )

    args = parser.parse_args()

    # Override device if specified
    if args.device:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        config["device"] = args.device
        with open(args.config, "w") as f:
            yaml.dump(config, f)

    main(args)
