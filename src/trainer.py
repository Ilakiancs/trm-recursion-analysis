"""
Training utilities for Tiny Recursive Models
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import time
from typing import Dict, Optional
import copy


class ExponentialMovingAverage:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class TRMTrainer:
    def __init__(
        self,
        model,
        optimizer,
        device="cuda",
        use_ema=True,
        ema_decay=0.999,
        grad_clip=1.0,
        n_supervision=16,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.grad_clip = grad_clip
        self.n_supervision = n_supervision
        self.use_ema = use_ema
        if use_ema:
            self.ema = ExponentialMovingAverage(model, decay=ema_decay)

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "test_acc": [],
            "avg_steps": [],
            "epoch": [],
        }

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        total_steps = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc="Training")
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            y_pred, losses = self.model(x, y, n_supervision=self.n_supervision)
            losses["total_loss"].backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            if self.use_ema:
                self.ema.update()

            pred_tokens = y_pred.argmax(dim=-1)
            batch_correct = (pred_tokens == y).sum().item()
            batch_tokens = y.numel()

            total_loss += losses["total_loss"].item()
            total_correct += batch_correct
            total_tokens += batch_tokens
            total_steps += losses["steps"]
            num_batches += 1

            pbar.set_postfix(
                {
                    "loss": f"{losses['total_loss'].item():.4f}",
                    "acc": f"{batch_correct / batch_tokens:.4f}",
                    "steps": losses["steps"],
                }
            )

        avg_loss = total_loss / num_batches
        avg_acc = total_correct / total_tokens
        avg_steps = total_steps / num_batches
        return {"loss": avg_loss, "accuracy": avg_acc, "avg_steps": avg_steps}

    @torch.no_grad()
    def evaluate(self, test_loader, use_ema=True):
        self.model.eval()
        if use_ema and self.use_ema:
            self.ema.apply_shadow()

        total_correct = 0
        total_tokens = 0

        for x, y in tqdm(test_loader, desc="Evaluating"):
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.model(x, n_supervision=self.n_supervision)
            pred_tokens = y_pred.argmax(dim=-1)
            total_correct += (pred_tokens == y).sum().item()
            total_tokens += y.numel()

        accuracy = total_correct / total_tokens

        if use_ema and self.use_ema:
            self.ema.restore()

        return {"accuracy": accuracy}

    def train(
        self, train_loader, test_loader, num_epochs, eval_freq=20, save_path=None
    ):
        """
        Full training loop with evaluation.

        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            num_epochs: Number of epochs to train
            eval_freq: Evaluate every N epochs
            save_path: Path to save best model

        Returns:
            Dictionary with training history
        """
        best_test_acc = 0.0
        start_time = time.time()

        print(f"\n{'=' * 70}")
        print(f"Starting Training: {num_epochs} epochs")
        print(f"{'=' * 70}\n")

        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)

            # Evaluate
            if epoch % eval_freq == 0 or epoch == num_epochs - 1:
                test_metrics = self.evaluate(test_loader)

                # Update history
                self.history["epoch"].append(epoch)
                self.history["train_loss"].append(train_metrics["loss"])
                self.history["train_acc"].append(train_metrics["accuracy"])
                self.history["test_acc"].append(test_metrics["accuracy"])
                self.history["avg_steps"].append(train_metrics["avg_steps"])

                # Print progress
                elapsed = time.time() - start_time
                print(
                    f"\nEpoch {epoch:4d} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Train Acc: {train_metrics['accuracy']:.4f} | "
                    f"Test Acc: {test_metrics['accuracy']:.4f} | "
                    f"Avg Steps: {train_metrics['avg_steps']:.1f} | "
                    f"Time: {elapsed:.1f}s"
                )

                # Save best model
                if test_metrics["accuracy"] > best_test_acc:
                    best_test_acc = test_metrics["accuracy"]
                    if save_path:
                        self.save_checkpoint(save_path, epoch, best_test_acc)

        print(f"\n{'=' * 70}")
        print(f"Training Complete!")
        print(f"Best Test Accuracy: {best_test_acc:.4f}")
        print(f"Total Time: {time.time() - start_time:.1f}s")
        print(f"{'=' * 70}\n")

        return {
            "best_test_acc": best_test_acc,
            "final_train_acc": train_metrics["accuracy"],
            "history": self.history,
        }

    def save_checkpoint(self, path, epoch, accuracy):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "accuracy": accuracy,
            "history": self.history,
        }

        if self.use_ema:
            checkpoint["ema_shadow"] = self.ema.shadow

        torch.save(checkpoint, path)
        print(f" Checkpoint saved: {path} (acc={accuracy:.4f})")

    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]

        if self.use_ema and "ema_shadow" in checkpoint:
            self.ema.shadow = checkpoint["ema_shadow"]

        print(f" Checkpoint loaded: {path}")
        return checkpoint


if __name__ == "__main__":
    # Test trainer
    print("Testing TRM Trainer...")

    from model import TinyRecursiveModel
    from torch.utils.data import TensorDataset, DataLoader

    # Create dummy data
    train_x = torch.randint(0, 10, (100, 81))
    train_y = torch.randint(1, 10, (100, 81))
    test_x = torch.randint(0, 10, (50, 81))
    test_y = torch.randint(1, 10, (50, 81))

    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Create model
    model = TinyRecursiveModel(
        vocab_size=10, hidden_size=128, num_layers=2, n_recursions=4, T_cycles=2
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

    # Create trainer
    trainer = TRMTrainer(
        model=model,
        optimizer=optimizer,
        device="cpu",  # Use CPU for testing
        use_ema=True,
        n_supervision=4,
    )

    # Train for 2 epochs
    results = trainer.train(
        train_loader=train_loader, test_loader=test_loader, num_epochs=2, eval_freq=1
    )

    print("\n Trainer test passed!")
    print(f"Final test accuracy: {results['best_test_acc']:.4f}")
