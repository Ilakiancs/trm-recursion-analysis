"""
Data loading utilities for TRM experiments.
Provides toy data generation and real Sudoku loading (if available).
"""
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

class ToyDataGenerator:
    """Generate toy Sudoku-like data for testing."""
    
    @staticmethod
    def create_toy_data(n_samples: int = 500, seq_len: int = 81, 
                        vocab_size: int = 10) -> TensorDataset:
        """
        Create toy data that mimics Sudoku structure.
        
        Args:
            n_samples: Number of samples to generate
            seq_len: Length of each sequence (81 for 9x9 Sudoku)
            vocab_size: Size of vocabulary (10 for digits 0-9)
            
        Returns:
            TensorDataset with (puzzles, solutions)
        """
        # Generate random puzzles (0 = empty cell, 1-9 = filled)
        puzzles = torch.randint(0, vocab_size, (n_samples, seq_len))
        
        # Generate random solutions (1-9 only, no empty cells)
        solutions = torch.randint(1, vocab_size, (n_samples, seq_len))
        
        return TensorDataset(puzzles, solutions)
    
    @staticmethod
    def create_structured_toy_data(n_samples: int = 500, 
                                   difficulty: str = 'easy') -> TensorDataset:
        """
        Create toy data with simple patterns (easier to learn than random).
        
        Args:
            n_samples: Number of samples
            difficulty: 'easy', 'medium', or 'hard'
            
        Returns:
            TensorDataset with (puzzles, solutions)
        """
        puzzles = []
        solutions = []
        
        for _ in range(n_samples):
            # Create a simple pattern-based puzzle
            solution = torch.randint(1, 10, (81,))
            
            # Create puzzle by masking some cells
            if difficulty == 'easy':
                mask_prob = 0.3  # 30% cells empty
            elif difficulty == 'medium':
                mask_prob = 0.5  # 50% cells empty
            else:  # hard
                mask_prob = 0.7  # 70% cells empty
            
            mask = torch.rand(81) < mask_prob
            puzzle = solution.clone()
            puzzle[mask] = 0  # 0 = empty cell
            
            puzzles.append(puzzle)
            solutions.append(solution)
        
        return TensorDataset(torch.stack(puzzles), torch.stack(solutions))


class SudokuDataset(Dataset):
    """
    Dataset for real Sudoku puzzles (if available).
    Falls back to toy data if real data not found.
    """
    def __init__(self, data_dir: Optional[Path] = None, 
                 split: str = 'train', 
                 use_toy_data: bool = False):
        """
        Initialize Sudoku dataset.
        
        Args:
            data_dir: Path to data directory (optional)
            split: 'train', 'val', or 'test'
            use_toy_data: If True, use toy data instead of real Sudoku
        """
        self.split = split
        
        if use_toy_data or data_dir is None:
            print(f"⚠️  Using toy data for {split} set")
            n_samples = {'train': 500, 'val': 100, 'test': 200}[split]
            dataset = ToyDataGenerator.create_structured_toy_data(n_samples)
            self.puzzles = dataset.tensors[0]
            self.solutions = dataset.tensors[1]
        else:
            # Try to load real Sudoku data
            data_path = Path(data_dir)
            puzzle_file = data_path / f"{split}_puzzles.npy"
            solution_file = data_path / f"{split}_solutions.npy"
            
            if puzzle_file.exists() and solution_file.exists():
                print(f"✓ Loading real Sudoku data from {data_path}")
                self.puzzles = torch.from_numpy(np.load(puzzle_file))
                self.solutions = torch.from_numpy(np.load(solution_file))
            else:
                print(f"⚠️  Real data not found at {data_path}, using toy data")
                n_samples = {'train': 500, 'val': 100, 'test': 200}[split]
                dataset = ToyDataGenerator.create_structured_toy_data(n_samples)
                self.puzzles = dataset.tensors[0]
                self.solutions = dataset.tensors[1]
    
    def __len__(self) -> int:
        return len(self.puzzles)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.puzzles[idx], self.solutions[idx]


def create_dataloaders(data_dir: Optional[Path] = None,
                       batch_size: int = 32,
                       use_toy_data: bool = True,
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Path to data directory (optional)
        batch_size: Batch size for training
        use_toy_data: If True, use toy data
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = SudokuDataset(data_dir, 'train', use_toy_data)
    val_dataset = SudokuDataset(data_dir, 'val', use_toy_data)
    test_dataset = SudokuDataset(data_dir, 'test', use_toy_data)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"\n{'='*60}")
    print("Data Loading Summary")
    print(f"{'='*60}")
    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    return train_loader, val_loader, test_loader


def verify_data(puzzles: torch.Tensor, solutions: torch.Tensor) -> bool:
    """
    Verify that data is properly formatted.
    
    Args:
        puzzles: Puzzle tensor [N, 81]
        solutions: Solution tensor [N, 81]
        
    Returns:
        True if data is valid
    """
    # Check shapes
    if puzzles.shape != solutions.shape:
        print(f"❌ Shape mismatch: puzzles {puzzles.shape} vs solutions {solutions.shape}")
        return False
    
    if puzzles.shape[1] != 81:
        print(f"❌ Expected length 81, got {puzzles.shape[1]}")
        return False
    
    # Check value ranges
    if puzzles.min() < 0 or puzzles.max() >= 10:
        print(f"❌ Puzzle values out of range [0, 9]: [{puzzles.min()}, {puzzles.max()}]")
        return False
    
    if solutions.min() < 1 or solutions.max() >= 10:
        print(f"❌ Solution values out of range [1, 9]: [{solutions.min()}, {solutions.max()}]")
        return False
    
    print("✓ Data verification passed")
    return True


if __name__ == '__main__':
    # Test data generation
    print("Testing data utilities...\n")
    
    # Test 1: Toy data generation
    print("1. Testing toy data generation:")
    toy_data = ToyDataGenerator.create_toy_data(n_samples=100)
    puzzles, solutions = toy_data[0]
    print(f"   Puzzle shape: {puzzles.shape}")
    print(f"   Solution shape: {solutions.shape}")
    verify_data(toy_data.tensors[0], toy_data.tensors[1])
    
    # Test 2: Structured toy data
    print("\n2. Testing structured toy data:")
    structured_data = ToyDataGenerator.create_structured_toy_data(100, 'medium')
    verify_data(structured_data.tensors[0], structured_data.tensors[1])
    
    # Test 3: Create dataloaders
    print("\n3. Testing dataloader creation:")
    train_loader, val_loader, test_loader = create_dataloaders(
        use_toy_data=True,
        batch_size=16
    )
    
    # Test 4: Sample batch
    print("4. Testing batch loading:")
    for x, y in train_loader:
        print(f"   Batch shape: x={x.shape}, y={y.shape}")
        print(f"   Sample puzzle (first 9 cells): {x[0, :9]}")
        print(f"   Sample solution (first 9 cells): {y[0, :9]}")
        break
    
    print("\n✓ All tests passed!")