"""
Results analysis and visualization for TRM experiments.
Generates publication-quality figures and summary statistics.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11

class TRMResultsAnalyzer:
    """Analyzer for TRM experiment results."""
    
    def __init__(self, results_dir: str = 'results'):
        """
        Initialize analyzer.
        
        Args:
            results_dir: Directory containing results
        """
        self.results_dir = Path(results_dir)
        self.results_df = None
        self.detailed_results = None
        
    def load_results(self) -> bool:
        """
        Load experiment results from CSV and JSON.
        
        Returns:
            True if successful, False otherwise
        """
        csv_path = self.results_dir / 'experiment_results.csv'
        json_path = self.results_dir / 'detailed_results.json'
        
        if not csv_path.exists():
            print(f" Results not found at {csv_path}")
            return False
        
        # Load summary
        self.results_df = pd.read_csv(csv_path)
        print(f" Loaded {len(self.results_df)} experiments from {csv_path}")
        
        # Load detailed results
        if json_path.exists():
            with open(json_path, 'r') as f:
                self.detailed_results = json.load(f)
            print(f" Loaded detailed results from {json_path}")
        
        return True
    
    def print_summary(self):
        """Print summary statistics."""
        if self.results_df is None:
            print(" No results loaded")
            return
        
        print("\n" + "="*70)
        print("EXPERIMENT SUMMARY")
        print("="*70)
        print(self.results_df.to_string(index=False))
        print("="*70)
        
        # Best configuration
        best_idx = self.results_df['test_acc'].idxmax()
        best = self.results_df.iloc[best_idx]
        
        print(f"\n BEST CONFIGURATION:")
        print(f"   Name: {best['name']}")
        print(f"   Layers: {best['layers']}")
        print(f"   Recursions: {best['n_recursions']}")
        print(f"   Test Acc: {best['test_acc']*100:.2f}%")
        print(f"   Gen Gap: {best['gen_gap']*100:.2f}%")
        
        # Worst overfitting
        worst_overfit_idx = self.results_df['gen_gap'].idxmax()
        worst = self.results_df.iloc[worst_overfit_idx]
        
        print(f"\n  HIGHEST OVERFITTING:")
        print(f"   Name: {worst['name']}")
        print(f"   Gen Gap: {worst['gen_gap']*100:.2f}%")
    
    def plot_layer_comparison(self, save_path: Optional[str] = None):
        """Plot comparison across different network sizes."""
        if self.results_df is None:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Get layer experiments
        layer_exps = self.results_df[
            self.results_df['name'].str.contains('Layer', case=False)
        ].sort_values('layers')
        
        if len(layer_exps) == 0:
            print("  No layer comparison experiments found")
            return
        
        # Plot 1: Accuracy comparison
        x = np.arange(len(layer_exps))
        width = 0.35
        
        axes[0].bar(x - width/2, layer_exps['test_acc'], width,
                   label='Test', alpha=0.8, color='steelblue')
        axes[0].bar(x + width/2, layer_exps['train_acc'], width,
                   label='Train', alpha=0.7, color='orange')
        axes[0].set_xlabel('Configuration', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Network Size vs Accuracy', fontsize=13, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f"{int(l)}L" for l in layer_exps['layers']])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Generalization gap
        axes[1].bar(range(len(layer_exps)), layer_exps['gen_gap'],
                   alpha=0.7, color='red')
        axes[1].set_xlabel('Configuration', fontsize=12)
        axes[1].set_ylabel('Generalization Gap', fontsize=12)
        axes[1].set_title('Overfitting Analysis', fontsize=13, fontweight='bold')
        axes[1].set_xticks(range(len(layer_exps)))
        axes[1].set_xticklabels([f"{int(l)}L" for l in layer_exps['layers']])
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Saved to {save_path}")
        plt.show()
    
    def plot_recursion_comparison(self, save_path: Optional[str] = None):
        """Plot comparison across different recursion depths."""
        if self.results_df is None:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Get recursion experiments (2 layers, varying n)
        recursion_exps = self.results_df[
            self.results_df['layers'] == 2
        ].sort_values('n_recursions')
        
        if len(recursion_exps) < 2:
            print("  Not enough recursion experiments found")
            return
        
        # Plot 1: Accuracy vs recursion
        axes[0].plot(recursion_exps['n_recursions'], 
                    recursion_exps['test_acc'],
                    marker='o', linewidth=2.5, markersize=12,
                    color='steelblue', label='Test')
        axes[0].plot(recursion_exps['n_recursions'],
                    recursion_exps['train_acc'],
                    marker='s', linewidth=2.5, markersize=12,
                    color='orange', alpha=0.7, label='Train')
        axes[0].set_xlabel('Recursion Steps (n)', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Recursion Depth vs Accuracy', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Generalization gap
        axes[1].plot(recursion_exps['n_recursions'],
                    recursion_exps['gen_gap'],
                    marker='D', linewidth=2.5, markersize=12,
                    color='red')
        axes[1].set_xlabel('Recursion Steps (n)', fontsize=12)
        axes[1].set_ylabel('Generalization Gap', fontsize=12)
        axes[1].set_title('Overfitting vs Recursion', fontsize=13, fontweight='bold')
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Saved to {save_path}")
        plt.show()
    
    def plot_learning_curves(self, save_path: Optional[str] = None):
        """Plot learning curves from detailed results."""
        if self.detailed_results is None:
            print("  No detailed results available")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for result in self.detailed_results:
            if 'history' not in result:
                continue
            
            history = result['history']
            label = f"{result['name']} (final: {result['test_acc']:.3f})"
            
            ax.plot(history['epoch'], history['test_acc'],
                   marker='o', label=label, linewidth=2, markersize=6)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title('Learning Curves: Test Accuracy Over Time', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Saved to {save_path}")
        plt.show()
    
    def plot_parameter_efficiency(self, save_path: Optional[str] = None):
        """Plot accuracy vs parameter count."""
        if self.results_df is None:
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        scatter = ax.scatter(
            self.results_df['params_M'],
            self.results_df['test_acc'],
            s=200, alpha=0.7,
            c=self.results_df['n_recursions'],
            cmap='viridis',
            edgecolors='black',
            linewidth=2
        )
        
        # Annotate points
        for _, row in self.results_df.iterrows():
            ax.annotate(
                f"{int(row['layers'])}L,n={int(row['n_recursions'])}",
                (row['params_M'], row['test_acc']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9
            )
        
        ax.set_xlabel('Parameters (M)', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title('Parameter Efficiency', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Recursion Depth (n)', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Saved to {save_path}")
        plt.show()
    
    def generate_all_figures(self, output_dir: Optional[str] = None):
        """Generate all analysis figures."""
        if output_dir is None:
            output_dir = self.results_dir / 'figures'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print("\n" + "="*70)
        print("GENERATING ANALYSIS FIGURES")
        print("="*70)
        
        print("\n1/4 - Layer Comparison...")
        self.plot_layer_comparison(output_dir / 'layer_comparison.png')
        
        print("\n2/4 - Recursion Comparison...")
        self.plot_recursion_comparison(output_dir / 'recursion_comparison.png')
        
        print("\n3/4 - Learning Curves...")
        self.plot_learning_curves(output_dir / 'learning_curves.png')
        
        print("\n4/4 - Parameter Efficiency...")
        self.plot_parameter_efficiency(output_dir / 'parameter_efficiency.png')
        
        print("\n" + "="*70)
        print(f" All figures saved to {output_dir}/")
        print("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze TRM experiment results'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory containing results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for figures (default: results/figures)'
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Only print summary, no figures'
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = TRMResultsAnalyzer(args.results_dir)
    
    # Load results
    if not analyzer.load_results():
        print("\n Failed to load results")
        print(f"   Make sure experiment_results.csv exists in {args.results_dir}/")
        return
    
    # Print summary
    analyzer.print_summary()
    
    # Generate figures
    if not args.summary_only:
        analyzer.generate_all_figures(args.output_dir)


if __name__ == '__main__':
    main()