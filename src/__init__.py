"""
Tiny Recursive Models (TRM) Package

Implementation of "Less is More: Recursive Reasoning with Tiny Networks"
by Alexia Jolicoeur-Martineau (Samsung SAIL Montreal, 2025)
"""

from .model import TinyRecursiveModel
from .trainer import TRMTrainer, ExponentialMovingAverage

__version__ = "1.0.0"
__author__ = "Your Name"
__all__ = ["TinyRecursiveModel", "TRMTrainer", "ExponentialMovingAverage"]
