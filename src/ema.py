"""
Exponential Moving Average (EMA) for PyTorch models.

Note: This module is kept for backward compatibility.
The EMA implementation is in trainer.py as ExponentialMovingAverage class.
This file re-exports it for convenience.
"""

from .trainer import ExponentialMovingAverage

__all__ = ['ExponentialMovingAverage']
