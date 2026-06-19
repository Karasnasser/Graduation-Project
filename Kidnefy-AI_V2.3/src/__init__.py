"""
Kidney Disease Prediction - Source Package

Submodules are imported lazily to avoid requiring all dependencies
when only a specific module is needed.

Usage:
    from src.staging import GFRCalculator       # works without pandas
    from src.preprocessing import DataLoader    # requires pandas
    from src.models import EnsembleModel        # requires sklearn, tensorflow
"""

__version__ = "1.0.0"
__author__ = "Graduation Project Team"

__all__ = [
    'preprocessing',
    'models',
    'staging',
    'rag',
    'reports',
    'explainability',
    'monitoring',
]
