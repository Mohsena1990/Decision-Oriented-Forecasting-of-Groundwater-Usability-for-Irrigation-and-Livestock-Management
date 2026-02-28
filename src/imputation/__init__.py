"""
Imputation module for missing hydrochemical observations.

Provides:
  RAEImputer   — Recurrent Autoencoder for temporally consistent imputation.
"""

from .recurrent_autoencoder import RAEImputer, TORCH_AVAILABLE

__all__ = ["RAEImputer", "TORCH_AVAILABLE"]
