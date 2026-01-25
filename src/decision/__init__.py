"""MCDM decision-making module."""

from .vikor import VIKOR, VIKORResult
from .model_selection import ModelSelector

__all__ = ["VIKOR", "VIKORResult", "ModelSelector"]
