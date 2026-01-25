"""Reproducibility utilities."""

import logging
import random
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed
        deterministic: Whether to set CUDA deterministic mode
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        logger.info(f"PyTorch seed set to {seed}")
    except ImportError:
        pass

    logger.info(f"Random seeds set to {seed}")
