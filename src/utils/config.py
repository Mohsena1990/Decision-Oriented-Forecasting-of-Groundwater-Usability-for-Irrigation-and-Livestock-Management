"""Configuration loading utilities."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def get_config_value(config: Dict, *keys: str, default: Any = None) -> Any:
    """
    Get nested configuration value.

    Args:
        config: Configuration dictionary
        *keys: Nested keys
        default: Default value if not found

    Returns:
        Configuration value
    """
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file."""
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Saved configuration to {config_path}")
