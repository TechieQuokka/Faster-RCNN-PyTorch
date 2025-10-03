"""
Configuration management utilities
"""

import yaml
import os
from typing import Dict, Any


class Config:
    """Configuration class with attribute access"""

    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config back to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to YAML config file

    Returns:
        Config object with attribute access
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def save_config(config: Config, save_path: str):
    """
    Save configuration to YAML file

    Args:
        config: Config object
        save_path: Path to save YAML file
    """
    config_dict = config.to_dict()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)


def merge_configs(base_config: Config, override_dict: Dict[str, Any]) -> Config:
    """
    Merge override dictionary into base config

    Args:
        base_config: Base configuration
        override_dict: Dictionary with overrides

    Returns:
        Merged Config object
    """
    base_dict = base_config.to_dict()

    def deep_merge(base, override):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                deep_merge(base[key], value)
            else:
                base[key] = value

    deep_merge(base_dict, override_dict)
    return Config(base_dict)
