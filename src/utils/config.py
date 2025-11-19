"""Configuration management module for GenAI CAD/CFD Studio.

This module provides a ConfigManager class for loading and managing YAML
configuration files with support for environment variables and dot notation
for accessing nested values.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


class ConfigManager:
    """Manages application configuration from YAML files.

    This class provides functionality to load, validate, and access configuration
    values from YAML files. It supports dot notation for nested values and
    environment variable substitution.

    Attributes:
        config_dir (Path): Directory containing configuration files.
        config (Dict): Loaded configuration data.

    Example:
        >>> config = ConfigManager("configs")
        >>> config.load("app.yaml")
        >>> db_host = config.get("database.host")
        >>> config.set("database.port", 5432)
    """

    def __init__(self, config_dir: Union[str, Path] = "configs"):
        """Initialize the ConfigManager.

        Args:
            config_dir: Path to the directory containing configuration files.
                Defaults to "configs".
        """
        self.config_dir = Path(config_dir)
        self.config: Dict[str, Any] = {}

        # Create config directory if it doesn't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def load(self, config_file: str) -> None:
        """Load a YAML configuration file.

        Args:
            config_file: Name of the configuration file to load.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            yaml.YAMLError: If the YAML file is invalid.
        """
        config_path = self.config_dir / config_file

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)

        if loaded_config is None:
            loaded_config = {}

        # Substitute environment variables
        self._substitute_env_vars(loaded_config)

        # Merge with existing config
        self.config.update(loaded_config)

    def _substitute_env_vars(self, data: Any) -> None:
        """Recursively substitute environment variables in config values.

        Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax.

        Args:
            data: Configuration data to process (dict, list, or primitive).
        """
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str):
                    data[key] = self._expand_env_var(value)
                else:
                    self._substitute_env_vars(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, str):
                    data[i] = self._expand_env_var(item)
                else:
                    self._substitute_env_vars(item)

    def _expand_env_var(self, value: str) -> str:
        """Expand environment variables in a string.

        Args:
            value: String potentially containing ${VAR} or ${VAR:default}.

        Returns:
            String with environment variables expanded.
        """
        if not isinstance(value, str):
            return value

        # Simple environment variable expansion
        import re
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

        def replacer(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(var_name, default_value)

        return re.sub(pattern, replacer, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation.

        Args:
            key: Configuration key in dot notation (e.g., "database.host").
            default: Default value to return if key doesn't exist.

        Returns:
            Configuration value or default if not found.

        Example:
            >>> config.get("database.host")
            'localhost'
            >>> config.get("database.connection.timeout", 30)
            30
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value using dot notation.

        Creates nested dictionaries as needed.

        Args:
            key: Configuration key in dot notation (e.g., "database.host").
            value: Value to set.

        Example:
            >>> config.set("database.host", "localhost")
            >>> config.set("database.connection.timeout", 30)
        """
        keys = key.split('.')
        current = self.config

        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]

        # Set the value
        current[keys[-1]] = value

    def validate_schema(self, schema: Dict[str, type]) -> bool:
        """Validate configuration against a schema.

        Args:
            schema: Dictionary mapping config keys (dot notation) to expected types.

        Returns:
            True if all required keys exist with correct types.

        Raises:
            ValueError: If validation fails.

        Example:
            >>> schema = {
            ...     "database.host": str,
            ...     "database.port": int,
            ...     "debug": bool
            ... }
            >>> config.validate_schema(schema)
            True
        """
        for key, expected_type in schema.items():
            value = self.get(key)

            if value is None:
                raise ValueError(f"Required configuration key missing: {key}")

            if not isinstance(value, expected_type):
                raise ValueError(
                    f"Invalid type for {key}: expected {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )

        return True

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration data.

        Returns:
            Complete configuration dictionary.
        """
        return self.config.copy()

    def clear(self) -> None:
        """Clear all configuration data."""
        self.config = {}

    def save(self, config_file: str) -> None:
        """Save current configuration to a YAML file.

        Args:
            config_file: Name of the file to save configuration to.
        """
        config_path = self.config_dir / config_file

        with open(config_path, 'w') as f:
            yaml.safe_dump(self.config, f, default_flow_style=False, indent=2)

    def __repr__(self) -> str:
        """String representation of ConfigManager."""
        return f"ConfigManager(config_dir='{self.config_dir}', keys={list(self.config.keys())})"
