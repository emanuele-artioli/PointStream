#!/usr/bin/env python3
"""
Direct configuration reading from config.ini

This replaces the complex config manager with simple direct INI file reading.
"""

import configparser
import json
import os
import re
from pathlib import Path
from typing import Any, List

# Global config parser - just one simple instance
config = configparser.ConfigParser()

def load_config(config_file: str = None):
    """Load configuration from INI file."""
    global config
    
    if config_file is None:
        # Look for config.ini in the parent directory (project root)
        config_file = Path(__file__).parent.parent / "config.ini"
    
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config.read(config_path)
    print(f"Loaded config from: {config_path}")

def get_str(section: str, key: str, default: str = "") -> str:
    """Get string value from config with environment variable substitution."""
    try:
        value = config.get(section, key)
        # Handle environment variable substitution ${VAR_NAME}
        if value and '${' in value:
            pattern = r'\$\{([^}]+)\}'
            def replace_env_var(match):
                env_var = match.group(1)
                return os.environ.get(env_var, '')
            value = re.sub(pattern, replace_env_var, value)
        return value
    except (configparser.NoSectionError, configparser.NoOptionError):
        return default

def get_int(section: str, key: str, default: int = 0) -> int:
    """Get integer value from config."""
    try:
        return config.getint(section, key)
    except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
        return default

def get_float(section: str, key: str, default: float = 0.0) -> float:
    """Get float value from config."""
    try:
        return config.getfloat(section, key)
    except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
        return default

def get_bool(section: str, key: str, default: bool = False) -> bool:
    """Get boolean value from config."""
    try:
        return config.getboolean(section, key)
    except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
        return default

def get_list(section: str, key: str, default: List = None) -> List:
    """Get list value from config (JSON format)."""
    if default is None:
        default = []
    try:
        value = config.get(section, key)
        if value.strip():
            return json.loads(value)
        return default
    except (configparser.NoSectionError, configparser.NoOptionError, json.JSONDecodeError):
        return default

# Load default config when module is imported
load_config()
