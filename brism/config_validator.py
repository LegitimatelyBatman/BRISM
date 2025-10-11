"""
Configuration schema validation for YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Set


class ConfigValidator:
    """Validator for BRISM configuration files."""
    
    # Define expected configuration schema
    REQUIRED_KEYS: Set[str] = {
        'symptom_vocab_size',
        'icd_vocab_size',
    }
    
    OPTIONAL_KEYS: Set[str] = {
        'symptom_embed_dim',
        'icd_embed_dim',
        'encoder_hidden_dim',
        'latent_dim',
        'decoder_hidden_dim',
        'max_symptom_length',
        'dropout_rate',
        'mc_samples',
        'temporal_encoding_type',
        'temperature',
        'beam_width',
        'n_ensemble_models',
    }
    
    # Type expectations
    TYPE_EXPECTATIONS: Dict[str, type] = {
        'symptom_vocab_size': int,
        'icd_vocab_size': int,
        'symptom_embed_dim': int,
        'icd_embed_dim': int,
        'encoder_hidden_dim': int,
        'latent_dim': int,
        'decoder_hidden_dim': int,
        'max_symptom_length': int,
        'dropout_rate': (int, float),
        'mc_samples': int,
        'temporal_encoding_type': str,
        'temperature': (int, float),
        'beam_width': int,
        'n_ensemble_models': int,
    }
    
    # Valid values for specific keys
    VALID_VALUES: Dict[str, Set[Any]] = {
        'temporal_encoding_type': {'positional', 'timestamp'},
    }
    
    # Common typos and corrections
    COMMON_TYPOS: Dict[str, str] = {
        'vocab_size': 'symptom_vocab_size or icd_vocab_size',
        'embedding_dim': 'symptom_embed_dim or icd_embed_dim',
        'hidden_dim': 'encoder_hidden_dim or decoder_hidden_dim',
        'dropout': 'dropout_rate',
        'samples': 'mc_samples',
        'temporal_type': 'temporal_encoding_type',
        'temp': 'temperature',
        'beam_size': 'beam_width',
        'n_models': 'n_ensemble_models',
    }
    
    def __init__(self, strict: bool = True):
        """
        Initialize validator.
        
        Args:
            strict: If True, raise errors for unknown keys. If False, only warn.
        """
        self.strict = strict
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_config_file(self, config_path: str) -> Dict[str, Any]:
        """
        Validate a YAML configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        # Check file exists
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load YAML
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML file: {e}")
        
        if config is None:
            raise ValueError("Configuration file is empty")
        
        if not isinstance(config, dict):
            raise ValueError(f"Configuration must be a dictionary, got {type(config)}")
        
        # Validate the configuration
        return self.validate_config_dict(config)
    
    def validate_config_dict(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
        """
        self.errors.clear()
        self.warnings.clear()
        
        # Check required keys
        missing_keys = self.REQUIRED_KEYS - set(config.keys())
        if missing_keys:
            self.errors.append(f"Missing required keys: {missing_keys}")
        
        # Check for unknown keys and suggest corrections
        all_valid_keys = self.REQUIRED_KEYS | self.OPTIONAL_KEYS
        unknown_keys = set(config.keys()) - all_valid_keys
        
        for key in unknown_keys:
            # Check if it's a common typo
            if key in self.COMMON_TYPOS:
                suggestion = self.COMMON_TYPOS[key]
                msg = f"Unknown key '{key}'. Did you mean '{suggestion}'?"
                if self.strict:
                    self.errors.append(msg)
                else:
                    self.warnings.append(msg)
            else:
                # Look for similar keys (simple string matching)
                suggestions = [k for k in all_valid_keys if key.lower() in k.lower() or k.lower() in key.lower()]
                if suggestions:
                    msg = f"Unknown key '{key}'. Similar valid keys: {suggestions}"
                else:
                    msg = f"Unknown key '{key}'"
                
                if self.strict:
                    self.errors.append(msg)
                else:
                    self.warnings.append(msg)
        
        # Validate types
        for key, value in config.items():
            if key in self.TYPE_EXPECTATIONS:
                expected_type = self.TYPE_EXPECTATIONS[key]
                if not isinstance(value, expected_type):
                    self.errors.append(
                        f"Key '{key}' has wrong type. Expected {expected_type}, got {type(value)}"
                    )
        
        # Validate specific values
        for key, valid_values in self.VALID_VALUES.items():
            if key in config and config[key] not in valid_values:
                self.errors.append(
                    f"Key '{key}' has invalid value '{config[key]}'. Valid values: {valid_values}"
                )
        
        # Validate ranges (only if type is correct)
        if 'dropout_rate' in config and isinstance(config['dropout_rate'], (int, float)):
            if not 0.0 <= config['dropout_rate'] <= 1.0:
                self.errors.append(f"dropout_rate must be between 0 and 1, got {config['dropout_rate']}")
        
        if 'mc_samples' in config and isinstance(config['mc_samples'], int):
            if config['mc_samples'] < 1:
                self.errors.append(f"mc_samples must be at least 1, got {config['mc_samples']}")
        
        if 'beam_width' in config and isinstance(config['beam_width'], int):
            if config['beam_width'] < 1:
                self.errors.append(f"beam_width must be at least 1, got {config['beam_width']}")
        
        # Check for positive integer values (only if type is correct)
        positive_int_keys = {
            'symptom_vocab_size', 'icd_vocab_size', 'symptom_embed_dim', 
            'icd_embed_dim', 'encoder_hidden_dim', 'latent_dim', 
            'decoder_hidden_dim', 'max_symptom_length', 'n_ensemble_models'
        }
        for key in positive_int_keys:
            if key in config and isinstance(config[key], int) and config[key] <= 0:
                self.errors.append(f"'{key}' must be positive, got {config[key]}")
        
        # Report results
        if self.warnings:
            print("Configuration warnings:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if self.errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in self.errors)
            raise ValueError(error_msg)
        
        return config
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get a default configuration dictionary.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'symptom_vocab_size': 1000,
            'icd_vocab_size': 500,
            'symptom_embed_dim': 128,
            'icd_embed_dim': 128,
            'encoder_hidden_dim': 256,
            'latent_dim': 64,
            'decoder_hidden_dim': 256,
            'max_symptom_length': 50,
            'dropout_rate': 0.2,
            'mc_samples': 20,
            'temporal_encoding_type': 'positional',
            'temperature': 1.0,
            'beam_width': 5,
            'n_ensemble_models': 5,
        }


def validate_config_file(config_path: str, strict: bool = True) -> Dict[str, Any]:
    """
    Convenience function to validate a configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        strict: If True, raise errors for unknown keys
        
    Returns:
        Validated configuration dictionary
    """
    validator = ConfigValidator(strict=strict)
    return validator.validate_config_file(config_path)
