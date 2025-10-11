"""
Tests for configuration validator.
"""

import unittest
import tempfile
import os
import yaml
from brism.config_validator import ConfigValidator, validate_config_file


class TestConfigValidator(unittest.TestCase):
    """Test configuration validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = ConfigValidator(strict=True)
    
    def test_valid_config_dict(self):
        """Test that valid config dictionary passes."""
        config = {
            'symptom_vocab_size': 1000,
            'icd_vocab_size': 500,
            'latent_dim': 64,
        }
        result = self.validator.validate_config_dict(config)
        self.assertEqual(result['symptom_vocab_size'], 1000)
    
    def test_missing_required_keys(self):
        """Test that missing required keys raises error."""
        config = {
            'latent_dim': 64,
        }
        with self.assertRaises(ValueError) as context:
            self.validator.validate_config_dict(config)
        self.assertIn("Missing required keys", str(context.exception))
    
    def test_invalid_type(self):
        """Test that invalid type raises error."""
        config = {
            'symptom_vocab_size': 1000,
            'icd_vocab_size': 500,
            'dropout_rate': "not a number",
        }
        with self.assertRaises(ValueError) as context:
            self.validator.validate_config_dict(config)
        self.assertIn("wrong type", str(context.exception))
    
    def test_invalid_value(self):
        """Test that invalid value raises error."""
        config = {
            'symptom_vocab_size': 1000,
            'icd_vocab_size': 500,
            'temporal_encoding_type': 'invalid_type',
        }
        with self.assertRaises(ValueError) as context:
            self.validator.validate_config_dict(config)
        self.assertIn("invalid value", str(context.exception))
    
    def test_dropout_rate_out_of_range(self):
        """Test that dropout_rate out of range raises error."""
        config = {
            'symptom_vocab_size': 1000,
            'icd_vocab_size': 500,
            'dropout_rate': 1.5,
        }
        with self.assertRaises(ValueError) as context:
            self.validator.validate_config_dict(config)
        self.assertIn("dropout_rate must be between 0 and 1", str(context.exception))
    
    def test_negative_value(self):
        """Test that negative values raise error."""
        config = {
            'symptom_vocab_size': -100,
            'icd_vocab_size': 500,
        }
        with self.assertRaises(ValueError) as context:
            self.validator.validate_config_dict(config)
        self.assertIn("must be positive", str(context.exception))
    
    def test_common_typo_detection(self):
        """Test that common typos are detected."""
        config = {
            'symptom_vocab_size': 1000,
            'icd_vocab_size': 500,
            'dropout': 0.2,  # Common typo
        }
        with self.assertRaises(ValueError) as context:
            self.validator.validate_config_dict(config)
        self.assertIn("dropout", str(context.exception))
        self.assertIn("dropout_rate", str(context.exception))
    
    def test_non_strict_mode(self):
        """Test that non-strict mode only warns for unknown keys."""
        validator = ConfigValidator(strict=False)
        config = {
            'symptom_vocab_size': 1000,
            'icd_vocab_size': 500,
            'unknown_key': 'value',
        }
        # Should not raise, just warn
        result = validator.validate_config_dict(config)
        self.assertEqual(result['symptom_vocab_size'], 1000)
        self.assertTrue(len(validator.warnings) > 0)
    
    def test_valid_yaml_file(self):
        """Test loading valid YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.yaml')
            
            config = {
                'symptom_vocab_size': 1000,
                'icd_vocab_size': 500,
                'latent_dim': 64,
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            result = validate_config_file(config_path)
            self.assertEqual(result['symptom_vocab_size'], 1000)
    
    def test_missing_yaml_file(self):
        """Test that missing YAML file raises error."""
        with self.assertRaises(FileNotFoundError) as context:
            validate_config_file('nonexistent.yaml')
        self.assertIn("Configuration file not found", str(context.exception))
    
    def test_invalid_yaml_file(self):
        """Test that invalid YAML raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.yaml')
            
            with open(config_path, 'w') as f:
                f.write("invalid: yaml: content: [")
            
            with self.assertRaises(ValueError) as context:
                validate_config_file(config_path)
            self.assertIn("Failed to parse YAML", str(context.exception))
    
    def test_empty_yaml_file(self):
        """Test that empty YAML file raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.yaml')
            
            with open(config_path, 'w') as f:
                f.write("")
            
            with self.assertRaises(ValueError) as context:
                validate_config_file(config_path)
            self.assertIn("empty", str(context.exception))
    
    def test_get_default_config(self):
        """Test getting default configuration."""
        config = self.validator.get_default_config()
        
        # Should have all required keys
        self.assertIn('symptom_vocab_size', config)
        self.assertIn('icd_vocab_size', config)
        
        # Should be valid
        result = self.validator.validate_config_dict(config)
        self.assertEqual(result, config)
    
    def test_mc_samples_validation(self):
        """Test mc_samples validation."""
        config = {
            'symptom_vocab_size': 1000,
            'icd_vocab_size': 500,
            'mc_samples': 0,
        }
        with self.assertRaises(ValueError) as context:
            self.validator.validate_config_dict(config)
        self.assertIn("mc_samples must be at least 1", str(context.exception))
    
    def test_beam_width_validation(self):
        """Test beam_width validation."""
        config = {
            'symptom_vocab_size': 1000,
            'icd_vocab_size': 500,
            'beam_width': 0,
        }
        with self.assertRaises(ValueError) as context:
            self.validator.validate_config_dict(config)
        self.assertIn("beam_width must be at least 1", str(context.exception))


if __name__ == '__main__':
    unittest.main()
