"""
Tests for BRISMConfig validation.
"""

import unittest
from brism.model import BRISMConfig


class TestConfigValidation(unittest.TestCase):
    """Test BRISMConfig validation."""
    
    def test_valid_config(self):
        """Test that valid config passes validation."""
        config = BRISMConfig(
            symptom_vocab_size=1000,
            icd_vocab_size=500,
            dropout_rate=0.2,
            mc_samples=20
        )
        self.assertEqual(config.symptom_vocab_size, 1000)
        self.assertEqual(config.icd_vocab_size, 500)
    
    def test_invalid_symptom_vocab_size(self):
        """Test that negative symptom_vocab_size raises error."""
        with self.assertRaises(ValueError) as context:
            BRISMConfig(symptom_vocab_size=0)
        self.assertIn("symptom_vocab_size must be positive", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            BRISMConfig(symptom_vocab_size=-1)
        self.assertIn("symptom_vocab_size must be positive", str(context.exception))
    
    def test_invalid_icd_vocab_size(self):
        """Test that non-positive icd_vocab_size raises error."""
        with self.assertRaises(ValueError) as context:
            BRISMConfig(icd_vocab_size=0)
        self.assertIn("icd_vocab_size must be positive", str(context.exception))
    
    def test_invalid_dropout_rate(self):
        """Test that dropout_rate outside [0, 1] raises error."""
        with self.assertRaises(ValueError) as context:
            BRISMConfig(dropout_rate=-0.1)
        self.assertIn("dropout_rate must be between 0 and 1", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            BRISMConfig(dropout_rate=1.5)
        self.assertIn("dropout_rate must be between 0 and 1", str(context.exception))
    
    def test_invalid_mc_samples(self):
        """Test that mc_samples less than 1 raises error."""
        with self.assertRaises(ValueError) as context:
            BRISMConfig(mc_samples=0)
        self.assertIn("mc_samples must be at least 1", str(context.exception))
    
    def test_invalid_dimensions(self):
        """Test that non-positive dimensions raise errors."""
        with self.assertRaises(ValueError) as context:
            BRISMConfig(latent_dim=0)
        self.assertIn("latent_dim must be positive", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            BRISMConfig(encoder_hidden_dim=-10)
        self.assertIn("encoder_hidden_dim must be positive", str(context.exception))
    
    def test_invalid_temporal_encoding_type(self):
        """Test that invalid temporal_encoding_type raises error."""
        with self.assertRaises(ValueError) as context:
            BRISMConfig(temporal_encoding_type='invalid')
        self.assertIn("temporal_encoding_type must be 'positional' or 'timestamp'", str(context.exception))
    
    def test_invalid_temperature(self):
        """Test that non-positive temperature raises error."""
        with self.assertRaises(ValueError) as context:
            BRISMConfig(temperature=0.0)
        self.assertIn("temperature must be positive", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            BRISMConfig(temperature=-1.0)
        self.assertIn("temperature must be positive", str(context.exception))
    
    def test_invalid_beam_width(self):
        """Test that beam_width less than 1 raises error."""
        with self.assertRaises(ValueError) as context:
            BRISMConfig(beam_width=0)
        self.assertIn("beam_width must be at least 1", str(context.exception))
    
    def test_invalid_n_ensemble_models(self):
        """Test that n_ensemble_models less than 1 raises error."""
        with self.assertRaises(ValueError) as context:
            BRISMConfig(n_ensemble_models=0)
        self.assertIn("n_ensemble_models must be at least 1", str(context.exception))
    
    def test_edge_case_valid_values(self):
        """Test edge cases with minimum valid values."""
        config = BRISMConfig(
            symptom_vocab_size=1,
            icd_vocab_size=1,
            dropout_rate=0.0,
            mc_samples=1,
            beam_width=1,
            n_ensemble_models=1
        )
        self.assertEqual(config.symptom_vocab_size, 1)
        self.assertEqual(config.dropout_rate, 0.0)
        
        config = BRISMConfig(dropout_rate=1.0)
        self.assertEqual(config.dropout_rate, 1.0)


if __name__ == '__main__':
    unittest.main()
