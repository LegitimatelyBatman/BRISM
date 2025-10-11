"""Unit tests for BRISM model components."""

import unittest
import torch
import numpy as np
from brism.model import BRISM, BRISMConfig, Encoder, Decoder, SequenceDecoder


class TestModelComponents(unittest.TestCase):
    """Test individual model components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.seq_len = 10
        self.latent_dim = 32
        self.hidden_dim = 64
        self.input_dim = 128
        self.output_dim = 50
        
    def test_encoder(self):
        """Test encoder forward pass."""
        encoder = Encoder(self.input_dim, self.hidden_dim, self.latent_dim)
        x = torch.randn(self.batch_size, self.input_dim)
        
        mu, logvar = encoder(x)
        
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))
        
    def test_decoder(self):
        """Test decoder forward pass."""
        decoder = Decoder(self.latent_dim, self.hidden_dim, self.output_dim)
        z = torch.randn(self.batch_size, self.latent_dim)
        
        output = decoder(z)
        
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        
    def test_sequence_decoder(self):
        """Test sequence decoder forward pass."""
        vocab_size = 100
        max_length = 20
        
        seq_decoder = SequenceDecoder(
            self.latent_dim, self.hidden_dim, vocab_size, max_length
        )
        z = torch.randn(self.batch_size, self.latent_dim)
        
        # Test with teacher forcing
        target = torch.randint(0, vocab_size, (self.batch_size, max_length))
        output = seq_decoder(z, target)
        
        self.assertEqual(output.shape, (self.batch_size, max_length, vocab_size))
        
        # Test autoregressive generation
        output_auto = seq_decoder(z, None)
        self.assertEqual(output_auto.shape, (self.batch_size, max_length, vocab_size))


class TestBRISMModel(unittest.TestCase):
    """Test BRISM model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BRISMConfig(
            symptom_vocab_size=100,
            icd_vocab_size=50,
            symptom_embed_dim=64,
            icd_embed_dim=64,
            encoder_hidden_dim=128,
            latent_dim=32,
            decoder_hidden_dim=128,
            max_symptom_length=20,
            dropout_rate=0.1,
            mc_samples=10
        )
        self.model = BRISM(self.config)
        self.batch_size = 4
        
    def test_reparameterization(self):
        """Test reparameterization trick."""
        mu = torch.randn(self.batch_size, self.config.latent_dim)
        logvar = torch.randn(self.batch_size, self.config.latent_dim)
        
        z = self.model.reparameterize(mu, logvar)
        
        self.assertEqual(z.shape, (self.batch_size, self.config.latent_dim))
        
    def test_forward_path(self):
        """Test forward path: symptoms -> ICD."""
        symptoms = torch.randint(
            0, self.config.symptom_vocab_size,
            (self.batch_size, self.config.max_symptom_length)
        )

        icd_logits, mu, logvar = self.model.forward_path(symptoms)

        self.assertEqual(icd_logits.shape, (self.batch_size, self.config.icd_vocab_size))
        self.assertEqual(mu.shape, (self.batch_size, self.config.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.config.latent_dim))

    def test_forward_method(self):
        """The module forward should mirror forward_path outputs."""
        symptoms = torch.randint(
            0, self.config.symptom_vocab_size,
            (self.batch_size, self.config.max_symptom_length)
        )
        timestamps = torch.rand(self.batch_size, self.config.max_symptom_length)

        self.model.eval()
        torch.manual_seed(42)
        probs, latent = self.model(symptoms, timestamps=timestamps)

        self.assertEqual(probs.shape, (self.batch_size, self.config.icd_vocab_size))
        self.assertEqual(latent.shape, (self.batch_size, self.config.latent_dim))

        # Probabilities should sum to one per sample
        torch.testing.assert_close(
            probs.sum(dim=1),
            torch.ones(self.batch_size),
            atol=1e-5,
            rtol=1e-5,
        )

        # Forward method should match softmax(logits) and latent mean from forward_path
        torch.manual_seed(42)
        icd_logits, mu, _ = self.model.forward_path(symptoms, timestamps)
        torch.testing.assert_close(probs, torch.softmax(icd_logits, dim=-1))
        torch.testing.assert_close(latent, mu)
        
    def test_reverse_path(self):
        """Test reverse path: ICD -> symptoms."""
        icd_codes = torch.randint(0, self.config.icd_vocab_size, (self.batch_size,))
        target_symptoms = torch.randint(
            0, self.config.symptom_vocab_size,
            (self.batch_size, self.config.max_symptom_length)
        )
        
        symptom_logits, mu, logvar = self.model.reverse_path(icd_codes, target_symptoms)
        
        self.assertEqual(
            symptom_logits.shape,
            (self.batch_size, self.config.max_symptom_length, self.config.symptom_vocab_size)
        )
        self.assertEqual(mu.shape, (self.batch_size, self.config.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.config.latent_dim))
        
    def test_cycle_forward(self):
        """Test forward cycle: symptoms -> ICD -> symptoms."""
        symptoms = torch.randint(
            0, self.config.symptom_vocab_size,
            (self.batch_size, self.config.max_symptom_length)
        )
        
        symptom_logits, icd_logits, mu1, logvar1, mu2, logvar2 = \
            self.model.cycle_forward(symptoms, symptoms)
        
        self.assertEqual(
            symptom_logits.shape,
            (self.batch_size, self.config.max_symptom_length, self.config.symptom_vocab_size)
        )
        self.assertEqual(icd_logits.shape, (self.batch_size, self.config.icd_vocab_size))
        self.assertEqual(mu1.shape, (self.batch_size, self.config.latent_dim))
        self.assertEqual(mu2.shape, (self.batch_size, self.config.latent_dim))
        
    def test_cycle_reverse(self):
        """Test reverse cycle: ICD -> symptoms -> ICD."""
        icd_codes = torch.randint(0, self.config.icd_vocab_size, (self.batch_size,))
        symptoms = torch.randint(
            0, self.config.symptom_vocab_size,
            (self.batch_size, self.config.max_symptom_length)
        )
        
        icd_logits, symptom_logits, mu1, logvar1, mu2, logvar2 = \
            self.model.cycle_reverse(icd_codes, symptoms)
        
        self.assertEqual(icd_logits.shape, (self.batch_size, self.config.icd_vocab_size))
        self.assertEqual(
            symptom_logits.shape,
            (self.batch_size, self.config.max_symptom_length, self.config.symptom_vocab_size)
        )
        
    def test_predict_with_uncertainty(self):
        """Test uncertainty estimation with Monte Carlo dropout."""
        symptoms = torch.randint(
            0, self.config.symptom_vocab_size,
            (self.batch_size, self.config.max_symptom_length)
        )
        
        mean_probs, std_probs = self.model.predict_with_uncertainty(symptoms, n_samples=5)
        
        self.assertEqual(mean_probs.shape, (self.batch_size, self.config.icd_vocab_size))
        self.assertEqual(std_probs.shape, (self.batch_size, self.config.icd_vocab_size))
        
        # Check probabilities sum to 1
        prob_sums = mean_probs.sum(dim=1)
        torch.testing.assert_close(prob_sums, torch.ones(self.batch_size), atol=1e-5, rtol=1e-5)
        
        # Check std is non-negative
        self.assertTrue((std_probs >= 0).all())


class TestSharedLatentSpace(unittest.TestCase):
    """Test that both encoders produce compatible latent representations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BRISMConfig(
            symptom_vocab_size=100,
            icd_vocab_size=50,
            latent_dim=32
        )
        self.model = BRISM(self.config)
        
    def test_latent_dimensions_match(self):
        """Test that both paths produce same latent dimensions."""
        batch_size = 4
        symptoms = torch.randint(0, self.config.symptom_vocab_size, (batch_size, 20))
        icd_codes = torch.randint(0, self.config.icd_vocab_size, (batch_size,))
        
        # Forward path
        _, mu1, logvar1 = self.model.forward_path(symptoms)
        
        # Reverse path
        _, mu2, logvar2 = self.model.reverse_path(icd_codes, symptoms)
        
        # Both should have same shape
        self.assertEqual(mu1.shape, mu2.shape)
        self.assertEqual(logvar1.shape, logvar2.shape)
        self.assertEqual(mu1.shape, (batch_size, self.config.latent_dim))


class TestModelBugFixes(unittest.TestCase):
    """Test bug fixes in model.py"""
    
    def test_beam_search_memory_limit(self):
        """Test that beam_search raises error when memory requirements are too large."""
        from brism import BRISM, BRISMConfig
        
        # Create config with larger vocab to make the limit easier to hit
        config = BRISMConfig(symptom_vocab_size=10000, icd_vocab_size=50, latent_dim=32)
        model = BRISM(config)
        
        z = torch.randn(1, 32)
        
        # Should work with reasonable parameters
        sequences, scores, lengths = model.symptom_decoder.beam_search(
            z, beam_width=5, max_length=50
        )
        self.assertEqual(sequences.shape[1], 5, "Beam width should be 5")
        
        # Should raise error with excessive parameters
        # With vocab_size=10000, max_length capped at 100 (2 * decoder.max_length):
        # beam_width * search_max_length * vocab_size = 1100 * 100 * 10000 = 1,100,000,000 > 100,000,000
        with self.assertRaises(ValueError) as context:
            model.symptom_decoder.beam_search(
                z, beam_width=1100, max_length=1000
            )
        self.assertIn("memory requirements too large", str(context.exception).lower())
    
    def test_predict_with_uncertainty_exception_safety(self):
        """Test that predict_with_uncertainty restores training mode on exception."""
        from brism import BRISM, BRISMConfig
        
        config = BRISMConfig(symptom_vocab_size=100, icd_vocab_size=50, latent_dim=32)
        model = BRISM(config)
        
        # Set to eval mode
        model.eval()
        initial_mode = model.training
        
        # Try with invalid input (should fail but restore mode)
        try:
            # This will fail due to shape mismatch, but should restore training mode
            invalid_symptoms = torch.tensor([])  # Empty tensor
            model.predict_with_uncertainty(invalid_symptoms, n_samples=5)
        except:
            pass
        
        # Check that training mode is still False (was restored)
        self.assertEqual(model.training, initial_mode, 
                        "Training mode should be restored even on exception")


if __name__ == '__main__':
    unittest.main()
