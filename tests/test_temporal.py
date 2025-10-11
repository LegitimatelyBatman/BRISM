"""Unit tests for temporal encoding."""

import unittest
import torch
from brism.temporal import TemporalEncoding, TemporalSymptomEncoder


class TestTemporalEncoding(unittest.TestCase):
    """Test temporal encoding layers."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.seq_len = 10
        self.embed_dim = 64
    
    def test_positional_encoding(self):
        """Test positional encoding."""
        encoder = TemporalEncoding(
            embed_dim=self.embed_dim,
            max_length=20,
            encoding_type='positional'
        )
        
        embeddings = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        output = encoder(embeddings)
        
        # Check shape
        self.assertEqual(output.shape, embeddings.shape)
        
        # Check that output differs from input
        self.assertFalse(torch.allclose(output, embeddings))
    
    def test_timestamp_encoding(self):
        """Test timestamp encoding."""
        encoder = TemporalEncoding(
            embed_dim=self.embed_dim,
            max_length=20,
            encoding_type='timestamp'
        )
        
        embeddings = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        timestamps = torch.randn(self.batch_size, self.seq_len)
        
        output = encoder(embeddings, timestamps)
        
        # Check shape
        self.assertEqual(output.shape, embeddings.shape)
        
        # Check that output differs from input
        self.assertFalse(torch.allclose(output, embeddings))
    
    def test_timestamp_encoding_without_timestamps(self):
        """Test timestamp encoding defaults to positional when no timestamps provided."""
        encoder = TemporalEncoding(
            embed_dim=self.embed_dim,
            max_length=20,
            encoding_type='timestamp'
        )
        
        embeddings = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        output = encoder(embeddings, timestamps=None)
        
        # Should still work and return same shape
        self.assertEqual(output.shape, embeddings.shape)
    
    def test_temporal_symptom_encoder(self):
        """Test temporal symptom encoder."""
        vocab_size = 100
        encoder = TemporalSymptomEncoder(
            vocab_size=vocab_size,
            embed_dim=64,
            hidden_dim=128,
            latent_dim=32,
            max_length=20,
            encoding_type='positional',
            use_lstm=False
        )
        
        symptom_ids = torch.randint(0, vocab_size, (self.batch_size, self.seq_len))
        mu, logvar = encoder(symptom_ids)
        
        # Check shapes
        self.assertEqual(mu.shape, (self.batch_size, 32))
        self.assertEqual(logvar.shape, (self.batch_size, 32))
    
    def test_temporal_symptom_encoder_with_timestamps(self):
        """Test temporal symptom encoder with timestamps."""
        vocab_size = 100
        encoder = TemporalSymptomEncoder(
            vocab_size=vocab_size,
            embed_dim=64,
            hidden_dim=128,
            latent_dim=32,
            max_length=20,
            encoding_type='timestamp',
            use_lstm=True
        )
        
        symptom_ids = torch.randint(0, vocab_size, (self.batch_size, self.seq_len))
        timestamps = torch.randn(self.batch_size, self.seq_len)
        
        mu, logvar = encoder(symptom_ids, timestamps)
        
        # Check shapes
        self.assertEqual(mu.shape, (self.batch_size, 32))
        self.assertEqual(logvar.shape, (self.batch_size, 32))


class TestTemporalBugFixes(unittest.TestCase):
    """Test bug fixes in temporal.py"""
    
    def test_timestamp_overflow_warning(self):
        """Test that warning is issued for very large timestamp values."""
        import warnings
        from brism.temporal import TemporalEncoding
        
        encoder = TemporalEncoding(
            embed_dim=64,
            max_length=20,
            encoding_type='timestamp'
        )
        
        embeddings = torch.randn(2, 10, 64)
        
        # Normal timestamps - no warning
        normal_timestamps = torch.randn(2, 10)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output = encoder(embeddings, normal_timestamps)
            self.assertEqual(len(w), 0, "No warning for normal timestamps")
        
        # Very large timestamps - should warn
        large_timestamps = torch.ones(2, 10) * 1e7
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output = encoder(embeddings, large_timestamps)
            self.assertGreater(len(w), 0, "Warning should be issued for large timestamps")
            self.assertIn("Timestamp values are very large", str(w[0].message))


if __name__ == '__main__':
    unittest.main()
