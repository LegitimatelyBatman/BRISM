"""Unit tests for bug fixes and improvements."""

import unittest
import torch
import warnings
from brism import BRISM, BRISMConfig, BRISMLoss
from brism.ensemble import BRISMEnsemble
from brism.temporal import TemporalEncoding
from brism.interpretability import IntegratedGradients
from brism.icd_hierarchy import ICDHierarchy


class TestLossComponentBugFixes(unittest.TestCase):
    """Test bug fixes in loss.py"""
    
    def test_suggest_weights_uses_vocab_size_ratio(self):
        """Test that suggest_weights properly uses vocab_size_ratio."""
        # Test with different vocab_size_ratios
        weights_ratio_1 = BRISMLoss.suggest_weights(vocab_size_ratio=1.0)
        weights_ratio_2 = BRISMLoss.suggest_weights(vocab_size_ratio=2.0)
        weights_ratio_4 = BRISMLoss.suggest_weights(vocab_size_ratio=4.0)
        
        # Hierarchical weight should decrease as vocab_size_ratio increases
        self.assertGreater(
            weights_ratio_1['hierarchical_weight'],
            weights_ratio_2['hierarchical_weight'],
            "Hierarchical weight should decrease with larger vocab_size_ratio"
        )
        self.assertGreater(
            weights_ratio_2['hierarchical_weight'],
            weights_ratio_4['hierarchical_weight'],
            "Hierarchical weight should decrease with larger vocab_size_ratio"
        )
    
    def test_distance_matrix_device_warning(self):
        """Test that device change warning is issued."""
        # Create a loss function with hierarchy
        icd_vocab_size = 3
        hierarchy = ICDHierarchy(icd_vocab_size)
        # Build hierarchy from mapping to initialize distance matrix
        hierarchy.build_from_mapping({
            0: 'A00',
            1: 'A01',
            2: 'B00'
        })
        loss_fn = BRISMLoss(
            kl_weight=0.1,
            cycle_weight=1.0,
            hierarchical_weight=0.3,
            icd_hierarchy=hierarchy
        )
        
        # First call - no warning (creating matrix)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            matrix1 = loss_fn._ensure_distance_matrix_device(torch.device('cpu'))
            self.assertEqual(len(w), 0, "No warning on first device access")
        
        # Second call with same device - no warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            matrix2 = loss_fn._ensure_distance_matrix_device(torch.device('cpu'))
            self.assertEqual(len(w), 0, "No warning when device unchanged")
        
        # Third call with different device - should warn
        if torch.cuda.is_available():
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                matrix3 = loss_fn._ensure_distance_matrix_device(torch.device('cuda'))
                self.assertGreater(len(w), 0, "Warning should be issued on device change")
                self.assertIn("Distance matrix device changed", str(w[0].message))
    
    def test_contrastive_loss_single_label(self):
        """Test that contrastive loss returns zero when all samples have same label."""
        config = BRISMConfig(symptom_vocab_size=100, icd_vocab_size=50, latent_dim=32)
        loss_fn = BRISMLoss(contrastive_weight=0.5)
        
        batch_size = 4
        latents = torch.randn(batch_size, 32)
        
        # All samples have the same label
        labels = torch.tensor([5, 5, 5, 5])
        
        # Should return zero loss (no negative pairs)
        loss = loss_fn.contrastive_loss_infonce(latents, labels)
        self.assertEqual(loss.item(), 0.0, "Contrastive loss should be zero with single label")
    
    def test_contrastive_loss_batch_size_one(self):
        """Test that contrastive loss returns zero for batch_size < 2."""
        loss_fn = BRISMLoss(contrastive_weight=0.5)
        
        # Batch size of 1
        latents = torch.randn(1, 32)
        labels = torch.tensor([5])
        
        loss = loss_fn.contrastive_loss_infonce(latents, labels)
        self.assertEqual(loss.item(), 0.0, "Contrastive loss should be zero for batch_size=1")


class TestModelBugFixes(unittest.TestCase):
    """Test bug fixes in model.py"""
    
    def test_beam_search_memory_limit(self):
        """Test that beam_search raises error when memory requirements are too large."""
        config = BRISMConfig(symptom_vocab_size=100, icd_vocab_size=50, latent_dim=32)
        model = BRISM(config)
        
        z = torch.randn(1, 32)
        
        # Should work with reasonable parameters
        sequences, scores, lengths = model.symptom_decoder.beam_search(
            z, beam_width=5, max_length=50
        )
        self.assertEqual(sequences.shape[1], 5, "Beam width should be 5")
        
        # Should raise error with excessive parameters
        with self.assertRaises(ValueError) as context:
            model.symptom_decoder.beam_search(
                z, beam_width=1000, max_length=1000
            )
        self.assertIn("memory requirements too large", str(context.exception).lower())


class TestTemporalBugFixes(unittest.TestCase):
    """Test bug fixes in temporal.py"""
    
    def test_timestamp_overflow_warning(self):
        """Test that warning is issued for very large timestamp values."""
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


class TestInterpretabilityBugFixes(unittest.TestCase):
    """Test bug fixes in interpretability.py"""
    
    def test_integrated_gradients_numerical_stability(self):
        """Test that integrated gradients warns when input equals baseline."""
        config = BRISMConfig(symptom_vocab_size=100, icd_vocab_size=50, latent_dim=32)
        model = BRISM(config)
        ig = IntegratedGradients(model)
        
        # Input identical to baseline
        symptoms = torch.zeros(5, dtype=torch.long)
        baseline = torch.zeros(5, dtype=torch.long)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            attributions = ig.attribute(symptoms, baseline=baseline)
            self.assertGreater(len(w), 0, "Warning should be issued when input equals baseline")
            self.assertIn("nearly identical", str(w[0].message).lower())
            # Should return zeros
            self.assertTrue(torch.allclose(attributions, torch.zeros_like(attributions)))


class TestEnsembleBugFixes(unittest.TestCase):
    """Test bug fixes in ensemble.py"""
    
    def test_pseudo_ensemble_minimum_n_models(self):
        """Test that pseudo-ensemble requires n_models >= 2."""
        config = BRISMConfig(symptom_vocab_size=100, icd_vocab_size=50, latent_dim=32)
        
        # Should work with n_models >= 2
        ensemble = BRISMEnsemble(
            config=config,
            n_models=2,
            use_pseudo_ensemble=True
        )
        self.assertEqual(ensemble.n_models, 2)
        
        # Should raise error with n_models < 2
        with self.assertRaises(ValueError) as context:
            BRISMEnsemble(
                config=config,
                n_models=1,
                use_pseudo_ensemble=True
            )
        self.assertIn("n_models >= 2", str(context.exception))
        self.assertIn("standard deviation will always be zero", str(context.exception))


if __name__ == '__main__':
    unittest.main()
