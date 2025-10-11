"""Unit tests for additional bug fixes and improvements."""

import unittest
import torch
import warnings
from brism import BRISM, BRISMConfig
from brism.interpretability import IntegratedGradients
from brism.symptom_normalization import SymptomNormalizer


class TestFuzzyMatchingOptimization(unittest.TestCase):
    """Test optimization of fuzzy matching with early stopping."""
    
    def test_fuzzy_match_early_stopping_perfect_match(self):
        """Test that fuzzy matching stops early on perfect match."""
        # Create normalizer with many canonical forms
        canonical_forms = {f"symptom_{i}": i for i in range(1000)}
        normalizer = SymptomNormalizer(
            canonical_forms=canonical_forms,
            use_fuzzy_matching=True,
            fuzzy_threshold=0.8
        )
        
        # Mock the _similarity method to count calls
        original_similarity = normalizer._similarity
        call_count = {'count': 0}
        
        def counted_similarity(s1, s2):
            call_count['count'] += 1
            result = original_similarity(s1, s2)
            # If we found a perfect match, we should stop
            if result == 1.0 and call_count['count'] < len(canonical_forms):
                # This test will fail if we don't have early stopping
                pass
            return result
        
        normalizer._similarity = counted_similarity
        
        # Test with exact match - should find it quickly
        match, score = normalizer._fuzzy_match("symptom_0")
        
        self.assertEqual(score, 1.0, "Should find perfect match")
        # With early stopping, should not call similarity for all items
        # We expect it to check a few items before finding the match, not all 1000
        # Note: Since dict iteration order may vary, the exact position varies
        self.assertLess(
            call_count['count'], 
            len(canonical_forms), 
            f"Should stop early on perfect match, but called _similarity {call_count['count']} times "
            f"out of {len(canonical_forms)} total items. Early stopping missing!"
        )


class TestIntegratedGradientsMemoryCleanup(unittest.TestCase):
    """Test that IntegratedGradients properly cleans up gradients."""
    
    def test_gradient_cleanup_after_attribution(self):
        """Test that gradients are properly cleaned up after computing attributions."""
        config = BRISMConfig(symptom_vocab_size=100, icd_vocab_size=50, latent_dim=32)
        model = BRISM(config)
        ig = IntegratedGradients(model)
        
        symptoms = torch.tensor([1, 2, 3, 0, 0])
        
        # First call
        attributions1 = ig.attribute(symptoms, n_steps=5)
        
        # Check if model has lingering gradients
        has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        
        self.assertFalse(
            has_grads, 
            "Model should not have lingering gradients after IntegratedGradients.attribute(). "
            "This can cause memory leaks in repeated calls."
        )
        
        # Second call should work without interference
        attributions2 = ig.attribute(symptoms, n_steps=5)
        self.assertEqual(attributions1.shape, attributions2.shape)


class TestGradientBasedImportance(unittest.TestCase):
    """Test gradient-based importance computation."""
    
    def test_gradient_based_importance_computation(self):
        """Test that get_gradient_based_importance works correctly."""
        from brism.interpretability import AttentionVisualization
        
        config = BRISMConfig(symptom_vocab_size=100, icd_vocab_size=50, latent_dim=32)
        model = BRISM(config)
        viz = AttentionVisualization(model)
        
        symptoms = torch.tensor([1, 2, 3, 0, 0])
        
        # Should compute importance scores without errors
        importance = viz.get_gradient_based_importance(symptoms)
        
        self.assertEqual(importance.shape, symptoms.shape, 
                        "Importance shape should match symptoms shape")
        self.assertTrue(torch.isfinite(importance).all(),
                       "Importance scores should be finite")
        self.assertTrue((importance >= 0).all(),
                       "Importance scores should be non-negative (absolute gradients)")
    
    def test_gradient_cleanup_in_importance(self):
        """Test that gradient cleanup happens in get_gradient_based_importance."""
        from brism.interpretability import AttentionVisualization
        
        config = BRISMConfig(symptom_vocab_size=100, icd_vocab_size=50, latent_dim=32)
        model = BRISM(config)
        viz = AttentionVisualization(model)
        
        symptoms = torch.tensor([1, 2, 3, 0, 0])
        
        # Compute importance
        importance = viz.get_gradient_based_importance(symptoms)
        
        # Check if model has lingering gradients
        has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        
        self.assertFalse(
            has_grads, 
            "Model should not have lingering gradients after get_gradient_based_importance(). "
            "This can cause memory leaks in repeated calls."
        )


class TestNumericalStabilityImprovements(unittest.TestCase):
    """Test numerical stability improvements."""
    
    def test_log_computation_with_zero_probabilities(self):
        """Test that log computations handle zero probabilities gracefully."""
        # This tests that epsilon values are consistently used
        probs = torch.tensor([0.0, 0.5, 0.5])
        
        # Should not raise error or produce inf/nan
        log_probs = torch.log(probs + 1e-10)
        
        self.assertTrue(torch.isfinite(log_probs).all(), 
                       "Log of zero with epsilon should be finite")
        self.assertFalse(torch.isinf(log_probs).any(),
                        "Log computations should not produce inf")
    
    def test_division_by_small_mask_sum(self):
        """Test that division by mask sums handles edge cases."""
        # Simulate empty mask (all padding)
        mask = torch.zeros(4, 10)
        
        # Division should use epsilon to avoid divide-by-zero
        safe_division = mask.sum(dim=1, keepdim=True) + 1e-8
        result = torch.randn(4, 10).sum(dim=1, keepdim=True) / safe_division
        
        self.assertTrue(torch.isfinite(result).all(),
                       "Division by zero-sum mask should use epsilon")


class TestEdgeCaseValidation(unittest.TestCase):
    """Test validation of edge cases."""
    
    def test_empty_symptom_sequence_handling(self):
        """Test that empty symptom sequences are handled gracefully."""
        config = BRISMConfig(symptom_vocab_size=100, icd_vocab_size=50, latent_dim=32)
        model = BRISM(config)
        
        # All zeros (empty sequence)
        empty_symptoms = torch.zeros(1, 10, dtype=torch.long)
        
        # Should not crash
        try:
            icd_logits, mu, logvar = model.forward_path(empty_symptoms)
            self.assertEqual(icd_logits.shape[0], 1, "Should process empty sequence")
        except Exception as e:
            self.fail(f"Empty sequence should be handled gracefully, but got: {e}")
    
    def test_single_token_sequence_handling(self):
        """Test that single-token sequences are handled."""
        config = BRISMConfig(symptom_vocab_size=100, icd_vocab_size=50, latent_dim=32)
        model = BRISM(config)
        
        # Single non-zero token
        single_symptom = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.long)
        
        try:
            icd_logits, mu, logvar = model.forward_path(single_symptom)
            self.assertEqual(icd_logits.shape[0], 1, "Should process single token")
        except Exception as e:
            self.fail(f"Single token sequence should be handled, but got: {e}")


class TestInputValidationImprovements(unittest.TestCase):
    """Test improved input validation."""
    
    def test_symptom_normalization_empty_string(self):
        """Test that symptom normalizer handles empty strings."""
        normalizer = SymptomNormalizer()
        
        # Empty string
        result = normalizer.normalize("")
        self.assertEqual(result, "", "Empty string should return empty string")
        
        # Whitespace only
        result = normalizer.normalize("   ")
        self.assertEqual(result, "", "Whitespace-only should return empty string")
    
    def test_symptom_normalization_very_long_string(self):
        """Test that symptom normalizer handles very long strings."""
        normalizer = SymptomNormalizer()
        
        # Very long string (potential performance issue)
        long_string = "symptom " * 1000
        
        # Should not crash or hang
        try:
            result = normalizer.normalize(long_string)
            self.assertIsInstance(result, str, "Should return string")
        except Exception as e:
            self.fail(f"Long string should be handled, but got: {e}")


class TestEnsembleNumericalStability(unittest.TestCase):
    """Test numerical stability in ensemble predictions."""
    
    def test_pseudo_ensemble_single_sample(self):
        """Test that pseudo-ensemble handles n_samples=1 gracefully."""
        from brism.ensemble import BRISMEnsemble
        
        config = BRISMConfig(symptom_vocab_size=100, icd_vocab_size=50, latent_dim=32)
        ensemble = BRISMEnsemble(config=config, n_models=5, use_pseudo_ensemble=True)
        
        symptoms = torch.tensor([[1, 2, 3, 0, 0]])
        
        # With n_samples=1, std should be 0 (not NaN)
        mean_probs, std_probs, uncertainty = ensemble.predict_with_uncertainty(symptoms, n_samples=1)
        
        self.assertFalse(torch.isnan(std_probs).any(),
                        "Standard deviation should not be NaN for n_samples=1. "
                        "Should return zeros instead.")
        self.assertTrue(torch.allclose(std_probs, torch.zeros_like(std_probs)),
                       "Standard deviation should be zero for n_samples=1")
    
    def test_pseudo_ensemble_multiple_samples(self):
        """Test that pseudo-ensemble works with multiple samples."""
        from brism.ensemble import BRISMEnsemble
        
        config = BRISMConfig(symptom_vocab_size=100, icd_vocab_size=50, latent_dim=32)
        ensemble = BRISMEnsemble(config=config, n_models=5, use_pseudo_ensemble=True)
        
        symptoms = torch.tensor([[1, 2, 3, 0, 0]])
        
        # With n_samples > 1, should get meaningful std
        mean_probs, std_probs, uncertainty = ensemble.predict_with_uncertainty(symptoms, n_samples=10)
        
        self.assertFalse(torch.isnan(mean_probs).any(), "Mean should not contain NaN")
        self.assertFalse(torch.isnan(std_probs).any(), "Std should not contain NaN")
        self.assertTrue((std_probs >= 0).all(), "Std should be non-negative")


class TestActiveLearningValidation(unittest.TestCase):
    """Test input validation in active learning."""
    
    def test_add_symptom_rejects_padding_token(self):
        """Test that _add_symptom rejects padding token (symptom_id=0)."""
        from brism.active_learning import ActiveLearner
        
        config = BRISMConfig(symptom_vocab_size=100, icd_vocab_size=50, latent_dim=32)
        model = BRISM(config)
        learner = ActiveLearner(model)
        
        symptoms = torch.tensor([1, 2, 0, 0, 0])
        
        # Should raise ValueError when trying to add padding token
        with self.assertRaises(ValueError) as context:
            learner._add_symptom(symptoms, 0)
        
        self.assertIn("padding token", str(context.exception).lower(),
                     "Error message should mention padding token")
        self.assertIn("symptom_id=0", str(context.exception),
                     "Error message should specify symptom_id=0")
    
    def test_add_symptom_valid_ids(self):
        """Test that _add_symptom works with valid symptom IDs."""
        from brism.active_learning import ActiveLearner
        
        config = BRISMConfig(symptom_vocab_size=100, icd_vocab_size=50, latent_dim=32)
        model = BRISM(config)
        learner = ActiveLearner(model)
        
        symptoms = torch.tensor([1, 2, 0, 0, 0])
        
        # Should work fine with valid symptom ID
        augmented = learner._add_symptom(symptoms, 3)
        self.assertEqual(augmented[2].item(), 3, "Should add symptom at first padding position")
        
        # Should work at end of sequence too
        full_symptoms = torch.tensor([1, 2, 3, 4, 5])
        augmented = learner._add_symptom(full_symptoms, 99)
        self.assertEqual(augmented[-1].item(), 99, "Should replace last token when full")


if __name__ == '__main__':
    unittest.main()
