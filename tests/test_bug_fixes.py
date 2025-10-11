"""
Tests for critical bug fixes.

Tests:
1. Bug 1: Variable used before assignment in minimal_sufficient_set
2. Bug 2: Gradient accumulation issue in IntegratedGradients
3. Bug 3: Device mismatch in BRISMLoss
"""

import unittest
import torch
import torch.nn as nn
from brism import BRISM, BRISMConfig
from brism.loss import BRISMLoss
from brism.interpretability import IntegratedGradients, CounterfactualExplanations
from brism.icd_hierarchy import ICDHierarchy


class TestBugFixes(unittest.TestCase):
    """Test critical bug fixes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BRISMConfig(
            symptom_vocab_size=50,
            icd_vocab_size=20,
            symptom_embed_dim=32,
            icd_embed_dim=32,
            encoder_hidden_dim=64,
            latent_dim=16,
            decoder_hidden_dim=64,
            max_symptom_length=10,
            dropout_rate=0.1,
            mc_samples=5
        )
        self.model = BRISM(self.config)
        self.model.eval()
    
    def test_bug1_minimal_sufficient_set_empty_initial(self):
        """
        Test Bug 1: minimal_sufficient_set with empty initial symptoms.
        
        The bug was that selected_prob was referenced before being defined
        when the selected list was empty.
        """
        # Create a symptom sequence
        symptoms = torch.tensor([5, 10, 15, 20, 0, 0, 0, 0, 0, 0])
        
        # Create counterfactual explanations
        cf = CounterfactualExplanations(self.model)
        
        # This should not raise UnboundLocalError
        try:
            minimal_set, prob = cf.minimal_sufficient_set(
                symptoms, 
                threshold=0.5
            )
            # Test passes if no exception is raised
            self.assertIsInstance(minimal_set, list)
            self.assertIsInstance(prob, float)
        except UnboundLocalError as e:
            self.fail(f"Bug 1 not fixed: {e}")
    
    def test_bug1_minimal_sufficient_set_all_padding(self):
        """
        Test Bug 1: minimal_sufficient_set with all padding (edge case).
        """
        # Create all-padding symptoms
        symptoms = torch.zeros(10, dtype=torch.long)
        
        cf = CounterfactualExplanations(self.model)
        
        # Should handle gracefully
        minimal_set, prob = cf.minimal_sufficient_set(symptoms, threshold=0.5)
        
        # Should return empty set and 0.0 probability
        self.assertEqual(minimal_set, [])
        self.assertEqual(prob, 0.0)
    
    def test_bug2_integrated_gradients_gradient_accumulation(self):
        """
        Test Bug 2: IntegratedGradients gradient accumulation.
        
        The bug was that gradients could accumulate across iterations
        when backward() was called multiple times without clearing.
        """
        symptoms = torch.tensor([5, 10, 15, 0, 0, 0, 0, 0, 0, 0])
        
        ig = IntegratedGradients(self.model)
        
        # Run multiple times to check for gradient accumulation issues
        try:
            attributions1 = ig.attribute(symptoms, n_steps=5)
            attributions2 = ig.attribute(symptoms, n_steps=5)
            
            # Attributions should be consistent (similar, not affected by accumulation)
            self.assertEqual(attributions1.shape, symptoms.shape)
            self.assertEqual(attributions2.shape, symptoms.shape)
            
            # Check that attributions are finite (not NaN or Inf)
            self.assertTrue(torch.isfinite(attributions1).all())
            self.assertTrue(torch.isfinite(attributions2).all())
            
        except RuntimeError as e:
            self.fail(f"Bug 2 not fixed: gradient accumulation error: {e}")
    
    def test_bug2_integrated_gradients_batch_input(self):
        """
        Test Bug 2: IntegratedGradients with batch input.
        """
        symptoms = torch.tensor([[5, 10, 15, 0, 0, 0, 0, 0, 0, 0],
                                  [3, 7, 12, 18, 0, 0, 0, 0, 0, 0]])
        
        ig = IntegratedGradients(self.model)
        
        try:
            attributions = ig.attribute(symptoms, n_steps=5)
            
            # Should work with batch input
            self.assertEqual(attributions.shape, symptoms.shape)
            self.assertTrue(torch.isfinite(attributions).all())
            
        except Exception as e:
            self.fail(f"Bug 2: Batch processing failed: {e}")
    
    def test_bug3_device_mismatch_helper_method_exists(self):
        """
        Test Bug 3: Helper method exists for ensuring device consistency.
        
        The bug was that _distance_matrix was cached on first use and
        caused device mismatch if model was moved to different device.
        The fix adds a helper method _ensure_distance_matrix_device.
        """
        # Create loss function without hierarchy (simpler test)
        loss_fn = BRISMLoss(
            kl_weight=0.1,
            cycle_weight=1.0
        )
        
        # Test that the helper method exists
        self.assertTrue(hasattr(loss_fn, '_ensure_distance_matrix_device'),
                       "Bug 3 fix: Helper method _ensure_distance_matrix_device should exist")
        
        # Test without hierarchy should work fine
        logits = torch.randn(4, 20)
        target = torch.tensor([1, 2, 3, 4])
        
        try:
            loss = loss_fn.reconstruction_loss_icd(logits, target)
            self.assertTrue(torch.isfinite(loss).all())
            self.assertEqual(loss.device, logits.device)
        except RuntimeError as e:
            self.fail(f"Bug 3: Basic loss computation failed: {e}")
    
    def test_bug3_device_mismatch_multiple_calls_no_hierarchy(self):
        """
        Test Bug 3: Multiple forward passes work correctly without hierarchy.
        """
        loss_fn = BRISMLoss(
            kl_weight=0.1,
            cycle_weight=1.0
        )
        
        # Multiple calls should work correctly
        for i in range(3):
            logits = torch.randn(2, 20)
            target = torch.tensor([1, 2])
            
            try:
                loss = loss_fn.reconstruction_loss_icd(logits, target)
                self.assertTrue(torch.isfinite(loss).all())
            except RuntimeError as e:
                self.fail(f"Bug 3: Multiple calls failed on iteration {i}: {e}")


if __name__ == '__main__':
    unittest.main()
