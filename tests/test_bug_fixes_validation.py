"""Unit tests to validate the implemented bug fixes and improvements."""

import unittest
import torch
from brism import BRISM, BRISMConfig
from brism.data_loader import ICDNormalizer, MedicalDataPreprocessor
from brism.icd_hierarchy import ICDHierarchy


class TestMemoryLeakFixes(unittest.TestCase):
    """Test that memory leak fixes are working."""
    
    def test_interpretability_gradient_cleanup(self):
        """Test that interpretability functions clean up gradients properly."""
        from brism.interpretability import IntegratedGradients, AttentionVisualization
        
        config = BRISMConfig(symptom_vocab_size=100, icd_vocab_size=50, latent_dim=32)
        model = BRISM(config)
        
        # Test IntegratedGradients
        ig = IntegratedGradients(model)
        symptoms = torch.tensor([1, 2, 3, 0, 0])
        
        attributions = ig.attribute(symptoms, n_steps=5)
        
        # Check no lingering gradients
        has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        self.assertFalse(has_grads, "IntegratedGradients should clean up gradients")
        
        # Test AttentionVisualization
        viz = AttentionVisualization(model)
        importance = viz.get_gradient_based_importance(symptoms)
        
        has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        self.assertFalse(has_grads, "AttentionVisualization should clean up gradients")


class TestExceptionSafety(unittest.TestCase):
    """Test that functions properly handle exceptions."""
    
    def test_predict_with_uncertainty_exception_safety(self):
        """Test that predict_with_uncertainty restores training mode on exception."""
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


class TestInputValidation(unittest.TestCase):
    """Test improved input validation."""
    
    def test_normalize_icd10_empty_string(self):
        """Test that normalize_icd10 rejects empty strings."""
        normalizer = ICDNormalizer()
        
        with self.assertRaises(ValueError) as context:
            normalizer.normalize_icd10("")
        
        self.assertIn("empty", str(context.exception).lower())
    
    def test_normalize_icd10_whitespace_only(self):
        """Test that normalize_icd10 rejects whitespace-only strings."""
        normalizer = ICDNormalizer()
        
        with self.assertRaises(ValueError) as context:
            normalizer.normalize_icd10("   ")
        
        self.assertIn("empty", str(context.exception).lower())
    
    def test_encode_icd_handles_empty_gracefully(self):
        """Test that encode_icd handles empty ICD codes gracefully."""
        preprocessor = MedicalDataPreprocessor(
            symptom_vocab={'fever': 1, 'cough': 2},
            icd_vocab={'E11.9': 0, 'I10': 1}
        )
        
        # Should return None, not raise exception
        result = preprocessor.encode_icd("")
        self.assertIsNone(result, "Empty ICD code should return None")
        
        result = preprocessor.encode_icd("   ")
        self.assertIsNone(result, "Whitespace ICD code should return None")
    
    def test_symptom_normalizer_empty_string(self):
        """Test that symptom normalizer handles empty strings."""
        from brism.symptom_normalization import SymptomNormalizer
        
        normalizer = SymptomNormalizer()
        
        # Should return empty string, not crash
        result = normalizer.normalize("")
        self.assertEqual(result, "", "Empty string should return empty string")
        
        result = normalizer.normalize("   ")
        self.assertEqual(result, "", "Whitespace should return empty string")


class TestIcdHierarchyRobustness(unittest.TestCase):
    """Test ICD hierarchy handling of edge cases."""
    
    def test_compute_tree_distance_empty_codes(self):
        """Test that tree distance handles empty codes."""
        hierarchy = ICDHierarchy(icd_vocab_size=10)
        
        # Empty codes should get maximum distance
        distance = hierarchy._compute_tree_distance("", "E11.9")
        self.assertEqual(distance, 4.0, "Empty code should have max distance")
        
        distance = hierarchy._compute_tree_distance("E11.9", "")
        self.assertEqual(distance, 4.0, "Empty code should have max distance")
        
        distance = hierarchy._compute_tree_distance("", "")
        self.assertEqual(distance, 0.0, "Two empty codes should have distance 0")
    
    def test_compute_tree_distance_short_codes(self):
        """Test that tree distance handles very short codes."""
        hierarchy = ICDHierarchy(icd_vocab_size=10)
        
        # Short codes should work
        distance = hierarchy._compute_tree_distance("E", "E11.9")
        self.assertIsInstance(distance, float, "Should return float for short codes")
        self.assertGreaterEqual(distance, 0.0, "Distance should be non-negative")
        self.assertLessEqual(distance, 4.0, "Distance should not exceed maximum")


class TestCalibrationValidation(unittest.TestCase):
    """Test calibration function validation."""
    
    def test_calibrate_temperature_empty_loader(self):
        """Test that calibrate_temperature validates empty loader."""
        from brism.calibration import calibrate_temperature
        from torch.utils.data import DataLoader, TensorDataset
        
        config = BRISMConfig(symptom_vocab_size=100, icd_vocab_size=50, latent_dim=32)
        model = BRISM(config)
        device = torch.device('cpu')
        
        # Create empty dataloader
        empty_dataset = TensorDataset(torch.tensor([]), torch.tensor([]))
        empty_loader = DataLoader(empty_dataset, batch_size=1)
        
        with self.assertRaises(ValueError) as context:
            calibrate_temperature(model, empty_loader, device)
        
        self.assertIn("empty", str(context.exception).lower())


if __name__ == '__main__':
    unittest.main()
