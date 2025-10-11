"""Unit tests for BRISM inference functions."""

import unittest
import torch
import numpy as np
from brism.model import BRISM, BRISMConfig
from brism.inference import (
    diagnose_with_confidence,
    generate_symptoms_with_uncertainty,
    evaluate_model_uncertainty
)
from torch.utils.data import Dataset, DataLoader


class DummyDataset(Dataset):
    """Dummy dataset for testing."""
    
    def __init__(self, num_samples=10):
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'symptoms': torch.randint(0, 100, (20,)),
            'icd_codes': torch.randint(0, 50, (1,)).squeeze()
        }


class TestInference(unittest.TestCase):
    """Test inference functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BRISMConfig(
            symptom_vocab_size=100,
            icd_vocab_size=50,
            latent_dim=32,
            mc_samples=10
        )
        self.model = BRISM(self.config)
        self.device = torch.device('cpu')
        
    def test_diagnose_with_confidence_single(self):
        """Test diagnosis with confidence intervals for single input."""
        symptoms = torch.randint(0, self.config.symptom_vocab_size, (20,))
        
        result = diagnose_with_confidence(
            model=self.model,
            symptoms=symptoms,
            device=self.device,
            n_samples=5,
            confidence_level=0.95,
            top_k=3
        )
        
        # Check structure
        self.assertIn('predictions', result)
        self.assertIn('uncertainty', result)
        self.assertIn('raw_probabilities', result)
        
        # Check predictions
        self.assertEqual(len(result['predictions']), 3)  # top_k=3
        
        for pred in result['predictions']:
            self.assertIn('icd_code', pred)
            self.assertIn('probability', pred)
            self.assertIn('std', pred)
            self.assertIn('confidence_interval', pred)
            
            # Check probability bounds
            self.assertGreaterEqual(pred['probability'], 0.0)
            self.assertLessEqual(pred['probability'], 1.0)
            
            # Check CI bounds
            ci_lower, ci_upper = pred['confidence_interval']
            self.assertGreaterEqual(ci_lower, 0.0)
            self.assertLessEqual(ci_upper, 1.0)
            self.assertLessEqual(ci_lower, ci_upper)
            
        # Check uncertainty metrics
        self.assertIn('entropy', result['uncertainty'])
        self.assertIn('average_std', result['uncertainty'])
        self.assertGreaterEqual(result['uncertainty']['entropy'], 0.0)
        self.assertGreaterEqual(result['uncertainty']['average_std'], 0.0)
        
    def test_diagnose_with_confidence_batch(self):
        """Test diagnosis with confidence intervals for batch input."""
        batch_size = 3
        symptoms = torch.randint(
            0, self.config.symptom_vocab_size,
            (batch_size, 20)
        )
        
        results = diagnose_with_confidence(
            model=self.model,
            symptoms=symptoms,
            device=self.device,
            n_samples=5,
            top_k=2
        )
        
        # Should return list for batch
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), batch_size)
        
        # Each result should have correct structure
        for result in results:
            self.assertIn('predictions', result)
            self.assertEqual(len(result['predictions']), 2)  # top_k=2
            
    def test_generate_symptoms_with_uncertainty(self):
        """Test symptom generation with uncertainty."""
        icd_code = torch.tensor(10)
        
        result = generate_symptoms_with_uncertainty(
            model=self.model,
            icd_code=icd_code,
            device=self.device,
            n_samples=10
        )
        
        # Check structure
        self.assertIn('mode_sequence', result)
        self.assertIn('token_probabilities', result)
        self.assertIn('all_sequences', result)
        self.assertIn('diversity', result)
        self.assertIn('n_unique_sequences', result)
        
        # Check mode sequence
        self.assertIsInstance(result['mode_sequence'], list)
        self.assertGreater(len(result['mode_sequence']), 0)
        
        # Check probabilities
        self.assertEqual(len(result['mode_sequence']), len(result['token_probabilities']))
        for prob in result['token_probabilities']:
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
            
        # Check diversity
        self.assertGreaterEqual(result['diversity'], 0.0)
        self.assertLessEqual(result['diversity'], 1.0)
        
    def test_evaluate_model_uncertainty(self):
        """Test model uncertainty evaluation."""
        dataset = DummyDataset(num_samples=10)
        dataloader = DataLoader(dataset, batch_size=2)
        
        result = evaluate_model_uncertainty(
            model=self.model,
            data_loader=dataloader,
            device=self.device,
            n_samples=5
        )
        
        # Check structure
        self.assertIn('mean_entropy', result)
        self.assertIn('std_entropy', result)
        self.assertIn('mean_std', result)
        self.assertIn('std_std', result)
        self.assertIn('accuracy', result)
        
        # Check values are reasonable
        self.assertGreaterEqual(result['mean_entropy'], 0.0)
        self.assertGreaterEqual(result['mean_std'], 0.0)
        self.assertGreaterEqual(result['accuracy'], 0.0)
        self.assertLessEqual(result['accuracy'], 1.0)


if __name__ == '__main__':
    unittest.main()
