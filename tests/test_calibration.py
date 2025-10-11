"""Unit tests for calibration utilities."""

import unittest
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from brism.calibration import TemperatureScaling, calibrate_temperature, evaluate_calibration_improvement
from brism import BRISM, BRISMConfig


class DummyDataset(Dataset):
    """Simple dataset for testing."""
    
    def __init__(self, num_samples=100, symptom_vocab_size=100, 
                 icd_vocab_size=50, max_length=20):
        self.num_samples = num_samples
        self.data = []
        
        for _ in range(num_samples):
            symptoms = np.random.randint(1, symptom_vocab_size, max_length)
            icd_code = np.random.randint(0, icd_vocab_size)
            self.data.append({
                'symptoms': symptoms,
                'icd_codes': icd_code
            })
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'symptoms': torch.tensor(sample['symptoms'], dtype=torch.long),
            'icd_codes': torch.tensor(sample['icd_codes'], dtype=torch.long)
        }


class TestCalibration(unittest.TestCase):
    """Test calibration utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BRISMConfig(
            symptom_vocab_size=100,
            icd_vocab_size=50,
            max_symptom_length=20
        )
        self.model = BRISM(self.config)
        self.device = torch.device('cpu')
    
    def test_temperature_scaling_init(self):
        """Test temperature scaling initialization."""
        temp_scaling = TemperatureScaling(temperature=2.0)
        
        self.assertEqual(temp_scaling.get_temperature(), 2.0)
    
    def test_temperature_scaling_forward(self):
        """Test temperature scaling forward pass."""
        temp_scaling = TemperatureScaling(temperature=2.0)
        
        logits = torch.randn(10, 50)
        scaled_logits = temp_scaling(logits)
        
        # Check that logits are scaled
        expected = logits / 2.0
        self.assertTrue(torch.allclose(scaled_logits, expected))
    
    def test_calibrate_temperature(self):
        """Test temperature calibration."""
        dataset = DummyDataset(num_samples=50)
        dataloader = DataLoader(dataset, batch_size=10)
        
        # Initial temperature
        initial_temp = self.model.temperature.item()
        
        # Calibrate
        optimal_temp = calibrate_temperature(
            self.model,
            dataloader,
            self.device,
            max_iter=10,
            lr=0.01
        )
        
        # Check that temperature was updated
        self.assertEqual(self.model.temperature.item(), optimal_temp)
        
        # Check that temperature is reasonable
        self.assertGreater(optimal_temp, 0.1)
        self.assertLess(optimal_temp, 10.0)
    
    def test_evaluate_calibration_improvement(self):
        """Test evaluation of calibration improvement."""
        dataset = DummyDataset(num_samples=50)
        dataloader = DataLoader(dataset, batch_size=10)
        
        # Evaluate calibration
        results = evaluate_calibration_improvement(
            self.model,
            dataloader,
            self.device,
            n_bins=5
        )
        
        # Check structure
        self.assertIn('before_scaling', results)
        self.assertIn('after_scaling', results)
        self.assertIn('temperature', results)
        self.assertIn('ece_improvement', results)
        
        # Check that both have calibration metrics
        self.assertIn('ece', results['before_scaling'])
        self.assertIn('mce', results['before_scaling'])
        self.assertIn('ece', results['after_scaling'])
        self.assertIn('mce', results['after_scaling'])


if __name__ == '__main__':
    unittest.main()
