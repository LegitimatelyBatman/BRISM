"""Unit tests for BRISM evaluation metrics."""

import unittest
import torch
import numpy as np
from brism.metrics import (
    top_k_accuracy,
    compute_auroc_per_class,
    compute_calibration_metrics,
    stratify_by_disease_frequency,
    comprehensive_evaluation
)
from brism import BRISM, BRISMConfig
from torch.utils.data import Dataset, DataLoader


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


class TestMetrics(unittest.TestCase):
    """Test evaluation metrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 20
        self.num_classes = 10
        
        # Create random predictions and targets
        self.predictions = torch.randn(self.batch_size, self.num_classes)
        self.targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
        # Convert to numpy
        probs = torch.softmax(self.predictions, dim=-1)
        self.predictions_np = probs.numpy()
        self.targets_np = self.targets.numpy()
    
    def test_top_k_accuracy_k1(self):
        """Test top-1 accuracy."""
        accuracy = top_k_accuracy(self.predictions, self.targets, k=1)
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
    
    def test_top_k_accuracy_k5(self):
        """Test top-5 accuracy."""
        accuracy = top_k_accuracy(self.predictions, self.targets, k=5)
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        
        # Top-5 should be >= top-1
        acc_1 = top_k_accuracy(self.predictions, self.targets, k=1)
        self.assertGreaterEqual(accuracy, acc_1)
    
    def test_top_k_accuracy_perfect(self):
        """Test top-k accuracy with perfect predictions."""
        predictions = torch.zeros(10, 5)
        targets = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        
        # Set correct class to have highest score
        for i, target in enumerate(targets):
            predictions[i, target] = 10.0
        
        accuracy = top_k_accuracy(predictions, targets, k=1)
        self.assertEqual(accuracy, 1.0)
    
    def test_compute_auroc_per_class(self):
        """Test AUROC computation per class."""
        auroc_scores = compute_auroc_per_class(self.predictions_np, self.targets_np)
        
        self.assertIsInstance(auroc_scores, dict)
        
        # Check that scores are valid
        for class_idx, score in auroc_scores.items():
            self.assertIsInstance(class_idx, (int, np.integer))
            self.assertIsInstance(score, (float, np.floating))
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_compute_calibration_metrics(self):
        """Test calibration metrics computation."""
        metrics = compute_calibration_metrics(self.predictions_np, self.targets_np, n_bins=10)
        
        # Check structure
        self.assertIn('ece', metrics)
        self.assertIn('mce', metrics)
        self.assertIn('bin_accuracies', metrics)
        self.assertIn('bin_confidences', metrics)
        self.assertIn('bin_counts', metrics)
        
        # Check values
        self.assertGreaterEqual(metrics['ece'], 0.0)
        self.assertLessEqual(metrics['ece'], 1.0)
        self.assertGreaterEqual(metrics['mce'], 0.0)
        self.assertLessEqual(metrics['mce'], 1.0)
        
        # Check array shapes
        self.assertEqual(len(metrics['bin_accuracies']), 10)
        self.assertEqual(len(metrics['bin_confidences']), 10)
        self.assertEqual(len(metrics['bin_counts']), 10)
    
    def test_stratify_by_disease_frequency(self):
        """Test stratified performance metrics."""
        stratified = stratify_by_disease_frequency(
            self.predictions_np, 
            self.targets_np
        )
        
        self.assertIsInstance(stratified, dict)
        
        # Should have at least some groups
        self.assertGreater(len(stratified), 0)
        
        # Check structure of each group
        for group_name, metrics in stratified.items():
            self.assertIn(group_name, ['rare', 'medium', 'common'])
            self.assertIn('accuracy', metrics)
            self.assertIn('top_5_accuracy', metrics)
            self.assertIn('avg_confidence', metrics)
            self.assertIn('n_samples', metrics)
            self.assertIn('n_classes', metrics)
            
            # Check values
            self.assertGreaterEqual(metrics['accuracy'], 0.0)
            self.assertLessEqual(metrics['accuracy'], 1.0)
            self.assertGreater(metrics['n_samples'], 0)
            self.assertGreater(metrics['n_classes'], 0)
    
    def test_comprehensive_evaluation(self):
        """Test comprehensive evaluation on model."""
        # Create a small model
        config = BRISMConfig(
            symptom_vocab_size=100,
            icd_vocab_size=50,
            max_symptom_length=20
        )
        model = BRISM(config)
        
        # Create dataset
        dataset = DummyDataset(num_samples=50)
        dataloader = DataLoader(dataset, batch_size=10)
        
        # Run evaluation
        device = torch.device('cpu')
        results = comprehensive_evaluation(
            model, dataloader, device,
            n_bins=5, compute_auroc=False  # Skip AUROC for speed
        )
        
        # Check structure
        self.assertIn('top_1_accuracy', results)
        self.assertIn('top_3_accuracy', results)
        self.assertIn('top_5_accuracy', results)
        self.assertIn('top_10_accuracy', results)
        self.assertIn('calibration', results)
        self.assertIn('ece', results)
        self.assertIn('mce', results)
        self.assertIn('stratified_performance', results)
        
        # Check accuracy ordering
        self.assertLessEqual(results['top_1_accuracy'], results['top_3_accuracy'])
        self.assertLessEqual(results['top_3_accuracy'], results['top_5_accuracy'])
        self.assertLessEqual(results['top_5_accuracy'], results['top_10_accuracy'])


if __name__ == '__main__':
    unittest.main()
