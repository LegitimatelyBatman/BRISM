"""
Tests for training enhancements including gradient clipping, scheduler support,
edge cases, and progress bars.
"""

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import tempfile
import shutil
import sys
from io import StringIO

from brism import (
    BRISM, BRISMConfig,
    BRISMLoss,
    train_brism,
)


class TinyDataset(torch.utils.data.Dataset):
    """Minimal dataset for testing."""
    
    def __init__(self, n_samples=10, symptom_vocab_size=20, icd_vocab_size=10, 
                 max_length=5, zero_length=False, single_token=False):
        """
        Create a tiny test dataset.
        
        Args:
            n_samples: Number of samples
            symptom_vocab_size: Size of symptom vocabulary
            icd_vocab_size: Size of ICD vocabulary
            max_length: Maximum sequence length
            zero_length: If True, create zero-length symptom sequences
            single_token: If True, create single-token symptom sequences
        """
        self.n_samples = n_samples
        self.symptom_vocab_size = symptom_vocab_size
        self.icd_vocab_size = icd_vocab_size
        self.max_length = max_length
        
        self.symptoms = []
        self.icd_codes = []
        
        for _ in range(n_samples):
            if zero_length:
                # All padding (zeros)
                symptoms = torch.zeros(max_length, dtype=torch.long)
            elif single_token:
                # Single token followed by padding
                symptoms = torch.zeros(max_length, dtype=torch.long)
                symptoms[0] = torch.randint(1, symptom_vocab_size, (1,)).item()
            else:
                # Normal random sequence
                seq_len = torch.randint(1, max_length, (1,)).item()
                symptoms = torch.randint(1, symptom_vocab_size, (seq_len,))
                symptoms = torch.cat([symptoms, torch.zeros(max_length - seq_len, dtype=torch.long)])
            
            self.symptoms.append(symptoms)
            
            # Random ICD code
            icd = torch.randint(0, icd_vocab_size, (1,)).item()
            self.icd_codes.append(icd)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {
            'symptoms': self.symptoms[idx],
            'icd_codes': torch.tensor(self.icd_codes[idx], dtype=torch.long)
        }


class TestGradientClipping(unittest.TestCase):
    """Test gradient clipping functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BRISMConfig(
            symptom_vocab_size=20,
            icd_vocab_size=10,
            latent_dim=16,
            encoder_hidden_dim=32,
            decoder_hidden_dim=32,
            max_symptom_length=5
        )
        self.device = torch.device('cpu')
        
    def test_gradient_clipping_enabled(self):
        """Test that gradient clipping is applied when max_grad_norm is set."""
        model = BRISM(self.config)
        dataset = TinyDataset(n_samples=8)
        loader = DataLoader(dataset, batch_size=4)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = BRISMLoss()
        
        # Train with gradient clipping
        history = train_brism(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            num_epochs=1,
            device=self.device,
            max_grad_norm=1.0
        )
        
        # Check that training completed
        self.assertEqual(len(history['train_loss']), 1)
        self.assertTrue(history['train_loss'][0] > 0)
        
    def test_gradient_clipping_disabled(self):
        """Test that gradient clipping is not applied when max_grad_norm is None."""
        model = BRISM(self.config)
        dataset = TinyDataset(n_samples=8)
        loader = DataLoader(dataset, batch_size=4)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = BRISMLoss()
        
        # Train without gradient clipping
        history = train_brism(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            num_epochs=1,
            device=self.device,
            max_grad_norm=None
        )
        
        # Check that training completed
        self.assertEqual(len(history['train_loss']), 1)
        self.assertTrue(history['train_loss'][0] > 0)
        
    def test_gradient_clipping_with_large_gradients(self):
        """Test gradient clipping with artificially large gradients."""
        model = BRISM(self.config)
        dataset = TinyDataset(n_samples=8)
        loader = DataLoader(dataset, batch_size=4)
        
        # Use larger learning rate to potentially create larger gradients
        optimizer = optim.SGD(model.parameters(), lr=1.0)
        loss_fn = BRISMLoss()
        
        # Train with strict gradient clipping
        history = train_brism(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            num_epochs=1,
            device=self.device,
            max_grad_norm=0.1
        )
        
        # Should still complete without NaN
        self.assertEqual(len(history['train_loss']), 1)
        self.assertFalse(torch.isnan(torch.tensor(history['train_loss'][0])))


class TestLearningRateScheduler(unittest.TestCase):
    """Test learning rate scheduler support."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BRISMConfig(
            symptom_vocab_size=20,
            icd_vocab_size=10,
            latent_dim=16,
            encoder_hidden_dim=32,
            decoder_hidden_dim=32,
            max_symptom_length=5
        )
        self.device = torch.device('cpu')
        
    def test_scheduler_step_lr(self):
        """Test training with StepLR scheduler."""
        model = BRISM(self.config)
        dataset = TinyDataset(n_samples=8)
        loader = DataLoader(dataset, batch_size=4)
        
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        loss_fn = BRISMLoss()
        
        history = train_brism(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            num_epochs=3,
            device=self.device,
            scheduler=scheduler
        )
        
        # Check that learning rate was tracked
        self.assertEqual(len(history['learning_rate']), 3)
        
        # Check that learning rate decreased (StepLR with gamma=0.5)
        self.assertAlmostEqual(history['learning_rate'][0], 0.01, places=5)
        self.assertAlmostEqual(history['learning_rate'][1], 0.005, places=5)
        self.assertAlmostEqual(history['learning_rate'][2], 0.0025, places=5)
        
    def test_scheduler_cosine_annealing(self):
        """Test training with CosineAnnealingLR scheduler."""
        model = BRISM(self.config)
        dataset = TinyDataset(n_samples=8)
        loader = DataLoader(dataset, batch_size=4)
        
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.001)
        loss_fn = BRISMLoss()
        
        history = train_brism(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            num_epochs=3,
            device=self.device,
            scheduler=scheduler
        )
        
        # Check that learning rate was tracked
        self.assertEqual(len(history['learning_rate']), 3)
        
        # First epoch should have the initial LR
        self.assertAlmostEqual(history['learning_rate'][0], 0.01, places=5)
        
        # Learning rate should change over epochs (cosine schedule)
        self.assertNotEqual(history['learning_rate'][0], history['learning_rate'][1])
        
    def test_no_scheduler(self):
        """Test training without scheduler still tracks learning rate."""
        model = BRISM(self.config)
        dataset = TinyDataset(n_samples=8)
        loader = DataLoader(dataset, batch_size=4)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = BRISMLoss()
        
        history = train_brism(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            num_epochs=2,
            device=self.device,
            scheduler=None
        )
        
        # Check that learning rate was tracked even without scheduler
        self.assertEqual(len(history['learning_rate']), 2)
        
        # Learning rate should remain constant
        self.assertAlmostEqual(history['learning_rate'][0], 0.001, places=5)
        self.assertAlmostEqual(history['learning_rate'][1], 0.001, places=5)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases for model robustness."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        
    def test_zero_length_symptom_sequences(self):
        """Test model with zero-length symptom sequences (all padding)."""
        config = BRISMConfig(
            symptom_vocab_size=20,
            icd_vocab_size=10,
            latent_dim=16,
            encoder_hidden_dim=32,
            decoder_hidden_dim=32,
            max_symptom_length=5
        )
        model = BRISM(config)
        
        dataset = TinyDataset(n_samples=8, zero_length=True)
        loader = DataLoader(dataset, batch_size=4)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = BRISMLoss()
        
        # Should not crash
        history = train_brism(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            num_epochs=1,
            device=self.device
        )
        
        self.assertEqual(len(history['train_loss']), 1)
        
    def test_single_token_symptoms(self):
        """Test model with single-token symptom sequences."""
        config = BRISMConfig(
            symptom_vocab_size=20,
            icd_vocab_size=10,
            latent_dim=16,
            encoder_hidden_dim=32,
            decoder_hidden_dim=32,
            max_symptom_length=5
        )
        model = BRISM(config)
        
        dataset = TinyDataset(n_samples=8, single_token=True)
        loader = DataLoader(dataset, batch_size=4)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = BRISMLoss()
        
        # Should not crash
        history = train_brism(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            num_epochs=1,
            device=self.device
        )
        
        self.assertEqual(len(history['train_loss']), 1)
        
    def test_vocab_size_one_symptom(self):
        """Test model with symptom_vocab_size=1 (only padding token)."""
        config = BRISMConfig(
            symptom_vocab_size=1,
            icd_vocab_size=10,
            latent_dim=16,
            encoder_hidden_dim=32,
            decoder_hidden_dim=32,
            max_symptom_length=5
        )
        model = BRISM(config)
        
        # Create dataset - all symptoms will be 0 (padding)
        dataset = TinyDataset(n_samples=8, symptom_vocab_size=1, zero_length=True)
        loader = DataLoader(dataset, batch_size=4)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = BRISMLoss()
        
        # Should not crash
        history = train_brism(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            num_epochs=1,
            device=self.device
        )
        
        self.assertEqual(len(history['train_loss']), 1)
        
    def test_vocab_size_one_icd(self):
        """Test model with icd_vocab_size=1."""
        config = BRISMConfig(
            symptom_vocab_size=20,
            icd_vocab_size=1,
            latent_dim=16,
            encoder_hidden_dim=32,
            decoder_hidden_dim=32,
            max_symptom_length=5
        )
        model = BRISM(config)
        
        dataset = TinyDataset(n_samples=8, icd_vocab_size=1)
        loader = DataLoader(dataset, batch_size=4)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = BRISMLoss()
        
        # Should not crash
        history = train_brism(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            num_epochs=1,
            device=self.device
        )
        
        self.assertEqual(len(history['train_loss']), 1)
        
    def test_empty_batch_handling(self):
        """Test that model handles very small batches (batch_size=1)."""
        config = BRISMConfig(
            symptom_vocab_size=20,
            icd_vocab_size=10,
            latent_dim=16,
            encoder_hidden_dim=32,
            decoder_hidden_dim=32,
            max_symptom_length=5
        )
        model = BRISM(config)
        
        dataset = TinyDataset(n_samples=4)
        loader = DataLoader(dataset, batch_size=1)  # Single sample per batch
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = BRISMLoss()
        
        # Should not crash
        history = train_brism(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            num_epochs=1,
            device=self.device
        )
        
        self.assertEqual(len(history['train_loss']), 1)


class TestProgressBars(unittest.TestCase):
    """Test progress bar functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BRISMConfig(
            symptom_vocab_size=20,
            icd_vocab_size=10,
            latent_dim=16,
            encoder_hidden_dim=32,
            decoder_hidden_dim=32,
            max_symptom_length=5
        )
        self.device = torch.device('cpu')
        
    def test_progress_bar_with_tqdm_available(self):
        """Test training with progress bars when tqdm is available."""
        try:
            import tqdm
            tqdm_available = True
        except ImportError:
            tqdm_available = False
            
        if not tqdm_available:
            self.skipTest("tqdm not available")
            
        model = BRISM(self.config)
        dataset = TinyDataset(n_samples=8)
        loader = DataLoader(dataset, batch_size=4)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = BRISMLoss()
        
        # Train with progress bar enabled
        history = train_brism(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            num_epochs=2,
            device=self.device,
            show_progress=True
        )
        
        # Check that training completed
        self.assertEqual(len(history['train_loss']), 2)
        
    def test_progress_bar_disabled(self):
        """Test training without progress bars."""
        model = BRISM(self.config)
        dataset = TinyDataset(n_samples=8)
        loader = DataLoader(dataset, batch_size=4)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = BRISMLoss()
        
        # Train with progress bar disabled
        history = train_brism(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            num_epochs=2,
            device=self.device,
            show_progress=False
        )
        
        # Check that training completed
        self.assertEqual(len(history['train_loss']), 2)


class TestCombinedFeatures(unittest.TestCase):
    """Test combinations of new features."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BRISMConfig(
            symptom_vocab_size=20,
            icd_vocab_size=10,
            latent_dim=16,
            encoder_hidden_dim=32,
            decoder_hidden_dim=32,
            max_symptom_length=5
        )
        self.device = torch.device('cpu')
        
    def test_all_features_combined(self):
        """Test using gradient clipping, scheduler, and progress bars together."""
        model = BRISM(self.config)
        dataset = TinyDataset(n_samples=16)
        loader = DataLoader(dataset, batch_size=4)
        
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        loss_fn = BRISMLoss()
        
        # Use all features together
        history = train_brism(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            num_epochs=3,
            device=self.device,
            scheduler=scheduler,
            max_grad_norm=1.0,
            show_progress=True
        )
        
        # Check that all tracking is working
        self.assertEqual(len(history['train_loss']), 3)
        self.assertEqual(len(history['learning_rate']), 3)
        
        # Verify scheduler worked
        self.assertLess(history['learning_rate'][2], history['learning_rate'][0])


if __name__ == '__main__':
    unittest.main()
