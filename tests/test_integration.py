"""
Integration tests for BRISM end-to-end training pipelines.

Tests complete workflows including:
- Data loading and preprocessing
- Model training and validation
- Checkpointing and resuming
- Memory usage monitoring
"""

import unittest
import torch
import torch.nn as nn
import tempfile
import shutil
import os
import gc
from pathlib import Path

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from brism import (
    BRISM, BRISMConfig,
    BRISMLoss,
    train_brism,
    load_checkpoint,
)


class ToyDataset(torch.utils.data.Dataset):
    """Simple toy dataset for integration testing."""
    
    def __init__(self, n_samples=100, symptom_vocab_size=50, icd_vocab_size=20, max_length=10):
        """
        Create a toy dataset with random data.
        
        Args:
            n_samples: Number of samples
            symptom_vocab_size: Size of symptom vocabulary
            icd_vocab_size: Size of ICD vocabulary
            max_length: Maximum sequence length
        """
        self.n_samples = n_samples
        self.symptom_vocab_size = symptom_vocab_size
        self.icd_vocab_size = icd_vocab_size
        self.max_length = max_length
        
        # Generate random data
        self.symptoms = []
        self.icd_codes = []
        
        for _ in range(n_samples):
            # Random symptom sequence (1 to max_length-1 symptoms, 0 is padding)
            seq_len = torch.randint(1, max_length, (1,)).item()
            symptoms = torch.randint(1, symptom_vocab_size, (seq_len,))
            # Pad to max_length
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


class TestEndToEndTraining(unittest.TestCase):
    """Test end-to-end training pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BRISMConfig(
            symptom_vocab_size=50,
            icd_vocab_size=20,
            latent_dim=16,
            symptom_embed_dim=32,
            icd_embed_dim=32,
            encoder_hidden_dim=32,
            decoder_hidden_dim=32,
            max_symptom_length=10,
            dropout_rate=0.1,
        )
        self.device = torch.device('cpu')  # Use CPU for testing
        
        # Create toy datasets
        self.train_dataset = ToyDataset(n_samples=100, 
                                       symptom_vocab_size=self.config.symptom_vocab_size,
                                       icd_vocab_size=self.config.icd_vocab_size,
                                       max_length=self.config.max_symptom_length)
        
        self.val_dataset = ToyDataset(n_samples=20,
                                     symptom_vocab_size=self.config.symptom_vocab_size,
                                     icd_vocab_size=self.config.icd_vocab_size,
                                     max_length=self.config.max_symptom_length)
    
    def test_simple_training_loop(self):
        """Test that a simple training loop completes without errors."""
        model = BRISM(self.config).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = BRISMLoss(kl_weight=0.1, cycle_weight=1.0)
        
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=16, shuffle=True)
        
        # Train for a few steps
        model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            symptoms = batch['symptoms'].to(self.device)
            icd_codes = batch['icd_codes'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            icd_logits, mu_f, logvar_f = model.forward_path(symptoms)
            
            # Compute forward loss
            forward_loss, forward_losses = loss_fn.forward_loss(
                (icd_logits, mu_f, logvar_f),
                symptoms,
                icd_codes
            )
            
            loss = forward_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            if n_batches >= 5:  # Only train for 5 batches
                break
        
        avg_loss = total_loss / n_batches
        
        # Check that loss is reasonable (not NaN or inf)
        self.assertFalse(torch.isnan(torch.tensor(avg_loss)))
        self.assertFalse(torch.isinf(torch.tensor(avg_loss)))
        self.assertGreater(avg_loss, 0)
    
    def test_training_with_validation(self):
        """Test training with validation loop."""
        model = BRISM(self.config).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = BRISMLoss(kl_weight=0.1, cycle_weight=1.0)
        
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=16, shuffle=True)
        val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=16, shuffle=False)
        
        # Train for one epoch
        model.train()
        train_losses = []
        batch_idx = 0
        
        for batch in train_loader:
            symptoms = batch['symptoms'].to(self.device)
            icd_codes = batch['icd_codes'].to(self.device)
            
            optimizer.zero_grad()
            
            # Alternate between forward and reverse paths
            if batch_idx % 2 == 0:
                icd_logits, mu_f, logvar_f = model.forward_path(symptoms)
                loss, _ = loss_fn.forward_loss(
                    (icd_logits, mu_f, logvar_f),
                    symptoms,
                    icd_codes
                )
            else:
                symptom_logits, mu_r, logvar_r = model.reverse_path(icd_codes, symptoms)
                loss, _ = loss_fn.reverse_loss(
                    (symptom_logits, mu_r, logvar_r),
                    icd_codes,
                    symptoms
                )
            
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            batch_idx += 1
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                symptoms = batch['symptoms'].to(self.device)
                icd_codes = batch['icd_codes'].to(self.device)
                
                icd_logits, mu_f, logvar_f = model.forward_path(symptoms)
                
                forward_loss, _ = loss_fn.forward_loss(
                    (icd_logits, mu_f, logvar_f),
                    symptoms,
                    icd_codes
                )
                
                loss = forward_loss
                val_losses.append(loss.item())
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        # Both losses should be reasonable
        self.assertFalse(torch.isnan(torch.tensor(avg_train_loss)))
        self.assertFalse(torch.isnan(torch.tensor(avg_val_loss)))
        self.assertGreater(avg_train_loss, 0)
        self.assertGreater(avg_val_loss, 0)
    
    def test_checkpoint_save_and_load(self):
        """Test saving and loading checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pt')
            
            # Create and train model
            model = BRISM(self.config).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Save checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': 1,
                'config': self.config,
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Create new model and load checkpoint
            new_model = BRISM(self.config).to(self.device)
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
            
            loaded_checkpoint = load_checkpoint(
                checkpoint_path,
                new_model,
                new_optimizer,
                device=self.device
            )
            
            # Check that checkpoint was loaded correctly
            self.assertEqual(loaded_checkpoint['epoch'], 1)
            
            # Check that model states match
            for key in model.state_dict().keys():
                torch.testing.assert_close(
                    model.state_dict()[key],
                    new_model.state_dict()[key]
                )
    
    def test_training_with_train_brism_function(self):
        """Test using the high-level train_brism function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = BRISM(self.config).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            loss_fn = BRISMLoss(kl_weight=0.1, cycle_weight=1.0)
            
            train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=16, shuffle=True)
            val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=16, shuffle=False)
            
            # Train for 2 epochs
            history = train_brism(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                num_epochs=2,
                device=self.device,
                checkpoint_dir=tmpdir,
                early_stopping_patience=10
            )
            
            # Check that history was recorded
            self.assertIn('train_loss', history)
            self.assertIn('val_loss', history)
            self.assertEqual(len(history['train_loss']), 2)
            self.assertEqual(len(history['val_loss']), 2)
            
            # Check that checkpoint was saved
            checkpoint_files = list(Path(tmpdir).glob('*.pt'))
            self.assertGreater(len(checkpoint_files), 0)


class TestMemoryLeaks(unittest.TestCase):
    """Test for memory leaks in training loops."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not PSUTIL_AVAILABLE:
            self.skipTest("psutil not available, skipping memory leak tests")
        
        self.config = BRISMConfig(
            symptom_vocab_size=50,
            icd_vocab_size=20,
            latent_dim=16,
            symptom_embed_dim=32,
            icd_embed_dim=32,
            encoder_hidden_dim=32,
            decoder_hidden_dim=32,
            max_symptom_length=10,
            dropout_rate=0.1,
        )
        self.device = torch.device('cpu')
    
    def test_no_memory_leak_in_training_loop(self):
        """Test that memory usage doesn't grow unbounded during training."""
        # Get initial memory usage
        process = psutil.Process()
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        model = BRISM(self.config).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = BRISMLoss(kl_weight=0.1, cycle_weight=1.0)
        
        # Create dataset
        dataset = ToyDataset(n_samples=50,
                           symptom_vocab_size=self.config.symptom_vocab_size,
                           icd_vocab_size=self.config.icd_vocab_size,
                           max_length=self.config.max_symptom_length)
        
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Run multiple training iterations
        memory_samples = []
        
        for iteration in range(3):
            model.train()
            
            for batch in train_loader:
                symptoms = batch['symptoms'].to(self.device)
                icd_codes = batch['icd_codes'].to(self.device)
                
                optimizer.zero_grad()
                icd_logits, mu_f, logvar_f = model.forward_path(symptoms)
                
                forward_loss, _ = loss_fn.forward_loss(
                    (icd_logits, mu_f, logvar_f),
                    symptoms,
                    icd_codes
                )
                
                loss = forward_loss
                loss.backward()
                optimizer.step()
            
            # Collect garbage and measure memory
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_samples.append(current_memory)
        
        # Check that memory growth is reasonable
        # Allow for some growth but not unbounded
        final_memory = memory_samples[-1]
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be less than 100 MB for this small test
        # (this is a conservative threshold)
        self.assertLess(memory_growth, 100, 
                       f"Memory grew by {memory_growth:.1f} MB, which may indicate a leak. "
                       f"Initial: {initial_memory:.1f} MB, Final: {final_memory:.1f} MB")
        
        # Also check that memory is not continuously growing
        if len(memory_samples) >= 3:
            # Growth from iteration 1 to 2
            growth_1_2 = memory_samples[1] - memory_samples[0]
            # Growth from iteration 2 to 3
            growth_2_3 = memory_samples[2] - memory_samples[1]
            
            # Growth should stabilize (second growth should not be much larger than first)
            # Allow for 50% increase in growth rate
            if growth_1_2 > 0:
                self.assertLess(growth_2_3, growth_1_2 * 1.5,
                               f"Memory growth is accelerating: {growth_1_2:.1f} MB -> {growth_2_3:.1f} MB")
    
    def test_no_memory_leak_with_multiple_models(self):
        """Test that creating and destroying models doesn't leak memory."""
        process = psutil.Process()
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and destroy multiple models
        for _ in range(5):
            model = BRISM(self.config).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Do some forward passes
            symptoms = torch.randint(0, self.config.symptom_vocab_size, 
                                    (4, self.config.max_symptom_length)).to(self.device)
            _ = model.forward_path(symptoms)
            
            # Explicitly delete
            del model
            del optimizer
            gc.collect()
        
        # Check final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Should not grow significantly (allow 50 MB)
        self.assertLess(memory_growth, 50,
                       f"Memory grew by {memory_growth:.1f} MB after creating/destroying 5 models")


class TestInferenceIntegration(unittest.TestCase):
    """Test inference workflows end-to-end."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BRISMConfig(
            symptom_vocab_size=50,
            icd_vocab_size=20,
            latent_dim=16,
            max_symptom_length=10,
        )
        self.device = torch.device('cpu')
        self.model = BRISM(self.config).to(self.device)
    
    def test_forward_inference(self):
        """Test forward inference from symptoms to ICD codes."""
        self.model.eval()
        
        # Create test symptoms
        symptoms = torch.tensor([1, 5, 12, 0, 0, 0, 0, 0, 0, 0]).to(self.device)
        
        with torch.no_grad():
            icd_logits, mu, logvar = self.model.forward_path(symptoms.unsqueeze(0))
            predictions = torch.softmax(icd_logits, dim=-1)
        
        # Check output shape
        self.assertEqual(predictions.shape, (1, self.config.icd_vocab_size))
        
        # Check that probabilities sum to 1
        torch.testing.assert_close(predictions.sum(), torch.tensor(1.0), atol=1e-5, rtol=1e-5)
    
    def test_reverse_inference(self):
        """Test reverse inference from ICD codes to symptoms."""
        self.model.eval()
        
        # Create test ICD code
        icd_code = torch.tensor([5]).to(self.device)
        
        with torch.no_grad():
            symptom_logits, mu, logvar = self.model.reverse_path(icd_code)
            predictions = torch.softmax(symptom_logits, dim=-1)
        
        # Check output shape
        self.assertEqual(predictions.shape, 
                        (1, self.config.max_symptom_length, self.config.symptom_vocab_size))
    
    def test_uncertainty_estimation(self):
        """Test uncertainty estimation with MC dropout."""
        self.model.train()  # Enable dropout for uncertainty
        
        symptoms = torch.tensor([1, 5, 12, 0, 0, 0, 0, 0, 0, 0]).to(self.device)
        
        # Get multiple predictions with dropout
        n_samples = 10
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                icd_logits, mu, logvar = self.model.forward_path(symptoms.unsqueeze(0))
                probs = torch.softmax(icd_logits, dim=-1)
                predictions.append(probs)
        
        predictions = torch.stack(predictions, dim=0)
        
        # Check shape
        self.assertEqual(predictions.shape, (n_samples, 1, self.config.icd_vocab_size))
        
        # Compute mean and std
        mean_probs = predictions.mean(dim=0)
        std_probs = predictions.std(dim=0)
        
        # Standard deviation should be non-zero (indicating uncertainty)
        self.assertGreater(std_probs.sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
