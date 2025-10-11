"""
Tests for checkpoint loading validation.
"""

import unittest
import tempfile
import os
import torch
from brism import BRISM, BRISMConfig
from brism.train import load_checkpoint, ModelCheckpoint


class TestCheckpointValidation(unittest.TestCase):
    """Test checkpoint loading validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BRISMConfig(
            symptom_vocab_size=100,
            icd_vocab_size=50,
            latent_dim=32
        )
        self.model = BRISM(self.config)
        self.device = torch.device('cpu')
    
    def test_load_missing_file(self):
        """Test that missing checkpoint file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError) as context:
            load_checkpoint('nonexistent.pt', self.model)
        self.assertIn("Checkpoint file not found", str(context.exception))
    
    def test_load_missing_required_keys(self):
        """Test that checkpoint missing required keys raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'checkpoint.pt')
            
            # Save checkpoint without required keys
            torch.save({'some_other_key': 'value'}, checkpoint_path)
            
            with self.assertRaises(ValueError) as context:
                load_checkpoint(checkpoint_path, self.model)
            self.assertIn("missing required keys", str(context.exception))
    
    def test_load_valid_checkpoint(self):
        """Test loading a valid checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'checkpoint.pt')
            
            # Save a valid checkpoint
            checkpoint = {
                'epoch': 5,
                'model_state_dict': self.model.state_dict(),
                'metrics': {'loss': 0.5}
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Load should succeed
            loaded = load_checkpoint(checkpoint_path, self.model, device=self.device)
            self.assertEqual(loaded['epoch'], 5)
    
    def test_load_with_architecture_mismatch(self):
        """Test that architecture mismatch raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'checkpoint.pt')
            
            # Create checkpoint with different architecture
            other_config = BRISMConfig(
                symptom_vocab_size=200,  # Different vocab size
                icd_vocab_size=50,
                latent_dim=32
            )
            other_model = BRISM(other_config)
            
            checkpoint = {
                'epoch': 5,
                'model_state_dict': other_model.state_dict(),
                'config': {
                    'symptom_vocab_size': 200,
                    'icd_vocab_size': 50,
                    'latent_dim': 32
                }
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Should raise error due to vocab size mismatch
            with self.assertRaises(ValueError) as context:
                load_checkpoint(checkpoint_path, self.model, device=self.device)
            self.assertIn("symptom_vocab_size", str(context.exception))
    
    def test_load_weights_only(self):
        """Test loading only model weights (ignoring optimizer)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'checkpoint.pt')
            
            optimizer = torch.optim.Adam(self.model.parameters())
            
            checkpoint = {
                'epoch': 5,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Load with weights_only=True
            new_optimizer = torch.optim.Adam(self.model.parameters())
            loaded = load_checkpoint(
                checkpoint_path, 
                self.model, 
                optimizer=new_optimizer,
                device=self.device,
                weights_only=True
            )
            
            # Optimizer state should not be loaded
            # (We just verify it doesn't crash)
            self.assertEqual(loaded['epoch'], 5)
    
    def test_load_with_version_info(self):
        """Test loading checkpoint with version information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'checkpoint.pt')
            
            checkpoint = {
                'epoch': 5,
                'model_state_dict': self.model.state_dict(),
                'version': '3.0.0'
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Should load successfully and print version
            loaded = load_checkpoint(checkpoint_path, self.model, device=self.device)
            self.assertEqual(loaded['version'], '3.0.0')
    
    def test_save_and_load_cycle(self):
        """Test that we can save and load a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpointer = ModelCheckpoint(
                checkpoint_dir=tmpdir,
                monitor='val_loss',
                verbose=False
            )
            
            optimizer = torch.optim.Adam(self.model.parameters())
            metrics = {'val_loss': 0.5}
            
            # Save checkpoint
            checkpointer.save_checkpoint(
                self.model, 
                optimizer, 
                epoch=1, 
                metrics=metrics,
                is_best=True
            )
            
            # Load checkpoint (it's saved as best_model.pt, not checkpoint_best.pt)
            checkpoint_path = os.path.join(tmpdir, 'best_model.pt')
            loaded = load_checkpoint(
                checkpoint_path, 
                self.model, 
                optimizer=optimizer,
                device=self.device
            )
            
            self.assertEqual(loaded['epoch'], 1)
    
    def test_malformed_checkpoint(self):
        """Test that malformed checkpoint raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'checkpoint.pt')
            
            # Write non-torch file
            with open(checkpoint_path, 'w') as f:
                f.write("not a pytorch checkpoint")
            
            with self.assertRaises(ValueError) as context:
                load_checkpoint(checkpoint_path, self.model)
            self.assertIn("Failed to load checkpoint", str(context.exception))


if __name__ == '__main__':
    unittest.main()
