"""
Tests for model export functionality.
"""

import unittest
import tempfile
import os
import torch
from brism import BRISM, BRISMConfig
from brism.export import (
    export_to_torchscript,
    quantize_model,
    prune_model,
    export_for_deployment
)


class TestModelExport(unittest.TestCase):
    """Test model export functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BRISMConfig(
            symptom_vocab_size=100,
            icd_vocab_size=50,
            latent_dim=32
        )
        self.model = BRISM(self.config)
    
    def test_export_to_torchscript_trace(self):
        """Test exporting with save method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'model.pt')
            
            # Should not raise
            export_to_torchscript(self.model, output_path, method='save')
            
            # File should exist
            self.assertTrue(os.path.exists(output_path))
            
            # Should be loadable
            checkpoint = torch.load(output_path, weights_only=False)
            self.assertIn('model_state_dict', checkpoint)
            self.assertIn('config', checkpoint)
    
    def test_export_invalid_method(self):
        """Test that invalid export method raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'model.pt')
            
            with self.assertRaises(ValueError) as context:
                export_to_torchscript(self.model, output_path, method='invalid')
            self.assertIn("method must be", str(context.exception))
    
    def test_quantize_model_dynamic(self):
        """Test dynamic quantization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'model_quantized.pt')
            
            # Quantize
            quantized = quantize_model(self.model, output_path, quantization_type='dynamic')
            
            # File should exist
            self.assertTrue(os.path.exists(output_path))
            
            # Should return a model
            self.assertIsNotNone(quantized)
    
    def test_quantize_invalid_type(self):
        """Test that invalid quantization type raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'model.pt')
            
            with self.assertRaises(ValueError) as context:
                quantize_model(self.model, output_path, quantization_type='invalid')
            self.assertIn("quantization_type must be", str(context.exception))
    
    def test_prune_model(self):
        """Test model pruning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'model_pruned.pt')
            
            # Prune
            pruned = prune_model(self.model, output_path, amount=0.3)
            
            # File should exist
            self.assertTrue(os.path.exists(output_path))
            
            # Should return a model
            self.assertIsNotNone(pruned)
    
    def test_prune_invalid_amount(self):
        """Test that invalid pruning amount raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'model.pt')
            
            with self.assertRaises(ValueError) as context:
                prune_model(self.model, output_path, amount=1.5)
            self.assertIn("amount must be between 0 and 1", str(context.exception))
    
    def test_prune_invalid_method(self):
        """Test that invalid pruning method raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'model.pt')
            
            with self.assertRaises(ValueError) as context:
                prune_model(self.model, output_path, method='invalid')
            self.assertIn("method must be", str(context.exception))
    
    def test_export_for_deployment(self):
        """Test exporting for deployment in multiple formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Export without ONNX (which requires extra dependencies)
            paths = export_for_deployment(
                self.model,
                tmpdir,
                formats=['torchscript'],
                quantize=True,
                prune=True
            )
            
            # Should have created torchscript, quantized, and pruned
            self.assertIn('torchscript', paths)
            self.assertIn('quantized', paths)
            self.assertIn('pruned', paths)
            
            # All files should exist
            for path in paths.values():
                self.assertTrue(os.path.exists(path))
    
    def test_export_creates_output_dir(self):
        """Test that export creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, 'nested', 'output')
            
            paths = export_for_deployment(
                self.model,
                output_dir,
                formats=['torchscript']
            )
            
            # Directory should be created
            self.assertTrue(os.path.exists(output_dir))
            
            # File should exist
            self.assertTrue(os.path.exists(paths['torchscript']))


if __name__ == '__main__':
    unittest.main()
