"""Unit tests for BRISM loss functions."""

import unittest
import torch
from brism.loss import BRISMLoss, FocalLoss, compute_class_weights


class TestBRISMLoss(unittest.TestCase):
    """Test BRISM loss functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loss_fn = BRISMLoss(kl_weight=0.1, cycle_weight=1.0)
        self.batch_size = 4
        self.latent_dim = 32
        self.icd_vocab_size = 50
        self.symptom_vocab_size = 100
        self.seq_len = 20
        
    def test_kl_divergence(self):
        """Test KL divergence calculation."""
        mu = torch.randn(self.batch_size, self.latent_dim)
        logvar = torch.randn(self.batch_size, self.latent_dim)
        
        kl = self.loss_fn.kl_divergence(mu, logvar)
        
        self.assertEqual(kl.shape, (self.batch_size,))
        # KL should be non-negative
        self.assertTrue((kl >= 0).all())
        
    def test_reconstruction_loss_icd(self):
        """Test ICD reconstruction loss."""
        logits = torch.randn(self.batch_size, self.icd_vocab_size)
        target = torch.randint(0, self.icd_vocab_size, (self.batch_size,))
        
        loss = self.loss_fn.reconstruction_loss_icd(logits, target)
        
        self.assertEqual(loss.shape, (self.batch_size,))
        self.assertTrue((loss >= 0).all())
        
    def test_reconstruction_loss_symptoms(self):
        """Test symptom reconstruction loss."""
        logits = torch.randn(self.batch_size, self.seq_len, self.symptom_vocab_size)
        target = torch.randint(0, self.symptom_vocab_size, (self.batch_size, self.seq_len))
        
        loss = self.loss_fn.reconstruction_loss_symptoms(logits, target)
        
        self.assertEqual(loss.shape, (self.batch_size,))
        self.assertTrue((loss >= 0).all())
        
    def test_reconstruction_loss_symptoms_with_padding(self):
        """Test symptom reconstruction loss handles padding correctly."""
        logits = torch.randn(self.batch_size, self.seq_len, self.symptom_vocab_size)
        target = torch.randint(0, self.symptom_vocab_size, (self.batch_size, self.seq_len))
        
        # Set some positions to padding (0)
        target[:, 10:] = 0
        
        loss = self.loss_fn.reconstruction_loss_symptoms(logits, target, pad_idx=0)
        
        self.assertEqual(loss.shape, (self.batch_size,))
        self.assertTrue((loss >= 0).all())
        
    def test_cycle_consistency_loss(self):
        """Test cycle consistency loss."""
        mu1 = torch.randn(self.batch_size, self.latent_dim)
        logvar1 = torch.randn(self.batch_size, self.latent_dim)
        mu2 = torch.randn(self.batch_size, self.latent_dim)
        logvar2 = torch.randn(self.batch_size, self.latent_dim)
        
        loss = self.loss_fn.cycle_consistency_loss(mu1, logvar1, mu2, logvar2)
        
        self.assertEqual(loss.shape, (self.batch_size,))
        
    def test_cycle_consistency_same_distribution(self):
        """Test cycle consistency is zero for identical distributions."""
        mu = torch.randn(self.batch_size, self.latent_dim)
        logvar = torch.randn(self.batch_size, self.latent_dim)
        
        loss = self.loss_fn.cycle_consistency_loss(mu, logvar, mu, logvar)
        
        # Should be close to zero
        self.assertTrue((loss.abs() < 1e-5).all())
        
    def test_forward_loss(self):
        """Test forward path loss."""
        icd_logits = torch.randn(self.batch_size, self.icd_vocab_size)
        mu = torch.randn(self.batch_size, self.latent_dim)
        logvar = torch.randn(self.batch_size, self.latent_dim)
        
        symptoms = torch.randint(0, self.symptom_vocab_size, (self.batch_size, self.seq_len))
        icd_codes = torch.randint(0, self.icd_vocab_size, (self.batch_size,))
        
        model_output = (icd_logits, mu, logvar)
        total_loss, loss_dict = self.loss_fn.forward_loss(model_output, symptoms, icd_codes)
        
        self.assertIsInstance(total_loss.item(), float)
        self.assertTrue(total_loss.item() >= 0)
        self.assertIn('forward_recon', loss_dict)
        self.assertIn('forward_kl', loss_dict)
        self.assertIn('forward_total', loss_dict)
        
    def test_reverse_loss(self):
        """Test reverse path loss."""
        symptom_logits = torch.randn(self.batch_size, self.seq_len, self.symptom_vocab_size)
        mu = torch.randn(self.batch_size, self.latent_dim)
        logvar = torch.randn(self.batch_size, self.latent_dim)
        
        icd_codes = torch.randint(0, self.icd_vocab_size, (self.batch_size,))
        symptoms = torch.randint(0, self.symptom_vocab_size, (self.batch_size, self.seq_len))
        
        model_output = (symptom_logits, mu, logvar)
        total_loss, loss_dict = self.loss_fn.reverse_loss(model_output, icd_codes, symptoms)
        
        self.assertIsInstance(total_loss.item(), float)
        self.assertTrue(total_loss.item() >= 0)
        self.assertIn('reverse_recon', loss_dict)
        self.assertIn('reverse_kl', loss_dict)
        
    def test_cycle_forward_loss(self):
        """Test forward cycle loss."""
        symptom_logits = torch.randn(self.batch_size, self.seq_len, self.symptom_vocab_size)
        icd_logits = torch.randn(self.batch_size, self.icd_vocab_size)
        mu1 = torch.randn(self.batch_size, self.latent_dim)
        logvar1 = torch.randn(self.batch_size, self.latent_dim)
        mu2 = torch.randn(self.batch_size, self.latent_dim)
        logvar2 = torch.randn(self.batch_size, self.latent_dim)
        
        symptoms = torch.randint(0, self.symptom_vocab_size, (self.batch_size, self.seq_len))
        icd_codes = torch.randint(0, self.icd_vocab_size, (self.batch_size,))
        
        model_output = (symptom_logits, icd_logits, mu1, logvar1, mu2, logvar2)
        total_loss, loss_dict = self.loss_fn.cycle_forward_loss(model_output, symptoms, icd_codes)
        
        self.assertIsInstance(total_loss.item(), float)
        self.assertTrue(total_loss.item() >= 0)
        self.assertIn('cycle_fwd_icd_recon', loss_dict)
        self.assertIn('cycle_fwd_symptom_recon', loss_dict)
        self.assertIn('cycle_fwd_cycle', loss_dict)
        
    def test_cycle_reverse_loss(self):
        """Test reverse cycle loss."""
        icd_logits = torch.randn(self.batch_size, self.icd_vocab_size)
        symptom_logits = torch.randn(self.batch_size, self.seq_len, self.symptom_vocab_size)
        mu1 = torch.randn(self.batch_size, self.latent_dim)
        logvar1 = torch.randn(self.batch_size, self.latent_dim)
        mu2 = torch.randn(self.batch_size, self.latent_dim)
        logvar2 = torch.randn(self.batch_size, self.latent_dim)
        
        icd_codes = torch.randint(0, self.icd_vocab_size, (self.batch_size,))
        symptoms = torch.randint(0, self.symptom_vocab_size, (self.batch_size, self.seq_len))
        
        model_output = (icd_logits, symptom_logits, mu1, logvar1, mu2, logvar2)
        total_loss, loss_dict = self.loss_fn.cycle_reverse_loss(model_output, icd_codes, symptoms)
        
        self.assertIsInstance(total_loss.item(), float)
        self.assertTrue(total_loss.item() >= 0)
        self.assertIn('cycle_rev_icd_recon', loss_dict)
        self.assertIn('cycle_rev_symptom_recon', loss_dict)
        self.assertIn('cycle_rev_cycle', loss_dict)


class TestFocalLoss(unittest.TestCase):
    """Test focal loss for class imbalance."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 10
        self.num_classes = 20
    
    def test_focal_loss_basic(self):
        """Test basic focal loss computation."""
        focal_loss = FocalLoss(gamma=2.0)
        
        logits = torch.randn(self.batch_size, self.num_classes)
        targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
        loss = focal_loss(logits, targets)
        
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0.0)
    
    def test_focal_loss_with_alpha(self):
        """Test focal loss with class weights."""
        alpha = torch.rand(self.num_classes)
        focal_loss = FocalLoss(alpha=alpha, gamma=2.0)
        
        logits = torch.randn(self.batch_size, self.num_classes)
        targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
        loss = focal_loss(logits, targets)
        
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0.0)
    
    def test_focal_loss_reduction_none(self):
        """Test focal loss with no reduction."""
        focal_loss = FocalLoss(gamma=2.0, reduction='none')
        
        logits = torch.randn(self.batch_size, self.num_classes)
        targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
        loss = focal_loss(logits, targets)
        
        self.assertEqual(loss.shape, (self.batch_size,))
        self.assertTrue((loss >= 0).all())
    
    def test_focal_loss_gamma_zero_equals_ce(self):
        """Test that focal loss with gamma=0 approximates cross-entropy."""
        focal_loss = FocalLoss(gamma=0.0)
        
        logits = torch.randn(self.batch_size, self.num_classes)
        targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
        focal = focal_loss(logits, targets)
        ce = torch.nn.functional.cross_entropy(logits, targets)
        
        # Should be very close
        self.assertTrue(torch.allclose(focal, ce, atol=1e-5))


class TestClassWeights(unittest.TestCase):
    """Test class weight computation."""
    
    def test_compute_class_weights(self):
        """Test class weight computation."""
        class_counts = {0: 100, 1: 50, 2: 10, 3: 200}
        num_classes = 5
        
        weights = compute_class_weights(class_counts, num_classes)
        
        # Check shape
        self.assertEqual(weights.shape, (num_classes,))
        
        # Check all positive
        self.assertTrue((weights > 0).all())
        
        # Rare classes should have higher weights
        self.assertGreater(weights[2].item(), weights[0].item())
        self.assertGreater(weights[2].item(), weights[3].item())
    
    def test_brism_loss_with_class_weights(self):
        """Test BRISM loss with class weights."""
        class_weights = torch.rand(50)
        loss_fn = BRISMLoss(
            kl_weight=0.1,
            cycle_weight=1.0,
            class_weights=class_weights,
            use_focal_loss=False
        )
        
        logits = torch.randn(4, 50)
        target = torch.randint(0, 50, (4,))
        
        loss = loss_fn.reconstruction_loss_icd(logits, target)
        
        self.assertEqual(loss.shape, (4,))
        self.assertTrue((loss >= 0).all())
    
    def test_brism_loss_with_focal_loss(self):
        """Test BRISM loss with focal loss."""
        loss_fn = BRISMLoss(
            kl_weight=0.1,
            cycle_weight=1.0,
            use_focal_loss=True,
            focal_gamma=2.0
        )
        
        logits = torch.randn(4, 50)
        target = torch.randint(0, 50, (4,))
        
        loss = loss_fn.reconstruction_loss_icd(logits, target)
        
        # Focal loss may return scalar or batch-wise
        self.assertTrue(loss.numel() >= 1)


if __name__ == '__main__':
    unittest.main()
