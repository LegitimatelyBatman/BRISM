"""
Loss functions for BRISM training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BRISMLoss(nn.Module):
    """
    Combined loss for BRISM model including:
    1. Reconstruction loss (VAE-style)
    2. KL divergence on latent distributions
    3. Cycle consistency loss
    """
    
    def __init__(self, kl_weight: float = 0.1, cycle_weight: float = 1.0):
        """
        Args:
            kl_weight: Weight for KL divergence term
            cycle_weight: Weight for cycle consistency term
        """
        super().__init__()
        self.kl_weight = kl_weight
        self.cycle_weight = cycle_weight
        
    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between latent distribution and standard normal.
        
        KL(N(mu, sigma^2) || N(0, 1)) = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        
        Args:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
            
        Returns:
            KL divergence [batch_size]
        """
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl
    
    def reconstruction_loss_icd(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Reconstruction loss for ICD predictions (cross-entropy).
        
        Args:
            logits: Predicted logits [batch_size, icd_vocab_size]
            target: Target ICD codes [batch_size]
            
        Returns:
            Loss [batch_size]
        """
        return F.cross_entropy(logits, target, reduction='none')
    
    def reconstruction_loss_symptoms(self, logits: torch.Tensor, target: torch.Tensor, 
                                    pad_idx: int = 0) -> torch.Tensor:
        """
        Reconstruction loss for symptom sequences (cross-entropy with masking).
        
        Args:
            logits: Predicted logits [batch_size, seq_len, vocab_size]
            target: Target sequences [batch_size, seq_len]
            pad_idx: Padding index to ignore
            
        Returns:
            Loss [batch_size]
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Reshape for cross entropy
        logits_flat = logits.reshape(-1, vocab_size)
        target_flat = target.reshape(-1)
        
        # Compute loss
        loss = F.cross_entropy(logits_flat, target_flat, reduction='none', ignore_index=pad_idx)
        loss = loss.reshape(batch_size, seq_len)
        
        # Average over sequence (excluding padding)
        mask = (target != pad_idx).float()
        loss = (loss * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        return loss
    
    def cycle_consistency_loss(self, mu1: torch.Tensor, logvar1: torch.Tensor,
                               mu2: torch.Tensor, logvar2: torch.Tensor) -> torch.Tensor:
        """
        Cycle consistency loss: latent distributions should be similar after cycle.
        
        Uses KL divergence between the two latent distributions.
        
        Args:
            mu1: Mean of first latent distribution [batch_size, latent_dim]
            logvar1: Log variance of first latent [batch_size, latent_dim]
            mu2: Mean of second latent distribution [batch_size, latent_dim]
            logvar2: Log variance of second latent [batch_size, latent_dim]
            
        Returns:
            Cycle consistency loss [batch_size]
        """
        # KL(N(mu1, var1) || N(mu2, var2))
        var1 = logvar1.exp()
        var2 = logvar2.exp()
        
        kl = 0.5 * torch.sum(
            logvar2 - logvar1 + (var1 + (mu1 - mu2).pow(2)) / var2 - 1,
            dim=1
        )
        
        return kl
    
    def forward_loss(self, model_output: Tuple, symptoms: torch.Tensor, 
                     icd_codes: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute loss for forward path: symptoms -> ICD.
        
        Args:
            model_output: (icd_logits, mu, logvar) from forward_path
            symptoms: Input symptoms [batch_size, seq_len]
            icd_codes: Target ICD codes [batch_size]
            
        Returns:
            total_loss: Scalar loss
            loss_dict: Dictionary with individual loss components
        """
        icd_logits, mu, logvar = model_output
        
        # Reconstruction loss
        recon_loss = self.reconstruction_loss_icd(icd_logits, icd_codes).mean()
        
        # KL divergence
        kl_loss = self.kl_divergence(mu, logvar).mean()
        
        # Total loss
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        loss_dict = {
            'forward_recon': recon_loss.item(),
            'forward_kl': kl_loss.item(),
            'forward_total': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def reverse_loss(self, model_output: Tuple, icd_codes: torch.Tensor,
                     symptoms: torch.Tensor, pad_idx: int = 0) -> Tuple[torch.Tensor, dict]:
        """
        Compute loss for reverse path: ICD -> symptoms.
        
        Args:
            model_output: (symptom_logits, mu, logvar) from reverse_path
            icd_codes: Input ICD codes [batch_size]
            symptoms: Target symptoms [batch_size, seq_len]
            pad_idx: Padding index
            
        Returns:
            total_loss: Scalar loss
            loss_dict: Dictionary with individual loss components
        """
        symptom_logits, mu, logvar = model_output
        
        # Reconstruction loss
        recon_loss = self.reconstruction_loss_symptoms(symptom_logits, symptoms, pad_idx).mean()
        
        # KL divergence
        kl_loss = self.kl_divergence(mu, logvar).mean()
        
        # Total loss
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        loss_dict = {
            'reverse_recon': recon_loss.item(),
            'reverse_kl': kl_loss.item(),
            'reverse_total': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def cycle_forward_loss(self, model_output: Tuple, symptoms: torch.Tensor,
                          icd_codes: torch.Tensor, pad_idx: int = 0) -> Tuple[torch.Tensor, dict]:
        """
        Compute loss for forward cycle: symptoms -> ICD -> symptoms'.
        
        Args:
            model_output: (symptom_logits, icd_logits, mu1, logvar1, mu2, logvar2)
            symptoms: Input/target symptoms [batch_size, seq_len]
            icd_codes: Target ICD codes [batch_size]
            pad_idx: Padding index
            
        Returns:
            total_loss: Scalar loss
            loss_dict: Dictionary with individual loss components
        """
        symptom_logits, icd_logits, mu1, logvar1, mu2, logvar2 = model_output
        
        # Reconstruction losses
        icd_recon = self.reconstruction_loss_icd(icd_logits, icd_codes).mean()
        symptom_recon = self.reconstruction_loss_symptoms(symptom_logits, symptoms, pad_idx).mean()
        
        # KL divergences
        kl1 = self.kl_divergence(mu1, logvar1).mean()
        kl2 = self.kl_divergence(mu2, logvar2).mean()
        
        # Cycle consistency
        cycle_loss = self.cycle_consistency_loss(mu1, logvar1, mu2, logvar2).mean()
        
        # Total loss
        total_loss = (icd_recon + symptom_recon + 
                     self.kl_weight * (kl1 + kl2) + 
                     self.cycle_weight * cycle_loss)
        
        loss_dict = {
            'cycle_fwd_icd_recon': icd_recon.item(),
            'cycle_fwd_symptom_recon': symptom_recon.item(),
            'cycle_fwd_kl1': kl1.item(),
            'cycle_fwd_kl2': kl2.item(),
            'cycle_fwd_cycle': cycle_loss.item(),
            'cycle_fwd_total': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def cycle_reverse_loss(self, model_output: Tuple, icd_codes: torch.Tensor,
                          symptoms: torch.Tensor, pad_idx: int = 0) -> Tuple[torch.Tensor, dict]:
        """
        Compute loss for reverse cycle: ICD -> symptoms -> ICD'.
        
        Args:
            model_output: (icd_logits, symptom_logits, mu1, logvar1, mu2, logvar2)
            icd_codes: Input/target ICD codes [batch_size]
            symptoms: Target symptoms [batch_size, seq_len]
            pad_idx: Padding index
            
        Returns:
            total_loss: Scalar loss
            loss_dict: Dictionary with individual loss components
        """
        icd_logits, symptom_logits, mu1, logvar1, mu2, logvar2 = model_output
        
        # Reconstruction losses
        icd_recon = self.reconstruction_loss_icd(icd_logits, icd_codes).mean()
        symptom_recon = self.reconstruction_loss_symptoms(symptom_logits, symptoms, pad_idx).mean()
        
        # KL divergences
        kl1 = self.kl_divergence(mu1, logvar1).mean()
        kl2 = self.kl_divergence(mu2, logvar2).mean()
        
        # Cycle consistency
        cycle_loss = self.cycle_consistency_loss(mu1, logvar1, mu2, logvar2).mean()
        
        # Total loss
        total_loss = (icd_recon + symptom_recon + 
                     self.kl_weight * (kl1 + kl2) + 
                     self.cycle_weight * cycle_loss)
        
        loss_dict = {
            'cycle_rev_icd_recon': icd_recon.item(),
            'cycle_rev_symptom_recon': symptom_recon.item(),
            'cycle_rev_kl1': kl1.item(),
            'cycle_rev_kl2': kl2.item(),
            'cycle_rev_cycle': cycle_loss.item(),
            'cycle_rev_total': total_loss.item()
        }
        
        return total_loss, loss_dict
