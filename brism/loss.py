"""
Loss functions for BRISM training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from .icd_hierarchy import ICDHierarchy, compute_hierarchical_loss


def compute_class_weights(
    class_counts: Dict[int, int],
    num_classes: int,
    smoothing: float = 1.0
) -> torch.Tensor:
    """
    Compute inverse frequency weights for class balancing.
    
    Args:
        class_counts: Dictionary mapping class index to count
        num_classes: Total number of classes
        smoothing: Smoothing factor to avoid extreme weights
        
    Returns:
        Class weights tensor [num_classes]
    """
    # Initialize all weights to smoothing value
    weights = torch.ones(num_classes) * smoothing
    
    # Compute total samples
    total_samples = sum(class_counts.values())
    
    # Compute inverse frequency weights
    for class_idx, count in class_counts.items():
        if count > 0:
            weights[class_idx] = total_samples / (num_classes * count)
    
    # Normalize weights to have mean of 1.0
    weights = weights / weights.mean()
    
    return weights


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Focuses training on hard examples and down-weights easy examples.
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Class weights [num_classes] or None
            gamma: Focusing parameter (0 = cross-entropy, higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Loss value
        """
        # Compute probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get probability of true class
        batch_size = targets.size(0)
        p_t = probs[torch.arange(batch_size), targets]
        
        # Compute focal term
        focal_term = (1 - p_t) ** self.gamma
        
        # Compute cross-entropy
        ce = F.cross_entropy(logits, targets, reduction='none')
        
        # Apply focal term
        loss = focal_term * ce
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BRISMLoss(nn.Module):
    """
    Combined loss for BRISM model including:
    1. Reconstruction loss (VAE-style) with focal loss
    2. KL divergence on latent distributions
    3. Cycle consistency loss
    4. Hierarchical ICD loss (mandatory)
    5. Contrastive loss for better latent space (mandatory)
    """
    
    def __init__(self, kl_weight: float = 0.1, cycle_weight: float = 1.0,
                 icd_hierarchy: Optional[ICDHierarchy] = None, 
                 hierarchical_weight: float = 0.3,
                 hierarchical_temperature: float = 1.0,
                 class_weights: Optional[torch.Tensor] = None,
                 focal_gamma: float = 2.0,
                 contrastive_weight: float = 0.5,
                 contrastive_margin: float = 1.0,
                 contrastive_temperature: float = 0.5):
        """
        Args:
            kl_weight: Weight for KL divergence term
            cycle_weight: Weight for cycle consistency term
            icd_hierarchy: Optional ICD hierarchy for hierarchical loss
            hierarchical_weight: Weight for hierarchical loss component (default 0.3)
            hierarchical_temperature: Temperature for hierarchical distance penalty
            class_weights: Optional class weights for imbalanced data [icd_vocab_size]
            focal_gamma: Gamma parameter for focal loss (always enabled)
            contrastive_weight: Weight for contrastive loss term (default 0.5)
            contrastive_margin: Margin for triplet loss
            contrastive_temperature: Temperature for contrastive loss
        """
        super().__init__()
        self.kl_weight = kl_weight
        self.cycle_weight = cycle_weight
        self.icd_hierarchy = icd_hierarchy
        self.hierarchical_weight = hierarchical_weight
        self.hierarchical_temperature = hierarchical_temperature
        self._distance_matrix = None
        
        # Focal loss (always enabled)
        self.register_buffer('class_weights', class_weights)
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma, reduction='none')
        
        # Contrastive learning (always enabled)
        self.contrastive_weight = contrastive_weight
        self.contrastive_margin = contrastive_margin
        self.contrastive_temperature = contrastive_temperature
        
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
    
    def _ensure_distance_matrix_device(self, device: torch.device) -> torch.Tensor:
        """
        Ensure distance matrix is on the correct device.
        
        Args:
            device: Target device
            
        Returns:
            Distance matrix on the correct device
        """
        # Create matrix if it doesn't exist
        if self._distance_matrix is None:
            self._distance_matrix = self.icd_hierarchy.get_distance_tensor(device=device)
            return self._distance_matrix
        
        # Check if device matches
        if self._distance_matrix.device != device:
            # Recreate on correct device
            self._distance_matrix = self.icd_hierarchy.get_distance_tensor(device=device)
        
        return self._distance_matrix
    
    def reconstruction_loss_icd(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Reconstruction loss for ICD predictions with optional class weighting and focal loss.
        
        Args:
            logits: Predicted logits [batch_size, icd_vocab_size]
            target: Target ICD codes [batch_size]
            
        Returns:
            Loss [batch_size]
        """
        # Use focal loss (always enabled)
        loss = self.focal_loss(logits, target)
        
        # Add hierarchical loss if hierarchy is provided
        if self.icd_hierarchy is not None and self.hierarchical_weight > 0:
            # Get distance matrix on the correct device
            distance_matrix = self._ensure_distance_matrix_device(logits.device)
            
            # Compute hierarchical loss
            hier_loss = compute_hierarchical_loss(
                logits, target, distance_matrix, 
                temperature=self.hierarchical_temperature
            )
            
            # Combine losses
            total_loss = (1 - self.hierarchical_weight) * loss + self.hierarchical_weight * hier_loss
            return total_loss
        
        return loss
    
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
    
    def contrastive_loss_triplet(
        self,
        latents: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Triplet loss for contrastive learning.
        
        Symptoms from the same disease should have similar latent representations,
        symptoms from different diseases should be far apart.
        
        Args:
            latents: Latent representations [batch_size, latent_dim]
            labels: ICD code labels [batch_size]
            
        Returns:
            Contrastive loss scalar
        """
        batch_size = latents.size(0)
        
        if batch_size < 2:
            return torch.tensor(0.0, device=latents.device)
        
        # Compute pairwise distances
        # [batch_size, batch_size]
        dists = torch.cdist(latents, latents, p=2)
        
        # Create label similarity matrix
        # [batch_size, batch_size]
        label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # Mask out diagonal (self-similarity)
        diagonal_mask = ~torch.eye(batch_size, dtype=torch.bool, device=latents.device)
        
        # Positive pairs: same label, different samples
        positive_mask = label_matrix & diagonal_mask
        
        # Negative pairs: different labels
        negative_mask = ~label_matrix
        
        # Check if we have positive and negative pairs
        if not positive_mask.any() or not negative_mask.any():
            return torch.tensor(0.0, device=latents.device)
        
        # For each anchor, find hardest positive and hardest negative
        losses = []
        
        for i in range(batch_size):
            # Get positive samples for anchor i
            pos_dists = dists[i][positive_mask[i]]
            if len(pos_dists) == 0:
                continue
            
            # Get negative samples for anchor i
            neg_dists = dists[i][negative_mask[i]]
            if len(neg_dists) == 0:
                continue
            
            # Hardest positive (furthest same-class sample)
            hardest_positive = pos_dists.max()
            
            # Hardest negative (closest different-class sample)
            hardest_negative = neg_dists.min()
            
            # Triplet loss: max(0, d(a,p) - d(a,n) + margin)
            loss = F.relu(hardest_positive - hardest_negative + self.contrastive_margin)
            losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=latents.device)
        
        return torch.stack(losses).mean()
    
    def contrastive_loss_infonce(
        self,
        latents: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        InfoNCE contrastive loss.
        
        Uses temperature-scaled similarity to pull together same-class samples
        and push apart different-class samples.
        
        Args:
            latents: Latent representations [batch_size, latent_dim]
            labels: ICD code labels [batch_size]
            
        Returns:
            Contrastive loss scalar
        """
        batch_size = latents.size(0)
        
        if batch_size < 2:
            return torch.tensor(0.0, device=latents.device)
        
        # Normalize latents
        latents_norm = F.normalize(latents, p=2, dim=1)
        
        # Compute similarity matrix
        # [batch_size, batch_size]
        similarity = torch.mm(latents_norm, latents_norm.t()) / self.contrastive_temperature
        
        # Create label similarity matrix
        label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # Mask out diagonal
        diagonal_mask = ~torch.eye(batch_size, dtype=torch.bool, device=latents.device)
        
        # Positive pairs: same label, different samples
        positive_mask = label_matrix & diagonal_mask
        
        # Check if we have positive pairs
        if not positive_mask.any():
            return torch.tensor(0.0, device=latents.device)
        
        # For each sample, compute InfoNCE loss
        losses = []
        
        for i in range(batch_size):
            # Get positive samples for anchor i
            pos_mask = positive_mask[i]
            if not pos_mask.any():
                continue
            
            # Numerator: exp(similarity to positives)
            pos_sim = similarity[i][pos_mask]
            
            # Denominator: exp(similarity to all except self)
            all_sim = similarity[i][diagonal_mask[i]]
            
            # InfoNCE loss: -log(sum(exp(pos)) / sum(exp(all)))
            loss = -torch.log(pos_sim.exp().sum() / all_sim.exp().sum() + 1e-8)
            losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=latents.device)
        
        return torch.stack(losses).mean()
    
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
        }
        
        # Contrastive loss
        if self.contrastive_weight > 0:
            contrastive_loss = self.contrastive_loss_triplet(mu, icd_codes)
            total_loss = total_loss + self.contrastive_weight * contrastive_loss
            loss_dict['forward_contrastive'] = contrastive_loss.item()
        
        loss_dict['forward_total'] = total_loss.item()
        
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
