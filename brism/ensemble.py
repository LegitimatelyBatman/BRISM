"""
Ensemble methods for better uncertainty quantification.

Supports:
1. Multi-model ensembles with different random seeds
2. Pseudo-ensembles using different dropout masks
3. Combined aleatoric and epistemic uncertainty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np
from .model import BRISM, BRISMConfig


class BRISMEnsemble:
    """
    Ensemble of BRISM models for improved uncertainty quantification.
    
    Can use either:
    - Multiple independently trained models (true ensemble)
    - Single model with multiple dropout masks (pseudo-ensemble)
    """
    
    def __init__(
        self,
        models: Optional[List[BRISM]] = None,
        config: Optional[BRISMConfig] = None,
        n_models: int = 5,
        use_pseudo_ensemble: bool = False
    ):
        """
        Initialize BRISM ensemble.
        
        Args:
            models: List of trained BRISM models (for true ensemble)
            config: Configuration for creating new models
            n_models: Number of models in ensemble
            use_pseudo_ensemble: Use pseudo-ensemble (dropout masks) instead
        """
        self.use_pseudo_ensemble = use_pseudo_ensemble
        
        if use_pseudo_ensemble:
            # Pseudo-ensemble: single model with dropout
            if models is not None and len(models) > 0:
                self.model = models[0]
            elif config is not None:
                self.model = BRISM(config)
            else:
                raise ValueError("Must provide either models or config for pseudo-ensemble")
            
            # Validate minimum n_models for meaningful uncertainty estimates
            if n_models < 2:
                raise ValueError(
                    f"Pseudo-ensemble requires n_models >= 2 for meaningful uncertainty estimates, "
                    f"got n_models={n_models}. With n_models=1, standard deviation will always be zero."
                )
            
            self.models = None
            self.n_models = n_models
        else:
            # True ensemble: multiple models
            if models is not None:
                self.models = models
                self.n_models = len(models)
            elif config is not None:
                # Create models with different initializations
                self.models = [BRISM(config) for _ in range(n_models)]
                self.n_models = n_models
            else:
                raise ValueError("Must provide either models or config for ensemble")
            
            self.model = None
    
    def predict_with_uncertainty(
        self,
        symptoms: torch.Tensor,
        n_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Make predictions with ensemble uncertainty.
        
        Args:
            symptoms: Symptom token IDs [batch_size, seq_len]
            n_samples: Number of samples for pseudo-ensemble (ignored for true ensemble)
            
        Returns:
            mean_probs: Mean probabilities [batch_size, icd_vocab_size]
            std_probs: Standard deviation [batch_size, icd_vocab_size]
            uncertainty_dict: Dictionary with uncertainty metrics
        """
        if self.use_pseudo_ensemble:
            return self._predict_pseudo_ensemble(symptoms, n_samples)
        else:
            return self._predict_true_ensemble(symptoms)
    
    def _predict_true_ensemble(
        self,
        symptoms: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Predict using true ensemble of models with incremental statistics.
        
        This implementation computes running mean and variance incrementally
        to reduce memory usage by a factor equal to the number of models.
        
        Args:
            symptoms: Symptom token IDs [batch_size, seq_len]
            
        Returns:
            mean_probs: Mean probabilities [batch_size, icd_vocab_size]
            std_probs: Standard deviation [batch_size, icd_vocab_size]
            uncertainty_dict: Dictionary with uncertainty metrics
        """
        device = symptoms.device
        batch_size = symptoms.size(0)
        
        # Initialize accumulators for incremental statistics
        mean_probs = None
        m2_probs = None  # Sum of squared differences from mean (for variance)
        aleatoric_sum = None  # Running sum of entropies
        
        # Store only the predictions needed for agreement computation
        # (top predictions per model, not full probability distributions)
        top_predictions = []
        
        # Get predictions from each model incrementally
        for idx, model in enumerate(self.models):
            model.to(device)
            model.eval()
            
            with torch.no_grad():
                icd_logits, _, _ = model.forward_path(symptoms)
                probs = F.softmax(icd_logits, dim=-1)
                
                # Update running statistics using Welford's algorithm
                if mean_probs is None:
                    # First iteration
                    mean_probs = probs
                    m2_probs = torch.zeros_like(probs)
                    aleatoric_sum = torch.zeros(batch_size, device=device)
                else:
                    # Incremental update
                    delta = probs - mean_probs
                    mean_probs = mean_probs + delta / (idx + 1)
                    delta2 = probs - mean_probs
                    m2_probs = m2_probs + delta * delta2
                
                # Accumulate aleatoric uncertainty (entropy)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                aleatoric_sum = aleatoric_sum + entropy
                
                # Store only top prediction for agreement (much smaller memory)
                top_predictions.append(probs.argmax(dim=-1).cpu().numpy())
        
        # Compute final statistics
        # Variance = m2 / n
        variance_probs = m2_probs / self.n_models
        std_probs = torch.sqrt(variance_probs + 1e-10)
        
        # Epistemic uncertainty: average std across classes
        epistemic_uncertainty = std_probs.mean(dim=-1)
        
        # Aleatoric uncertainty: average entropy
        aleatoric_uncertainty = aleatoric_sum / self.n_models
        
        # Total uncertainty: entropy of mean prediction
        total_uncertainty = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)
        
        # Store only top predictions for agreement, not all probabilities
        top_predictions = np.stack(top_predictions)  # [n_models, batch_size]
        
        uncertainty_dict = {
            'epistemic': epistemic_uncertainty.cpu().numpy(),
            'aleatoric': aleatoric_uncertainty.cpu().numpy(),
            'total': total_uncertainty.cpu().numpy(),
            'all_predictions': top_predictions  # Only top predictions, not full distributions
        }
        
        return mean_probs, std_probs, uncertainty_dict
    
    def _predict_pseudo_ensemble(
        self,
        symptoms: torch.Tensor,
        n_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Predict using pseudo-ensemble (multiple dropout masks).
        
        Args:
            symptoms: Symptom token IDs [batch_size, seq_len]
            n_samples: Number of samples (defaults to n_models)
            
        Returns:
            mean_probs: Mean probabilities [batch_size, icd_vocab_size]
            std_probs: Standard deviation [batch_size, icd_vocab_size]
            uncertainty_dict: Dictionary with uncertainty metrics
        """
        if n_samples is None:
            n_samples = self.n_models
        
        device = symptoms.device
        self.model.to(device)
        self.model.train()  # Enable dropout
        
        all_probs = []
        
        # Get predictions with different dropout masks
        with torch.no_grad():
            for _ in range(n_samples):
                icd_logits, _, _ = self.model.forward_path(symptoms)
                probs = F.softmax(icd_logits, dim=-1)
                all_probs.append(probs)
        
        # Stack predictions: [n_samples, batch_size, icd_vocab_size]
        all_probs = torch.stack(all_probs)
        
        # Compute mean and std
        mean_probs = all_probs.mean(dim=0)
        std_probs = all_probs.std(dim=0)
        
        # Compute uncertainty metrics
        epistemic_uncertainty = std_probs.mean(dim=-1)
        entropies = -(all_probs * torch.log(all_probs + 1e-10)).sum(dim=-1)
        aleatoric_uncertainty = entropies.mean(dim=0)
        total_uncertainty = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)
        
        uncertainty_dict = {
            'epistemic': epistemic_uncertainty.cpu().numpy(),
            'aleatoric': aleatoric_uncertainty.cpu().numpy(),
            'total': total_uncertainty.cpu().numpy(),
            'all_predictions': all_probs.cpu().numpy()
        }
        
        return mean_probs, std_probs, uncertainty_dict
    
    def diagnose_with_ensemble(
        self,
        symptoms: torch.Tensor,
        top_k: int = 5,
        n_samples: Optional[int] = None
    ) -> Dict:
        """
        Diagnose with ensemble uncertainty quantification.
        
        Args:
            symptoms: Symptom token IDs [batch_size, seq_len] or [seq_len]
            top_k: Number of top predictions to return
            n_samples: Number of samples for pseudo-ensemble
            
        Returns:
            Dictionary with predictions and uncertainty metrics
        """
        # Add batch dimension if needed
        if symptoms.dim() == 1:
            symptoms = symptoms.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Get predictions
        mean_probs, std_probs, uncertainty_dict = self.predict_with_uncertainty(
            symptoms, n_samples
        )
        
        # Get top-k predictions
        mean_probs_np = mean_probs.cpu().numpy()
        std_probs_np = std_probs.cpu().numpy()
        
        batch_size = symptoms.size(0)
        results = []
        
        for i in range(batch_size):
            top_k_indices = np.argsort(mean_probs_np[i])[-top_k:][::-1]
            
            predictions = []
            for idx in top_k_indices:
                predictions.append({
                    'icd_code': int(idx),
                    'probability': float(mean_probs_np[i, idx]),
                    'std': float(std_probs_np[i, idx]),
                    'coefficient_of_variation': float(std_probs_np[i, idx] / (mean_probs_np[i, idx] + 1e-10))
                })
            
            result = {
                'predictions': predictions,
                'uncertainty': {
                    'epistemic': float(uncertainty_dict['epistemic'][i]),
                    'aleatoric': float(uncertainty_dict['aleatoric'][i]),
                    'total': float(uncertainty_dict['total'][i])
                },
                'ensemble_agreement': self._compute_agreement(
                    uncertainty_dict['all_predictions'][:, i, :]
                )
            }
            
            results.append(result)
        
        if squeeze_output:
            return results[0]
        
        return results
    
    def _compute_agreement(self, predictions: np.ndarray) -> float:
        """
        Compute ensemble agreement (fraction of models predicting same top class).
        
        Args:
            predictions: Top predictions from all models [n_models] (already argmax'ed)
            
        Returns:
            Agreement score [0, 1]
        """
        # predictions is already argmax'ed in the new implementation
        if predictions.ndim == 2:
            # Old format: [n_models, icd_vocab_size] - take argmax
            top_classes = predictions.argmax(axis=-1)
        else:
            # New format: [n_models] - already argmax'ed
            top_classes = predictions
        
        most_common = np.bincount(top_classes).argmax()
        agreement = (top_classes == most_common).mean()
        return float(agreement)


def train_ensemble(
    config: BRISMConfig,
    train_loader,
    optimizer_fn,
    loss_fn,
    num_epochs: int,
    device: torch.device,
    n_models: int = 5,
    val_loader = None,
    random_seeds: Optional[List[int]] = None
) -> List[BRISM]:
    """
    Train an ensemble of BRISM models with different random seeds.
    
    Args:
        config: Model configuration
        train_loader: Training data loader
        optimizer_fn: Function that creates optimizer given model
        loss_fn: Loss function
        num_epochs: Number of epochs
        device: Device to train on
        n_models: Number of models in ensemble
        val_loader: Optional validation loader
        random_seeds: Optional list of random seeds (one per model)
        
    Returns:
        List of trained models
    """
    from .train import train_brism
    
    if random_seeds is None:
        random_seeds = [42 + i * 1000 for i in range(n_models)]
    
    models = []
    
    for i, seed in enumerate(random_seeds[:n_models]):
        print(f"\n{'='*60}")
        print(f"Training Ensemble Model {i+1}/{n_models} (seed={seed})")
        print('='*60)
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create model
        model = BRISM(config)
        model.to(device)
        
        # Create optimizer
        optimizer = optimizer_fn(model)
        
        # Train model
        history = train_brism(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            num_epochs=num_epochs,
            device=device,
            val_loader=val_loader
        )
        
        models.append(model)
        
        print(f"Model {i+1} final loss: {history['train_loss'][-1]:.4f}")
    
    return models


def load_ensemble(
    checkpoint_paths: List[str],
    config: BRISMConfig,
    device: torch.device
) -> BRISMEnsemble:
    """
    Load ensemble from saved checkpoints.
    
    Args:
        checkpoint_paths: List of paths to model checkpoints
        config: Model configuration
        device: Device to load models on
        
    Returns:
        BRISMEnsemble with loaded models
    """
    from .train import load_checkpoint
    
    models = []
    
    for path in checkpoint_paths:
        model = BRISM(config)
        checkpoint = load_checkpoint(path, model=model, device=device)
        models.append(model)
    
    return BRISMEnsemble(models=models)
