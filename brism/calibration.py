"""
Uncertainty calibration utilities including temperature scaling.

Temperature scaling adjusts model confidence to match actual accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
import numpy as np


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for model calibration.
    
    Learns a single scalar temperature parameter that scales logits
    to improve calibration (make predicted probabilities match actual frequencies).
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: Initial temperature value
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Model logits [batch_size, num_classes]
            
        Returns:
            Scaled logits [batch_size, num_classes]
        """
        return logits / self.temperature
    
    def get_temperature(self) -> float:
        """Get current temperature value."""
        return self.temperature.item()


def calibrate_temperature(
    model,
    calibration_loader: DataLoader,
    device: torch.device,
    max_iter: int = 50,
    lr: float = 0.01
) -> float:
    """
    Learn optimal temperature parameter on calibration set.
    
    Args:
        model: BRISM model with temperature parameter
        calibration_loader: DataLoader with calibration data
        device: Device to run on
        max_iter: Maximum optimization iterations
        lr: Learning rate
        
    Returns:
        Optimal temperature value
    """
    model.eval()
    model.to(device)
    
    # Store original temperature
    original_temp = model.temperature.detach().clone()
    
    # Only optimize temperature parameter
    optimizer = torch.optim.LBFGS(
        [model.temperature],
        lr=lr,
        max_iter=max_iter
    )
    
    # Collect all predictions and labels
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in calibration_loader:
            symptoms = batch['symptoms'].to(device)
            icd_codes = batch['icd_codes'].to(device)
            
            # Get logits (before temperature scaling)
            # Temporarily set temperature to 1.0
            temp_backup = model.temperature.detach().clone()
            with torch.no_grad():
                model.temperature.fill_(1.0)

            icd_logits, _, _ = model.forward_path(symptoms)

            # Restore temperature
            with torch.no_grad():
                model.temperature.copy_(temp_backup)
            
            all_logits.append(icd_logits)
            all_labels.append(icd_codes)
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Define optimization closure
    def eval_loss():
        optimizer.zero_grad()
        
        # Apply temperature scaling
        scaled_logits = all_logits / model.temperature
        
        # Negative log likelihood loss
        loss = F.cross_entropy(scaled_logits, all_labels)
        
        loss.backward()
        return loss
    
    # Optimize temperature
    optimizer.step(eval_loss)
    
    optimal_temp = model.temperature.item()
    
    # Ensure temperature is reasonable (between 0.1 and 10)
    if optimal_temp < 0.1:
        print(f"Warning: Temperature {optimal_temp:.4f} too low, clipping to 0.1")
        with torch.no_grad():
            model.temperature.fill_(0.1)
        optimal_temp = 0.1
    elif optimal_temp > 10.0:
        print(f"Warning: Temperature {optimal_temp:.4f} too high, clipping to 10.0")
        with torch.no_grad():
            model.temperature.fill_(10.0)
        optimal_temp = 10.0

    # Validate that calibration improved ECE
    from .metrics import compute_calibration_metrics

    # Compute ECE before calibration (with original temperature)
    with torch.no_grad():
        model.temperature.copy_(original_temp)
    with torch.no_grad():
        probs_before = F.softmax(all_logits / model.temperature, dim=-1)
    ece_before = compute_calibration_metrics(
        probs_before.cpu().numpy(),
        all_labels.cpu().numpy(),
        n_bins=10
    )['ece']
    
    # Compute ECE after calibration
    with torch.no_grad():
        model.temperature.fill_(optimal_temp)
    with torch.no_grad():
        probs_after = F.softmax(all_logits / model.temperature, dim=-1)
    ece_after = compute_calibration_metrics(
        probs_after.cpu().numpy(),
        all_labels.cpu().numpy(),
        n_bins=10
    )['ece']
    
    # Log results
    print(f"Temperature calibration: {original_temp.item():.4f} -> {optimal_temp:.4f}")
    print(f"ECE: {ece_before:.4f} -> {ece_after:.4f} (improvement: {ece_before - ece_after:.4f})")
    
    # Warn if calibration made things worse
    if ece_after > ece_before:
        print(f"WARNING: Calibration increased ECE by {ece_after - ece_before:.4f}. "
              f"Consider keeping original temperature or using more calibration data.")
    
    return optimal_temp


def evaluate_calibration_improvement(
    model,
    data_loader: DataLoader,
    device: torch.device,
    n_bins: int = 10
) -> dict:
    """
    Evaluate calibration before and after temperature scaling.
    
    Args:
        model: Model with temperature parameter
        data_loader: Evaluation data loader
        device: Device
        n_bins: Number of bins for calibration
        
    Returns:
        Dictionary with calibration metrics
    """
    from .metrics import compute_calibration_metrics
    
    model.eval()
    model.to(device)
    
    # Evaluate with current temperature
    all_probs_scaled = []
    all_labels = []
    
    # Also evaluate without temperature scaling
    all_probs_unscaled = []
    
    with torch.no_grad():
        for batch in data_loader:
            symptoms = batch['symptoms'].to(device)
            icd_codes = batch['icd_codes'].to(device)
            
            # Get predictions with temperature scaling
            icd_logits, _, _ = model.forward_path(symptoms)
            probs_scaled = F.softmax(icd_logits, dim=-1)
            
            # Get predictions without temperature scaling
            temp_backup = model.temperature.detach().clone()
            with torch.no_grad():
                model.temperature.fill_(1.0)
            icd_logits_unscaled, _, _ = model.forward_path(symptoms)
            probs_unscaled = F.softmax(icd_logits_unscaled, dim=-1)
            with torch.no_grad():
                model.temperature.copy_(temp_backup)
            
            all_probs_scaled.append(probs_scaled.cpu())
            all_probs_unscaled.append(probs_unscaled.cpu())
            all_labels.append(icd_codes.cpu())
    
    probs_scaled = torch.cat(all_probs_scaled, dim=0).numpy()
    probs_unscaled = torch.cat(all_probs_unscaled, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    
    # Compute calibration metrics
    cal_before = compute_calibration_metrics(probs_unscaled, labels, n_bins)
    cal_after = compute_calibration_metrics(probs_scaled, labels, n_bins)
    
    return {
        'before_scaling': cal_before,
        'after_scaling': cal_after,
        'temperature': model.temperature.item(),
        'ece_improvement': cal_before['ece'] - cal_after['ece']
    }
