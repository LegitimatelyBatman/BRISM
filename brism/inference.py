"""
Inference functions for BRISM model with uncertainty quantification.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import stats
from .model import BRISM


def diagnose_with_confidence(
    model: BRISM,
    symptoms: torch.Tensor,
    device: torch.device,
    n_samples: Optional[int] = None,
    confidence_level: float = 0.95,
    top_k: int = 5
) -> Dict:
    """
    Diagnose from symptoms with confidence intervals using Monte Carlo dropout.

    Args:
        model: Trained BRISM model
        symptoms: Symptom token IDs [batch_size, seq_len] or [seq_len]
        device: Device to run on
        n_samples: Number of MC samples (defaults to model config)
        confidence_level: Confidence level for intervals (default 0.95)
        top_k: Number of top diagnoses to return

    Returns:
        Dictionary containing:
            - predictions: List of top-k ICD predictions with probabilities
            - confidence_intervals: Confidence intervals for each prediction
            - uncertainty: Overall uncertainty metrics
            - raw_probabilities: Full probability distribution with std
    """
    model.to(device)

    # Validate and convert symptoms to tensor if needed
    if not isinstance(symptoms, torch.Tensor):
        try:
            symptoms = torch.tensor(symptoms, dtype=torch.long)
        except (TypeError, ValueError) as e:
            raise TypeError(f"symptoms must be a torch.Tensor or convertible to one: {e}")

    symptoms = symptoms.to(device)

    # Add batch dimension if needed
    if symptoms.dim() == 1:
        symptoms = symptoms.unsqueeze(0)
    
    batch_size = symptoms.size(0)
    
    # Get predictions with uncertainty
    mean_probs, std_probs = model.predict_with_uncertainty(symptoms, n_samples)
    
    # Convert to numpy for statistical analysis
    mean_probs_np = mean_probs.cpu().numpy()
    std_probs_np = std_probs.cpu().numpy()
    
    results = []
    
    for i in range(batch_size):
        # Get top-k predictions
        top_k_indices = np.argsort(mean_probs_np[i])[-top_k:][::-1]
        
        predictions = []
        for idx in top_k_indices:
            mean = mean_probs_np[i, idx]
            std = std_probs_np[i, idx]
            
            # Compute confidence interval assuming normal distribution
            # For probabilities, we use normal approximation
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            ci_lower = max(0.0, mean - z_score * std)
            ci_upper = min(1.0, mean + z_score * std)
            
            predictions.append({
                'icd_code': int(idx),
                'probability': float(mean),
                'std': float(std),
                'confidence_interval': (float(ci_lower), float(ci_upper)),
                'confidence_level': confidence_level
            })
        
        # Overall uncertainty metrics
        # Entropy of mean distribution
        entropy = -np.sum(mean_probs_np[i] * np.log(mean_probs_np[i] + 1e-10))
        
        # Average standard deviation (epistemic uncertainty)
        avg_std = np.mean(std_probs_np[i])
        
        # Predictive entropy (total uncertainty)
        predictive_entropy = entropy
        
        result = {
            'predictions': predictions,
            'uncertainty': {
                'entropy': float(entropy),
                'average_std': float(avg_std),
                'predictive_entropy': float(predictive_entropy)
            },
            'raw_probabilities': {
                'mean': mean_probs_np[i].tolist(),
                'std': std_probs_np[i].tolist()
            }
        }
        
        results.append(result)
    
    # Return single result if single input
    if batch_size == 1:
        return results[0]
    
    return results


def generate_symptoms_with_uncertainty(
    model: BRISM,
    icd_code: torch.Tensor,
    device: torch.device,
    n_samples: Optional[int] = None,
    max_length: Optional[int] = None
) -> Dict:
    """
    Generate symptom sequences from ICD code with uncertainty.
    
    Args:
        model: Trained BRISM model
        icd_code: ICD code token ID [batch_size] or scalar
        device: Device to run on
        n_samples: Number of samples to generate
        max_length: Maximum sequence length
        
    Returns:
        Dictionary containing generated sequences and statistics
    """
    model.to(device)
    icd_code = icd_code.to(device)

    if n_samples is None:
        n_samples = model.config.mc_samples
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}")

    # Add batch dimension if needed
    if icd_code.dim() == 0:
        icd_code = icd_code.unsqueeze(0)

    batch_size = icd_code.size(0)

    previous_mode = model.training
    model.train()  # Enable dropout

    generated_sequences = []

    try:
        with torch.no_grad():
            for _ in range(n_samples):
                symptom_logits, _, _ = model.reverse_path(icd_code, target_symptoms=None)
                symptom_ids = symptom_logits.argmax(dim=-1)
                generated_sequences.append(symptom_ids.cpu().numpy())
    finally:
        if not previous_mode:
            model.eval()

    generated_sequences = np.array(generated_sequences)  # [n_samples, batch_size, seq_len]

    results = []
    
    for i in range(batch_size):
        # Most common sequence
        sequences_i = generated_sequences[:, i, :]
        
        # Token-wise mode (most common token at each position)
        mode_sequence = []
        token_probabilities = []
        
        seq_len = sequences_i.shape[1]
        for pos in range(seq_len):
            tokens_at_pos = sequences_i[:, pos]
            unique, counts = np.unique(tokens_at_pos, return_counts=True)
            mode_idx = unique[np.argmax(counts)]
            mode_prob = counts[np.argmax(counts)] / n_samples
            
            mode_sequence.append(int(mode_idx))
            token_probabilities.append(float(mode_prob))
        
        # Sequence diversity (unique sequences / total sequences)
        unique_sequences = len(set(tuple(seq) for seq in sequences_i))
        diversity = unique_sequences / n_samples
        
        result = {
            'mode_sequence': mode_sequence,
            'token_probabilities': token_probabilities,
            'all_sequences': sequences_i.tolist(),
            'diversity': float(diversity),
            'n_unique_sequences': unique_sequences
        }
        
        results.append(result)
    
    if batch_size == 1:
        return results[0]
    
    return results


def batch_diagnose(
    model: BRISM,
    symptoms_list: List[torch.Tensor],
    device: torch.device,
    **kwargs
) -> List[Dict]:
    """
    Diagnose multiple symptom sequences.
    
    Args:
        model: Trained BRISM model
        symptoms_list: List of symptom tensors
        device: Device
        **kwargs: Additional arguments for diagnose_with_confidence
        
    Returns:
        List of diagnosis results
    """
    results = []
    
    for symptoms in symptoms_list:
        result = diagnose_with_confidence(model, symptoms, device, **kwargs)
        results.append(result)
    
    return results


def generate_symptoms_beam_search(
    model: BRISM,
    icd_code: torch.Tensor,
    device: torch.device,
    beam_width: int = 5,
    temperature: float = 1.0,
    length_penalty: float = 1.0,
    return_all_beams: bool = False
) -> Dict:
    """
    Generate symptom sequences from ICD code using beam search.
    
    Args:
        model: Trained BRISM model
        icd_code: ICD code token ID [batch_size] or scalar
        device: Device to run on
        beam_width: Number of beams to keep
        temperature: Temperature for softmax (higher = more diverse)
        length_penalty: Penalty for sequence length (>1 favors longer sequences)
        return_all_beams: If True, return all beams; if False, return only best
        
    Returns:
        Dictionary containing generated sequences and scores
    """
    model.to(device)
    icd_code = icd_code.to(device)
    
    # Add batch dimension if needed
    if icd_code.dim() == 0:
        icd_code = icd_code.unsqueeze(0)
    
    batch_size = icd_code.size(0)
    
    previous_mode = model.training
    model.eval()  # Use eval mode for deterministic beam search

    try:
        with torch.no_grad():
            # Encode ICD to latent
            icd_embeds = model.icd_embedding(icd_code)
            mu, logvar = model.icd_encoder(icd_embeds)
            z = model.reparameterize(mu, logvar)

            # Generate with beam search
            sequences, scores, lengths = model.symptom_decoder.beam_search(
                z,
                beam_width=beam_width,
                temperature=temperature,
                length_penalty=length_penalty
            )
    finally:
        if previous_mode:
            model.train()
        else:
            model.eval()
    
    results = []
    
    for i in range(batch_size):
        if return_all_beams:
            # Return all beams
            beam_results = []
            for k in range(beam_width):
                beam_results.append({
                    'sequence': sequences[i, k].cpu().numpy().tolist(),
                    'score': float(scores[i, k]),
                    'length': int(lengths[i, k])
                })
            
            result = {
                'beams': beam_results,
                'best_sequence': sequences[i, 0].cpu().numpy().tolist(),
                'best_score': float(scores[i, 0])
            }
        else:
            # Return only best beam
            result = {
                'sequence': sequences[i, 0].cpu().numpy().tolist(),
                'score': float(scores[i, 0]),
                'length': int(lengths[i, 0])
            }
        
        results.append(result)
    
    if batch_size == 1:
        return results[0]
    
    return results


def evaluate_model_uncertainty(
    model: BRISM,
    data_loader,
    device: torch.device,
    n_samples: int = 20
) -> Dict:
    """
    Evaluate model uncertainty on a dataset.
    
    Args:
        model: Trained BRISM model
        data_loader: Data loader
        device: Device
        n_samples: Number of MC samples
        
    Returns:
        Dictionary of uncertainty statistics
    """
    model.to(device)
    
    all_entropies = []
    all_stds = []
    all_accuracies = []
    
    for batch in data_loader:
        symptoms = batch['symptoms'].to(device)
        icd_codes = batch['icd_codes'].to(device)
        
        # Get predictions
        mean_probs, std_probs = model.predict_with_uncertainty(symptoms, n_samples)
        
        # Predicted classes
        pred_classes = mean_probs.argmax(dim=-1)
        
        # Accuracy
        accuracy = (pred_classes == icd_codes).float().mean().item()
        all_accuracies.append(accuracy)
        
        # Uncertainty metrics
        mean_probs_np = mean_probs.cpu().numpy()
        std_probs_np = std_probs.cpu().numpy()
        
        for i in range(mean_probs_np.shape[0]):
            entropy = -np.sum(mean_probs_np[i] * np.log(mean_probs_np[i] + 1e-10))
            avg_std = np.mean(std_probs_np[i])
            
            all_entropies.append(entropy)
            all_stds.append(avg_std)
    
    return {
        'mean_entropy': float(np.mean(all_entropies)),
        'std_entropy': float(np.std(all_entropies)),
        'mean_std': float(np.mean(all_stds)),
        'std_std': float(np.std(all_stds)),
        'accuracy': float(np.mean(all_accuracies))
    }
