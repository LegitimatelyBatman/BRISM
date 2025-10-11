"""
Interpretability tools for BRISM model.

Provides:
1. Gradient-based attention visualization
2. Integrated gradients for feature attribution
3. Counterfactual explanations
4. Attention rollout for multi-layer attention
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from .model import BRISM


class IntegratedGradients:
    """
    Integrated Gradients for feature attribution.
    
    Computes the attribution of each input feature to the model's prediction
    by integrating gradients along a path from a baseline to the input.
    """
    
    def __init__(self, model: BRISM):
        """
        Initialize Integrated Gradients.
        
        Args:
            model: BRISM model to explain
        """
        self.model = model
    
    def attribute(
        self,
        symptoms: torch.Tensor,
        target_icd: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
        n_steps: int = 50
    ) -> torch.Tensor:
        """
        Compute integrated gradients for symptom attributions.
        
        Args:
            symptoms: Symptom token IDs [batch_size, seq_len] or [seq_len]
            target_icd: Target ICD code to explain (if None, uses predicted class)
            baseline: Baseline input (if None, uses zeros/padding)
            n_steps: Number of integration steps
            
        Returns:
            Attribution scores [batch_size, seq_len] or [seq_len]
        """
        # Add batch dimension if needed
        if symptoms.dim() == 1:
            symptoms = symptoms.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len = symptoms.shape
        device = symptoms.device
        
        # Create baseline (all zeros/padding)
        if baseline is None:
            baseline = torch.zeros_like(symptoms)
        
        # Get embeddings for symptoms and baseline
        self.model.eval()
        symptom_embeds = self.model.symptom_embedding(symptoms)
        baseline_embeds = self.model.symptom_embedding(baseline)
        
        # Compute path from baseline to input
        alphas = torch.linspace(0, 1, n_steps + 1, device=device).view(-1, 1, 1, 1)
        
        # Interpolate embeddings
        path_embeds = baseline_embeds.unsqueeze(0) + alphas * (symptom_embeds.unsqueeze(0) - baseline_embeds.unsqueeze(0))
        path_embeds = path_embeds.view(-1, seq_len, symptom_embeds.size(-1))
        
        # Enable gradients for embeddings
        path_embeds.requires_grad_(True)
        
        # Forward pass through the model for each step
        # We need to manually run through the model components
        gradients = []
        
        for step in range(n_steps + 1):
            # Get embedding for this step
            step_embeds = path_embeds[step * batch_size:(step + 1) * batch_size]
            step_embeds = step_embeds.detach().requires_grad_(True)
            
            # Apply temporal encoding (always enabled)
            step_embeds_temporal = self.model.temporal_encoding(step_embeds, None)
            
            # Aggregate symptoms with attention (always enabled)
            mask = (symptoms != 0).float()
            aggregated, _ = self.model.symptom_attention(step_embeds_temporal, mask)
            
            # Encode to latent
            mu, logvar = self.model.symptom_encoder(aggregated)
            z = self.model.reparameterize(mu, logvar)
            
            # Decode to ICD logits
            icd_logits = self.model.icd_decoder(z)
            
            # Get target class
            if target_icd is None:
                target_icd_val = icd_logits.argmax(dim=-1)[0].item()
            else:
                target_icd_val = target_icd
            
            # Compute gradients w.r.t. target class and input embeddings
            target_logits = icd_logits[:, target_icd_val].sum()
            target_logits.backward()
            
            # Store gradients w.r.t. original step_embeds
            if step_embeds.grad is not None:
                gradients.append(step_embeds.grad.detach().clone())
            else:
                # If no gradients, use zeros
                gradients.append(torch.zeros_like(step_embeds))
        
        # Average gradients along path
        gradients = torch.stack(gradients)
        avg_gradients = gradients.mean(dim=0)
        
        # Compute integrated gradients
        integrated_grads = (symptom_embeds - baseline_embeds) * avg_gradients
        
        # Sum over embedding dimension to get per-token attribution
        attributions = integrated_grads.sum(dim=-1)
        
        if squeeze_output:
            attributions = attributions.squeeze(0)
        
        return attributions


class AttentionVisualization:
    """
    Gradient-based attention visualization.
    
    Shows which symptoms most influenced each diagnosis prediction.
    """
    
    def __init__(self, model: BRISM):
        """
        Initialize attention visualization.
        
        Args:
            model: BRISM model with attention mechanism
        """
        self.model = model
    
    def get_attention_weights(
        self,
        symptoms: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract attention weights from the model.
        
        Args:
            symptoms: Symptom token IDs [batch_size, seq_len] or [seq_len]
            timestamps: Optional timestamps [batch_size, seq_len] or [seq_len]
            
        Returns:
            attention_weights: Attention weights [batch_size, seq_len]
            predictions: ICD predictions [batch_size, icd_vocab_size]
        """
        # Add batch dimension if needed
        if symptoms.dim() == 1:
            symptoms = symptoms.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        self.model.eval()
        
        with torch.no_grad():
            # Embed symptoms
            symptom_embeds = self.model.symptom_embedding(symptoms)
            
            # Apply temporal encoding (always enabled)
            symptom_embeds = self.model.temporal_encoding(symptom_embeds, timestamps)
            
            # Get attention weights
            mask = (symptoms != 0).float()
            aggregated, attention_weights = self.model.symptom_attention(symptom_embeds, mask)
            
            # Get predictions
            mu, logvar = self.model.symptom_encoder(aggregated)
            z = self.model.reparameterize(mu, logvar)
            icd_logits = self.model.icd_decoder(z)
            predictions = F.softmax(icd_logits, dim=-1)
        
        if squeeze_output:
            attention_weights = attention_weights.squeeze(0)
            predictions = predictions.squeeze(0)
        
        return attention_weights, predictions
    
    def get_gradient_based_importance(
        self,
        symptoms: torch.Tensor,
        target_icd: Optional[int] = None,
        timestamps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute gradient-based importance scores for symptoms.
        
        Args:
            symptoms: Symptom token IDs [batch_size, seq_len] or [seq_len]
            target_icd: Target ICD code (if None, uses predicted class)
            timestamps: Optional timestamps [batch_size, seq_len] or [seq_len]
            
        Returns:
            importance_scores: Gradient-based importance [batch_size, seq_len] or [seq_len]
        """
        # Add batch dimension if needed
        if symptoms.dim() == 1:
            symptoms = symptoms.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        self.model.eval()
        
        # Get symptom embeddings with gradients
        symptom_embeds = self.model.symptom_embedding(symptoms)
        symptom_embeds.requires_grad_(True)
        
        # Forward pass with temporal encoding (always enabled)
        symptom_embeds_temporal = self.model.temporal_encoding(symptom_embeds, timestamps)
        
        mask = (symptoms != 0).float()
        aggregated, _ = self.model.symptom_attention(symptom_embeds_temporal, mask)
        
        mu, logvar = self.model.symptom_encoder(aggregated)
        z = self.model.reparameterize(mu, logvar)
        icd_logits = self.model.icd_decoder(z)
        
        # Get target class
        if target_icd is None:
            target_icd = icd_logits.argmax(dim=-1)
        else:
            target_icd = torch.tensor([target_icd] * symptoms.size(0), device=symptoms.device)
        
        # Compute gradients
        target_logits = icd_logits.gather(1, target_icd.unsqueeze(-1)).sum()
        target_logits.backward()
        
        # Get gradient magnitude
        importance = symptom_embeds.grad.abs().sum(dim=-1)
        
        if squeeze_output:
            importance = importance.squeeze(0)
        
        return importance


class CounterfactualExplanations:
    """
    Generate counterfactual explanations.
    
    Shows how removing or changing symptoms affects prediction probabilities.
    """
    
    def __init__(self, model: BRISM):
        """
        Initialize counterfactual explanations.
        
        Args:
            model: BRISM model
        """
        self.model = model
    
    def explain_by_removal(
        self,
        symptoms: torch.Tensor,
        target_icd: Optional[int] = None,
        timestamps: Optional[torch.Tensor] = None
    ) -> List[Dict]:
        """
        Explain prediction by removing each symptom one at a time.
        
        Args:
            symptoms: Symptom token IDs [seq_len]
            target_icd: Target ICD code (if None, uses predicted class)
            timestamps: Optional timestamps [seq_len]
            
        Returns:
            List of dictionaries with symptom removal effects
        """
        self.model.eval()
        
        # Get original prediction
        with torch.no_grad():
            original_probs, _ = self.model.predict_with_uncertainty(
                symptoms.unsqueeze(0), n_samples=1
            )
            original_probs = original_probs.squeeze(0)
        
        # Get target ICD
        if target_icd is None:
            target_icd = original_probs.argmax().item()
        
        original_prob = original_probs[target_icd].item()
        
        # Test removing each symptom
        explanations = []
        symptom_ids = symptoms.cpu().numpy()
        
        for i, symptom_id in enumerate(symptom_ids):
            if symptom_id == 0:  # Skip padding
                continue
            
            # Create modified symptoms with this symptom removed
            modified_symptoms = symptoms.clone()
            modified_symptoms[i] = 0  # Set to padding
            
            # Get prediction without this symptom
            with torch.no_grad():
                modified_probs, _ = self.model.predict_with_uncertainty(
                    modified_symptoms.unsqueeze(0), n_samples=1
                )
                modified_probs = modified_probs.squeeze(0)
            
            modified_prob = modified_probs[target_icd].item()
            prob_drop = original_prob - modified_prob
            prob_drop_pct = (prob_drop / (original_prob + 1e-10)) * 100
            
            explanations.append({
                'symptom_position': i,
                'symptom_id': int(symptom_id),
                'original_probability': original_prob,
                'modified_probability': modified_prob,
                'probability_drop': prob_drop,
                'probability_drop_percentage': prob_drop_pct
            })
        
        # Sort by probability drop (most important first)
        explanations.sort(key=lambda x: x['probability_drop'], reverse=True)
        
        return explanations
    
    def minimal_sufficient_set(
        self,
        symptoms: torch.Tensor,
        target_icd: Optional[int] = None,
        threshold: float = 0.9,
        timestamps: Optional[torch.Tensor] = None
    ) -> Tuple[List[int], float]:
        """
        Find minimal set of symptoms sufficient for prediction.
        
        Args:
            symptoms: Symptom token IDs [seq_len]
            target_icd: Target ICD code (if None, uses predicted class)
            threshold: Probability threshold for sufficiency
            timestamps: Optional timestamps [seq_len]
            
        Returns:
            minimal_set: Indices of sufficient symptoms
            probability: Prediction probability with minimal set
        """
        self.model.eval()
        
        # Get original prediction
        with torch.no_grad():
            original_probs, _ = self.model.predict_with_uncertainty(
                symptoms.unsqueeze(0), n_samples=1
            )
            original_probs = original_probs.squeeze(0)
        
        # Get target ICD
        if target_icd is None:
            target_icd = original_probs.argmax().item()
        
        original_prob = original_probs[target_icd].item()
        target_prob = original_prob * threshold
        
        # Get importance scores for ranking
        symptom_ids = symptoms.cpu().numpy()
        non_padding = [i for i, sid in enumerate(symptom_ids) if sid != 0]
        
        # Greedy selection starting with empty set
        selected = []
        
        for idx in non_padding:
            # Try adding this symptom
            candidate_set = selected + [idx]
            
            # Create symptom sequence with only selected symptoms
            test_symptoms = torch.zeros_like(symptoms)
            for pos in candidate_set:
                test_symptoms[pos] = symptoms[pos]
            
            # Get prediction
            with torch.no_grad():
                test_probs, _ = self.model.predict_with_uncertainty(
                    test_symptoms.unsqueeze(0), n_samples=1
                )
                test_probs = test_probs.squeeze(0)
            
            test_prob = test_probs[target_icd].item()
            
            # Add if increases probability
            if test_prob > test_probs[target_icd].item() if len(selected) == 0 else selected_prob:
                selected.append(idx)
                selected_prob = test_prob
                
                # Check if sufficient
                if test_prob >= target_prob:
                    break
        
        return selected, selected_prob if selected else 0.0


class AttentionRollout:
    """
    Attention rollout for multi-layer attention visualization.
    
    Aggregates attention weights across multiple layers to show
    the flow of information through the network.
    """
    
    def __init__(self, model: BRISM):
        """
        Initialize attention rollout.
        
        Args:
            model: BRISM model with attention mechanism
        """
        self.model = model
    
    def compute_rollout(
        self,
        symptoms: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute attention rollout.
        
        For the current BRISM architecture with single attention layer,
        this returns the attention weights. In multi-layer architectures,
        this would aggregate attention across layers.
        
        Args:
            symptoms: Symptom token IDs [batch_size, seq_len] or [seq_len]
            timestamps: Optional timestamps [batch_size, seq_len] or [seq_len]
            
        Returns:
            rollout_attention: Aggregated attention [batch_size, seq_len] or [seq_len]
        """
        # For single-layer attention, rollout is just the attention weights
        vis = AttentionVisualization(self.model)
        attention_weights, _ = vis.get_attention_weights(symptoms, timestamps)
        
        return attention_weights


def explain_prediction(
    model: BRISM,
    symptoms: torch.Tensor,
    symptom_vocab: Optional[Dict[int, str]] = None,
    top_k: int = 5,
    method: str = 'all'
) -> Dict:
    """
    Comprehensive prediction explanation.
    
    Args:
        model: BRISM model
        symptoms: Symptom token IDs [seq_len]
        symptom_vocab: Optional mapping from IDs to symptom names
        top_k: Number of top ICDs to explain
        method: Explanation method ('attention', 'gradients', 'counterfactual', 'all')
        
    Returns:
        Dictionary containing various explanations
    """
    model.eval()
    device = next(model.parameters()).device
    symptoms = symptoms.to(device)
    
    explanations = {}
    
    # Get prediction
    with torch.no_grad():
        probs, _ = model.predict_with_uncertainty(symptoms.unsqueeze(0), n_samples=1)
        probs = probs.squeeze(0)
    
    top_k_probs, top_k_icds = probs.topk(top_k)
    
    explanations['predictions'] = [
        {'icd_code': int(icd), 'probability': float(prob)}
        for icd, prob in zip(top_k_icds, top_k_probs)
    ]
    
    # Attention-based explanation (always available)
    if method in ['attention', 'all']:
        vis = AttentionVisualization(model)
        attention_weights, _ = vis.get_attention_weights(symptoms)
        
        explanations['attention_weights'] = attention_weights.cpu().numpy().tolist()
        
        # Get top attended symptoms
        non_padding = (symptoms != 0).cpu().numpy()
        attended_indices = attention_weights.argsort(descending=True).cpu().numpy()
        attended_indices = [int(i) for i in attended_indices if non_padding[i]][:top_k]
        
        explanations['top_attended_symptoms'] = [
            {
                'position': pos,
                'symptom_id': int(symptoms[pos]),
                'symptom_name': symptom_vocab.get(int(symptoms[pos]), f"Symptom_{int(symptoms[pos])}") if symptom_vocab else f"Symptom_{int(symptoms[pos])}",
                'attention_weight': float(attention_weights[pos])
            }
            for pos in attended_indices
        ]
    
    # Gradient-based explanation
    if method in ['gradients', 'all']:
        ig = IntegratedGradients(model)
        attributions = ig.attribute(symptoms, target_icd=int(top_k_icds[0]))
        
        explanations['gradient_attributions'] = attributions.detach().cpu().numpy().tolist()
        
        # Get top gradient-attributed symptoms
        non_padding = (symptoms != 0).cpu().numpy()
        important_indices = attributions.abs().argsort(descending=True).cpu().numpy()
        important_indices = [int(i) for i in important_indices if non_padding[i]][:top_k]
        
        explanations['top_gradient_symptoms'] = [
            {
                'position': pos,
                'symptom_id': int(symptoms[pos]),
                'symptom_name': symptom_vocab.get(int(symptoms[pos]), f"Symptom_{int(symptoms[pos])}") if symptom_vocab else f"Symptom_{int(symptoms[pos])}",
                'attribution_score': float(attributions[pos].detach())
            }
            for pos in important_indices
        ]
    
    # Counterfactual explanation
    if method in ['counterfactual', 'all']:
        cf = CounterfactualExplanations(model)
        removal_effects = cf.explain_by_removal(symptoms, target_icd=int(top_k_icds[0]))
        
        explanations['counterfactual_explanations'] = [
            {
                'symptom_id': eff['symptom_id'],
                'symptom_name': symptom_vocab.get(eff['symptom_id'], f"Symptom_{eff['symptom_id']}") if symptom_vocab else f"Symptom_{eff['symptom_id']}",
                'probability_drop_percentage': eff['probability_drop_percentage']
            }
            for eff in removal_effects[:top_k]
        ]
    
    return explanations
