"""
Active Learning interface for BRISM.

Supports:
1. Uncertainty-based query selection
2. Information gain ranking for symptoms
3. Query-by-committee
4. Expected model change
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Set
import numpy as np
from .model import BRISM


class ActiveLearner:
    """
    Active learning module for querying most informative symptoms.
    
    When the model is uncertain, suggests which additional symptoms
    would be most informative to query.
    """
    
    def __init__(
        self,
        model: BRISM,
        symptom_vocab: Optional[Dict[int, str]] = None,
        icd_vocab: Optional[Dict[int, str]] = None
    ):
        """
        Initialize active learner.
        
        Args:
            model: BRISM model
            symptom_vocab: Optional mapping from token IDs to symptom names
            icd_vocab: Optional mapping from token IDs to ICD code names
        """
        self.model = model
        self.symptom_vocab = symptom_vocab or {}
        self.icd_vocab = icd_vocab or {}
    
    def query_next_symptom(
        self,
        current_symptoms: torch.Tensor,
        candidate_symptoms: Optional[List[int]] = None,
        method: str = 'entropy',
        top_k: int = 5,
        n_samples: int = 20,
        already_queried: Optional[Set[int]] = None
    ) -> List[Dict]:
        """
        Recommend next symptoms to query based on information gain.
        
        Args:
            current_symptoms: Current symptom sequence [seq_len]
            candidate_symptoms: List of candidate symptom IDs to consider
            method: Query strategy ('entropy', 'bald', 'variance', 'eig')
            top_k: Number of symptoms to recommend
            n_samples: Number of MC samples for uncertainty estimation
            already_queried: Set of symptom IDs that have been previously queried (to exclude)
            
        Returns:
            List of recommended symptoms with scores
        """
        device = next(self.model.parameters()).device
        current_symptoms = current_symptoms.to(device)
        
        # Get all possible symptoms if not specified
        if candidate_symptoms is None:
            candidate_symptoms = list(range(1, self.model.config.symptom_vocab_size))
        
        # Remove symptoms already present
        current_symptom_set = set(current_symptoms.cpu().numpy().tolist())
        candidate_symptoms = [s for s in candidate_symptoms if s not in current_symptom_set]
        
        # Remove already queried symptoms if provided
        if already_queried is not None:
            candidate_symptoms = [s for s in candidate_symptoms if s not in already_queried]
        
        if len(candidate_symptoms) == 0:
            return []
        
        # Compute scores for each candidate
        scores = []
        
        if method == 'entropy':
            scores = self._entropy_based_selection(
                current_symptoms, candidate_symptoms, n_samples
            )
        elif method == 'bald':
            scores = self._bald_selection(
                current_symptoms, candidate_symptoms, n_samples
            )
        elif method == 'variance':
            scores = self._variance_based_selection(
                current_symptoms, candidate_symptoms, n_samples
            )
        elif method == 'eig':
            scores = self._expected_information_gain(
                current_symptoms, candidate_symptoms, n_samples
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Sort by score and get top-k
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        
        recommendations = []
        for idx in sorted_indices:
            symptom_id = candidate_symptoms[idx]
            recommendations.append({
                'symptom_id': symptom_id,
                'symptom_name': self.symptom_vocab.get(symptom_id, f"Symptom_{symptom_id}"),
                'score': float(scores[idx]),
                'method': method
            })
        
        return recommendations
    
    def query_next_k_symptoms(
        self,
        current_symptoms: torch.Tensor,
        k: int = 3,
        candidate_symptoms: Optional[List[int]] = None,
        method: str = 'entropy',
        n_samples: int = 20,
        batch_mode: str = 'joint',
        already_queried: Optional[Set[int]] = None
    ) -> List[Dict]:
        """
        Recommend the top k most informative symptoms to query together.
        
        This is more practical for clinical workflows where multiple questions
        can be asked simultaneously, rather than one at a time.
        
        Args:
            current_symptoms: Current symptom sequence [seq_len]
            k: Number of symptoms to recommend
            candidate_symptoms: List of candidate symptom IDs to consider
            method: Query strategy ('entropy', 'bald', 'variance', 'eig')
            n_samples: Number of MC samples for uncertainty estimation
            batch_mode: Selection strategy:
                - 'independent': Select top k independently (fastest)
                - 'joint': Consider joint information gain (more accurate but slower)
                - 'greedy': Greedy sequential selection (balanced)
            already_queried: Set of symptom IDs that have been previously queried (to exclude)
            
        Returns:
            List of k recommended symptoms with individual and joint scores
        """
        device = next(self.model.parameters()).device
        current_symptoms = current_symptoms.to(device)
        
        # Get all possible symptoms if not specified
        if candidate_symptoms is None:
            candidate_symptoms = list(range(1, self.model.config.symptom_vocab_size))
        
        # Remove symptoms already present
        current_symptom_set = set(current_symptoms.cpu().numpy().tolist())
        candidate_symptoms = [s for s in candidate_symptoms if s not in current_symptom_set]
        
        # Remove already queried symptoms if provided
        if already_queried is not None:
            candidate_symptoms = [s for s in candidate_symptoms if s not in already_queried]
        
        if len(candidate_symptoms) == 0:
            return []
        
        # Limit k to available candidates
        k = min(k, len(candidate_symptoms))
        
        if batch_mode == 'independent':
            # Fastest: just return top k from single query
            return self.query_next_symptom(
                current_symptoms, candidate_symptoms, method, k, n_samples, already_queried
            )
        
        elif batch_mode == 'greedy':
            # Greedy sequential selection
            selected = []
            remaining_candidates = candidate_symptoms.copy()
            current = current_symptoms
            
            for _ in range(k):
                # Query best symptom given current state
                recommendations = self.query_next_symptom(
                    current, remaining_candidates, method, 1, n_samples
                )
                
                if not recommendations:
                    break
                
                best = recommendations[0]
                selected.append(best)
                
                # Update current symptoms
                current = self._add_symptom(current, best['symptom_id'])
                
                # Remove from candidates
                remaining_candidates.remove(best['symptom_id'])
            
            # Add batch score (joint information gain)
            if selected:
                final_symptoms = current
                joint_score = self._compute_information_gain(
                    current_symptoms, final_symptoms, n_samples
                )
                for item in selected:
                    item['joint_score'] = float(joint_score)
            
            return selected
        
        elif batch_mode == 'joint':
            # Most accurate: evaluate all combinations of size k
            # For large k or candidate sets, this can be slow
            if len(candidate_symptoms) > 50 or k > 5:
                # Fallback to greedy for large problems
                return self.query_next_k_symptoms(
                    current_symptoms, k, candidate_symptoms, method, n_samples, 'greedy'
                )
            
            from itertools import combinations
            
            best_score = -float('inf')
            best_combo = None
            
            # Evaluate all combinations
            for combo in combinations(candidate_symptoms, k):
                # Create symptom sequence with all symptoms in combo
                augmented = current_symptoms
                for symptom_id in combo:
                    augmented = self._add_symptom(augmented, symptom_id)
                
                # Compute joint information gain
                score = self._compute_information_gain(
                    current_symptoms, augmented, n_samples
                )
                
                if score > best_score:
                    best_score = score
                    best_combo = combo
            
            # Format results
            recommendations = []
            for symptom_id in best_combo:
                recommendations.append({
                    'symptom_id': symptom_id,
                    'symptom_name': self.symptom_vocab.get(symptom_id, f"Symptom_{symptom_id}"),
                    'joint_score': float(best_score),
                    'method': f'{method}_joint'
                })
            
            return recommendations
        
        else:
            raise ValueError(f"Unknown batch_mode: {batch_mode}")
    
    def _compute_information_gain(
        self,
        symptoms_before: torch.Tensor,
        symptoms_after: torch.Tensor,
        n_samples: int
    ) -> float:
        """
        Compute information gain from adding symptoms.
        
        Args:
            symptoms_before: Symptoms before adding [seq_len]
            symptoms_after: Symptoms after adding [seq_len]
            n_samples: Number of MC samples
            
        Returns:
            Information gain (reduction in entropy)
        """
        entropy_before = self._compute_prediction_entropy(
            symptoms_before.unsqueeze(0), n_samples
        )[0]
        
        entropy_after = self._compute_prediction_entropy(
            symptoms_after.unsqueeze(0), n_samples
        )[0]
        
        return entropy_before - entropy_after
    
    def _entropy_based_selection(
        self,
        current_symptoms: torch.Tensor,
        candidates: List[int],
        n_samples: int
    ) -> np.ndarray:
        """
        Select symptoms that maximize prediction entropy reduction.
        
        Args:
            current_symptoms: Current symptoms [seq_len]
            candidates: Candidate symptom IDs
            n_samples: Number of MC samples
            
        Returns:
            Scores for each candidate
        """
        device = current_symptoms.device
        
        # Get current prediction uncertainty
        current_entropy = self._compute_prediction_entropy(
            current_symptoms.unsqueeze(0), n_samples
        )[0]
        
        scores = []
        
        for symptom_id in candidates:
            # Add symptom to sequence
            augmented_symptoms = self._add_symptom(current_symptoms, symptom_id)
            
            # Get new prediction uncertainty
            new_entropy = self._compute_prediction_entropy(
                augmented_symptoms.unsqueeze(0), n_samples
            )[0]
            
            # Information gain = reduction in entropy
            info_gain = current_entropy - new_entropy
            scores.append(info_gain)
        
        return np.array(scores)
    
    def _bald_selection(
        self,
        current_symptoms: torch.Tensor,
        candidates: List[int],
        n_samples: int
    ) -> np.ndarray:
        """
        Bayesian Active Learning by Disagreement (BALD).
        
        Measures mutual information between predictions and model parameters.
        
        Args:
            current_symptoms: Current symptoms [seq_len]
            candidates: Candidate symptom IDs
            n_samples: Number of MC samples
            
        Returns:
            Scores for each candidate
        """
        scores = []
        
        for symptom_id in candidates:
            # Add symptom to sequence
            augmented_symptoms = self._add_symptom(current_symptoms, symptom_id)
            
            # Get predictions with dropout
            all_probs = self._get_mc_predictions(
                augmented_symptoms.unsqueeze(0), n_samples
            )[0]  # [n_samples, vocab_size]
            
            # Mean prediction
            mean_probs = all_probs.mean(axis=0)
            
            # Expected entropy: E[H(y|x,w)]
            entropies = -(all_probs * np.log(all_probs + 1e-10)).sum(axis=-1)
            expected_entropy = entropies.mean()
            
            # Entropy of expectation: H(E[y|x])
            entropy_of_expectation = -(mean_probs * np.log(mean_probs + 1e-10)).sum()
            
            # BALD score: mutual information
            bald_score = entropy_of_expectation - expected_entropy
            scores.append(bald_score)
        
        return np.array(scores)
    
    def _variance_based_selection(
        self,
        current_symptoms: torch.Tensor,
        candidates: List[int],
        n_samples: int
    ) -> np.ndarray:
        """
        Select symptoms that maximize prediction variance.
        
        Args:
            current_symptoms: Current symptoms [seq_len]
            candidates: Candidate symptom IDs
            n_samples: Number of MC samples
            
        Returns:
            Scores for each candidate
        """
        scores = []
        
        for symptom_id in candidates:
            # Add symptom to sequence
            augmented_symptoms = self._add_symptom(current_symptoms, symptom_id)
            
            # Get predictions with dropout
            all_probs = self._get_mc_predictions(
                augmented_symptoms.unsqueeze(0), n_samples
            )[0]  # [n_samples, vocab_size]
            
            # Compute variance
            variance = all_probs.var(axis=0).mean()
            scores.append(variance)
        
        return np.array(scores)
    
    def _expected_information_gain(
        self,
        current_symptoms: torch.Tensor,
        candidates: List[int],
        n_samples: int
    ) -> np.ndarray:
        """
        Expected Information Gain (EIG).
        
        Estimates how much information we expect to gain about the true label.
        
        Args:
            current_symptoms: Current symptoms [seq_len]
            candidates: Candidate symptom IDs
            n_samples: Number of MC samples
            
        Returns:
            Scores for each candidate
        """
        # Get current prediction distribution
        current_probs = self._get_mc_predictions(
            current_symptoms.unsqueeze(0), n_samples
        )[0].mean(axis=0)
        
        scores = []
        
        for symptom_id in candidates:
            # Add symptom
            augmented_symptoms = self._add_symptom(current_symptoms, symptom_id)
            
            # Get new prediction distribution
            new_probs = self._get_mc_predictions(
                augmented_symptoms.unsqueeze(0), n_samples
            )[0].mean(axis=0)
            
            # KL divergence from current to new
            kl_div = (new_probs * np.log((new_probs + 1e-10) / (current_probs + 1e-10))).sum()
            scores.append(kl_div)
        
        return np.array(scores)
    
    def _add_symptom(
        self,
        symptoms: torch.Tensor,
        symptom_id: int
    ) -> torch.Tensor:
        """
        Add a symptom to the sequence.
        
        Args:
            symptoms: Current symptom sequence [seq_len]
            symptom_id: Symptom ID to add
            
        Returns:
            Augmented sequence [seq_len]
            
        Raises:
            ValueError: If symptom_id is 0 (padding token)
        """
        device = symptoms.device
        
        # Validate symptom_id
        if symptom_id == 0:
            raise ValueError(
                "Cannot add padding token (symptom_id=0) to sequence. "
                "Padding tokens are reserved for empty positions."
            )
        
        # Find first padding position
        non_zero = (symptoms != 0).cpu().numpy()
        if non_zero.all():
            # Sequence is full, replace last token
            augmented = symptoms.clone()
            augmented[-1] = symptom_id
        else:
            # Add to first padding position
            first_pad = np.where(~non_zero)[0][0]
            augmented = symptoms.clone()
            augmented[first_pad] = symptom_id
        
        return augmented
    
    def _compute_prediction_entropy(
        self,
        symptoms: torch.Tensor,
        n_samples: int
    ) -> np.ndarray:
        """
        Compute prediction entropy.
        
        Args:
            symptoms: Symptom sequences [batch_size, seq_len]
            n_samples: Number of MC samples
            
        Returns:
            Entropy for each sequence [batch_size]
        """
        all_probs = self._get_mc_predictions(symptoms, n_samples)
        mean_probs = all_probs.mean(axis=1)  # [batch_size, vocab_size]
        
        # Compute entropy
        entropy = -(mean_probs * np.log(mean_probs + 1e-10)).sum(axis=-1)
        
        return entropy
    
    def _get_mc_predictions(
        self,
        symptoms: torch.Tensor,
        n_samples: Optional[int]
    ) -> np.ndarray:
        """
        Get Monte Carlo predictions.
        
        Args:
            symptoms: Symptom sequences [batch_size, seq_len]
            n_samples: Number of samples
            
        Returns:
            Predictions [batch_size, n_samples, vocab_size]
        """
        if n_samples is None:
            n_samples = self.model.config.mc_samples
        if n_samples < 1:
            raise ValueError(f"n_samples must be at least 1, got {n_samples}")

        device = symptoms.device
        self.model.to(device)
        symptoms = symptoms.to(device)

        previous_mode = self.model.training
        self.model.train()  # Enable dropout

        batch_size = symptoms.size(0)
        vocab_size = self.model.config.icd_vocab_size

        all_probs = np.zeros((batch_size, n_samples, vocab_size), dtype=np.float32)

        try:
            with torch.no_grad():
                for i in range(n_samples):
                    icd_logits, _, _ = self.model.forward_path(symptoms)
                    probs = F.softmax(icd_logits, dim=-1)
                    all_probs[:, i, :] = probs.cpu().numpy()
        finally:
            if not previous_mode:
                self.model.eval()

        return all_probs
    
    def interactive_diagnosis(
        self,
        initial_symptoms: Optional[torch.Tensor] = None,
        max_queries: int = 10,
        uncertainty_threshold: float = 0.5,
        confidence_threshold: float = 0.9
    ) -> Dict:
        """
        Interactive diagnosis with active querying.
        
        Iteratively queries for symptoms until confident or max queries reached.
        
        Args:
            initial_symptoms: Initial symptom sequence [seq_len]
            max_queries: Maximum number of symptoms to query
            uncertainty_threshold: Query if uncertainty above this
            confidence_threshold: Stop if confidence above this
            
        Returns:
            Dictionary with diagnosis and query history
        """
        device = next(self.model.parameters()).device
        
        # Initialize with empty or given symptoms
        if initial_symptoms is None:
            current_symptoms = torch.zeros(
                self.model.config.max_symptom_length,
                dtype=torch.long,
                device=device
            )
        else:
            current_symptoms = initial_symptoms.to(device)
        
        query_history = []
        
        for query_num in range(max_queries):
            # Get current prediction
            self.model.eval()
            with torch.no_grad():
                icd_logits, _, _ = self.model.forward_path(current_symptoms.unsqueeze(0))
                probs = F.softmax(icd_logits, dim=-1).squeeze(0)
            
            max_prob = probs.max().item()
            predicted_icd = probs.argmax().item()
            
            # Compute uncertainty
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            normalized_entropy = entropy / np.log(self.model.config.icd_vocab_size)
            
            # Record current state
            state = {
                'query_num': query_num,
                'current_symptoms': current_symptoms.cpu().numpy().tolist(),
                'predicted_icd': predicted_icd,
                'confidence': max_prob,
                'uncertainty': normalized_entropy
            }
            
            # Check stopping criteria
            if max_prob >= confidence_threshold:
                state['status'] = 'confident'
                query_history.append(state)
                break
            
            if normalized_entropy < uncertainty_threshold:
                state['status'] = 'low_uncertainty'
                query_history.append(state)
                break
            
            # Query next symptom
            recommendations = self.query_next_symptom(
                current_symptoms,
                method='bald',
                top_k=3,
                n_samples=20
            )
            
            if len(recommendations) == 0:
                state['status'] = 'no_more_symptoms'
                query_history.append(state)
                break
            
            # For simulation, take top recommendation
            # In practice, this would prompt user
            next_symptom = recommendations[0]
            state['queried_symptom'] = next_symptom
            state['status'] = 'querying'
            query_history.append(state)
            
            # Add symptom to sequence (simulated)
            current_symptoms = self._add_symptom(
                current_symptoms,
                next_symptom['symptom_id']
            )
        else:
            # Max queries reached
            query_history[-1]['status'] = 'max_queries_reached'
        
        # Final prediction
        self.model.eval()
        with torch.no_grad():
            icd_logits, _, _ = self.model.forward_path(current_symptoms.unsqueeze(0))
            probs = F.softmax(icd_logits, dim=-1).squeeze(0)
        
        top_k_probs, top_k_icds = probs.topk(5)
        
        predictions = [
            {
                'icd_code': int(icd),
                'icd_name': self.icd_vocab.get(int(icd), f"ICD_{int(icd)}"),
                'probability': float(prob)
            }
            for icd, prob in zip(top_k_icds, top_k_probs)
        ]
        
        return {
            'final_symptoms': current_symptoms.cpu().numpy().tolist(),
            'predictions': predictions,
            'query_history': query_history,
            'num_queries': len(query_history)
        }


def demonstrate_active_learning(
    model: BRISM,
    test_symptoms: torch.Tensor,
    symptom_vocab: Optional[Dict[int, str]] = None
):
    """
    Demonstrate active learning capabilities.
    
    Args:
        model: Trained BRISM model
        test_symptoms: Test symptom sequence
        symptom_vocab: Symptom vocabulary
    """
    learner = ActiveLearner(model, symptom_vocab=symptom_vocab)
    
    # Start with partial symptoms
    n_initial = len(test_symptoms) // 2
    initial_symptoms = torch.cat([
        test_symptoms[:n_initial],
        torch.zeros(len(test_symptoms) - n_initial, dtype=torch.long)
    ])
    
    print("Initial symptoms:", initial_symptoms[:n_initial].numpy().tolist())
    print()
    
    # Query recommendations
    recommendations = learner.query_next_symptom(
        initial_symptoms,
        method='bald',
        top_k=5
    )
    
    print("Recommended symptoms to query:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['symptom_name']} (score: {rec['score']:.4f})")
    
    # Interactive diagnosis
    print("\n" + "="*60)
    print("Interactive Diagnosis Simulation")
    print("="*60)
    
    result = learner.interactive_diagnosis(
        initial_symptoms=initial_symptoms,
        max_queries=5,
        uncertainty_threshold=0.3,
        confidence_threshold=0.85
    )
    
    print(f"\nCompleted in {result['num_queries']} queries")
    print("\nFinal predictions:")
    for i, pred in enumerate(result['predictions'], 1):
        print(f"{i}. {pred['icd_name']}: {pred['probability']:.4f}")
