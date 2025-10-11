"""
ICD-10 hierarchical distance computation for hierarchical loss.
"""

import torch
import yaml
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


class ICDHierarchy:
    """
    ICD-10 code hierarchy manager for computing hierarchical distances.
    
    ICD-10 codes have a hierarchical structure:
    - Chapter (e.g., E = Endocrine)
    - Category (e.g., E11 = Type 2 diabetes)
    - Subcategory (e.g., E11.6 = with complications)
    - Code (e.g., E11.65 = with hyperglycemia)
    """
    
    def __init__(self, icd_vocab_size: int):
        """
        Initialize ICD hierarchy.
        
        Args:
            icd_vocab_size: Number of ICD codes in vocabulary
        """
        self.icd_vocab_size = icd_vocab_size
        self.distance_matrix = None
        self.code_to_idx = {}
        self.idx_to_code = {}
        
    def _compute_tree_distance(self, code1: str, code2: str) -> float:
        """
        Compute tree distance between two ICD-10 codes.
        
        Distance is based on hierarchical similarity:
        - Same code: distance = 0
        - Same subcategory: distance = 1
        - Same category: distance = 2
        - Same chapter: distance = 3
        - Different chapter: distance = 4
        
        Args:
            code1: First ICD-10 code (e.g., "E11.65")
            code2: Second ICD-10 code (e.g., "E11.9")
            
        Returns:
            Hierarchical distance
        """
        if code1 == code2:
            return 0.0
        
        # Extract hierarchy levels
        def get_levels(code):
            # Chapter: first character
            chapter = code[0] if len(code) > 0 else ""
            # Category: first 3 characters (e.g., E11)
            category = code[:3] if len(code) >= 3 else code
            # Subcategory: first 4-5 characters (e.g., E11.6)
            subcategory = code[:5] if len(code) >= 5 and '.' in code else category
            return chapter, category, subcategory
        
        ch1, cat1, sub1 = get_levels(code1)
        ch2, cat2, sub2 = get_levels(code2)
        
        # Compare hierarchies
        if sub1 == sub2:
            return 1.0  # Same subcategory
        elif cat1 == cat2:
            return 2.0  # Same category
        elif ch1 == ch2:
            return 3.0  # Same chapter
        else:
            return 4.0  # Different chapter
    
    def build_from_mapping(self, idx_to_code: Dict[int, str]):
        """
        Build distance matrix from index-to-code mapping.
        
        Args:
            idx_to_code: Dictionary mapping vocab indices to ICD-10 codes
        """
        self.idx_to_code = idx_to_code
        self.code_to_idx = {code: idx for idx, code in idx_to_code.items()}
        
        # Build distance matrix
        self.distance_matrix = np.zeros((self.icd_vocab_size, self.icd_vocab_size))
        
        for i in range(self.icd_vocab_size):
            code_i = idx_to_code.get(i, f"UNKNOWN_IDX{i}")
            for j in range(self.icd_vocab_size):
                code_j = idx_to_code.get(j, f"UNKNOWN_IDX{j}")
                self.distance_matrix[i, j] = self._compute_tree_distance(code_i, code_j)
    
    def build_from_yaml(self, yaml_path: str):
        """
        Build distance matrix from YAML configuration file.
        
        Expected YAML format:
        ```yaml
        icd_codes:
          0: "E11.65"
          1: "E11.9"
          2: "I10"
          ...
        ```
        
        Args:
            yaml_path: Path to YAML configuration file
        """
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        idx_to_code = {int(k): v for k, v in config.get('icd_codes', {}).items()}
        self.build_from_mapping(idx_to_code)
    
    def build_synthetic(self):
        """
        Build synthetic hierarchical structure for testing/demo.
        
        Creates a simple hierarchy where codes are grouped:
        - Codes 0-99: Chapter A (groups of 10)
        - Codes 100-199: Chapter B (groups of 10)
        - etc.
        """
        idx_to_code = {}
        
        for i in range(self.icd_vocab_size):
            chapter_num = i // 100
            category_num = (i % 100) // 10
            code_num = i % 10
            
            chapter = chr(65 + (chapter_num % 26))  # A-Z
            code = f"{chapter}{category_num:02d}.{code_num}"
            idx_to_code[i] = code
        
        self.build_from_mapping(idx_to_code)
    
    def get_distance_tensor(self, device: torch.device = None) -> torch.Tensor:
        """
        Get distance matrix as PyTorch tensor.
        
        Args:
            device: Device to place tensor on
            
        Returns:
            Distance matrix [vocab_size, vocab_size]
        """
        if self.distance_matrix is None:
            raise ValueError("Distance matrix not built. Call build_from_* first.")
        
        tensor = torch.from_numpy(self.distance_matrix).float()
        if device is not None:
            tensor = tensor.to(device)
        return tensor
    
    def save_yaml(self, yaml_path: str):
        """
        Save current ICD code mapping to YAML file.
        
        Args:
            yaml_path: Path to save YAML file
        """
        config = {
            'icd_codes': {str(idx): code for idx, code in self.idx_to_code.items()}
        }
        
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)


def compute_hierarchical_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    distance_matrix: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute hierarchical loss that penalizes based on ICD code distance.
    
    The loss is reduced when predictions are hierarchically close to the target,
    even if not exactly correct.
    
    Args:
        logits: Predicted logits [batch_size, vocab_size]
        targets: Target ICD indices [batch_size]
        distance_matrix: Hierarchical distance matrix [vocab_size, vocab_size]
        temperature: Temperature for softening distances (higher = more penalty reduction)
        
    Returns:
        Hierarchical loss [batch_size]
    """
    batch_size = logits.size(0)
    vocab_size = logits.size(1)
    
    # Get predicted probabilities
    probs = torch.softmax(logits, dim=-1)  # [B, V]
    
    # Get target distances for each sample
    target_distances = distance_matrix[targets]  # [B, V]
    
    # Weight by distance (closer codes have lower penalty)
    # Use exponential decay: penalty = exp(-distance / temperature)
    distance_weights = torch.exp(-target_distances / temperature)  # [B, V]
    
    # Compute weighted cross-entropy
    # Instead of only penalizing incorrect predictions, we weight by hierarchical distance
    log_probs = torch.log_softmax(logits, dim=-1)  # [B, V]
    
    # Standard cross-entropy for exact match
    ce_loss = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
    
    # Hierarchical component: reward predictions close to target
    # Expected distance: sum of (prob * distance) for each prediction
    expected_distance = (probs * target_distances).sum(dim=1)  # [B]
    
    # Combine: CE loss + scaled expected distance
    hierarchical_loss = ce_loss + expected_distance / temperature
    
    return hierarchical_loss
