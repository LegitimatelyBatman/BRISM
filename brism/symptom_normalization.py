"""
Symptom synonym handling and normalization.

Supports:
1. Mapping symptom variants to canonical forms
2. UMLS/SNOMED-CT integration (when available)
3. Rule-based normalization
4. Fuzzy matching for symptom names
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set, Tuple
import re
from collections import defaultdict


class SymptomNormalizer:
    """
    Normalize symptom descriptions to canonical forms.
    
    Handles:
    - Synonym mapping (e.g., "SOB" -> "shortness of breath")
    - Abbreviation expansion
    - Case normalization
    - Medical term standardization
    """
    
    def __init__(
        self,
        synonym_dict: Optional[Dict[str, str]] = None,
        canonical_forms: Optional[Dict[str, int]] = None,
        use_fuzzy_matching: bool = False,
        fuzzy_threshold: float = 0.8
    ):
        """
        Initialize symptom normalizer.
        
        Args:
            synonym_dict: Mapping from synonyms to canonical forms
            canonical_forms: Mapping from canonical forms to token IDs
            use_fuzzy_matching: Enable fuzzy string matching
            fuzzy_threshold: Similarity threshold for fuzzy matching
        """
        self.synonym_dict = synonym_dict or {}
        self.canonical_forms = canonical_forms or {}
        self.use_fuzzy_matching = use_fuzzy_matching
        self.fuzzy_threshold = fuzzy_threshold
        
        # Build reverse mapping
        self.canonical_to_id = self.canonical_forms
        self.id_to_canonical = {v: k for k, v in self.canonical_forms.items()} if self.canonical_forms else {}
        
        # Common medical abbreviations (can be extended)
        self.default_abbreviations = {
            'sob': 'shortness of breath',
            'cp': 'chest pain',
            'ha': 'headache',
            'n/v': 'nausea and vomiting',
            'abd': 'abdominal',
            'lle': 'left lower extremity',
            'rle': 'right lower extremity',
            'lue': 'left upper extremity',
            'rue': 'right upper extremity',
            'lbp': 'low back pain',
            'doi': 'difficulty breathing',
            'doe': 'dyspnea on exertion',
            'rom': 'range of motion',
            'wt': 'weight',
            'bp': 'blood pressure',
            'hr': 'heart rate',
            'rr': 'respiratory rate',
            'temp': 'temperature',
            'htn': 'hypertension',
            'dm': 'diabetes mellitus',
            'cad': 'coronary artery disease',
            'chf': 'congestive heart failure',
            'copd': 'chronic obstructive pulmonary disease',
            'afib': 'atrial fibrillation',
            'mi': 'myocardial infarction',
            'cva': 'cerebrovascular accident',
            'uri': 'upper respiratory infection',
            'uti': 'urinary tract infection',
            'gi': 'gastrointestinal',
        }
        
        # Merge with provided synonyms
        self.synonym_dict = {**self.default_abbreviations, **self.synonym_dict}
    
    def normalize(self, symptom_text: str, preserve_case: bool = False) -> str:
        """
        Normalize a symptom description to canonical form.
        
        Args:
            symptom_text: Raw symptom description
            preserve_case: If True, preserve original case when returning normalized text
            
        Returns:
            Normalized canonical form (empty string if input is empty or whitespace-only)
        """
        # Store original for case preservation
        original_text = symptom_text.strip()
        
        # Handle empty string early
        if not original_text:
            return ""
        
        # Convert to lowercase for matching
        text = symptom_text.lower().strip()
        
        # Remove special characters except spaces and hyphens
        text = re.sub(r'[^\w\s\-/]', ' ', text)
        
        # After cleaning, check if empty again
        text = text.strip()
        if not text:
            return ""
        
        # Check direct synonym match
        if text in self.synonym_dict:
            normalized = self.synonym_dict[text]
            # If preserve_case and the original was an exact match in synonym_dict, keep it
            if preserve_case and original_text.lower() == text:
                # Check if the original matches exactly (case-insensitive) the synonym key
                # If it does, return the original case for acronyms like 'SOB' -> 'SOB' (not normalized form)
                # But still return the normalized form, just preserve case if possible
                return normalized
            return normalized
        
        # Try multi-word synonyms
        words = text.split()
        for i in range(len(words), 0, -1):
            for j in range(len(words) - i + 1):
                phrase = ' '.join(words[j:j+i])
                if phrase in self.synonym_dict:
                    # Replace the phrase
                    before = ' '.join(words[:j])
                    after = ' '.join(words[j+i:])
                    result = ' '.join(filter(None, [before, self.synonym_dict[phrase], after]))
                    return result
        
        # Check if already canonical
        if text in self.canonical_forms:
            return original_text if preserve_case else text
        
        # Fuzzy matching (if enabled)
        if self.use_fuzzy_matching:
            best_match, score = self._fuzzy_match(text)
            if score >= self.fuzzy_threshold:
                return best_match
        
        # Return as-is, preserving case if requested
        return original_text if preserve_case else text
    
    def normalize_to_id(self, symptom_text: str) -> Optional[int]:
        """
        Normalize symptom and return its token ID.
        
        Args:
            symptom_text: Raw symptom description
            
        Returns:
            Token ID or None if not found
        """
        normalized = self.normalize(symptom_text)
        return self.canonical_to_id.get(normalized)
    
    def normalize_sequence(self, symptom_texts: List[str], preserve_case: bool = False) -> List[str]:
        """
        Normalize a sequence of symptoms.
        
        Args:
            symptom_texts: List of symptom descriptions
            preserve_case: If True, preserve original case when possible
            
        Returns:
            List of normalized symptoms
        """
        return [self.normalize(text, preserve_case=preserve_case) for text in symptom_texts]
    
    def normalize_sequence_to_ids(
        self,
        symptom_texts: List[str],
        max_length: Optional[int] = None,
        pad_id: int = 0
    ) -> List[int]:
        """
        Normalize symptoms and convert to token IDs.
        
        Args:
            symptom_texts: List of symptom descriptions
            max_length: Maximum sequence length (pad or truncate)
            pad_id: Padding token ID
            
        Returns:
            List of token IDs
        """
        ids = []
        for text in symptom_texts:
            token_id = self.normalize_to_id(text)
            if token_id is not None:
                ids.append(token_id)
        
        # Apply max_length
        if max_length is not None:
            if len(ids) > max_length:
                ids = ids[:max_length]
            else:
                ids = ids + [pad_id] * (max_length - len(ids))
        
        return ids
    
    def _fuzzy_match(self, text: str) -> Tuple[str, float]:
        """
        Find best fuzzy match in canonical forms.
        
        Args:
            text: Query text
            
        Returns:
            best_match: Best matching canonical form
            score: Similarity score [0, 1]
        """
        best_match = text
        best_score = 0.0
        
        for canonical in self.canonical_forms.keys():
            score = self._similarity(text, canonical)
            if score > best_score:
                best_score = score
                best_match = canonical
                # Early stopping: if we found a perfect match, no need to continue
                if score == 1.0:
                    break
        
        return best_match, best_score
    
    def _similarity(self, s1: str, s2: str) -> float:
        """
        Compute string similarity using Levenshtein distance.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Similarity score [0, 1]
        """
        # Simple implementation (can be replaced with better algorithm)
        if s1 == s2:
            return 1.0
        
        # Jaccard similarity on character sets
        set1 = set(s1)
        set2 = set(s2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def add_synonym(self, synonym: str, canonical: str):
        """
        Add a new synonym mapping.
        
        Args:
            synonym: Synonym or abbreviation
            canonical: Canonical form
        """
        self.synonym_dict[synonym.lower()] = canonical.lower()
    
    def add_canonical_form(self, canonical: str, token_id: int):
        """
        Add a new canonical form.
        
        Args:
            canonical: Canonical symptom description
            token_id: Token ID for this symptom
        """
        canonical = canonical.lower()
        self.canonical_forms[canonical] = token_id
        self.canonical_to_id[canonical] = token_id
        self.id_to_canonical[token_id] = canonical
    
    def build_from_umls(self, umls_mapping: Dict[str, List[str]]):
        """
        Build synonym dictionary from UMLS concept mappings.
        
        Args:
            umls_mapping: Dictionary mapping canonical concepts to lists of synonyms
                         Example: {'shortness of breath': ['SOB', 'dyspnea', 'breathlessness']}
        """
        for canonical, synonyms in umls_mapping.items():
            canonical_lower = canonical.lower()
            for synonym in synonyms:
                self.synonym_dict[synonym.lower()] = canonical_lower
    
    def save(self, path: str):
        """
        Save normalizer to file.
        
        Args:
            path: Path to save file
        """
        import json
        
        data = {
            'synonym_dict': self.synonym_dict,
            'canonical_forms': self.canonical_forms,
            'use_fuzzy_matching': self.use_fuzzy_matching,
            'fuzzy_threshold': self.fuzzy_threshold
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'SymptomNormalizer':
        """
        Load normalizer from file.
        
        Args:
            path: Path to saved file
            
        Returns:
            Loaded SymptomNormalizer
        """
        import json
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(
            synonym_dict=data['synonym_dict'],
            canonical_forms=data['canonical_forms'],
            use_fuzzy_matching=data['use_fuzzy_matching'],
            fuzzy_threshold=data['fuzzy_threshold']
        )


class SymptomNormalizationLayer(nn.Module):
    """
    Neural network layer for symptom normalization.
    
    Learns to map symptom embeddings to a normalized space where
    synonyms have similar representations.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout_rate: float = 0.2
    ):
        """
        Initialize normalization layer.
        
        Args:
            input_dim: Input embedding dimension
            output_dim: Output normalized dimension
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Normalize symptom embeddings.
        
        Args:
            embeddings: Input embeddings [batch_size, seq_len, input_dim]
            
        Returns:
            Normalized embeddings [batch_size, seq_len, output_dim]
        """
        # Project to normalized space
        h = torch.relu(self.fc1(embeddings))
        h = self.dropout(h)
        h = self.fc2(h)
        
        # Layer normalization
        h = self.layer_norm(h)
        
        return h


def build_symptom_normalizer_from_vocab(
    symptom_vocab: Dict[str, int],
    synonym_lists: Optional[Dict[str, List[str]]] = None
) -> SymptomNormalizer:
    """
    Build a symptom normalizer from vocabulary.
    
    Args:
        symptom_vocab: Mapping from symptom names to token IDs
        synonym_lists: Optional mapping from canonical forms to synonym lists
        
    Returns:
        SymptomNormalizer
    """
    # Create canonical forms mapping
    canonical_forms = {name.lower(): idx for name, idx in symptom_vocab.items()}
    
    # Create synonym dictionary
    synonym_dict = {}
    if synonym_lists:
        for canonical, synonyms in synonym_lists.items():
            canonical_lower = canonical.lower()
            for synonym in synonyms:
                synonym_dict[synonym.lower()] = canonical_lower
    
    return SymptomNormalizer(
        synonym_dict=synonym_dict,
        canonical_forms=canonical_forms,
        use_fuzzy_matching=True,
        fuzzy_threshold=0.8
    )


def create_default_medical_synonyms() -> Dict[str, List[str]]:
    """
    Create default medical symptom synonym mappings.
    
    Returns:
        Dictionary mapping canonical symptoms to synonym lists
    """
    return {
        'shortness of breath': ['SOB', 'dyspnea', 'breathlessness', 'difficulty breathing', 'labored breathing'],
        'chest pain': ['CP', 'thoracic pain', 'angina', 'chest discomfort'],
        'headache': ['HA', 'cephalgia', 'head pain', 'migraine'],
        'abdominal pain': ['abd pain', 'stomach pain', 'belly pain', 'stomach ache'],
        'nausea': ['feeling sick', 'queasiness', 'upset stomach'],
        'vomiting': ['emesis', 'throwing up', 'being sick'],
        'dizziness': ['vertigo', 'lightheadedness', 'feeling faint'],
        'fatigue': ['tiredness', 'exhaustion', 'weakness', 'lethargy'],
        'fever': ['pyrexia', 'elevated temperature', 'high temperature'],
        'cough': ['coughing', 'tussis'],
        'back pain': ['backache', 'dorsalgia', 'lumbago'],
        'joint pain': ['arthralgia', 'joint ache'],
        'muscle pain': ['myalgia', 'muscle ache', 'muscle soreness'],
        'swelling': ['edema', 'inflammation', 'puffiness'],
        'rash': ['skin eruption', 'dermatitis', 'skin rash'],
        'palpitations': ['heart racing', 'rapid heartbeat', 'irregular heartbeat'],
        'confusion': ['disorientation', 'altered mental status', 'delirium'],
        'weakness': ['muscle weakness', 'asthenia', 'lack of strength'],
        'numbness': ['paresthesia', 'tingling', 'loss of sensation'],
        'vision changes': ['blurred vision', 'visual disturbance', 'sight problems'],
    }
