"""
Medical data preprocessing utilities for MIMIC-III/IV and other clinical datasets.
"""

import pandas as pd
import numpy as np
import torch
import logging
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import re
from collections import defaultdict
import warnings

# Configure logger
logger = logging.getLogger(__name__)


class ICDNormalizer:
    """Normalize ICD codes to ICD-10-CM format."""
    
    @staticmethod
    def normalize_icd9_to_icd10(icd9_code: str) -> Optional[str]:
        """
        Convert ICD-9 code to ICD-10-CM (simplified mapping).
        
        Note: This is a basic conversion. For production use, please use
        official CMS General Equivalence Mappings (GEMs).
        
        Args:
            icd9_code: ICD-9 code (e.g., "250.00")
            
        Returns:
            ICD-10-CM code or None if unmappable
        """
        # Remove dots and spaces
        code = icd9_code.replace('.', '').replace(' ', '').upper()
        
        # Basic mapping examples (would need full GEM tables for production)
        # This is a simplified demonstration
        icd9_to_icd10_map = {
            # Diabetes
            '25000': 'E11.9',   # Type 2 diabetes without complications
            '25001': 'E10.9',   # Type 1 diabetes without complications
            '25002': 'E11.65',  # Type 2 diabetes with hyperglycemia
            # Hypertension
            '4019': 'I10',      # Essential hypertension
            '4011': 'I11.9',    # Hypertensive heart disease
            # COPD
            '4960': 'J44.0',    # COPD with acute lower respiratory infection
            '4961': 'J44.1',    # COPD with acute exacerbation
            # Pneumonia
            '486': 'J18.9',     # Pneumonia, unspecified
            # MI
            '41001': 'I21.0',   # STEMI
            '41011': 'I21.4',   # NSTEMI
        }
        
        return icd9_to_icd10_map.get(code[:5], None)
    
    @staticmethod
    def normalize_icd10(icd10_code: str) -> str:
        """
        Normalize ICD-10 code to standard format.
        
        Args:
            icd10_code: ICD-10 code in any format
            
        Returns:
            Normalized ICD-10-CM code
        """
        # Remove whitespace
        code = icd10_code.strip().upper()
        
        # Add dot if missing and code is long enough
        if '.' not in code and len(code) > 3:
            code = code[:3] + '.' + code[3:]
        
        return code
    
    @staticmethod
    def is_valid_icd10(code: str) -> bool:
        """
        Check if code looks like valid ICD-10 format.
        
        Args:
            code: ICD-10 code
            
        Returns:
            True if valid format
        """
        # ICD-10 pattern: Letter + 2 digits + optional (. + 1-4 chars)
        pattern = r'^[A-Z]\d{2}(\.\d{1,4})?$'
        return bool(re.match(pattern, code))


class MedicalDataPreprocessor:
    """
    Preprocess medical records from MIMIC-III/IV format.
    
    Handles:
    - ICD code extraction and normalization
    - Clinical note tokenization
    - Missing data handling
    - Patient-level splitting
    """
    
    def __init__(self, 
                 symptom_vocab: Optional[Dict[str, int]] = None,
                 icd_vocab: Optional[Dict[str, int]] = None,
                 max_symptom_length: int = 50):
        """
        Args:
            symptom_vocab: Mapping from symptom tokens to indices
            icd_vocab: Mapping from ICD codes to indices
            max_symptom_length: Maximum length of symptom sequence
        """
        self.symptom_vocab = symptom_vocab or {}
        self.icd_vocab = icd_vocab or {}
        self.max_symptom_length = max_symptom_length
        self.normalizer = ICDNormalizer()
        
    def build_vocabularies_from_data(self, 
                                     symptoms_data: List[List[str]], 
                                     icd_data: List[str],
                                     min_symptom_freq: int = 5,
                                     min_icd_freq: int = 3):
        """
        Build vocabularies from raw data.
        
        Args:
            symptoms_data: List of symptom token lists
            icd_data: List of ICD codes
            min_symptom_freq: Minimum frequency for symptom to be included
            min_icd_freq: Minimum frequency for ICD code to be included
        """
        # Count symptom frequencies
        symptom_counts = defaultdict(int)
        for symptoms in symptoms_data:
            for symptom in symptoms:
                symptom_counts[symptom] += 1
        
        # Count ICD frequencies
        icd_counts = defaultdict(int)
        for icd in icd_data:
            icd_counts[icd] += 1
        
        # Build symptom vocab (reserve 0 for padding)
        self.symptom_vocab = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        for symptom, count in sorted(symptom_counts.items()):
            if count >= min_symptom_freq:
                self.symptom_vocab[symptom] = idx
                idx += 1
        
        # Build ICD vocab
        self.icd_vocab = {}
        idx = 0
        for icd, count in sorted(icd_counts.items()):
            if count >= min_icd_freq:
                self.icd_vocab[icd] = idx
                idx += 1
    
    def tokenize_symptoms(self, symptom_text: str) -> List[str]:
        """
        Tokenize clinical text into symptom tokens.
        
        Args:
            symptom_text: Raw clinical text
            
        Returns:
            List of symptom tokens
        """
        # Simple tokenization (would use medical NLP in production)
        # Convert to lowercase and split
        text = symptom_text.lower()
        
        # Remove special characters
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Split into tokens
        tokens = text.split()
        
        # Filter stopwords (simplified)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
        
        return tokens
    
    def encode_symptoms(self, symptom_tokens: List[str]) -> np.ndarray:
        """
        Encode symptom tokens to indices.
        
        Args:
            symptom_tokens: List of symptom tokens
            
        Returns:
            Array of symptom indices [max_symptom_length]
        """
        # Convert to indices
        indices = []
        for token in symptom_tokens[:self.max_symptom_length]:
            indices.append(self.symptom_vocab.get(token, self.symptom_vocab.get('<UNK>', 1)))
        
        # Pad to max length
        encoded = np.zeros(self.max_symptom_length, dtype=np.int64)
        encoded[:len(indices)] = indices
        
        return encoded
    
    def encode_icd(self, icd_code: str) -> Optional[int]:
        """
        Encode ICD code to index.
        
        Args:
            icd_code: ICD-10 code
            
        Returns:
            ICD index or None if not in vocabulary
        """
        # Normalize code
        normalized = self.normalizer.normalize_icd10(icd_code)
        
        # Get index
        return self.icd_vocab.get(normalized, None)
    
    def process_mimic_diagnoses(self, diagnoses_df: pd.DataFrame) -> Dict[int, List[str]]:
        """
        Process MIMIC diagnoses table.
        
        Expected columns: hadm_id (or subject_id), icd_code, icd_version
        
        Args:
            diagnoses_df: DataFrame with diagnosis information
            
        Returns:
            Dictionary mapping admission/subject ID to list of ICD-10 codes
        """
        # Determine ID column
        id_col = 'hadm_id' if 'hadm_id' in diagnoses_df.columns else 'subject_id'
        
        if id_col not in diagnoses_df.columns:
            raise ValueError("DataFrame must contain 'hadm_id' or 'subject_id' column")
        
        result = defaultdict(list)
        
        for _, row in diagnoses_df.iterrows():
            patient_id = row[id_col]
            icd_code = str(row['icd_code'])
            icd_version = row.get('icd_version', 10)  # Default to ICD-10
            
            # Convert ICD-9 to ICD-10 if needed
            if icd_version == 9:
                icd10_code = self.normalizer.normalize_icd9_to_icd10(icd_code)
                if icd10_code is None:
                    warnings.warn(f"Could not map ICD-9 code {icd_code} to ICD-10")
                    continue
            else:
                icd10_code = self.normalizer.normalize_icd10(icd_code)
            
            # Validate
            if self.normalizer.is_valid_icd10(icd10_code):
                result[patient_id].append(icd10_code)
        
        return dict(result)
    
    def process_mimic_notes(self, notes_df: pd.DataFrame, 
                           text_column: str = 'text') -> Dict[int, str]:
        """
        Process MIMIC clinical notes table.
        
        Expected columns: hadm_id (or subject_id), text
        
        Args:
            notes_df: DataFrame with clinical notes
            text_column: Name of column containing note text
            
        Returns:
            Dictionary mapping admission/subject ID to concatenated note text
        """
        # Determine ID column
        id_col = 'hadm_id' if 'hadm_id' in notes_df.columns else 'subject_id'
        
        if id_col not in notes_df.columns:
            raise ValueError("DataFrame must contain 'hadm_id' or 'subject_id' column")
        
        # Group notes by patient
        result = {}
        for patient_id, group in notes_df.groupby(id_col):
            # Concatenate all notes for this patient
            notes = ' '.join(group[text_column].astype(str).tolist())
            result[patient_id] = notes
        
        return result
    
    def create_patient_splits(self, 
                            patient_ids: List[int],
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.15,
                            test_ratio: float = 0.15,
                            random_seed: int = 42) -> Tuple[Set[int], Set[int], Set[int]]:
        """
        Create train/val/test splits at patient level to avoid data leakage.
        
        Args:
            patient_ids: List of all patient IDs
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_ids, val_ids, test_ids)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        # Shuffle patient IDs
        rng = np.random.RandomState(random_seed)
        shuffled_ids = list(patient_ids)
        rng.shuffle(shuffled_ids)
        
        # Calculate split points
        n = len(shuffled_ids)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # Split
        train_ids = set(shuffled_ids[:train_end])
        val_ids = set(shuffled_ids[train_end:val_end])
        test_ids = set(shuffled_ids[val_end:])
        
        return train_ids, val_ids, test_ids


class MedicalRecordDataset(Dataset):
    """PyTorch Dataset for medical records."""
    
    def __init__(self, 
                 symptoms: List[np.ndarray],
                 icd_codes: List[int],
                 patient_ids: Optional[List[int]] = None):
        """
        Args:
            symptoms: List of encoded symptom sequences
            icd_codes: List of encoded ICD codes
            patient_ids: Optional list of patient IDs
        """
        assert len(symptoms) == len(icd_codes), \
            "Number of symptoms and ICD codes must match"
        
        self.symptoms = symptoms
        self.icd_codes = icd_codes
        self.patient_ids = patient_ids or list(range(len(symptoms)))
    
    def __len__(self) -> int:
        return len(self.symptoms)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'symptoms': torch.tensor(self.symptoms[idx], dtype=torch.long),
            'icd_codes': torch.tensor(self.icd_codes[idx], dtype=torch.long),
            'patient_id': self.patient_ids[idx]
        }


def load_mimic_data(
    diagnoses_path: str,
    notes_path: str,
    max_symptom_length: int = 50,
    min_symptom_freq: int = 5,
    min_icd_freq: int = 3,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    dry_run: bool = False
) -> Tuple[MedicalRecordDataset, MedicalRecordDataset, MedicalRecordDataset, MedicalDataPreprocessor]:
    """
    Load and preprocess MIMIC data.
    
    Args:
        diagnoses_path: Path to diagnoses CSV
        notes_path: Path to notes CSV
        max_symptom_length: Maximum symptom sequence length
        min_symptom_freq: Minimum frequency for symptoms
        min_icd_freq: Minimum frequency for ICD codes
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed
        dry_run: If True, only validates data without full processing
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, preprocessor)
        
    Raises:
        FileNotFoundError: If input files don't exist
        ValueError: If required columns are missing or data is malformed
    """
    # Validate file paths
    if not Path(diagnoses_path).exists():
        raise FileNotFoundError(f"Diagnoses file not found: {diagnoses_path}")
    if not Path(notes_path).exists():
        raise FileNotFoundError(f"Notes file not found: {notes_path}")
    
    # Load data with error handling
    try:
        diagnoses_df = pd.read_csv(diagnoses_path)
    except Exception as e:
        raise ValueError(f"Failed to load diagnoses CSV from {diagnoses_path}: {str(e)}")
    
    try:
        notes_df = pd.read_csv(notes_path)
    except Exception as e:
        raise ValueError(f"Failed to load notes CSV from {notes_path}: {str(e)}")
    
    # Validate required columns exist
    # For diagnoses, we need either 'hadm_id' or 'subject_id', plus 'icd_code'
    id_col = 'hadm_id' if 'hadm_id' in diagnoses_df.columns else 'subject_id'
    if id_col not in diagnoses_df.columns:
        raise ValueError(f"Diagnoses file must contain 'hadm_id' or 'subject_id' column. Found columns: {list(diagnoses_df.columns)}")
    if 'icd_code' not in diagnoses_df.columns:
        raise ValueError(f"Diagnoses file must contain 'icd_code' column. Found columns: {list(diagnoses_df.columns)}")
    
    # For notes, we need either 'hadm_id' or 'subject_id', plus 'text'
    notes_id_col = 'hadm_id' if 'hadm_id' in notes_df.columns else 'subject_id'
    if notes_id_col not in notes_df.columns:
        raise ValueError(f"Notes file must contain 'hadm_id' or 'subject_id' column. Found columns: {list(notes_df.columns)}")
    if 'text' not in notes_df.columns:
        raise ValueError(f"Notes file must contain 'text' column. Found columns: {list(notes_df.columns)}")
    
    # Dry run: validate data without full processing
    if dry_run:
        n_diagnoses = len(diagnoses_df)
        n_notes = len(notes_df)
        logger.info(f"Dry run - Data validation successful:")
        logger.info(f"  Diagnoses: {n_diagnoses} records")
        logger.info(f"  Notes: {n_notes} records")
        logger.info(f"  Diagnoses columns: {list(diagnoses_df.columns)}")
        logger.info(f"  Notes columns: {list(notes_df.columns)}")
        # Return empty datasets for dry run
        empty_dataset = MedicalRecordDataset([], [], [])
        return empty_dataset, empty_dataset, empty_dataset, MedicalDataPreprocessor(max_symptom_length=max_symptom_length)
    
    # Initialize preprocessor
    preprocessor = MedicalDataPreprocessor(max_symptom_length=max_symptom_length)
    
    # Process diagnoses and notes
    patient_diagnoses = preprocessor.process_mimic_diagnoses(diagnoses_df)
    patient_notes = preprocessor.process_mimic_notes(notes_df)
    
    # Find common patient IDs
    common_ids = set(patient_diagnoses.keys()) & set(patient_notes.keys())
    
    # Prepare data for vocabulary building
    all_symptoms = []
    all_icds = []
    
    for patient_id in common_ids:
        note_text = patient_notes[patient_id]
        symptom_tokens = preprocessor.tokenize_symptoms(note_text)
        all_symptoms.append(symptom_tokens)
        
        # Use first diagnosis for now (could be extended)
        if patient_diagnoses[patient_id]:
            all_icds.append(patient_diagnoses[patient_id][0])
    
    # Build vocabularies
    preprocessor.build_vocabularies_from_data(all_symptoms, all_icds, 
                                             min_symptom_freq, min_icd_freq)
    
    # Create patient splits
    patient_list = list(common_ids)
    train_ids, val_ids, test_ids = preprocessor.create_patient_splits(
        patient_list, train_ratio, val_ratio, test_ratio, random_seed
    )
    
    # Encode data by split
    def encode_split(patient_ids):
        symptoms = []
        icds = []
        ids = []
        
        for pid in patient_ids:
            note_text = patient_notes[pid]
            symptom_tokens = preprocessor.tokenize_symptoms(note_text)
            encoded_symptoms = preprocessor.encode_symptoms(symptom_tokens)
            
            # Use first diagnosis
            if patient_diagnoses[pid]:
                icd_code = patient_diagnoses[pid][0]
                encoded_icd = preprocessor.encode_icd(icd_code)
                
                if encoded_icd is not None:
                    symptoms.append(encoded_symptoms)
                    icds.append(encoded_icd)
                    ids.append(pid)
        
        return MedicalRecordDataset(symptoms, icds, ids)
    
    train_dataset = encode_split(train_ids)
    val_dataset = encode_split(val_ids)
    test_dataset = encode_split(test_ids)
    
    # Validate that splits contain samples
    if len(train_dataset) == 0:
        raise ValueError(
            f"Training split resulted in zero samples. "
            f"This may be due to: (1) no common patient IDs between diagnoses and notes, "
            f"(2) all ICD codes filtered out by min_icd_freq={min_icd_freq}, "
            f"or (3) encoding failures. Common IDs found: {len(common_ids)}, "
            f"Train patient IDs: {len(train_ids)}"
        )
    if len(val_dataset) == 0:
        raise ValueError(
            f"Validation split resulted in zero samples. "
            f"This may be due to: (1) insufficient data for val_ratio={val_ratio}, "
            f"(2) all ICD codes in validation set filtered out by min_icd_freq={min_icd_freq}, "
            f"or (3) encoding failures. Common IDs found: {len(common_ids)}, "
            f"Val patient IDs: {len(val_ids)}"
        )
    if len(test_dataset) == 0:
        raise ValueError(
            f"Test split resulted in zero samples. "
            f"This may be due to: (1) insufficient data for test_ratio={test_ratio}, "
            f"(2) all ICD codes in test set filtered out by min_icd_freq={min_icd_freq}, "
            f"or (3) encoding failures. Common IDs found: {len(common_ids)}, "
            f"Test patient IDs: {len(test_ids)}"
        )
    
    return train_dataset, val_dataset, test_dataset, preprocessor
