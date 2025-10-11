"""
BRISM: Bayesian Reciprocal ICD-Symptom Model

A dual encoder-decoder architecture with shared latent space for bidirectional
mapping between symptoms and ICD codes with uncertainty quantification.
"""

from .model import BRISM, BRISMConfig
from .train import train_brism, EarlyStopping, ModelCheckpoint, load_checkpoint
from .inference import diagnose_with_confidence
from .loss import BRISMLoss
from .icd_hierarchy import ICDHierarchy
from .data_loader import (
    MedicalDataPreprocessor, 
    ICDNormalizer, 
    MedicalRecordDataset,
    load_mimic_data
)

__version__ = "0.1.0"
__all__ = [
    "BRISM", 
    "BRISMConfig", 
    "train_brism",
    "EarlyStopping",
    "ModelCheckpoint",
    "load_checkpoint",
    "diagnose_with_confidence",
    "BRISMLoss",
    "ICDHierarchy",
    "MedicalDataPreprocessor",
    "ICDNormalizer",
    "MedicalRecordDataset",
    "load_mimic_data"
]
