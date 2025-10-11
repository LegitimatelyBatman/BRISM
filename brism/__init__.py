"""
BRISM: Bayesian Reciprocal ICD-Symptom Model

A dual encoder-decoder architecture with shared latent space for bidirectional
mapping between symptoms and ICD codes with uncertainty quantification.
"""

from .model import BRISM, BRISMConfig
from .train import train_brism, EarlyStopping, ModelCheckpoint, load_checkpoint
from .inference import diagnose_with_confidence
from .loss import BRISMLoss, FocalLoss, compute_class_weights
from .icd_hierarchy import ICDHierarchy
from .data_loader import (
    MedicalDataPreprocessor, 
    ICDNormalizer, 
    MedicalRecordDataset,
    load_mimic_data
)
from .metrics import (
    top_k_accuracy,
    compute_auroc_per_class,
    compute_calibration_metrics,
    plot_reliability_diagram,
    stratify_by_disease_frequency,
    comprehensive_evaluation,
    print_evaluation_summary
)
from .temporal import TemporalEncoding, TemporalSymptomEncoder
from .calibration import (
    TemperatureScaling,
    calibrate_temperature,
    evaluate_calibration_improvement
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
    "FocalLoss",
    "compute_class_weights",
    "ICDHierarchy",
    "MedicalDataPreprocessor",
    "ICDNormalizer",
    "MedicalRecordDataset",
    "load_mimic_data",
    "top_k_accuracy",
    "compute_auroc_per_class",
    "compute_calibration_metrics",
    "plot_reliability_diagram",
    "stratify_by_disease_frequency",
    "comprehensive_evaluation",
    "print_evaluation_summary",
    "TemporalEncoding",
    "TemporalSymptomEncoder",
    "TemperatureScaling",
    "calibrate_temperature",
    "evaluate_calibration_improvement"
]
