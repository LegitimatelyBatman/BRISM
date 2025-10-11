"""
BRISM: Bayesian Reciprocal ICD-Symptom Model

A dual encoder-decoder architecture with shared latent space for bidirectional
mapping between symptoms and ICD codes with uncertainty quantification.
"""

from .model import BRISM, BRISMConfig
from .train import train_brism, EarlyStopping, ModelCheckpoint, load_checkpoint
from .inference import diagnose_with_confidence, generate_symptoms_beam_search
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
from .interpretability import (
    IntegratedGradients,
    AttentionVisualization,
    CounterfactualExplanations,
    AttentionRollout,
    explain_prediction
)
from .ensemble import (
    BRISMEnsemble,
    train_ensemble,
    load_ensemble
)
from .symptom_normalization import (
    SymptomNormalizer,
    SymptomNormalizationLayer,
    build_symptom_normalizer_from_vocab,
    create_default_medical_synonyms
)
from .active_learning import (
    ActiveLearner,
    demonstrate_active_learning
)

__version__ = "3.0.1"
__all__ = [
    "BRISM", 
    "BRISMConfig", 
    "train_brism",
    "EarlyStopping",
    "ModelCheckpoint",
    "load_checkpoint",
    "diagnose_with_confidence",
    "generate_symptoms_beam_search",
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
    "evaluate_calibration_improvement",
    # New features
    "IntegratedGradients",
    "AttentionVisualization",
    "CounterfactualExplanations",
    "AttentionRollout",
    "explain_prediction",
    "BRISMEnsemble",
    "train_ensemble",
    "load_ensemble",
    "SymptomNormalizer",
    "SymptomNormalizationLayer",
    "build_symptom_normalizer_from_vocab",
    "create_default_medical_synonyms",
    "ActiveLearner",
    "demonstrate_active_learning",
]
