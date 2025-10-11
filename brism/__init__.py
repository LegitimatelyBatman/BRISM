"""
BRISM: Bayesian Reciprocal ICD-Symptom Model

A dual encoder-decoder architecture with shared latent space for bidirectional
mapping between symptoms and ICD codes with uncertainty quantification.
"""

from .model import BRISM, BRISMConfig
from .train import train_brism
from .inference import diagnose_with_confidence

__version__ = "0.1.0"
__all__ = ["BRISM", "BRISMConfig", "train_brism", "diagnose_with_confidence"]
