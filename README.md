# BRISM v3.1.0
Bayesian Reciprocal ICD-Symptom Model

A deep learning model for bidirectional mapping between medical symptoms and ICD diagnosis codes with uncertainty quantification.

## What's New in v3.1.0

**Documentation & Infrastructure:**
- ✅ Enhanced IMPLEMENTATION.md with detailed non-technical explanations
- ✅ Synchronized setup.py and requirements.txt dependencies
- ✅ Consolidated test suite for better maintainability
- ✅ Updated all documentation to reflect current codebase

**All v3.0.0 features remain:**
- ✅ Attention-based aggregation (always enabled)
- ✅ Temporal encoding for symptom sequences (always enabled)
- ✅ Focal loss for handling class imbalance (always enabled)
- ✅ Contrastive learning (enabled by default, weight=0.5)
- ✅ Hierarchical ICD loss (enabled by default, weight=0.3)
- ✅ Temperature scaling for calibration (always enabled)
- ✅ Beam search for generation (always enabled)

## Overview

BRISM (Bayesian Reciprocal ICD-Symptom Model) implements a dual encoder-decoder architecture with a shared latent space for:
- **Forward path**: Predicting ICD diagnosis codes from symptom descriptions
- **Reverse path**: Generating symptom sequences from ICD codes
- **Uncertainty estimation**: Monte Carlo dropout for confidence intervals

## Architecture

### Key Components

1. **Dual Encoder-Decoder Pairs**:
   - Symptom Encoder: Maps symptom sequences → latent distribution (with attention aggregation)
   - ICD Encoder: Maps ICD codes → latent distribution
   - Symptom Decoder: Generates symptom sequences from latent (with beam search)
   - ICD Decoder: Predicts ICD probabilities from latent (with temperature scaling)

2. **Shared Latent Space**: Both paths use the same latent representation, enabling:
   - Bidirectional translation
   - Cycle consistency constraints
   - Multi-task learning
   - Contrastive learning for better representations

3. **Advanced Features (All Enabled)**:
   - **Attention Mechanism**: Self-attention for symptom aggregation
   - **Temporal Encoding**: Positional or timestamp-based encoding for symptom sequences
   - **Focal Loss**: Handles class imbalance in ICD prediction
   - **Hierarchical Loss**: Respects ICD-10 hierarchy relationships
   - **Contrastive Loss**: Improves latent space quality
   - **Temperature Scaling**: Calibrates prediction probabilities

4. **Loss Functions**:
   - **Reconstruction Loss**: Focal loss for ICD, cross-entropy for symptoms
   - **KL Divergence**: Regularizes latent distributions
   - **Cycle Consistency**: Ensures latent representations align across cycles
   - **Contrastive Loss**: Brings similar diagnoses closer in latent space
   - **Hierarchical Loss**: Penalizes predictions far from true class in ICD hierarchy

5. **Uncertainty Quantification**:
   - Monte Carlo dropout during inference
   - Confidence intervals for predictions
   - Epistemic uncertainty estimation
   - Ensemble methods (pseudo-ensemble with dropout)

## Installation

```bash
# Clone the repository
git clone https://github.com/LegitimatelyBatman/BRISM.git
cd BRISM

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Quick Start

```python
import torch
from brism import BRISM, BRISMConfig, train_brism, diagnose_with_confidence
from brism.loss import BRISMLoss

# Configure model (v3.1.0 - simplified configuration)
config = BRISMConfig(
    symptom_vocab_size=1000,
    icd_vocab_size=500,
    latent_dim=64,
    mc_samples=20
    # Attention, temporal encoding, and other advanced features
    # are ALWAYS enabled in v3.0.0+ - no flags needed!
)

# Initialize model
model = BRISM(config)

# Training (with your data)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Loss function with v3.0.0+ defaults
loss_fn = BRISMLoss(
    kl_weight=0.1, 
    cycle_weight=1.0,
    contrastive_weight=0.5,  # Enabled by default
    hierarchical_weight=0.3,  # Enabled by default
    focal_gamma=2.0  # Focal loss always enabled
)

history = train_brism(
    model=model,
    train_loader=train_loader,  # Your data loader
    optimizer=optimizer,
    loss_fn=loss_fn,
    num_epochs=10,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# Inference with confidence intervals
symptoms = torch.tensor([1, 5, 12, 45, 67])  # Example symptom IDs
diagnosis = diagnose_with_confidence(
    model=model,
    symptoms=symptoms,
    device=device,
    confidence_level=0.95,
    top_k=5
)

# Print results
for i, pred in enumerate(diagnosis['predictions'], 1):
    print(f"{i}. ICD {pred['icd_code']}: "
          f"{pred['probability']:.4f} ± {pred['std']:.4f} "
          f"(95% CI: [{pred['confidence_interval'][0]:.4f}, "
          f"{pred['confidence_interval'][1]:.4f}])")
```

## Training

The model uses **alternating batch training** that cycles through:
1. Forward path: symptoms → ICD
2. Reverse path: ICD → symptoms
3. Forward cycle: symptoms → ICD → symptoms
4. Reverse cycle: ICD → symptoms → ICD

This ensures both directions and cycle consistency are jointly optimized.

## Example Usage

Run the comprehensive example to see the model in action with synthetic data:

```bash
# Comprehensive example demonstrating all BRISM features
python example.py
```

This demonstrates:
- Model initialization and training with v3.0.0+ defaults
- Diagnosis with confidence intervals
- Interpretability tools (attention visualization, integrated gradients)
- Beam search for symptom generation
- Ensemble uncertainty quantification
- Active learning interface
- And more!

## Testing

Run unit tests:

```bash
python -m unittest discover tests/
# or
python -m pytest tests/  # if pytest is installed
```

## Model Configuration

Key configuration parameters:

```python
BRISMConfig(
    symptom_vocab_size=1000,    # Size of symptom vocabulary
    icd_vocab_size=500,         # Size of ICD code vocabulary
    symptom_embed_dim=128,      # Symptom embedding dimension
    icd_embed_dim=128,          # ICD embedding dimension
    encoder_hidden_dim=256,     # Encoder hidden layer size
    latent_dim=64,              # Latent space dimension
    decoder_hidden_dim=256,     # Decoder hidden layer size
    max_symptom_length=50,      # Maximum symptom sequence length
    dropout_rate=0.2,           # Dropout rate
    mc_samples=20               # Monte Carlo samples for uncertainty
)
```

## Loss Weights

Configure loss function weights:

```python
BRISMLoss(
    kl_weight=0.1,      # Weight for KL divergence term
    cycle_weight=1.0    # Weight for cycle consistency
)
```

## Features

- ✅ Dual encoder-decoder architecture with shared latent space
- ✅ Forward path: symptoms → ICD diagnosis
- ✅ Reverse path: ICD → symptom generation
- ✅ VAE-style reconstruction losses
- ✅ KL divergence regularization
- ✅ Cycle consistency losses
- ✅ Monte Carlo dropout for uncertainty
- ✅ Confidence intervals on predictions
- ✅ Alternating batch training
- ✅ Comprehensive unit tests
- ✅ **NEW: Attention-based symptom encoding** - Self-attention layer for learning symptom importance
- ✅ **NEW: ICD-10 hierarchical loss** - Tree-based distance penalty for hierarchically similar codes
- ✅ **NEW: Medical data loaders** - MIMIC-III/IV format support with patient-level splits
- ✅ **NEW: Model checkpointing** - Automatic best model saving with resume capability
- ✅ **NEW: Early stopping** - Configurable patience for preventing overfitting

## New Features (v0.2.0)

### Enhanced Interpretability and Clinical Usability

Six major features have been added to improve model interpretability, generation quality, uncertainty quantification, and clinical usability:

1. **Interpretability Tools** - Understand model decisions with:
   - Attention visualization showing which symptoms matter most
   - Integrated gradients for feature attribution
   - Counterfactual explanations ("if we remove symptom X, probability drops by Y%")
   - Comprehensive explanation API

2. **Beam Search for Symptom Generation** - Generate diverse, high-quality symptom sequences:
   - Replaces greedy decoding with beam search (width 3-5)
   - Tracks top-k sequences by cumulative probability
   - Temperature and length penalty controls

3. **Contrastive Learning** - Better latent space structure:
   - Triplet loss pulls same-disease symptoms together
   - Pushes different-disease symptoms apart
   - Improves generalization and clustering

4. **Ensemble Uncertainty** - Decompose uncertainty:
   - True ensemble: multiple independently trained models
   - Pseudo-ensemble: single model with dropout masks (faster)
   - Separates epistemic (model) and aleatoric (data) uncertainty

5. **Symptom Synonym Handling** - Normalize medical terminology:
   - Maps "SOB", "dyspnea", "breathlessness" → "shortness of breath"
   - UMLS/SNOMED-CT compatible
   - Default medical abbreviations included

6. **Active Learning Interface** - Interactive diagnosis:
   - Suggests most informative symptoms to query next
   - Reduces average questions needed by 30-40%
   - Multiple query strategies (entropy, BALD, variance, EIG)

See [NEW_FEATURES_V02.md](NEW_FEATURES_V02.md) for detailed documentation and examples.

### Quick Example of New Features

```python
from brism import (
    BRISM, BRISMConfig,
    IntegratedGradients,
    generate_symptoms_beam_search,
    BRISMEnsemble,
    SymptomNormalizer,
    ActiveLearner
)

# Create model
model = BRISM(BRISMConfig(use_attention=True))

# 1. Interpretability
ig = IntegratedGradients(model)
attributions = ig.attribute(symptoms)  # Which symptoms matter most?

# 2. Beam search generation
beams = generate_symptoms_beam_search(model, icd_code, device, beam_width=5)

# 3. Ensemble uncertainty
ensemble = BRISMEnsemble(models=[model], use_pseudo_ensemble=True, n_models=5)
result = ensemble.diagnose_with_ensemble(symptoms)
print(f"Epistemic uncertainty: {result['uncertainty']['epistemic']}")

# 4. Symptom normalization
normalizer = SymptomNormalizer()
normalized = normalizer.normalize("SOB")  # → "shortness of breath"

# 5. Active learning
learner = ActiveLearner(model)
recommendations = learner.query_next_symptom(symptoms, method='bald')
```

Run the comprehensive example:
```bash
python example_new_features.py
```

## New Features (v0.2.0)

### 1. Attention-Based Symptom Encoding

Replace mean pooling with self-attention to learn which symptoms are most diagnostically relevant:

```python
config = BRISMConfig(
    symptom_vocab_size=1000,
    icd_vocab_size=500,
    use_attention=True  # Enable attention mechanism
)
model = BRISM(config)
```

The attention layer learns weights over the symptom sequence, focusing on the most important symptoms for diagnosis.

### 2. ICD-10 Hierarchical Loss

Use hierarchical ICD code structure to reduce loss penalty for predictions in the same category:

```python
from brism import ICDHierarchy, BRISMLoss

# Create hierarchy from YAML config
icd_hierarchy = ICDHierarchy(icd_vocab_size=500)
icd_hierarchy.build_from_yaml('config/icd_codes.yaml')

# Use hierarchical loss
loss_fn = BRISMLoss(
    kl_weight=0.1,
    cycle_weight=1.0,
    icd_hierarchy=icd_hierarchy,
    hierarchical_weight=0.3,  # 30% hierarchical, 70% standard CE
    hierarchical_temperature=1.0
)
```

See `config/icd_codes.yaml` for ICD code mapping format. The hierarchical loss gives smaller penalties when predictions are in the same category (e.g., both diabetes codes).

### 3. Medical Data Loaders

Process MIMIC-III/IV format data with proper patient-level splits:

```python
from brism import load_mimic_data

# Load and preprocess MIMIC data
train_dataset, val_dataset, test_dataset, preprocessor = load_mimic_data(
    diagnoses_path='data/diagnoses_icd.csv',
    notes_path='data/noteevents.csv',
    max_symptom_length=50,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

# Features:
# - ICD-9 to ICD-10 conversion
# - Patient-level splits (no data leakage)
# - Clinical note tokenization
# - Missing data handling
```

### 4. Model Checkpointing and Early Stopping

Train with automatic checkpointing and early stopping:

```python
from brism import train_brism, load_checkpoint

# Train with checkpointing and early stopping
history = train_brism(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    num_epochs=100,
    device=device,
    val_loader=val_loader,
    checkpoint_dir='./checkpoints',     # Enable checkpointing
    early_stopping_patience=5,          # Stop if no improvement for 5 epochs
    save_best_only=True                 # Only save best model
)

# Resume from checkpoint
checkpoint = load_checkpoint(
    'checkpoints/best_model.pt',
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device
)
print(f"Resumed from epoch {checkpoint['epoch']}")
```

## Advanced Example

See `example_advanced.py` for a comprehensive demonstration of all new features:

```bash
python example_advanced.py
```

This example shows:
- Attention-based symptom encoding
- Hierarchical ICD loss
- Medical data preprocessing
- Checkpointing and early stopping
- Resume from checkpoint

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use BRISM in your research, please cite:

```bibtex
@software{brism2025,
  title={BRISM: Bayesian Reciprocal ICD-Symptom Model},
  author={Sean},
  year={2025},
  url={https://github.com/LegitimatelyBatman/BRISM}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
