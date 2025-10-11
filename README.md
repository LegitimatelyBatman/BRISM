# BRISM
Bayesian Reciprocal ICD-Symptom Model

A deep learning model for bidirectional mapping between medical symptoms and ICD diagnosis codes with uncertainty quantification.

## Overview

BRISM (Bayesian Reciprocal ICD-Symptom Model) implements a dual encoder-decoder architecture with a shared latent space for:
- **Forward path**: Predicting ICD diagnosis codes from symptom descriptions
- **Reverse path**: Generating symptom sequences from ICD codes
- **Uncertainty estimation**: Monte Carlo dropout for confidence intervals

## Architecture

### Key Components

1. **Dual Encoder-Decoder Pairs**:
   - Symptom Encoder: Maps symptom sequences → latent distribution
   - ICD Encoder: Maps ICD codes → latent distribution
   - Symptom Decoder: Generates symptom sequences from latent
   - ICD Decoder: Predicts ICD probabilities from latent

2. **Shared Latent Space**: Both paths use the same latent representation, enabling:
   - Bidirectional translation
   - Cycle consistency constraints
   - Multi-task learning

3. **Loss Functions**:
   - **Reconstruction Loss**: VAE-style reconstruction for both directions
   - **KL Divergence**: Regularizes latent distributions
   - **Cycle Consistency**: Ensures latent representations align across cycles

4. **Uncertainty Quantification**:
   - Monte Carlo dropout during inference
   - Confidence intervals for predictions
   - Epistemic uncertainty estimation

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

# Configure model
config = BRISMConfig(
    symptom_vocab_size=1000,
    icd_vocab_size=500,
    latent_dim=64,
    mc_samples=20
)

# Initialize model
model = BRISM(config)

# Training (with your data)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = BRISMLoss(kl_weight=0.1, cycle_weight=1.0)

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

Run the example script to see the model in action with synthetic data:

```bash
python example.py
```

This demonstrates:
- Model initialization and training
- Diagnosis with confidence intervals
- Symptom generation from ICD codes
- Uncertainty quantification

## Testing

Run unit tests:

```bash
python -m pytest tests/
# or
python -m unittest discover tests/
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
