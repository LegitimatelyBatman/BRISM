# BRISM Implementation Summary

## Overview
This document provides a comprehensive summary of the BRISM (Bayesian Reciprocal ICD-Symptom Model) implementation.

## Problem Statement
Build BRISM with two encoder-decoder pairs sharing one latent space with:
- Forward path: embed symptoms → encode to latent → decode to ICD probabilities
- Reverse path: embed ICD code → encode to latent → decode to symptom sequence
- Three losses: reconstruction (VAE-style), KL divergence, cycle consistency
- Monte Carlo dropout for uncertainty
- Alternating batch training on each direction
- Output diagnostic probabilities with confidence intervals

## Architecture Implementation

### 1. Dual Encoder-Decoder Pairs with Shared Latent Space

**Symptom Encoder** (`Encoder` class in `brism/model.py`)
- Input: Symptom sequence embeddings (mean-pooled)
- Output: Latent distribution parameters (mu, logvar)
- Architecture: 2-layer MLP with dropout

**ICD Encoder** (`Encoder` class in `brism/model.py`)
- Input: ICD code embeddings
- Output: Latent distribution parameters (mu, logvar)
- Architecture: Same as symptom encoder (shared class)

**Symptom Decoder** (`SequenceDecoder` class in `brism/model.py`)
- Input: Latent sample
- Output: Symptom sequence logits
- Architecture: LSTM-based sequence generator with teacher forcing support

**ICD Decoder** (`Decoder` class in `brism/model.py`)
- Input: Latent sample
- Output: ICD probability distribution
- Architecture: 2-layer MLP with dropout

**Shared Latent Space**
- Both encoders produce distributions in the same latent space
- Reparameterization trick enables backpropagation
- Dimension configurable via `BRISMConfig.latent_dim`

### 2. Forward Path (Symptoms → ICD)

Implementation: `BRISM.forward_path()` in `brism/model.py`

```
symptoms [B, L] 
  → embedding [B, L, D] 
  → mean pooling [B, D]
  → symptom_encoder → (mu, logvar) [B, latent_dim]
  → reparameterize → z [B, latent_dim]
  → icd_decoder → icd_logits [B, icd_vocab_size]
```

### 3. Reverse Path (ICD → Symptoms)

Implementation: `BRISM.reverse_path()` in `brism/model.py`

```
icd_codes [B]
  → embedding [B, D]
  → icd_encoder → (mu, logvar) [B, latent_dim]
  → reparameterize → z [B, latent_dim]
  → symptom_decoder → symptom_logits [B, L, symptom_vocab_size]
```

### 4. Three Loss Functions

Implementation: `BRISMLoss` class in `brism/loss.py`

**Reconstruction Loss (VAE-style)**
- Forward: Cross-entropy between predicted and true ICD codes
- Reverse: Masked cross-entropy between predicted and true symptom sequences
- Handles padding in symptom sequences

**KL Divergence Loss**
- Regularizes latent distributions toward standard normal N(0, 1)
- Formula: KL(N(μ, σ²) || N(0, 1)) = 0.5 * Σ(1 + log(σ²) - μ² - σ²)
- Applied to both forward and reverse path latents

**Cycle Consistency Loss**
- Ensures latent distributions align after complete cycles
- Forward cycle: symptoms → ICD → symptoms (compare latents)
- Reverse cycle: ICD → symptoms → ICD (compare latents)
- Uses KL divergence between latent distributions

### 5. Monte Carlo Dropout for Uncertainty

Implementation: `BRISM.predict_with_uncertainty()` in `brism/model.py`

- Keeps dropout active during inference (`model.train()`)
- Performs N forward passes (default: 20)
- Computes mean and standard deviation of predictions
- Standard deviation represents epistemic uncertainty

**Confidence Intervals**: `diagnose_with_confidence()` in `brism/inference.py`
- Assumes normal distribution of predictions
- Computes confidence intervals using z-scores
- Default: 95% confidence level
- Bounds: [mean - z*std, mean + z*std] clipped to [0, 1]

### 6. Alternating Batch Training

Implementation: `train_brism()` in `brism/train.py`

Training cycles through four directions per batch:
1. **Batch 0 (mod 4)**: Forward path only
   - Loss: reconstruction + KL divergence
   
2. **Batch 1 (mod 4)**: Reverse path only
   - Loss: reconstruction + KL divergence
   
3. **Batch 2 (mod 4)**: Forward cycle
   - Loss: both reconstructions + both KL divergences + cycle consistency
   
4. **Batch 3 (mod 4)**: Reverse cycle
   - Loss: both reconstructions + both KL divergences + cycle consistency

All losses are jointly optimized through backpropagation.

### 7. Diagnostic Output with Confidence Intervals

Implementation: `diagnose_with_confidence()` in `brism/inference.py`

Output structure:
```python
{
    'predictions': [
        {
            'icd_code': int,
            'probability': float,
            'std': float,
            'confidence_interval': (lower, upper),
            'confidence_level': 0.95
        },
        ...  # top_k predictions
    ],
    'uncertainty': {
        'entropy': float,           # Predictive entropy
        'average_std': float,       # Average epistemic uncertainty
        'predictive_entropy': float # Total uncertainty
    },
    'raw_probabilities': {
        'mean': [...],  # Full probability distribution
        'std': [...]    # Standard deviations for all classes
    }
}
```

## File Structure

```
BRISM/
├── brism/
│   ├── __init__.py          # Package exports
│   ├── model.py             # Core model architecture
│   ├── loss.py              # Loss functions
│   ├── train.py             # Training loops
│   └── inference.py         # Inference with uncertainty
├── tests/
│   ├── __init__.py
│   ├── test_model.py        # Model architecture tests
│   ├── test_loss.py         # Loss function tests
│   └── test_inference.py    # Inference tests
├── example.py               # Demonstration script
├── requirements.txt         # Dependencies
├── setup.py                 # Package setup
├── README.md                # Documentation
└── .gitignore              # Git ignore rules
```

## Testing

24 comprehensive unit tests covering:
- Individual model components (encoders, decoders)
- Complete model paths (forward, reverse, cycles)
- Shared latent space verification
- All loss functions
- Monte Carlo dropout uncertainty
- Confidence interval computation
- Batch inference

All tests pass successfully.

## Example Usage

See `example.py` for a complete demonstration including:
- Synthetic data generation
- Model initialization and configuration
- Training with alternating batches
- Diagnosis with confidence intervals
- Symptom generation from ICD codes
- Uncertainty quantification

Run: `python example.py`

## Key Configuration Parameters

```python
BRISMConfig(
    symptom_vocab_size=1000,    # Number of unique symptoms
    icd_vocab_size=500,         # Number of unique ICD codes
    symptom_embed_dim=128,      # Symptom embedding dimension
    icd_embed_dim=128,          # ICD embedding dimension
    encoder_hidden_dim=256,     # Hidden layer size in encoders
    latent_dim=64,              # Shared latent space dimension
    decoder_hidden_dim=256,     # Hidden layer size in decoders
    max_symptom_length=50,      # Max length of symptom sequences
    dropout_rate=0.2,           # Dropout probability
    mc_samples=20               # Monte Carlo samples for uncertainty
)

BRISMLoss(
    kl_weight=0.1,              # Weight for KL divergence term
    cycle_weight=1.0            # Weight for cycle consistency
)
```

## Dependencies

- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- SciPy >= 1.10.0 (for confidence interval computation)

## Performance

Model size with default configuration:
- Total parameters: ~1.7M
- Latent dimension: 64
- Training speed: ~50-100 batches/second (CPU)

## Future Enhancements

Potential improvements not included in current implementation:
- Attention mechanisms for symptom encoding
- Hierarchical latent spaces
- Adversarial training for better cycle consistency
- Transformer-based sequence models
- Integration with real medical datasets
- Model interpretability features

## Verification

All requirements from the problem statement have been implemented and verified:
- ✅ Two encoder-decoder pairs with shared latent space
- ✅ Forward path: symptoms → ICD
- ✅ Reverse path: ICD → symptoms
- ✅ Three loss functions (reconstruction, KL, cycle consistency)
- ✅ Monte Carlo dropout for uncertainty
- ✅ Alternating batch training
- ✅ Confidence intervals on predictions

Run `python -m unittest discover tests/` to verify all tests pass.
