# Breaking Changes in BRISM v3.0.0

This document describes breaking changes when migrating from v0.2.0 to v3.0.0.

## Overview

BRISM v3.0.0 is a major refactoring that simplifies the codebase by making advanced features mandatory. Previous "optional" or "experimental" features are now standard, with conditional code removed for clarity and maintainability.

## ⚠️ Breaking Changes

### 1. Model Architecture (brism/model.py)

#### Removed Configuration Flags

**`use_attention` flag removed**
- **Before (v0.2.0):**
  ```python
  config = BRISMConfig(
      symptom_vocab_size=1000,
      icd_vocab_size=500,
      use_attention=True  # Optional flag
  )
  ```
- **After (v3.0.0):**
  ```python
  config = BRISMConfig(
      symptom_vocab_size=1000,
      icd_vocab_size=500
      # Attention is ALWAYS enabled - no flag needed
  )
  ```
- **Impact:** Models ALWAYS use attention-based symptom aggregation. Mean pooling is no longer available.
- **Migration:** Remove `use_attention=True/False` from all config instantiations.

**`use_temporal_encoding` flag removed**
- **Before (v0.2.0):**
  ```python
  config = BRISMConfig(
      symptom_vocab_size=1000,
      use_temporal_encoding=True,  # Optional flag
      temporal_encoding_type='positional'
  )
  ```
- **After (v3.0.0):**
  ```python
  config = BRISMConfig(
      symptom_vocab_size=1000,
      temporal_encoding_type='positional'  # Always enabled
  )
  ```
- **Impact:** Temporal encoding is ALWAYS applied to symptom sequences.
- **Migration:** Remove `use_temporal_encoding=True/False` from all config instantiations.

### 2. Loss Functions (brism/loss.py)

#### Focal Loss is Now Mandatory

**`use_focal_loss` flag removed**
- **Before (v0.2.0):**
  ```python
  loss_fn = BRISMLoss(
      kl_weight=0.1,
      use_focal_loss=True,  # Optional flag
      focal_gamma=2.0
  )
  ```
- **After (v3.0.0):**
  ```python
  loss_fn = BRISMLoss(
      kl_weight=0.1,
      focal_gamma=2.0  # Focal loss ALWAYS enabled
  )
  ```
- **Impact:** Focal loss is ALWAYS used for ICD classification. Standard cross-entropy is no longer available.
- **Migration:** Remove `use_focal_loss=True/False` from all BRISMLoss instantiations.

#### Updated Default Weights

**Contrastive learning default changed**
- **Before (v0.2.0):** `contrastive_weight=0.0` (disabled by default)
- **After (v3.0.0):** `contrastive_weight=0.5` (enabled by default)
- **Impact:** Contrastive loss is now applied by default for better latent space learning.
- **Migration:** If you want to disable contrastive learning, explicitly set `contrastive_weight=0.0`.

**Hierarchical loss default changed**
- **Before (v0.2.0):** `hierarchical_weight=0.0` (disabled by default)
- **After (v3.0.0):** `hierarchical_weight=0.3` (enabled by default)
- **Impact:** Hierarchical ICD relationships are now considered by default.
- **Migration:** If you want to disable hierarchical loss, explicitly set `hierarchical_weight=0.0`.

### 3. Configuration Changes

#### New Default Configuration File

A new YAML configuration file is available at `config/general_configuration.yaml` that documents all default hyperparameters.

**Key changes in BRISMConfig:**
```python
@dataclass
class BRISMConfig:
    # Removed fields:
    # - use_attention: bool
    # - use_temporal_encoding: bool
    
    # New fields:
    beam_width: int = 5  # For beam search generation
    n_ensemble_models: int = 5  # For pseudo-ensemble
```

### 4. Backward Compatibility with Checkpoints

**Loading v0.2.0 checkpoints:**
- ✅ Model weights are compatible (architecture unchanged)
- ⚠️ Config objects need migration (remove deprecated flags)
- ⚠️ Loss functions may need reinitialization with new defaults

**Example migration:**
```python
# Load old checkpoint
checkpoint = torch.load('model_v0.2.0.pt')

# Create new config (without deprecated flags)
new_config = BRISMConfig(
    symptom_vocab_size=checkpoint['config'].symptom_vocab_size,
    icd_vocab_size=checkpoint['config'].icd_vocab_size,
    # ... other parameters (don't include use_attention or use_temporal_encoding)
)

# Load weights
model = BRISM(new_config)
model.load_state_dict(checkpoint['model_state_dict'])
```

## Migration Checklist

- [ ] Remove all `use_attention=True/False` from BRISMConfig instantiations
- [ ] Remove all `use_temporal_encoding=True/False` from BRISMConfig instantiations
- [ ] Remove all `use_focal_loss=True/False` from BRISMLoss instantiations
- [ ] Review `contrastive_weight` and `hierarchical_weight` defaults (now 0.5 and 0.3)
- [ ] Update any code that checks these flags (e.g., `if config.use_attention:`)
- [ ] Update checkpoint loading code if needed
- [ ] Test all code paths with the new defaults
- [ ] Update documentation and examples

## Rationale

These changes were made to:

1. **Simplify the codebase:** Remove conditionals and branching logic
2. **Improve code quality:** Fewer edge cases and better test coverage
3. **Better defaults:** Make proven features standard rather than optional
4. **Clearer intent:** "Standard" vs "optional" is now explicit in the API

The features that are now mandatory (attention, temporal encoding, focal loss, etc.) have been proven effective in v0.2.0 and are now considered essential parts of the BRISM architecture.

## Support

If you encounter issues migrating from v0.2.0 to v3.0.0, please:
1. Check this document first
2. Review the updated examples in the repository
3. Open an issue on GitHub with details about your migration problem

## Deprecation Timeline

- **v0.2.0:** Optional features introduced
- **v3.0.0:** Optional features made mandatory (current release)
- **Future:** v0.2.0 will be considered legacy and unsupported

We recommend migrating to v3.0.0 as soon as possible to benefit from improved code quality and future updates.
