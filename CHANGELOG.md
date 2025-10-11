# Changelog

All notable changes to BRISM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.1] - 2025-10-11

### 📚 Documentation and Example Consolidation

This is a documentation and example consolidation release that combines redundant files while maintaining all functionality from v3.0.0.

### Changed

#### Documentation
- **Consolidated** all implementation documentation into single `IMPLEMENTATION.md`
  - Merged `IMPLEMENTATION_SUMMARY.md` (Enhancement Phase 1 - v0.2.0 Part 1)
  - Merged `IMPLEMENTATION_DETAILS.md` (Enhancement Phase 2 - v0.2.0 Part 2)
  - Merged `IMPLEMENTATION_SUMMARY_V02.md` (Enhancement Phase 3 - v0.2.0 Part 3)
  - Merged original `IMPLEMENTATION.md` (Core Architecture - v0.1.0)
  - Result: Single comprehensive implementation document covering all phases

#### Examples
- **Consolidated** all example scripts into single comprehensive `example.py`
  - Merged basic example (original `example.py`)
  - Merged `example_enhanced_features.py` (metrics, temporal, calibration, focal loss)
  - Merged `example_new_features.py` (interpretability, beam search, ensemble, active learning)
  - Result: Single comprehensive example demonstrating all BRISM v3.0.0 features in organized sections

#### This Changelog
- **Incorporated** breaking changes details from `BREAKING_CHANGES.md`
- **Incorporated** refactoring summary from `REFACTORING_SUMMARY.md`
- All migration information now centralized in this changelog

### Removed Files (Redundant)
- `IMPLEMENTATION_SUMMARY.md` - Content moved to `IMPLEMENTATION.md`
- `IMPLEMENTATION_DETAILS.md` - Content moved to `IMPLEMENTATION.md`
- `IMPLEMENTATION_SUMMARY_V02.md` - Content moved to `IMPLEMENTATION.md`
- `example_enhanced_features.py` - Content moved to `example.py`
- `example_new_features.py` - Content moved to `example.py`
- `BREAKING_CHANGES.md` - Content incorporated into this changelog
- `REFACTORING_SUMMARY.md` - Content incorporated into this changelog

### Improved
- Cleaner repository structure with less file duplication
- Single source of truth for implementation documentation
- Single comprehensive example covering all features
- Easier navigation and maintenance

### Migration Notes
- All code functionality remains unchanged from v3.0.0
- No API changes or breaking changes
- Simply documentation and example reorganization
- See v3.0.0 section below for feature details and migration from v0.2.0

---

## [3.0.0] - 2025-10-11

### 🚀 Major Refactoring Release

This is a major refactoring that simplifies the codebase by making advanced features mandatory and removing optional flags.

### Changed

#### Model Architecture
- **BREAKING:** Removed `use_attention` flag - attention-based symptom aggregation is now ALWAYS enabled
- **BREAKING:** Removed `use_temporal_encoding` flag - temporal encoding is now ALWAYS enabled
- Temperature scaling remains always enabled (no change from v0.2.0)
- Added `beam_width` parameter to BRISMConfig (default: 5)
- Added `n_ensemble_models` parameter to BRISMConfig (default: 5)

#### Loss Functions
- **BREAKING:** Removed `use_focal_loss` flag - focal loss is now ALWAYS used for ICD classification
- **BREAKING:** Changed `contrastive_weight` default from 0.0 to 0.5 (now enabled by default)
- **BREAKING:** Changed `hierarchical_weight` default from 0.0 to 0.3 (now enabled by default)
- Contrastive learning is now a standard part of training (was optional)
- Hierarchical ICD loss is now a standard part of training (was optional)

#### Configuration
- Added `config/general_configuration.yaml` with documented default hyperparameters
- Simplified BRISMConfig dataclass with fewer conditional parameters

### Fixed
- Fixed integrated gradients computation with temporal encoding
- Improved gradient flow in interpretability tools

### Removed
- Mean pooling option for symptom aggregation (use attention only)
- Standard cross-entropy option (use focal loss only)
- Optional contrastive and hierarchical loss (now mandatory with configurable weights)

### Improved
- Cleaner codebase with less conditional logic
- Better default hyperparameters based on v0.2.0 results
- More consistent API across the library
- Improved test coverage

### Migration Guide
For detailed migration instructions from v0.2.0, see the sections below.

#### Breaking Changes Details

**1. Model Architecture (brism/model.py)**

*Removed Configuration Flags:*

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

**2. Loss Functions (brism/loss.py)**

*Focal Loss is Now Mandatory:*

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

*Updated Default Weights:*

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

**3. Configuration Changes**

*New Default Configuration File:*

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

**4. Backward Compatibility with Checkpoints**

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

#### Migration Checklist

- [ ] Remove all `use_attention=True/False` from BRISMConfig instantiations
- [ ] Remove all `use_temporal_encoding=True/False` from BRISMConfig instantiations
- [ ] Remove all `use_focal_loss=True/False` from BRISMLoss instantiations
- [ ] Review `contrastive_weight` and `hierarchical_weight` defaults (now 0.5 and 0.3)
- [ ] Update any code that checks these flags (e.g., `if config.use_attention:`)
- [ ] Update checkpoint loading code if needed
- [ ] Test all code paths with the new defaults
- [ ] Update documentation and examples

#### Rationale for Changes

These changes were made to:

1. **Simplify the codebase:** Remove conditionals and branching logic
2. **Improve code quality:** Fewer edge cases and better test coverage
3. **Better defaults:** Make proven features standard rather than optional
4. **Clearer intent:** "Standard" vs "optional" is now explicit in the API

The features that are now mandatory (attention, temporal encoding, focal loss, etc.) have been proven effective in v0.2.0 and are now considered essential parts of the BRISM architecture.

#### Refactoring Summary

**Files Changed:**
- **Modified**: 10 files
  - `brism/model.py` - Removed use_attention and use_temporal_encoding flags
  - `brism/loss.py` - Removed use_focal_loss flag, updated defaults
  - `brism/interpretability.py` - Fixed for always-on features
  - `tests/test_loss.py` - Updated for new API
  - `tests/test_new_features.py` - Updated for new API
  - `example_enhanced_features.py` - Removed deprecated flags (later consolidated)
  - `example_new_features.py` - Removed deprecated flags (later consolidated)
  - `brism/__init__.py` - Version bump to 3.0.0
  - `setup.py` - Version bump to 3.0.0
  - `README.md` - Updated with v3.0.0 changes

- **Created**: 4 files
  - `BREAKING_CHANGES.md` - Comprehensive migration guide (now in this changelog)
  - `CHANGELOG.md` - Detailed version history (this file)
  - `config/general_configuration.yaml` - Default hyperparameters
  - `REFACTORING_SUMMARY.md` - Summary of changes (now in this changelog)

- **Deleted**: 5 files (in v3.0.0)
  - `NEW_FEATURES.md` - Obsolete (features are now standard)
  - `NEW_FEATURES_V02.md` - Obsolete
  - `ENHANCED_FEATURES.md` - Obsolete
  - `example_advanced.py` - Obsolete (everything is "advanced" now)
  - Various old documentation files

**Code Quality Improvements:**
1. **Reduced Complexity**: Removed ~100+ lines of conditional logic
2. **Better Defaults**: Advanced features enabled by default (proven effective in v0.2.0)
3. **Clearer Intent**: No ambiguity about what's "optional" vs "standard"
4. **Improved Maintainability**: Fewer branches and edge cases
5. **Better Documentation**: Clear migration path and breaking changes documented

**Key Takeaways:**
1. All "optional" features from v0.2.0 are now standard in v3.0.0
2. The codebase is significantly simpler with less branching logic
3. Better defaults based on empirical results from v0.2.0
4. Comprehensive documentation for migration
5. All tests passing with improved code quality

**Testing Results:**
All 71 tests passing: ✅

```
Ran 71 tests in 2.712s
OK
```

#### Deprecation Timeline

- **v0.2.0:** Optional features introduced
- **v3.0.0:** Optional features made mandatory (October 2025)
- **v3.0.1:** Documentation and examples consolidated (current release)
- **Future:** v0.2.0 will be considered legacy and unsupported

We recommend migrating to v3.0.0+ as soon as possible to benefit from improved code quality and future updates.

#### Support

If you encounter issues migrating from v0.2.0 to v3.0.0, please:
1. Check this migration guide first
2. Review the updated examples in the repository
3. Consult `IMPLEMENTATION.md` for comprehensive documentation
4. Open an issue on GitHub with details about your migration problem

---

### Migration Guide

---

## [0.2.0] - 2025-10 (Previous Release)

### Added
- Interpretability tools (integrated gradients, attention visualization, counterfactuals)
- Beam search for symptom generation
- Ensemble methods for uncertainty quantification
- Symptom normalization and preprocessing
- Active learning capabilities
- Temporal encoding for symptom sequences
- Attention-based symptom aggregation
- Temperature scaling for calibration
- Focal loss for class imbalance
- Contrastive learning for latent space
- Hierarchical ICD loss
- Comprehensive evaluation metrics

### Features (Now Made Mandatory in v3.0.0)
- Optional attention mechanism (`use_attention` flag)
- Optional temporal encoding (`use_temporal_encoding` flag)
- Optional focal loss (`use_focal_loss` flag)
- Optional contrastive loss (default disabled)
- Optional hierarchical loss (default disabled)

---

## [0.1.0] - 2024 (Initial Release)

### Added
- Initial BRISM architecture with dual encoder-decoder
- Shared latent space for bidirectional symptom-ICD mapping
- VAE-based uncertainty quantification
- Monte Carlo dropout for predictions
- Basic training utilities
- Basic evaluation metrics
- Example scripts

### Features
- Forward path: symptoms → ICD codes
- Reverse path: ICD codes → symptoms
- Cycle consistency loss
- KL divergence regularization
- Basic data preprocessing

---

## Version Numbering

- **Major version (X.0.0):** Breaking changes, significant refactoring
- **Minor version (x.Y.0):** New features, backward compatible
- **Patch version (x.y.Z):** Bug fixes, documentation updates, backward compatible

## Links
- [Latest Release (v3.0.1)](https://github.com/LegitimatelyBatman/BRISM)
- [Implementation Documentation](IMPLEMENTATION.md)
- [GitHub Repository](https://github.com/LegitimatelyBatman/BRISM)
