# Changelog

All notable changes to BRISM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
See [BREAKING_CHANGES.md](BREAKING_CHANGES.md) for detailed migration instructions from v0.2.0.

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
- **Patch version (x.y.Z):** Bug fixes, backward compatible

## Links
- [v3.0.0 Release Notes](BREAKING_CHANGES.md)
- [GitHub Repository](https://github.com/LegitimatelyBatman/BRISM)
