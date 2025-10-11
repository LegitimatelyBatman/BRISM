# BRISM Coding Agent Instructions

## Repository Overview

BRISM (Bayesian Reciprocal ICD-Symptom Model) is a PyTorch-based deep learning project implementing a dual encoder-decoder architecture for bidirectional mapping between medical symptoms and ICD diagnosis codes with uncertainty quantification.

**Current Version:** v3.0.1  
**Python Version:** 3.8+  
**Primary Framework:** PyTorch 2.0+  
**Project Size:** ~7,500 lines of Python code  
**Test Suite:** 71 unit tests (run time: ~3 seconds)  

### Key Technologies
- **Framework:** PyTorch (neural network architecture)
- **Scientific Computing:** NumPy, SciPy, Pandas
- **Machine Learning:** scikit-learn (metrics, preprocessing)
- **Visualization:** matplotlib
- **Configuration:** YAML (PyYAML)

### Architecture
The model uses:
- Dual encoder-decoder pairs with shared latent space
- VAE-style latent distributions with reparameterization
- Attention-based symptom aggregation (always enabled in v3.0.0)
- Temporal encoding for sequences (always enabled in v3.0.0)
- Focal loss for ICD classification (always enabled in v3.0.0)
- Monte Carlo dropout for uncertainty quantification

---

## Critical Setup Instructions

### Installation Steps (ALWAYS RUN IN THIS ORDER)

1. **Install dependencies first:**
   ```bash
   pip install -r requirements.txt
   ```
   - Time: ~30-60 seconds
   - Required packages: torch, numpy, scipy, pyyaml, pandas, scikit-learn, matplotlib
   - **IMPORTANT:** Do NOT install individual packages - use requirements.txt to ensure compatible versions

2. **Install package in editable mode:**
   ```bash
   pip install -e .
   ```
   - Time: ~5 seconds
   - Makes `brism` package importable
   - **ALWAYS do this after modifying setup.py or adding new modules**

### Environment Requirements
- Python 3.8+ (tested with 3.12.3)
- PyTorch 2.0+ with CUDA support (CPU fallback available)
- No GPU required for tests (but recommended for training)

---

## Build and Test Instructions

### Running Tests

**Primary test command (ALWAYS USE THIS):**
```bash
python -m unittest discover tests/
```
- Runs all 71 tests
- Expected time: ~3 seconds
- All tests should pass in baseline
- **IMPORTANT:** Run tests before AND after making changes

**Alternative with pytest (if installed):**
```bash
python -m pytest tests/
```

### Expected Test Output
```
Ran 71 tests in 2.7s
OK
```

**Known warnings (SAFE TO IGNORE):**
- `UserWarning: Converting a tensor with requires_grad=True to a scalar` (from LBFGS optimizer in test_calibration.py)
- `UserWarning: std(): degrees of freedom is <= 0` (from ensemble predictions with single sample)
- `Temperature calibration: 1.0000 -> X.XXXX` (informational output, not an error)

### Test Organization
- `test_model.py` - Core model architecture (encoders, decoders, forward/reverse paths)
- `test_loss.py` - Loss functions (reconstruction, KL, cycle, focal, contrastive, hierarchical)
- `test_inference.py` - Inference and prediction functions
- `test_metrics.py` - Evaluation metrics
- `test_temporal.py` - Temporal encoding
- `test_calibration.py` - Temperature scaling for calibration
- `test_new_features.py` - Interpretability, beam search, ensemble, active learning, symptom normalization

### Running Examples

**DO NOT run example.py directly** - it has a known import issue:
```bash
python example.py  # FAILS - missing import
```

The example file exists for reference only. To test functionality, use the test suite.

---

## Project Layout

### Root Directory Structure
```
BRISM/
├── brism/              # Main package (14 Python modules)
├── tests/              # Test suite (7 test modules, 71 tests)
├── config/             # Configuration files (YAML)
├── example.py          # Demonstration code (reference only - has import issues)
├── requirements.txt    # Dependencies (7 packages)
├── setup.py            # Package installation
├── README.md           # User documentation
├── CHANGELOG.md        # Version history and migration guides
├── IMPLEMENTATION.md   # Detailed implementation documentation
├── LICENSE             # MIT license
└── .gitignore         # Git ignore patterns
```

### Package Architecture (`brism/`)

**Core modules (modify these for model changes):**
- `model.py` (551 lines) - Core BRISM architecture, encoders, decoders, attention mechanism, temporal encoding
- `loss.py` (586 lines) - All loss functions (reconstruction, KL, cycle, focal, contrastive, hierarchical)
- `train.py` (475 lines) - Training loop, checkpointing, early stopping
- `inference.py` (358 lines) - Inference functions, beam search generation

**Feature modules (modify for specific features):**
- `interpretability.py` (563 lines) - Integrated gradients, attention visualization, counterfactual explanations
- `ensemble.py` (379 lines) - Ensemble methods, pseudo-ensemble with dropout
- `active_learning.py` (498 lines) - Active learning strategies (entropy, BALD, variance, EIG)
- `symptom_normalization.py` (431 lines) - Synonym handling for medical terms
- `metrics.py` (409 lines) - Evaluation metrics (accuracy, AUROC, calibration)
- `calibration.py` (178 lines) - Temperature scaling
- `temporal.py` (222 lines) - Temporal encoding implementations
- `data_loader.py` (441 lines) - Medical data preprocessing (MIMIC format)

**Configuration & Utilities:**
- `icd_hierarchy.py` (224 lines) - ICD-10 hierarchy tree structure
- `__init__.py` (102 lines) - Package exports (42 exported symbols)

### Configuration Files (`config/`)
- `general_configuration.yaml` - Default hyperparameters for v3.0.0
- `icd_codes.yaml` - ICD-10 hierarchy relationships for hierarchical loss

### Key Files to Reference
- **CHANGELOG.md** - Version history, breaking changes, migration guides (especially v0.2.0→v3.0.0)
- **IMPLEMENTATION.md** - Detailed technical documentation of all features and implementation phases
- **README.md** - Installation, quick start, API reference

---

## Code Modification Guidelines

### v3.0.0 Breaking Changes (CRITICAL)

**In v3.0.0, these features became MANDATORY (flags removed):**

1. **Attention mechanism** - ALWAYS enabled, no `use_attention` flag
2. **Temporal encoding** - ALWAYS enabled, no `use_temporal_encoding` flag  
3. **Focal loss** - ALWAYS enabled, no `use_focal_loss` flag
4. **Contrastive learning** - Enabled by default (weight=0.5)
5. **Hierarchical ICD loss** - Enabled by default (weight=0.3)

**When creating BRISMConfig:**
```python
# CORRECT (v3.0.0)
config = BRISMConfig(
    symptom_vocab_size=1000,
    icd_vocab_size=500,
    latent_dim=64
    # No use_attention, use_temporal_encoding flags!
)

# WRONG (v0.2.0 style - will cause errors)
config = BRISMConfig(
    use_attention=True,  # REMOVED in v3.0.0
    use_temporal_encoding=True  # REMOVED in v3.0.0
)
```

**When creating BRISMLoss:**
```python
# CORRECT (v3.0.0 defaults)
loss_fn = BRISMLoss(
    kl_weight=0.1,
    cycle_weight=1.0,
    contrastive_weight=0.5,  # Default changed from 0.0
    hierarchical_weight=0.3,  # Default changed from 0.0
    focal_gamma=2.0  # Always applied
)
```

### Device Handling

**ALWAYS move tensors to device explicitly:**
```python
symptoms = batch['symptoms'].to(device)
icd_codes = batch['icd_codes'].to(device)
model.to(device)
```

**NEVER assume GPU is available:**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Common Pitfalls

1. **Import errors** - Always import from `brism` package, not direct file imports
   ```python
   # CORRECT
   from brism import BRISM, BRISMConfig, train_brism
   
   # WRONG
   from brism.model import BRISM  # Works but not consistent with package API
   ```

2. **Missing package installation** - Always run `pip install -e .` after modifying code structure

3. **Dropout at inference** - For uncertainty quantification, model uses dropout at inference time via `model.train()` mode

4. **Sequence padding** - Symptom sequences support variable lengths with padding token 0

### Where to Make Changes

**To add/modify model architecture:**
- Edit `brism/model.py`
- Update tests in `tests/test_model.py`
- Update `BRISMConfig` dataclass if adding new hyperparameters

**To add/modify loss functions:**
- Edit `brism/loss.py`
- Update tests in `tests/test_loss.py`
- Update `BRISMLoss` class if adding new loss terms

**To add/modify training:**
- Edit `brism/train.py`
- May need to update checkpointing/early stopping logic

**To add/modify inference:**
- Edit `brism/inference.py`
- Update tests in `tests/test_inference.py`

**To add new feature module:**
1. Create new file in `brism/` directory
2. Add tests in `tests/test_<feature>.py`
3. Export symbols in `brism/__init__.py` (both import and `__all__` list)
4. Update CHANGELOG.md with changes

---

## Validation Workflow

### Pre-Change Validation
1. Run tests to establish baseline: `python -m unittest discover tests/`
2. Verify all 71 tests pass

### Post-Change Validation
1. **ALWAYS run tests after ANY code change**
   ```bash
   python -m unittest discover tests/
   ```

2. **For model/loss changes, also test specific modules:**
   ```bash
   python -m unittest tests.test_model
   python -m unittest tests.test_loss
   ```

3. **Check import integrity:**
   ```bash
   python -c "from brism import *; print('All imports successful')"
   ```

4. **Verify package exports are correct:**
   ```bash
   python -c "from brism import __all__; print(f'{len(__all__)} exports')"
   ```
   - Expected: 42 exports

### Common Test Failures and Fixes

**"ModuleNotFoundError: No module named 'brism'"**
- Fix: Run `pip install -e .`

**"ImportError: cannot import name 'X' from 'brism'"**
- Fix: Check if symbol is exported in `brism/__init__.py`
- Add to both imports and `__all__` list

**Tests fail after adding new dependency**
- Fix: Add to `requirements.txt` AND `setup.py` install_requires
- Reinstall: `pip install -r requirements.txt`

---

## File Organization Rules

### Files to NEVER Delete
- Any file in `brism/` directory (all are used)
- Any file in `tests/` directory
- `config/general_configuration.yaml` - referenced by documentation
- `config/icd_codes.yaml` - used by hierarchical loss

### Files You Can Modify Safely
- Test files (add new tests)
- Documentation files (README.md, IMPLEMENTATION.md)
- CHANGELOG.md (add new entries at top)

### Changelog Protocol
**ALWAYS update CHANGELOG.md for code changes:**
- Add new section at top with version and date
- Use format: `## [X.Y.Z] - YYYY-MM-DD`
- Categorize changes: Added, Changed, Deprecated, Removed, Fixed
- Reference issue/PR numbers
- **DO NOT create new markdown files** - all changes go in CHANGELOG.md

---

## Dependency Management

### Adding New Dependencies

1. Add to `requirements.txt` with version constraint:
   ```
   new-package>=X.Y.Z
   ```

2. Add to `setup.py` install_requires list:
   ```python
   install_requires=[
       "torch>=2.0.0",
       "new-package>=X.Y.Z",  # Add here
   ]
   ```

3. Reinstall package:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

### Current Dependencies (DO NOT REMOVE)
```
torch>=2.0.0          # Core framework
numpy>=1.24.0         # Numerical operations
scipy>=1.10.0         # Scientific computing
pyyaml>=6.0          # Config file parsing
pandas>=2.0.0        # Data manipulation
scikit-learn>=1.3.0  # Metrics and preprocessing
matplotlib>=3.7.0    # Visualization (used in metrics.py)
```

---

## Performance Expectations

### Test Execution Times
- Full test suite: ~3 seconds
- Individual test modules: <1 second each
- **If tests take >10 seconds, something is wrong**

### Build/Install Times
- `pip install -r requirements.txt`: 30-60 seconds (first time)
- `pip install -e .`: ~5 seconds
- Subsequent installs: faster (cached)

### Memory Usage
- Tests run comfortably in 2GB RAM
- Training requires more (depends on batch size)
- No special memory considerations needed for tests

---

## Trust These Instructions

**When working on this repository:**

1. **ALWAYS start by installing dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

2. **ALWAYS run tests before making changes:**
   ```bash
   python -m unittest discover tests/
   ```

3. **ALWAYS run tests after making changes**

4. **Reference CHANGELOG.md for breaking changes** - especially v3.0.0 refactoring

5. **Use IMPLEMENTATION.md** for detailed technical information about each feature

6. **DON'T search for information already documented here** - trust these instructions first

7. **DON'T try to run example.py** - it has known import issues; use tests instead

8. **ALWAYS update CHANGELOG.md** for code changes (not new markdown files)

9. **For v3.0.0 codebase:** Attention, temporal encoding, and focal loss are ALWAYS enabled - don't add flags for these

10. **When in doubt about API usage:** Check `brism/__init__.py` for exported symbols and their source modules

---

## Quick Reference Commands

```bash
# Setup (run once)
pip install -r requirements.txt
pip install -e .

# Run all tests
python -m unittest discover tests/

# Run specific test module
python -m unittest tests.test_model

# Check imports
python -c "from brism import *"

# Check Python version
python --version  # Should be 3.8+

# Check PyTorch version
python -c "import torch; print(torch.__version__)"  # Should be 2.0+

# Check device availability
python -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')"
```

---

**Last Updated:** 2025-10-11 (v3.0.1)
