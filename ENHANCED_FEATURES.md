# Enhanced Medical Diagnostic Features

This document describes the new features added to BRISM for improved medical diagnosis.

## Overview

Four major enhancements have been added:

1. **Better Evaluation Metrics** - Comprehensive medical diagnostic metrics
2. **Temporal Modeling** - Support for symptom progression over time
3. **Uncertainty Calibration** - Temperature scaling for probability calibration
4. **Class Balancing** - Focal loss and class weights for rare diseases

---

## 1. Better Evaluation Metrics

### Features

#### Top-k Accuracy
Measures if the correct diagnosis appears in the top-k predictions - crucial for clinical decision support where a differential diagnosis list is provided.

```python
from brism.metrics import top_k_accuracy

# Check if correct diagnosis is in top-5
accuracy = top_k_accuracy(predictions, targets, k=5)
```

#### AUROC Per Class
Computes Area Under ROC Curve for each disease class independently (one-vs-rest).

```python
from brism.metrics import compute_auroc_per_class

auroc_scores = compute_auroc_per_class(predictions_np, targets_np)
mean_auroc = np.mean(list(auroc_scores.values()))
```

#### Calibration Metrics
Evaluates whether predicted probabilities match actual frequencies using Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

```python
from brism.metrics import compute_calibration_metrics, plot_reliability_diagram

cal_metrics = compute_calibration_metrics(predictions_np, targets_np, n_bins=10)
print(f"ECE: {cal_metrics['ece']:.4f}")

# Visualize calibration
fig = plot_reliability_diagram(cal_metrics, save_path='calibration.png')
```

#### Stratified Performance
Analyzes model performance separately for rare, medium, and common diseases.

```python
from brism.metrics import stratify_by_disease_frequency

stratified = stratify_by_disease_frequency(
    predictions_np, targets_np, class_counts
)

for group in ['rare', 'medium', 'common']:
    metrics = stratified[group]
    print(f"{group}: Acc={metrics['accuracy']:.3f}")
```

#### Comprehensive Evaluation
One function to compute all metrics at once:

```python
from brism import comprehensive_evaluation, print_evaluation_summary

results = comprehensive_evaluation(
    model, val_loader, device,
    class_counts=class_counts,
    n_bins=10,
    compute_auroc=True
)

print_evaluation_summary(results)
```

### Example Output

```
======================================================================
COMPREHENSIVE EVALUATION RESULTS
======================================================================

📊 Top-k Accuracies:
  Top-1:  0.0600
  Top-3:  0.1800
  Top-5:  0.2500
  Top-10: 0.4500

🎯 Mean AUROC: 0.4981
   (across 42 classes)

📈 Calibration Metrics:
  ECE (Expected Calibration Error): 0.0048
  MCE (Maximum Calibration Error):  0.0048

🔍 Performance by Disease Frequency:

  RARE diseases:
    Accuracy:       0.0000
    Top-5 Accuracy: 0.0000
    Avg Confidence: 0.0551
    Samples:        25
    Classes:        16

  MEDIUM diseases:
    Accuracy:       0.0000
    Top-5 Accuracy: 0.0000
    Avg Confidence: 0.0547
    Samples:        38
    Classes:        17

  COMMON diseases:
    Accuracy:       0.0876
    Top-5 Accuracy: 0.3650
    Avg Confidence: 0.0554
    Samples:        137
    Classes:        17
======================================================================
```

---

## 2. Temporal Modeling for Symptom Progression

### Overview

Instead of treating symptoms as a bag of tokens, temporal encoding captures **when** symptoms appear. This is critical for diseases where symptom progression matters (e.g., fever → cough → shortness of breath indicates different conditions than all appearing simultaneously).

### Configuration

```python
from brism import BRISMConfig, BRISM

config = BRISMConfig(
    symptom_vocab_size=1000,
    icd_vocab_size=500,
    use_temporal_encoding=True,
    temporal_encoding_type='positional'  # or 'timestamp'
)

model = BRISM(config)
```

### Encoding Types

#### Positional Encoding (Transformer-style)
Captures relative ordering of symptoms without absolute time:

```python
config.temporal_encoding_type = 'positional'
```

Best for: Sequential ordering matters but absolute time doesn't (symptom A before B before C).

#### Timestamp Encoding
Uses actual time values (e.g., hours since symptom onset):

```python
config.temporal_encoding_type = 'timestamp'
```

Best for: Absolute timing is important (symptom A at hour 1, B at hour 5).

### Data Format

When using timestamps:

```python
# In your dataset
def __getitem__(self, idx):
    return {
        'symptoms': torch.tensor([101, 205, 308]),  # Symptom IDs
        'timestamps': torch.tensor([1.0, 3.5, 7.2]),  # Hours since onset
        'icd_codes': torch.tensor(15)
    }
```

### Standalone Temporal Encoder

For custom architectures:

```python
from brism.temporal import TemporalSymptomEncoder

encoder = TemporalSymptomEncoder(
    vocab_size=1000,
    embed_dim=128,
    hidden_dim=256,
    latent_dim=64,
    encoding_type='timestamp',
    use_lstm=True  # or False for attention pooling
)

mu, logvar = encoder(symptom_ids, timestamps)
```

---

## 3. Uncertainty Calibration

### Overview

Temperature scaling adjusts model confidence to match actual accuracy. Without calibration, models may be overconfident or underconfident.

### Model Configuration

Temperature is automatically added to the model:

```python
config = BRISMConfig(
    symptom_vocab_size=1000,
    icd_vocab_size=500,
    temperature=1.0  # Initial temperature
)

model = BRISM(config)
```

### Calibration Process

**Step 1: Split data**
```python
train_loader = DataLoader(train_dataset, batch_size=32)
cal_loader = DataLoader(cal_dataset, batch_size=32)  # Held-out calibration set
test_loader = DataLoader(test_dataset, batch_size=32)
```

**Step 2: Train model normally**
```python
# Standard training
train_brism(model, train_loader, optimizer, loss_fn, num_epochs, device)
```

**Step 3: Calibrate temperature**
```python
from brism import calibrate_temperature

optimal_temp = calibrate_temperature(
    model, cal_loader, device,
    max_iter=50, lr=0.01
)
# Temperature updated in-place
```

**Step 4: Evaluate improvement**
```python
from brism import evaluate_calibration_improvement

results = evaluate_calibration_improvement(
    model, test_loader, device, n_bins=10
)

print(f"ECE before: {results['before_scaling']['ece']:.4f}")
print(f"ECE after:  {results['after_scaling']['ece']:.4f}")
print(f"Improvement: {results['ece_improvement']:.4f}")
```

### Reliability Diagrams

Visualize calibration quality:

```python
from brism.metrics import plot_reliability_diagram

fig = plot_reliability_diagram(
    results['after_scaling'],
    save_path='reliability_diagram.png',
    title='Model Calibration After Temperature Scaling'
)
```

A perfectly calibrated model has predictions that fall on the diagonal line.

---

## 4. Class Balancing for Rare Diseases

### Overview

Medical data is heavily imbalanced - common colds vs rare genetic disorders. Standard training leads to models that ignore rare diseases. Two solutions are provided:

1. **Class-weighted Cross-Entropy** - Weight loss by inverse class frequency
2. **Focal Loss** - Focus on hard-to-classify examples

### Computing Class Weights

```python
from brism import compute_class_weights

# Get class counts from training data
class_counts = {0: 1000, 1: 500, 2: 50, 3: 10, ...}

weights = compute_class_weights(
    class_counts, 
    num_classes=500,
    smoothing=1.0  # Avoid extreme weights
)

# weights[rare_class] > weights[common_class]
```

### Using Class Weights

```python
from brism import BRISMLoss

loss_fn = BRISMLoss(
    kl_weight=0.1,
    cycle_weight=1.0,
    class_weights=weights,  # Add class weights
    use_focal_loss=False
)
```

### Using Focal Loss

Focal loss automatically focuses on hard examples:

```python
loss_fn = BRISMLoss(
    kl_weight=0.1,
    cycle_weight=1.0,
    class_weights=weights,  # Optional with focal loss
    use_focal_loss=True,
    focal_gamma=2.0  # Higher = more focus on hard examples
)
```

**Focal Loss Formula:**
```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```

- `γ = 0`: Standard cross-entropy
- `γ = 2`: Default, good balance
- `γ > 2`: More aggressive focus on hard examples

### Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Class Weights** | Simple, interpretable | May overweight very rare classes | Moderate imbalance |
| **Focal Loss** | Focuses on hard examples | Adds hyperparameter (gamma) | Severe imbalance |
| **Both Combined** | Most powerful | Most complex | Production systems |

### Example

```python
from brism import BRISM, BRISMConfig, BRISMLoss, compute_class_weights

# Compute weights
class_counts = train_dataset.get_class_counts()
class_weights = compute_class_weights(class_counts, config.icd_vocab_size)

# Create loss with focal loss + class weights
loss_fn = BRISMLoss(
    kl_weight=0.1,
    cycle_weight=1.0,
    class_weights=class_weights,
    use_focal_loss=True,
    focal_gamma=2.0
)

# Train
train_brism(model, train_loader, optimizer, loss_fn, num_epochs, device)

# Evaluate on rare diseases
results = comprehensive_evaluation(model, val_loader, device, class_counts=class_counts)
print("Rare disease performance:", results['stratified_performance']['rare'])
```

---

## Complete Example

See `example_enhanced_features.py` for a full demonstration of all features.

```bash
python example_enhanced_features.py
```

---

## API Reference

### Metrics Module (`brism.metrics`)

- `top_k_accuracy(predictions, targets, k)` - Top-k accuracy
- `compute_auroc_per_class(predictions, targets)` - AUROC per class
- `compute_calibration_metrics(predictions, targets, n_bins)` - Calibration metrics
- `plot_reliability_diagram(metrics, save_path)` - Reliability diagram
- `stratify_by_disease_frequency(predictions, targets, class_counts)` - Stratified performance
- `comprehensive_evaluation(model, loader, device, ...)` - All metrics at once
- `print_evaluation_summary(results)` - Pretty print results

### Temporal Module (`brism.temporal`)

- `TemporalEncoding` - Adds temporal information to embeddings
- `TemporalSymptomEncoder` - Full encoder with temporal encoding

### Calibration Module (`brism.calibration`)

- `TemperatureScaling` - Temperature scaling layer
- `calibrate_temperature(model, cal_loader, device)` - Learn optimal temperature
- `evaluate_calibration_improvement(model, loader, device)` - Evaluate calibration

### Loss Module (`brism.loss`)

- `FocalLoss` - Focal loss for class imbalance
- `compute_class_weights(class_counts, num_classes)` - Compute inverse frequency weights
- `BRISMLoss` - Enhanced with class_weights and use_focal_loss parameters

---

## Migration Guide

All new features are **opt-in** and **backward compatible**.

### Before (still works)
```python
config = BRISMConfig(symptom_vocab_size=1000, icd_vocab_size=500)
model = BRISM(config)
loss_fn = BRISMLoss(kl_weight=0.1, cycle_weight=1.0)
```

### After (with enhancements)
```python
# Enable temporal encoding
config = BRISMConfig(
    symptom_vocab_size=1000,
    icd_vocab_size=500,
    use_temporal_encoding=True,
    temporal_encoding_type='positional'
)

# Add class balancing
class_weights = compute_class_weights(class_counts, config.icd_vocab_size)
loss_fn = BRISMLoss(
    kl_weight=0.1,
    cycle_weight=1.0,
    class_weights=class_weights,
    use_focal_loss=True,
    focal_gamma=2.0
)

# Model with temperature scaling
model = BRISM(config)
# ... train ...
calibrate_temperature(model, cal_loader, device)

# Comprehensive evaluation
results = comprehensive_evaluation(model, val_loader, device)
print_evaluation_summary(results)
```

---

## Performance Considerations

- **Top-k Accuracy**: Very fast, O(n log k) per sample
- **AUROC**: Can be slow for many classes, set `compute_auroc=False` if needed
- **Calibration**: ECE/MCE are fast, O(n) with binning
- **Temporal Encoding**: Minimal overhead (~2-5% slower)
- **Focal Loss**: Same speed as cross-entropy
- **Temperature Calibration**: Fast, typically converges in <50 iterations

---

## Citations

If you use these features in your research, please cite:

```bibtex
@software{brism2025_enhanced,
  title={BRISM: Bayesian Reciprocal ICD-Symptom Model with Enhanced Medical Diagnostics},
  author={Your Name},
  year={2025},
  url={https://github.com/LegitimatelyBatman/BRISM}
}
```

### Key References

- **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
- **Temperature Scaling**: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017
- **Positional Encoding**: Vaswani et al., "Attention Is All You Need", NeurIPS 2017
- **Top-k Accuracy**: Standard metric in information retrieval and recommendation systems
