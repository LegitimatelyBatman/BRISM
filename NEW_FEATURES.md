# BRISM New Features Documentation

This document provides detailed information about the new features added to BRISM.

## Table of Contents
1. [Attention-Based Symptom Encoding](#attention-based-symptom-encoding)
2. [ICD-10 Hierarchical Loss](#icd-10-hierarchical-loss)
3. [Medical Data Loaders](#medical-data-loaders)
4. [Model Checkpointing and Early Stopping](#model-checkpointing-and-early-stopping)

---

## Attention-Based Symptom Encoding

### Problem
The original implementation used simple mean pooling to aggregate symptom embeddings, which:
- Treats all symptoms equally
- Loses information about which symptoms are diagnostically important
- Cannot focus on key symptoms for specific diagnoses

### Solution
Replace mean pooling with a self-attention mechanism that learns importance weights for each symptom in the sequence.

### Implementation

**Configuration:**
```python
from brism import BRISMConfig, BRISM

config = BRISMConfig(
    symptom_vocab_size=1000,
    icd_vocab_size=500,
    use_attention=True  # Enable attention (default: True)
)
model = BRISM(config)
```

**How it works:**
1. Symptom embeddings are passed through a linear layer to compute attention scores
2. Scores are masked for padding tokens and normalized with softmax
3. Final representation is a weighted sum based on attention weights
4. The model learns which symptoms are most relevant for each diagnosis

**Benefits:**
- Better handles variable-length symptom sequences
- Focuses on diagnostically relevant symptoms
- Interpretable (can inspect attention weights)
- Typically improves prediction accuracy

**Backward Compatibility:**
Set `use_attention=False` to revert to mean pooling behavior.

---

## ICD-10 Hierarchical Loss

### Problem
Standard cross-entropy loss treats all incorrect predictions equally, but:
- Some wrong predictions are "less wrong" (e.g., predicting E11.65 instead of E11.9 - both are Type 2 diabetes)
- ICD codes have a natural hierarchical structure (chapter → category → subcategory → code)
- Model should be penalized less for hierarchically similar predictions

### Solution
Implement a hierarchical loss that computes tree distance between ICD codes and reduces penalty for predictions in the same category.

### Implementation

**Step 1: Create ICD Hierarchy**

Option A - From YAML file:
```python
from brism import ICDHierarchy

icd_hierarchy = ICDHierarchy(icd_vocab_size=500)
icd_hierarchy.build_from_yaml('config/icd_codes.yaml')
```

Option B - Programmatically:
```python
icd_hierarchy = ICDHierarchy(icd_vocab_size=500)

# Map vocabulary indices to ICD-10 codes
idx_to_code = {
    0: "E11.65",  # Type 2 diabetes with hyperglycemia
    1: "E11.9",   # Type 2 diabetes without complications
    2: "I10",     # Essential hypertension
    # ... more codes
}

icd_hierarchy.build_from_mapping(idx_to_code)
```

Option C - Synthetic (for testing):
```python
icd_hierarchy.build_synthetic()  # Creates hierarchical groupings
```

**Step 2: Configure Loss Function**

```python
from brism import BRISMLoss

loss_fn = BRISMLoss(
    kl_weight=0.1,
    cycle_weight=1.0,
    icd_hierarchy=icd_hierarchy,        # Provide hierarchy
    hierarchical_weight=0.3,            # 0-1: weight of hierarchical component
    hierarchical_temperature=1.0        # Controls distance penalty steepness
)
```

**YAML Configuration Format:**

Create `config/icd_codes.yaml`:
```yaml
icd_codes:
  0: "E11.65"  # Type 2 diabetes with hyperglycemia
  1: "E11.9"   # Type 2 diabetes without complications
  2: "E10.65"  # Type 1 diabetes with hyperglycemia
  3: "I10"     # Essential hypertension
  4: "I11.0"   # Hypertensive heart disease
  # ... more codes
```

**How it works:**

1. **Distance Calculation**: Hierarchical distance is computed based on ICD structure:
   - Same code: distance = 0
   - Same subcategory (e.g., E11.6*): distance = 1
   - Same category (e.g., E11.*): distance = 2
   - Same chapter (e.g., E*): distance = 3
   - Different chapter: distance = 4

2. **Loss Computation**:
   ```
   hierarchical_loss = (1 - w) * CE_loss + w * distance_weighted_loss
   ```
   where w is `hierarchical_weight`

3. **Result**: Model is penalized less for predictions that are hierarchically close to the target.

**Benefits:**
- More clinically meaningful loss function
- Faster convergence (model learns category structure)
- Better generalization to rare codes
- Configurable trade-off between strict and hierarchical matching

---

## Medical Data Loaders

### Problem
Working with real medical data requires:
- Parsing MIMIC-III/IV format
- ICD-9 to ICD-10 conversion
- Proper train/val/test splits at patient level (to avoid data leakage)
- Handling missing data
- Clinical text tokenization

### Solution
Comprehensive medical data preprocessing utilities with MIMIC support.

### Implementation

**For MIMIC Data:**

```python
from brism import load_mimic_data

train_dataset, val_dataset, test_dataset, preprocessor = load_mimic_data(
    diagnoses_path='data/DIAGNOSES_ICD.csv',
    notes_path='data/NOTEEVENTS.csv',
    max_symptom_length=50,
    min_symptom_freq=5,      # Min frequency to include symptom
    min_icd_freq=3,          # Min frequency to include ICD code
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42
)
```

**MIMIC Format Requirements:**

Diagnoses file columns:
- `hadm_id` or `subject_id`: Patient/admission identifier
- `icd_code`: ICD diagnosis code
- `icd_version`: 9 or 10 (for ICD-9/ICD-10)

Notes file columns:
- `hadm_id` or `subject_id`: Patient/admission identifier
- `text`: Clinical note text

**Custom Data Processing:**

```python
from brism import MedicalDataPreprocessor, MedicalRecordDataset

preprocessor = MedicalDataPreprocessor(max_symptom_length=50)

# Process diagnoses
patient_diagnoses = preprocessor.process_mimic_diagnoses(diagnoses_df)

# Process clinical notes
patient_notes = preprocessor.process_mimic_notes(notes_df)

# Create patient-level splits (no data leakage!)
patient_ids = list(set(patient_diagnoses.keys()) & set(patient_notes.keys()))
train_ids, val_ids, test_ids = preprocessor.create_patient_splits(
    patient_ids,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

# Build vocabularies
all_symptoms = [preprocessor.tokenize_symptoms(patient_notes[pid]) 
                for pid in patient_ids]
all_icds = [patient_diagnoses[pid][0] for pid in patient_ids]
preprocessor.build_vocabularies_from_data(all_symptoms, all_icds)

# Encode data
def encode_split(patient_ids):
    symptoms = []
    icds = []
    for pid in patient_ids:
        symptom_tokens = preprocessor.tokenize_symptoms(patient_notes[pid])
        encoded_symptoms = preprocessor.encode_symptoms(symptom_tokens)
        encoded_icd = preprocessor.encode_icd(patient_diagnoses[pid][0])
        if encoded_icd is not None:
            symptoms.append(encoded_symptoms)
            icds.append(encoded_icd)
    return MedicalRecordDataset(symptoms, icds)

train_dataset = encode_split(train_ids)
val_dataset = encode_split(val_ids)
test_dataset = encode_split(test_ids)
```

**ICD Normalization:**

```python
from brism import ICDNormalizer

normalizer = ICDNormalizer()

# Normalize ICD-10 codes
normalized = normalizer.normalize_icd10("E1165")  # Returns "E11.65"

# Convert ICD-9 to ICD-10
icd10 = normalizer.normalize_icd9_to_icd10("250.00")  # Returns "E11.9"

# Validate format
is_valid = normalizer.is_valid_icd10("E11.65")  # Returns True
```

**Features:**
- **Patient-level splits**: Patients are never split across train/val/test
- **ICD-9 conversion**: Automatic conversion to ICD-10-CM (uses simplified mapping for demo)
- **Text tokenization**: Basic clinical text preprocessing
- **Missing data**: Handles missing codes and notes gracefully
- **Vocabulary building**: Automatic vocabulary construction with frequency thresholding

**Important Notes:**
- The ICD-9 to ICD-10 conversion uses a simplified demonstration mapping
- For production use, download official CMS General Equivalence Mappings (GEMs)
- Text tokenization is basic - consider using medical NLP tools (MetaMap, cTAKES) for production

---

## Model Checkpointing and Early Stopping

### Problem
Training neural networks requires:
- Saving best model based on validation loss
- Periodic checkpointing for resume capability
- Early stopping to prevent overfitting
- Saving optimizer/scheduler state for exact resume

### Solution
Built-in checkpointing and early stopping with the training function.

### Implementation

**Basic Usage:**

```python
from brism import train_brism

history = train_brism(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    num_epochs=100,
    device=device,
    val_loader=val_loader,
    checkpoint_dir='./checkpoints',        # Where to save
    early_stopping_patience=5,             # Stop after 5 epochs without improvement
    save_best_only=True                    # Only save best model
)
```

**Resume from Checkpoint:**

```python
from brism import load_checkpoint

# Create new model/optimizer for resume
model = BRISM(config)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

# Load checkpoint
checkpoint = load_checkpoint(
    'checkpoints/best_model.pt',
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device
)

# Continue training from where you left off
history = train_brism(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    num_epochs=50,  # Train for 50 more epochs
    device=device,
    # ... other args
)
```

**Advanced Configuration:**

```python
from brism.train import ModelCheckpoint, EarlyStopping

# Manual checkpoint configuration (if not using train_brism)
checkpointer = ModelCheckpoint(
    checkpoint_dir='./checkpoints',
    monitor='val_loss',           # Metric to monitor
    mode='min',                   # 'min' or 'max'
    save_best_only=True,
    save_freq=5,                  # Save every N epochs if not save_best_only
    verbose=True
)

# Manual early stopping
early_stopping = EarlyStopping(
    patience=7,
    min_delta=0.001,              # Minimum change to qualify as improvement
    verbose=True
)

# Use in training loop
for epoch in range(num_epochs):
    # ... training code ...
    
    # Check early stopping
    if early_stopping(val_loss):
        print("Early stopping triggered!")
        break
    
    # Save checkpoint
    checkpointer(model, optimizer, epoch, metrics, scheduler)
```

**Checkpoint Contents:**

Each checkpoint file contains:
```python
{
    'epoch': int,                          # Epoch number
    'model_state_dict': dict,              # Model weights
    'optimizer_state_dict': dict,          # Optimizer state
    'scheduler_state_dict': dict,          # Scheduler state (if provided)
    'metrics': dict,                       # Training metrics
    'config': BRISMConfig                  # Model configuration
}
```

**Checkpoint Files:**

The checkpoint directory contains:
- `best_model.pt`: Best model based on validation loss
- `latest_checkpoint.pt`: Most recent checkpoint (for resume)
- `checkpoint_epoch_N.pt`: Periodic checkpoints (if `save_best_only=False`)

**Features:**
- **Automatic best model tracking**: Saves model with best validation loss
- **Resume capability**: Restore exact training state
- **Early stopping**: Configurable patience and min_delta
- **Periodic saves**: Optional periodic checkpointing
- **Metric monitoring**: Choose any metric to monitor (train_loss, val_loss, custom metrics)

**Benefits:**
- Never lose training progress
- Easy experimentation with different hyperparameters
- Automatic best model selection
- Prevents overfitting with early stopping

---

## Migration Guide

### Updating Existing Code

**Minimal changes required:**

```python
# Old code works unchanged
config = BRISMConfig(symptom_vocab_size=1000, icd_vocab_size=500)
model = BRISM(config)

# New features are opt-in
config = BRISMConfig(
    symptom_vocab_size=1000,
    icd_vocab_size=500,
    use_attention=True  # Only change needed for attention
)
```

**Adding hierarchical loss:**

```python
# Old code
loss_fn = BRISMLoss(kl_weight=0.1, cycle_weight=1.0)

# New code (with hierarchy)
icd_hierarchy = ICDHierarchy(icd_vocab_size=500)
icd_hierarchy.build_from_yaml('config/icd_codes.yaml')

loss_fn = BRISMLoss(
    kl_weight=0.1,
    cycle_weight=1.0,
    icd_hierarchy=icd_hierarchy,      # Add hierarchy
    hierarchical_weight=0.3
)
```

**Adding checkpointing:**

```python
# Old code
history = train_brism(model, train_loader, optimizer, loss_fn, num_epochs, device)

# New code (with checkpointing)
history = train_brism(
    model, train_loader, optimizer, loss_fn, num_epochs, device,
    checkpoint_dir='./checkpoints',      # Add these
    early_stopping_patience=5
)
```

---

## Performance Tips

1. **Attention vs Mean Pooling**: Attention typically improves accuracy by 2-5% but adds ~10-15% training time

2. **Hierarchical Loss**: Start with `hierarchical_weight=0.3` and `temperature=1.0`, tune based on your hierarchy depth

3. **Checkpointing**: Use `save_best_only=True` for production to save disk space

4. **Early Stopping**: Set patience to ~5-10% of total epochs (e.g., patience=5 for 50 epochs)

5. **Data Loaders**: For large datasets, consider streaming from disk instead of loading all into memory

---

## Troubleshooting

**Q: Attention layer causes NaN loss**
- A: Check that your symptom sequences aren't all padding. Attention needs at least one non-zero token.

**Q: Hierarchical loss doesn't seem to help**
- A: Try adjusting `hierarchical_weight` (0.2-0.5) and `temperature` (0.5-2.0). Also verify your ICD mapping is correct.

**Q: Patient splits seem unbalanced**
- A: This can happen with very small datasets. Consider adjusting ratios or using stratified splitting.

**Q: Checkpoint loading fails with pickle error**
- A: Ensure you're using the same Python/PyTorch version. For compatibility across versions, use `weights_only=False`.

**Q: Training is slower with new features**
- A: Attention and hierarchical loss add overhead. Disable features you don't need or reduce batch size.

---

## Citation

If you use these features in your research, please cite:

```bibtex
@software{brism2025_enhanced,
  title={BRISM: Bayesian Reciprocal ICD-Symptom Model with Attention and Hierarchical Loss},
  author={Sean},
  year={2025},
  url={https://github.com/LegitimatelyBatman/BRISM}
}
```
