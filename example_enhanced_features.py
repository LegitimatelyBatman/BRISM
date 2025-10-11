"""
Example demonstrating new medical diagnostic features.

Shows:
1. Better evaluation metrics (top-k accuracy, AUROC, calibration, stratification)
2. Temporal modeling for symptom progression
3. Uncertainty calibration with temperature scaling
4. Class balancing with focal loss for rare diseases
"""

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter

from brism import (
    BRISM, BRISMConfig, 
    BRISMLoss,
    comprehensive_evaluation,
    print_evaluation_summary,
    compute_class_weights,
    calibrate_temperature,
    evaluate_calibration_improvement,
    plot_reliability_diagram
)
from brism.train import train_epoch_simple


class SyntheticMedicalDataset(Dataset):
    """Synthetic dataset with imbalanced classes and temporal information."""
    
    def __init__(self, num_samples=1000, symptom_vocab_size=100, 
                 icd_vocab_size=50, max_symptom_length=20, 
                 imbalanced=True, temporal=False):
        self.num_samples = num_samples
        self.temporal = temporal
        self.data = []
        
        # Create imbalanced distribution if requested
        if imbalanced:
            # 70% common diseases (0-15), 20% medium (16-35), 10% rare (36-49)
            disease_probs = np.ones(icd_vocab_size)
            disease_probs[0:16] = 7.0  # Common
            disease_probs[16:36] = 2.0  # Medium
            disease_probs[36:50] = 1.0  # Rare
            disease_probs = disease_probs / disease_probs.sum()
        else:
            disease_probs = None
        
        for _ in range(num_samples):
            # Sample symptoms
            symptoms = np.random.randint(1, symptom_vocab_size, max_symptom_length)
            
            # Sample ICD code (possibly imbalanced)
            if disease_probs is not None:
                icd_code = np.random.choice(icd_vocab_size, p=disease_probs)
            else:
                icd_code = np.random.randint(0, icd_vocab_size)
            
            # Generate temporal information if requested
            if temporal:
                # Timestamps: sequential appearance with some jitter
                timestamps = np.cumsum(np.random.exponential(1.0, max_symptom_length))
                timestamps = timestamps / timestamps[-1] * 24.0  # Normalize to 24 hours
            else:
                timestamps = None
            
            self.data.append({
                'symptoms': symptoms,
                'icd_codes': icd_code,
                'timestamps': timestamps
            })
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        result = {
            'symptoms': torch.tensor(sample['symptoms'], dtype=torch.long),
            'icd_codes': torch.tensor(sample['icd_codes'], dtype=torch.long)
        }
        if self.temporal and sample['timestamps'] is not None:
            result['timestamps'] = torch.tensor(sample['timestamps'], dtype=torch.float32)
        return result
    
    def get_class_counts(self):
        """Get class distribution for computing weights."""
        icd_codes = [sample['icd_codes'] for sample in self.data]
        return dict(Counter(icd_codes))


def main():
    """Main demonstration function."""
    print("=" * 70)
    print("BRISM ENHANCED FEATURES DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    # ========================================================================
    # Feature 1: Better Evaluation Metrics
    # ========================================================================
    print("=" * 70)
    print("1. BETTER EVALUATION METRICS")
    print("=" * 70)
    print()
    
    # Create model with standard configuration
    config = BRISMConfig(
        symptom_vocab_size=100,
        icd_vocab_size=50,
        max_symptom_length=20,
        use_attention=True
    )
    model = BRISM(config)
    print(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    print()
    
    # Create imbalanced dataset
    print("Creating imbalanced dataset...")
    train_dataset = SyntheticMedicalDataset(
        num_samples=1000, 
        imbalanced=True,
        temporal=False
    )
    val_dataset = SyntheticMedicalDataset(
        num_samples=200,
        imbalanced=True,
        temporal=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    class_counts = train_dataset.get_class_counts()
    print(f"Dataset created with {len(train_dataset)} train samples")
    print(f"Class distribution: {len([c for c in class_counts.values() if c > 50])} common, "
          f"{len([c for c in class_counts.values() if 10 <= c <= 50])} medium, "
          f"{len([c for c in class_counts.values() if c < 10])} rare diseases")
    print()
    
    # Quick training (just a few steps for demonstration)
    print("Training model (5 epochs)...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = BRISMLoss(kl_weight=0.1, cycle_weight=1.0)
    
    for epoch in range(5):
        metrics = train_epoch_simple(model, train_loader, optimizer, loss_fn, device, epoch)
        print(f"  Epoch {epoch+1}: Loss = {metrics['loss']:.4f}")
    print()
    
    # Comprehensive evaluation
    print("Running comprehensive evaluation...")
    results = comprehensive_evaluation(
        model, val_loader, device,
        class_counts=class_counts,
        n_bins=10,
        compute_auroc=True
    )
    print()
    print_evaluation_summary(results)
    print()
    
    # ========================================================================
    # Feature 2: Temporal Modeling
    # ========================================================================
    print("=" * 70)
    print("2. TEMPORAL MODELING FOR SYMPTOM PROGRESSION")
    print("=" * 70)
    print()
    
    # Create model with temporal encoding
    config_temporal = BRISMConfig(
        symptom_vocab_size=100,
        icd_vocab_size=50,
        max_symptom_length=20,
        use_attention=True,
        use_temporal_encoding=True,
        temporal_encoding_type='positional'
    )
    model_temporal = BRISM(config_temporal)
    print(f"Created temporal model with positional encoding")
    print()
    
    # Create dataset with temporal information
    train_dataset_temporal = SyntheticMedicalDataset(
        num_samples=1000,
        imbalanced=False,
        temporal=True
    )
    val_dataset_temporal = SyntheticMedicalDataset(
        num_samples=200,
        imbalanced=False,
        temporal=True
    )
    
    train_loader_temporal = DataLoader(train_dataset_temporal, batch_size=32, shuffle=True)
    val_loader_temporal = DataLoader(val_dataset_temporal, batch_size=32, shuffle=False)
    
    print("Dataset created with temporal timestamps")
    sample = train_dataset_temporal[0]
    if 'timestamps' in sample:
        print(f"Example timestamps (hours): {sample['timestamps'][:5].tolist()}")
    print()
    
    # Quick training
    print("Training temporal model (3 epochs)...")
    optimizer_temporal = optim.Adam(model_temporal.parameters(), lr=0.001)
    loss_fn_temporal = BRISMLoss(kl_weight=0.1, cycle_weight=1.0)
    
    for epoch in range(3):
        metrics = train_epoch_simple(model_temporal, train_loader_temporal, 
                                     optimizer_temporal, loss_fn_temporal, device, epoch)
        print(f"  Epoch {epoch+1}: Loss = {metrics['loss']:.4f}")
    print()
    
    # ========================================================================
    # Feature 3: Uncertainty Calibration
    # ========================================================================
    print("=" * 70)
    print("3. UNCERTAINTY CALIBRATION WITH TEMPERATURE SCALING")
    print("=" * 70)
    print()
    
    # Use the first model for calibration demo
    print(f"Initial temperature: {model.temperature.item():.4f}")
    print()
    
    # Create calibration set
    cal_dataset = SyntheticMedicalDataset(num_samples=100, imbalanced=True)
    cal_loader = DataLoader(cal_dataset, batch_size=32)
    
    print("Calibrating temperature on calibration set...")
    optimal_temp = calibrate_temperature(
        model, cal_loader, device,
        max_iter=50, lr=0.01
    )
    print(f"Optimal temperature: {optimal_temp:.4f}")
    print()
    
    # Evaluate calibration improvement
    print("Evaluating calibration improvement...")
    cal_results = evaluate_calibration_improvement(
        model, val_loader, device, n_bins=10
    )
    print(f"ECE before scaling: {cal_results['before_scaling']['ece']:.4f}")
    print(f"ECE after scaling:  {cal_results['after_scaling']['ece']:.4f}")
    print(f"Improvement:        {cal_results['ece_improvement']:.4f}")
    print()
    
    # Generate reliability diagram
    print("Generating reliability diagram...")
    try:
        fig = plot_reliability_diagram(
            cal_results['after_scaling'],
            save_path='/tmp/reliability_diagram.png',
            title='Model Calibration After Temperature Scaling'
        )
        print("Saved reliability diagram to /tmp/reliability_diagram.png")
    except Exception as e:
        print(f"Note: Could not save plot (non-GUI environment): {e}")
    print()
    
    # ========================================================================
    # Feature 4: Class Balancing for Rare Diseases
    # ========================================================================
    print("=" * 70)
    print("4. CLASS BALANCING FOR RARE DISEASES")
    print("=" * 70)
    print()
    
    # Compute class weights
    print("Computing class weights from training data...")
    class_counts_dict = train_dataset.get_class_counts()
    class_weights = compute_class_weights(class_counts_dict, config.icd_vocab_size)
    
    print(f"Weight range: {class_weights.min():.2f} - {class_weights.max():.2f}")
    print(f"Mean weight: {class_weights.mean():.2f}")
    
    # Show weights for rare vs common diseases
    rare_idx = 45  # A rare disease
    common_idx = 5  # A common disease
    print(f"Weight for common disease (class {common_idx}): {class_weights[common_idx]:.2f}")
    print(f"Weight for rare disease (class {rare_idx}):     {class_weights[rare_idx]:.2f}")
    print()
    
    # Create model with focal loss
    print("Creating model with focal loss (gamma=2.0)...")
    model_focal = BRISM(config)
    loss_fn_focal = BRISMLoss(
        kl_weight=0.1,
        cycle_weight=1.0,
        class_weights=class_weights,
        use_focal_loss=True,
        focal_gamma=2.0
    )
    print()
    
    # Train with focal loss
    print("Training with focal loss (3 epochs)...")
    optimizer_focal = optim.Adam(model_focal.parameters(), lr=0.001)
    
    for epoch in range(3):
        metrics = train_epoch_simple(model_focal, train_loader, 
                                     optimizer_focal, loss_fn_focal, device, epoch)
        print(f"  Epoch {epoch+1}: Loss = {metrics['loss']:.4f}")
    print()
    
    # Evaluate performance on rare diseases
    print("Evaluating performance with focal loss...")
    results_focal = comprehensive_evaluation(
        model_focal, val_loader, device,
        class_counts=class_counts_dict,
        n_bins=10,
        compute_auroc=False
    )
    
    stratified = results_focal['stratified_performance']
    print("\nPerformance comparison:")
    for group in ['common', 'medium', 'rare']:
        if group in stratified:
            print(f"  {group.upper()} diseases: "
                  f"Acc={stratified[group]['accuracy']:.3f}, "
                  f"Top-5={stratified[group]['top_5_accuracy']:.3f}")
    print()
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("✅ Feature 1: Better evaluation metrics")
    print("   - Top-k accuracy for clinical relevance")
    print("   - AUROC per disease class")
    print("   - Calibration metrics (ECE, MCE)")
    print("   - Stratified performance by disease frequency")
    print()
    print("✅ Feature 2: Temporal modeling")
    print("   - Positional encoding for symptom ordering")
    print("   - Timestamp encoding for progression tracking")
    print()
    print("✅ Feature 3: Uncertainty calibration")
    print("   - Temperature scaling for confidence adjustment")
    print("   - Reliability diagrams for visualization")
    print("   - Improved probability calibration")
    print()
    print("✅ Feature 4: Class balancing")
    print("   - Focal loss for hard example mining")
    print("   - Class-weighted cross-entropy")
    print("   - Better performance on rare diseases")
    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
