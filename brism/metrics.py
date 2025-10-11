"""
Evaluation metrics for medical diagnosis.

Includes:
- Top-k accuracy
- AUROC per disease class
- Calibration metrics and reliability diagrams
- Stratified performance by disease frequency
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from collections import Counter


def top_k_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5
) -> float:
    """
    Compute top-k accuracy: is correct diagnosis in top-k predictions?
    
    Args:
        predictions: Model predictions [batch_size, num_classes] (logits or probabilities)
        targets: Ground truth labels [batch_size]
        k: Number of top predictions to consider
        
    Returns:
        Top-k accuracy as a float between 0 and 1
    """
    batch_size = targets.size(0)
    
    # Get top-k predictions
    _, top_k_preds = predictions.topk(k, dim=1, largest=True, sorted=True)
    
    # Check if target is in top-k
    targets_expanded = targets.unsqueeze(1).expand_as(top_k_preds)
    correct = (top_k_preds == targets_expanded).any(dim=1)
    
    accuracy = correct.float().mean().item()
    return accuracy


def compute_auroc_per_class(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: Optional[int] = None
) -> Dict[int, float]:
    """
    Compute AUROC for each disease class (one-vs-rest).
    
    Args:
        predictions: Model predictions [batch_size, num_classes] (probabilities)
        targets: Ground truth labels [batch_size]
        num_classes: Number of classes (auto-detected if None)
        
    Returns:
        Dictionary mapping class index to AUROC score
    """
    if num_classes is None:
        num_classes = predictions.shape[1]
    
    auroc_scores = {}
    
    for class_idx in range(num_classes):
        # Create binary labels for this class
        binary_targets = (targets == class_idx).astype(int)
        
        # Skip if class has no positive or negative samples
        if len(np.unique(binary_targets)) < 2:
            continue
        
        # Get predictions for this class
        class_predictions = predictions[:, class_idx]
        
        # Compute AUROC
        try:
            auroc = roc_auc_score(binary_targets, class_predictions)
            auroc_scores[class_idx] = auroc
        except ValueError:
            # Handle edge cases (e.g., all samples are one class)
            continue
    
    return auroc_scores


def compute_calibration_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute calibration metrics including Expected Calibration Error (ECE).
    
    Args:
        predictions: Model predictions [batch_size, num_classes] (probabilities)
        targets: Ground truth labels [batch_size]
        n_bins: Number of bins for calibration
        
    Returns:
        Dictionary containing:
            - ece: Expected Calibration Error
            - mce: Maximum Calibration Error
            - bin_accuracies: Accuracy per bin
            - bin_confidences: Average confidence per bin
            - bin_counts: Number of samples per bin
    """
    # Get predicted class and confidence
    confidences = np.max(predictions, axis=1)
    predicted_classes = np.argmax(predictions, axis=1)
    accuracies = (predicted_classes == targets).astype(float)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Initialize arrays for binned statistics
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    ece = 0.0
    mce = 0.0
    
    for bin_idx, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_count = np.sum(in_bin)
        
        if bin_count > 0:
            # Compute bin statistics
            bin_accuracy = np.mean(accuracies[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            
            bin_accuracies[bin_idx] = bin_accuracy
            bin_confidences[bin_idx] = bin_confidence
            bin_counts[bin_idx] = bin_count
            
            # Update ECE and MCE
            ece += (bin_count / len(predictions)) * np.abs(bin_accuracy - bin_confidence)
            mce = max(mce, np.abs(bin_accuracy - bin_confidence))
    
    return {
        'ece': ece,
        'mce': mce,
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts
    }


def plot_reliability_diagram(
    calibration_metrics: Dict,
    save_path: Optional[str] = None,
    title: str = "Reliability Diagram"
) -> plt.Figure:
    """
    Create a reliability diagram (calibration plot).
    
    Args:
        calibration_metrics: Output from compute_calibration_metrics
        save_path: Optional path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    bin_confidences = calibration_metrics['bin_confidences']
    bin_accuracies = calibration_metrics['bin_accuracies']
    bin_counts = calibration_metrics['bin_counts']
    
    # Filter out empty bins
    mask = bin_counts > 0
    bin_confidences = bin_confidences[mask]
    bin_accuracies = bin_accuracies[mask]
    bin_counts = bin_counts[mask]
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    
    # Plot actual calibration
    ax.plot(bin_confidences, bin_accuracies, 'o-', label='Model', 
            markersize=8, linewidth=2)
    
    # Add bin counts as bar widths
    bar_width = 0.02
    for conf, acc, count in zip(bin_confidences, bin_accuracies, bin_counts):
        ax.bar(conf, acc, width=bar_width, alpha=0.3, 
               color='blue', edgecolor='none')
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f"{title}\nECE: {calibration_metrics['ece']:.4f}, "
                f"MCE: {calibration_metrics['mce']:.4f}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def stratify_by_disease_frequency(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_counts: Optional[Dict[int, int]] = None
) -> Dict[str, Dict]:
    """
    Compute performance metrics stratified by disease frequency (rare vs common).
    
    Args:
        predictions: Model predictions [batch_size, num_classes] (probabilities)
        targets: Ground truth labels [batch_size]
        class_counts: Dictionary mapping class index to total count in training set
                     If None, uses counts from targets
        
    Returns:
        Dictionary with performance for rare, medium, and common diseases
    """
    # Compute class counts if not provided
    if class_counts is None:
        class_counts = dict(Counter(targets))
    
    # Sort classes by frequency
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
    
    # Split into tertiles (rare, medium, common)
    n_classes = len(sorted_classes)
    rare_classes = set([c for c, _ in sorted_classes[:n_classes // 3]])
    common_classes = set([c for c, _ in sorted_classes[2 * n_classes // 3:]])
    medium_classes = set([c for c, _ in sorted_classes[n_classes // 3:2 * n_classes // 3]])
    
    # Get predictions and targets
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Compute metrics for each group
    results = {}
    
    for group_name, group_classes in [
        ('rare', rare_classes),
        ('medium', medium_classes),
        ('common', common_classes)
    ]:
        # Filter samples belonging to this group
        mask = np.isin(targets, list(group_classes))
        
        if np.sum(mask) == 0:
            continue
        
        group_targets = targets[mask]
        group_predictions = predictions[mask]
        group_predicted = predicted_classes[mask]
        
        # Compute accuracy
        accuracy = np.mean(group_predicted == group_targets)
        
        # Compute top-k accuracy
        top_k_acc = top_k_accuracy(
            torch.tensor(group_predictions),
            torch.tensor(group_targets),
            k=5
        )
        
        # Compute average confidence
        avg_confidence = np.mean(np.max(group_predictions, axis=1))
        
        results[group_name] = {
            'accuracy': accuracy,
            'top_5_accuracy': top_k_acc,
            'avg_confidence': avg_confidence,
            'n_samples': np.sum(mask),
            'n_classes': len(group_classes)
        }
    
    return results


def comprehensive_evaluation(
    model,
    data_loader,
    device: torch.device,
    class_counts: Optional[Dict[int, int]] = None,
    n_bins: int = 10,
    compute_auroc: bool = True
) -> Dict:
    """
    Perform comprehensive evaluation with all metrics.
    
    Args:
        model: BRISM model
        data_loader: Data loader for evaluation
        device: Device to run on
        class_counts: Optional class counts for stratification
        n_bins: Number of bins for calibration
        compute_auroc: Whether to compute AUROC (can be slow for many classes)
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            symptoms = batch['symptoms'].to(device)
            icd_codes = batch['icd_codes'].to(device)
            
            # Get predictions
            icd_logits, _, _ = model.forward_path(symptoms)
            probs = F.softmax(icd_logits, dim=-1)
            
            all_predictions.append(probs.cpu())
            all_targets.append(icd_codes.cpu())
    
    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Convert to numpy for some metrics
    predictions_np = predictions.numpy()
    targets_np = targets.numpy()
    
    # Compute metrics
    results = {}
    
    # 1. Top-k accuracies
    results['top_1_accuracy'] = top_k_accuracy(predictions, targets, k=1)
    results['top_3_accuracy'] = top_k_accuracy(predictions, targets, k=3)
    results['top_5_accuracy'] = top_k_accuracy(predictions, targets, k=5)
    results['top_10_accuracy'] = top_k_accuracy(predictions, targets, k=10)
    
    # 2. AUROC per class
    if compute_auroc:
        auroc_scores = compute_auroc_per_class(predictions_np, targets_np)
        results['auroc_per_class'] = auroc_scores
        results['mean_auroc'] = np.mean(list(auroc_scores.values())) if auroc_scores else 0.0
    
    # 3. Calibration metrics
    calibration_metrics = compute_calibration_metrics(predictions_np, targets_np, n_bins)
    results['calibration'] = calibration_metrics
    results['ece'] = calibration_metrics['ece']
    results['mce'] = calibration_metrics['mce']
    
    # 4. Stratified performance
    stratified_metrics = stratify_by_disease_frequency(
        predictions_np, targets_np, class_counts
    )
    results['stratified_performance'] = stratified_metrics
    
    return results


def print_evaluation_summary(results: Dict):
    """
    Print a formatted summary of evaluation results.
    
    Args:
        results: Output from comprehensive_evaluation
    """
    print("=" * 70)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 70)
    
    # Top-k accuracies
    print("\n📊 Top-k Accuracies:")
    print(f"  Top-1:  {results['top_1_accuracy']:.4f}")
    print(f"  Top-3:  {results['top_3_accuracy']:.4f}")
    print(f"  Top-5:  {results['top_5_accuracy']:.4f}")
    print(f"  Top-10: {results['top_10_accuracy']:.4f}")
    
    # AUROC
    if 'mean_auroc' in results:
        print(f"\n🎯 Mean AUROC: {results['mean_auroc']:.4f}")
        print(f"   (across {len(results.get('auroc_per_class', {}))} classes)")
    
    # Calibration
    print(f"\n📈 Calibration Metrics:")
    print(f"  ECE (Expected Calibration Error): {results['ece']:.4f}")
    print(f"  MCE (Maximum Calibration Error):  {results['mce']:.4f}")
    
    # Stratified performance
    if 'stratified_performance' in results:
        print("\n🔍 Performance by Disease Frequency:")
        for group_name, metrics in results['stratified_performance'].items():
            print(f"\n  {group_name.upper()} diseases:")
            print(f"    Accuracy:       {metrics['accuracy']:.4f}")
            print(f"    Top-5 Accuracy: {metrics['top_5_accuracy']:.4f}")
            print(f"    Avg Confidence: {metrics['avg_confidence']:.4f}")
            print(f"    Samples:        {metrics['n_samples']}")
            print(f"    Classes:        {metrics['n_classes']}")
    
    print("\n" + "=" * 70)
