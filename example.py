"""
Comprehensive example demonstrating all BRISM v3.0.0 features.

This is the main example file for BRISM v3.0.0, combining demonstrations of:

SECTION 1: Basic Usage
  1.1 Creating synthetic data
  1.2 Model initialization and training
  1.3 Basic predictions with confidence intervals
  1.4 Generating symptom sequences from ICD codes

SECTION 2: Interpretability Tools
  2.1 Attention visualization
  2.2 Integrated gradients for feature attribution
  2.3 Counterfactual explanations
  2.4 Comprehensive explanations

SECTION 3: Advanced Generation
  3.1 Beam search for diverse symptom generation
  3.2 Ensemble uncertainty quantification

SECTION 4: Clinical Features
  4.1 Symptom synonym handling
  4.2 Active learning interface
  4.3 Better evaluation metrics
  4.4 Temporal modeling for symptom progression
  4.5 Uncertainty calibration with temperature scaling
  4.6 Class balancing with focal loss for rare diseases

All features shown here are now standard in v3.0.0 (no optional flags needed).
"""

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter

from brism import (
    BRISM, BRISMConfig, train_brism, diagnose_with_confidence,
    BRISMLoss,
    # Interpretability
    IntegratedGradients, AttentionVisualization, CounterfactualExplanations,
    AttentionRollout, explain_prediction,
    # Generation
    generate_symptoms_beam_search,
    # Ensemble
    BRISMEnsemble, train_ensemble,
    # Synonym Handling
    SymptomNormalizer, create_default_medical_synonyms,
    build_symptom_normalizer_from_vocab,
    # Active Learning
    ActiveLearner,
    # Evaluation
    comprehensive_evaluation, print_evaluation_summary, compute_class_weights,
    # Calibration
    calibrate_temperature, evaluate_calibration_improvement, plot_reliability_diagram
)
from brism.inference import generate_symptoms_with_uncertainty
from brism.train import train_epoch_simple


class SyntheticMedicalDataset(Dataset):
    """
    Comprehensive synthetic dataset for demonstration.
    
    Supports:
    - Basic symptom-ICD pairs
    - Imbalanced class distributions
    - Temporal information (symptom progression)
    """
    
    def __init__(self, num_samples: int = 1000, symptom_vocab_size: int = 1000,
                 icd_vocab_size: int = 500, max_symptom_length: int = 50,
                 imbalanced: bool = False, temporal: bool = False):
        self.num_samples = num_samples
        self.symptom_vocab_size = symptom_vocab_size
        self.icd_vocab_size = icd_vocab_size
        self.max_symptom_length = max_symptom_length
        self.temporal = temporal
        
        # Generate synthetic data
        self.data = self._generate_data(imbalanced)
    
    def _generate_data(self, imbalanced: bool = False):
        """Generate synthetic symptom-ICD pairs."""
        data = []
        
        # Create imbalanced distribution if requested
        if imbalanced:
            # 70% common diseases, 20% medium, 10% rare
            common_size = int(self.icd_vocab_size * 0.3)
            medium_size = int(self.icd_vocab_size * 0.4)
            
            disease_probs = np.ones(self.icd_vocab_size)
            disease_probs[:common_size] = 7.0  # Common
            disease_probs[common_size:common_size+medium_size] = 2.0  # Medium
            disease_probs[common_size+medium_size:] = 1.0  # Rare
            disease_probs = disease_probs / disease_probs.sum()
        else:
            disease_probs = None
        
        for _ in range(self.num_samples):
            # Random ICD code (possibly imbalanced)
            if disease_probs is not None:
                icd_code = np.random.choice(self.icd_vocab_size, p=disease_probs)
            else:
                icd_code = np.random.randint(0, self.icd_vocab_size)
            
            # Generate symptom sequence (5-20 symptoms)
            seq_len = np.random.randint(5, 21)
            symptoms = np.zeros(self.max_symptom_length, dtype=np.int64)
            
            # Add some correlated symptoms
            base_symptom = (icd_code * 2) % self.symptom_vocab_size
            for i in range(min(seq_len, 5)):
                symptoms[i] = (base_symptom + i) % self.symptom_vocab_size
            
            # Add random symptoms
            for i in range(5, seq_len):
                symptoms[i] = np.random.randint(1, self.symptom_vocab_size)
            
            # Generate temporal information if requested
            if self.temporal:
                timestamps = np.cumsum(np.random.exponential(1.0, self.max_symptom_length))
                timestamps = timestamps / timestamps[-1] * 24.0  # Normalize to 24 hours
            else:
                timestamps = None
            
            data.append({
                'symptoms': symptoms,
                'icd_codes': icd_code,
                'seq_len': seq_len,
                'timestamps': timestamps
            })
        
        return data
    
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
    """Main example function demonstrating all BRISM v3.0.0 features."""
    
    print("=" * 80)
    print("BRISM v3.0.0: Comprehensive Feature Demonstration")
    print("=" * 80)
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    # ========================================================================
    # SECTION 1: BASIC USAGE
    # ========================================================================
    print("=" * 80)
    print("SECTION 1: BASIC USAGE")
    print("=" * 80)
    print()
    
    # Configuration
    config = BRISMConfig(
        symptom_vocab_size=1000,
        icd_vocab_size=500,
        symptom_embed_dim=128,
        icd_embed_dim=128,
        encoder_hidden_dim=256,
        latent_dim=64,
        decoder_hidden_dim=256,
        max_symptom_length=50,
        dropout_rate=0.2,
        mc_samples=20
    )
    
    print("Configuration:")
    print(f"  Symptom vocabulary size: {config.symptom_vocab_size}")
    print(f"  ICD vocabulary size: {config.icd_vocab_size}")
    print(f"  Latent dimension: {config.latent_dim}")
    print(f"  MC samples for uncertainty: {config.mc_samples}")
    print()
    
    # Create model
    model = BRISM(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print()
    
    # Create synthetic dataset
    print("Creating synthetic dataset...")
    train_dataset = SyntheticMedicalDataset(num_samples=1000, 
                                           symptom_vocab_size=config.symptom_vocab_size,
                                           icd_vocab_size=config.icd_vocab_size,
                                           max_symptom_length=config.max_symptom_length)
    val_dataset = SyntheticMedicalDataset(num_samples=200,
                                         symptom_vocab_size=config.symptom_vocab_size,
                                         icd_vocab_size=config.icd_vocab_size,
                                         max_symptom_length=config.max_symptom_length)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print()
    
    # Train for a few epochs (shortened for demo)
    print("1.1 Training Model (3 epochs for demo)")
    print("-" * 80)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = BRISMLoss(
        kl_weight=0.1, 
        cycle_weight=1.0,
        contrastive_weight=0.5,  # Contrastive learning enabled by default
        hierarchical_weight=0.3  # Hierarchical loss enabled by default
    )
    
    history = train_brism(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=3,
        device=device,
        val_loader=val_loader,
        log_interval=50
    )
    
    print("Training completed!")
    print()
    
    # 1.2 Basic Diagnosis
    print("1.2 Basic Diagnosis with Confidence Intervals")
    print("-" * 80)
    
    sample = val_dataset[0]
    symptoms = sample['symptoms'].unsqueeze(0)
    true_icd = sample['icd_codes'].item()
    
    print(f"Input symptoms (first 10 non-zero): {symptoms[0][:10].tolist()}")
    print(f"True ICD code: {true_icd}")
    print()
    
    diagnosis = diagnose_with_confidence(
        model=model,
        symptoms=symptoms,
        device=device,
        confidence_level=0.95,
        top_k=5
    )
    
    print("Top 5 predicted ICD codes with confidence intervals:")
    for i, pred in enumerate(diagnosis['predictions'], 1):
        print(f"  {i}. ICD {pred['icd_code']}: "
              f"{pred['probability']:.4f} ± {pred['std']:.4f} "
              f"(95% CI: [{pred['confidence_interval'][0]:.4f}, "
              f"{pred['confidence_interval'][1]:.4f}])")
    print()
    
    # 1.3 Generate symptoms from ICD
    print("1.3 Generate Symptom Sequences from ICD Code")
    print("-" * 80)
    
    icd_code = torch.tensor(true_icd)
    generated = generate_symptoms_with_uncertainty(
        model=model,
        icd_code=icd_code,
        device=device,
        n_samples=20
    )
    
    print(f"Generated symptom sequence (mode): {generated['mode_sequence'][:15]}")
    print(f"Sequence diversity: {generated['diversity']:.4f}")
    print(f"Unique sequences: {generated['n_unique_sequences']}/{20}")
    print()
    
    # ========================================================================
    # SECTION 2: INTERPRETABILITY TOOLS
    # ========================================================================
    print("=" * 80)
    print("SECTION 2: INTERPRETABILITY TOOLS")
    print("=" * 80)
    print()
    
    test_symptoms = val_dataset[0]['symptoms'].to(device)
    test_icd = val_dataset[0]['icd_codes'].item()
    
    # 2.1 Attention Visualization
    print("2.1 Attention Visualization")
    print("-" * 80)
    vis = AttentionVisualization(model)
    attention_weights, predictions = vis.get_attention_weights(test_symptoms)
    
    print(f"Attention weights (first 8): {attention_weights[:8].cpu().numpy()}")
    print(f"Top prediction: ICD {predictions.argmax().item()} "
          f"(prob: {predictions.max().item():.4f})")
    print()
    
    # 2.2 Integrated Gradients
    print("2.2 Integrated Gradients for Feature Attribution")
    print("-" * 80)
    ig = IntegratedGradients(model)
    attributions = ig.attribute(test_symptoms, n_steps=20)
    
    top_symptoms = attributions.abs().argsort(descending=True)[:3]
    print(f"Most important symptoms (indices): {top_symptoms.cpu().numpy().tolist()}")
    print(f"Attribution scores: {attributions[:8].detach().cpu().numpy()}")
    print()
    
    # 2.3 Counterfactual Explanations
    print("2.3 Counterfactual Explanations")
    print("-" * 80)
    cf = CounterfactualExplanations(model)
    explanations = cf.explain_by_removal(test_symptoms)
    
    if len(explanations) > 0:
        print("Impact of removing each symptom:")
        for exp in explanations[:3]:
            print(f"  Symptom {exp['symptom_id']}: "
                  f"{exp['probability_drop_percentage']:.2f}% probability drop")
    print()
    
    # 2.4 Comprehensive Explanation
    print("2.4 Comprehensive Explanation")
    print("-" * 80)
    full_explanation = explain_prediction(
        model, test_symptoms, top_k=3, method='all'
    )
    
    print("Predictions:")
    for pred in full_explanation['predictions'][:3]:
        print(f"  ICD {pred['icd_code']}: {pred['probability']:.4f}")
    
    if 'top_attended_symptoms' in full_explanation:
        print("\nTop attended symptoms:")
        for symp in full_explanation['top_attended_symptoms'][:3]:
            print(f"  Position {symp['position']}: "
                  f"weight={symp['attention_weight']:.4f}")
    print()
    
    # ========================================================================
    # SECTION 3: ADVANCED GENERATION
    # ========================================================================
    print("=" * 80)
    print("SECTION 3: ADVANCED GENERATION")
    print("=" * 80)
    print()
    
    # 3.1 Beam Search
    print("3.1 Beam Search for Diverse Symptom Generation")
    print("-" * 80)
    
    test_icd_tensor = torch.tensor([test_icd], device=device)
    beam_results = generate_symptoms_beam_search(
        model, test_icd_tensor, device,
        beam_width=5, temperature=1.0, return_all_beams=True
    )
    
    print(f"Generated {len(beam_results['beams'])} symptom sequences:")
    for i, beam in enumerate(beam_results['beams'][:3], 1):
        symptoms_list = [s for s in beam['sequence'] if s != 0][:5]
        print(f"  Beam {i} (score={beam['score']:.4f}): {symptoms_list}")
    print()
    
    # 3.2 Ensemble Uncertainty
    print("3.2 Ensemble Uncertainty Quantification")
    print("-" * 80)
    
    ensemble = BRISMEnsemble(
        models=[model],
        use_pseudo_ensemble=True,
        n_models=5
    )
    
    ensemble_result = ensemble.diagnose_with_ensemble(
        test_symptoms, top_k=3, n_samples=5
    )
    
    print("Ensemble Predictions:")
    for pred in ensemble_result['predictions']:
        print(f"  ICD {pred['icd_code']}: {pred['probability']:.4f} ± "
              f"{pred['std']:.4f}")
    
    print("\nUncertainty Decomposition:")
    print(f"  Epistemic (model uncertainty): "
          f"{ensemble_result['uncertainty']['epistemic']:.4f}")
    print(f"  Aleatoric (data uncertainty): "
          f"{ensemble_result['uncertainty']['aleatoric']:.4f}")
    print(f"  Ensemble agreement: {ensemble_result['ensemble_agreement']:.2%}")
    print()
    
    # ========================================================================
    # SECTION 4: CLINICAL FEATURES
    # ========================================================================
    print("=" * 80)
    print("SECTION 4: CLINICAL FEATURES")
    print("=" * 80)
    print()
    
    # 4.1 Symptom Synonym Handling
    print("4.1 Symptom Synonym Handling")
    print("-" * 80)
    
    normalizer = SymptomNormalizer()
    synonym_dict = create_default_medical_synonyms()
    normalizer.build_from_umls(synonym_dict)
    
    test_symptoms_text = ["SOB", "dyspnea", "CP", "chest pain", "HA"]
    print("Symptom normalization examples:")
    for symptom in test_symptoms_text:
        normalized = normalizer.normalize(symptom)
        print(f"  '{symptom}' -> '{normalized}'")
    print()
    
    # 4.2 Active Learning
    print("4.2 Active Learning Interface")
    print("-" * 80)
    
    learner = ActiveLearner(model)
    partial_symptoms = test_symptoms.clone()
    partial_symptoms[5:] = 0  # Remove half
    
    recommendations = learner.query_next_symptom(
        partial_symptoms, method='bald', top_k=3, n_samples=10
    )
    
    print("Most informative symptoms to query (BALD method):")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. Symptom {rec['symptom_id']}: score={rec['score']:.4f}")
    print()
    
    # 4.3 Better Evaluation Metrics
    print("4.3 Better Evaluation Metrics")
    print("-" * 80)
    
    # Create imbalanced dataset for evaluation
    eval_dataset = SyntheticMedicalDataset(
        num_samples=200,
        symptom_vocab_size=100,
        icd_vocab_size=50,
        max_symptom_length=20,
        imbalanced=True
    )
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
    
    # Create smaller model for evaluation demo
    eval_config = BRISMConfig(
        symptom_vocab_size=100,
        icd_vocab_size=50,
        max_symptom_length=20
    )
    eval_model = BRISM(eval_config)
    
    # Quick training
    eval_optimizer = optim.Adam(eval_model.parameters(), lr=0.001)
    eval_loss = BRISMLoss(kl_weight=0.1, cycle_weight=1.0)
    for _ in range(2):
        train_epoch_simple(eval_model, eval_loader, eval_optimizer, 
                          eval_loss, device, 0)
    
    # Comprehensive evaluation
    class_counts = eval_dataset.get_class_counts()
    results = comprehensive_evaluation(
        eval_model, eval_loader, device,
        class_counts=class_counts,
        n_bins=10,
        compute_auroc=False  # Skip for speed
    )
    
    print_evaluation_summary(results)
    print()
    
    # 4.4 Temporal Modeling
    print("4.4 Temporal Modeling for Symptom Progression")
    print("-" * 80)
    
    temporal_config = BRISMConfig(
        symptom_vocab_size=100,
        icd_vocab_size=50,
        max_symptom_length=20,
        temporal_encoding_type='positional'  # Always enabled
    )
    temporal_model = BRISM(temporal_config)
    
    temporal_dataset = SyntheticMedicalDataset(
        num_samples=200, symptom_vocab_size=100, icd_vocab_size=50,
        max_symptom_length=20, temporal=True
    )
    
    sample_temporal = temporal_dataset[0]
    if 'timestamps' in sample_temporal:
        print(f"Example timestamps (hours): "
              f"{sample_temporal['timestamps'][:5].tolist()}")
    print(f"Temporal encoding enabled: positional")
    print()
    
    # 4.5 Uncertainty Calibration
    print("4.5 Uncertainty Calibration with Temperature Scaling")
    print("-" * 80)
    
    print(f"Initial temperature: {eval_model.temperature.item():.4f}")
    
    cal_dataset = SyntheticMedicalDataset(
        num_samples=50, symptom_vocab_size=100, icd_vocab_size=50,
        max_symptom_length=20, imbalanced=True
    )
    cal_loader = DataLoader(cal_dataset, batch_size=32)
    
    optimal_temp = calibrate_temperature(
        eval_model, cal_loader, device, max_iter=30, lr=0.01
    )
    print(f"Optimal temperature: {optimal_temp:.4f}")
    
    cal_results = evaluate_calibration_improvement(
        eval_model, eval_loader, device, n_bins=10
    )
    print(f"ECE improvement: {cal_results['ece_improvement']:.4f}")
    print()
    
    # 4.6 Class Balancing with Focal Loss
    print("4.6 Class Balancing with Focal Loss for Rare Diseases")
    print("-" * 80)
    
    class_weights = compute_class_weights(class_counts, eval_config.icd_vocab_size)
    print(f"Class weights range: {class_weights.min():.2f} - "
          f"{class_weights.max():.2f}")
    
    focal_loss = BRISMLoss(
        kl_weight=0.1,
        cycle_weight=1.0,
        class_weights=class_weights,
        focal_gamma=2.0  # Focal loss always enabled
    )
    print("Focal loss created with gamma=2.0 for rare disease focus")
    print()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 80)
    print("SUMMARY - All BRISM v3.0.0 Features Demonstrated")
    print("=" * 80)
    print()
    print("✅ SECTION 1: Basic Usage")
    print("   - Model training and basic diagnosis")
    print("   - Symptom generation from ICD codes")
    print()
    print("✅ SECTION 2: Interpretability Tools")
    print("   - Attention visualization and integrated gradients")
    print("   - Counterfactual explanations")
    print()
    print("✅ SECTION 3: Advanced Generation")
    print("   - Beam search for diverse sequences")
    print("   - Ensemble uncertainty quantification")
    print()
    print("✅ SECTION 4: Clinical Features")
    print("   - Symptom synonym handling and normalization")
    print("   - Active learning for efficient diagnosis")
    print("   - Better evaluation metrics with stratification")
    print("   - Temporal modeling for symptom progression")
    print("   - Uncertainty calibration with temperature scaling")
    print("   - Class balancing with focal loss for rare diseases")
    print()
    print("All features are now standard in v3.0.0!")
    print("=" * 80)


if __name__ == '__main__':
    main()
