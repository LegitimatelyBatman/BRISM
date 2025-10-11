"""
Comprehensive example demonstrating all 6 new BRISM features.

Features demonstrated:
1. Interpretability Tools (attention visualization, integrated gradients, counterfactuals)
2. Beam Search for Symptom Generation
3. Contrastive Learning for Better Latent Space
4. Ensemble Uncertainty Quantification
5. Symptom Synonym Handling
6. Active Learning Interface
"""

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from brism import (
    BRISM, BRISMConfig,
    BRISMLoss,
    # Feature 1: Interpretability
    IntegratedGradients,
    AttentionVisualization,
    CounterfactualExplanations,
    AttentionRollout,
    explain_prediction,
    # Feature 2: Beam Search
    generate_symptoms_beam_search,
    # Feature 4: Ensemble
    BRISMEnsemble,
    train_ensemble,
    # Feature 5: Synonym Handling
    SymptomNormalizer,
    create_default_medical_synonyms,
    build_symptom_normalizer_from_vocab,
    # Feature 6: Active Learning
    ActiveLearner,
)


# Synthetic dataset for demonstration
class SyntheticMedicalDataset(Dataset):
    """Synthetic medical dataset for testing."""
    
    def __init__(self, num_samples=1000, symptom_vocab_size=100, icd_vocab_size=50, max_length=20):
        self.num_samples = num_samples
        self.symptom_vocab_size = symptom_vocab_size
        self.icd_vocab_size = icd_vocab_size
        self.max_length = max_length
        
        # Generate synthetic data
        np.random.seed(42)
        
        self.symptoms = []
        self.icd_codes = []
        
        for _ in range(num_samples):
            # Random ICD code
            icd = np.random.randint(1, icd_vocab_size)
            
            # Generate symptoms based on ICD (with some correlation)
            n_symptoms = np.random.randint(3, 8)
            base_symptoms = [(icd * 2 + i) % symptom_vocab_size for i in range(n_symptoms)]
            # Add some random symptoms
            random_symptoms = np.random.randint(1, symptom_vocab_size, size=2).tolist()
            all_symptoms = base_symptoms + random_symptoms
            
            # Pad to max_length
            symptoms = all_symptoms[:max_length]
            if len(symptoms) < max_length:
                symptoms = symptoms + [0] * (max_length - len(symptoms))
            
            self.symptoms.append(symptoms)
            self.icd_codes.append(icd)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'symptoms': torch.tensor(self.symptoms[idx], dtype=torch.long),
            'icd_code': torch.tensor(self.icd_codes[idx], dtype=torch.long)
        }


def main():
    """Demonstrate all new features."""
    
    print("="*80)
    print("BRISM NEW FEATURES DEMONSTRATION")
    print("="*80)
    print()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    # Create model
    config = BRISMConfig(
        symptom_vocab_size=100,
        icd_vocab_size=50,
        latent_dim=32,
        max_symptom_length=20,
        use_attention=True,
        mc_samples=20
    )
    model = BRISM(config).to(device)
    print(f"Created BRISM model with {sum(p.numel() for p in model.parameters())} parameters")
    print()
    
    # Create dataset
    train_dataset = SyntheticMedicalDataset(num_samples=500)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # ========================================================================
    # Feature 3: Contrastive Learning for Better Latent Space
    # ========================================================================
    print("="*80)
    print("FEATURE 3: CONTRASTIVE LEARNING")
    print("="*80)
    print()
    
    # Train with contrastive loss
    print("Training model with contrastive learning...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Loss with contrastive learning
    loss_fn = BRISMLoss(
        kl_weight=0.1,
        cycle_weight=1.0,
        contrastive_weight=0.5,  # Enable contrastive learning
        contrastive_margin=1.0,
        contrastive_temperature=0.5
    )
    
    # Train for a few epochs
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in train_loader:
            symptoms = batch['symptoms'].to(device)
            icd_codes = batch['icd_code'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            icd_logits, mu, logvar = model.forward_path(symptoms)
            
            # Compute loss with contrastive term
            loss, loss_dict = loss_fn.forward_loss((icd_logits, mu, logvar), symptoms, icd_codes)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        if epoch == 0 and 'forward_contrastive' in loss_dict:
            print(f"  Contrastive loss component: {loss_dict['forward_contrastive']:.4f}")
    
    print()
    
    # ========================================================================
    # Feature 1: Interpretability Tools
    # ========================================================================
    print("="*80)
    print("FEATURE 1: INTERPRETABILITY TOOLS")
    print("="*80)
    print()
    
    # Get a test sample
    test_sample = train_dataset[0]
    test_symptoms = test_sample['symptoms'].to(device)
    test_icd = test_sample['icd_code'].item()
    
    print(f"Test symptoms: {test_symptoms[:8].cpu().numpy().tolist()}")
    print(f"True ICD code: {test_icd}")
    print()
    
    # 1.1 Attention Visualization
    print("1.1 Attention Visualization")
    print("-" * 40)
    vis = AttentionVisualization(model)
    attention_weights, predictions = vis.get_attention_weights(test_symptoms)
    
    print(f"Attention weights: {attention_weights[:8].cpu().numpy()}")
    print(f"Top prediction: ICD {predictions.argmax().item()} (prob: {predictions.max().item():.4f})")
    print()
    
    # 1.2 Integrated Gradients
    print("1.2 Integrated Gradients")
    print("-" * 40)
    ig = IntegratedGradients(model)
    attributions = ig.attribute(test_symptoms, n_steps=20)
    
    print(f"Attribution scores: {attributions[:8].detach().cpu().numpy()}")
    top_symptoms = attributions.abs().argsort(descending=True)[:3]
    print(f"Most important symptoms (indices): {top_symptoms.cpu().numpy().tolist()}")
    print()
    
    # 1.3 Counterfactual Explanations
    print("1.3 Counterfactual Explanations")
    print("-" * 40)
    cf = CounterfactualExplanations(model)
    explanations = cf.explain_by_removal(test_symptoms)
    
    if len(explanations) > 0:
        print("Impact of removing each symptom:")
        for exp in explanations[:3]:
            print(f"  Symptom {exp['symptom_id']}: "
                  f"{exp['probability_drop_percentage']:.2f}% probability drop")
    print()
    
    # 1.4 Comprehensive Explanation
    print("1.4 Comprehensive Explanation")
    print("-" * 40)
    full_explanation = explain_prediction(
        model,
        test_symptoms,
        top_k=3,
        method='all'
    )
    
    print("Predictions:")
    for pred in full_explanation['predictions'][:3]:
        print(f"  ICD {pred['icd_code']}: {pred['probability']:.4f}")
    
    if 'top_attended_symptoms' in full_explanation:
        print("\nTop attended symptoms:")
        for symp in full_explanation['top_attended_symptoms'][:3]:
            print(f"  Position {symp['position']}: weight={symp['attention_weight']:.4f}")
    print()
    
    # ========================================================================
    # Feature 2: Beam Search for Symptom Generation
    # ========================================================================
    print("="*80)
    print("FEATURE 2: BEAM SEARCH FOR SYMPTOM GENERATION")
    print("="*80)
    print()
    
    # Generate symptoms using beam search
    test_icd_tensor = torch.tensor([test_icd], device=device)
    
    print("Generating symptoms with beam search (beam_width=5)...")
    beam_results = generate_symptoms_beam_search(
        model,
        test_icd_tensor,
        device,
        beam_width=5,
        temperature=1.0,
        return_all_beams=True
    )
    
    print(f"\nGenerated {len(beam_results['beams'])} symptom sequences:")
    for i, beam in enumerate(beam_results['beams'][:3], 1):
        symptoms = [s for s in beam['sequence'] if s != 0][:5]
        print(f"  Beam {i} (score={beam['score']:.4f}): {symptoms}")
    
    print(f"\nBest sequence: {[s for s in beam_results['best_sequence'] if s != 0][:8]}")
    print()
    
    # ========================================================================
    # Feature 4: Ensemble Uncertainty Quantification
    # ========================================================================
    print("="*80)
    print("FEATURE 4: ENSEMBLE UNCERTAINTY")
    print("="*80)
    print()
    
    # Create pseudo-ensemble (faster than training multiple models)
    print("Creating pseudo-ensemble with 5 dropout masks...")
    ensemble = BRISMEnsemble(
        models=[model],
        use_pseudo_ensemble=True,
        n_models=5
    )
    
    # Get ensemble predictions
    print("Making ensemble prediction...")
    ensemble_result = ensemble.diagnose_with_ensemble(
        test_symptoms,
        top_k=3,
        n_samples=5
    )
    
    print("\nEnsemble Predictions:")
    for pred in ensemble_result['predictions']:
        cv = pred['coefficient_of_variation']
        print(f"  ICD {pred['icd_code']}: {pred['probability']:.4f} ± {pred['std']:.4f} "
              f"(CV: {cv:.3f})")
    
    print("\nUncertainty Decomposition:")
    print(f"  Epistemic (model uncertainty): {ensemble_result['uncertainty']['epistemic']:.4f}")
    print(f"  Aleatoric (data uncertainty): {ensemble_result['uncertainty']['aleatoric']:.4f}")
    print(f"  Total uncertainty: {ensemble_result['uncertainty']['total']:.4f}")
    print(f"  Ensemble agreement: {ensemble_result['ensemble_agreement']:.2%}")
    print()
    
    # ========================================================================
    # Feature 5: Symptom Synonym Handling
    # ========================================================================
    print("="*80)
    print("FEATURE 5: SYMPTOM SYNONYM HANDLING")
    print("="*80)
    print()
    
    # Create symptom normalizer
    print("Creating symptom normalizer with medical synonyms...")
    normalizer = SymptomNormalizer()
    
    # Add custom synonyms
    synonym_dict = create_default_medical_synonyms()
    normalizer.build_from_umls(synonym_dict)
    
    # Test normalization
    test_symptoms_text = [
        "SOB",
        "shortness of breath",
        "dyspnea",
        "CP",
        "chest pain",
        "ha",
        "headache"
    ]
    
    print("Symptom normalization examples:")
    for symptom in test_symptoms_text:
        normalized = normalizer.normalize(symptom)
        print(f"  '{symptom}' -> '{normalized}'")
    
    print()
    
    # Create canonical vocabulary mapping
    symptom_vocab = {f"symptom_{i}": i for i in range(100)}
    symptom_vocab.update({
        'shortness of breath': 1,
        'chest pain': 2,
        'headache': 3,
        'abdominal pain': 4,
        'fever': 5
    })
    
    # Build normalizer with vocab
    vocab_normalizer = build_symptom_normalizer_from_vocab(
        symptom_vocab,
        synonym_lists=synonym_dict
    )
    
    # Test ID conversion
    test_text = "SOB and CP"
    print(f"\nConverting '{test_text}' to token IDs:")
    normalized_texts = ['shortness of breath', 'chest pain']  # After normalization
    ids = [vocab_normalizer.normalize_to_id(t) for t in normalized_texts]
    print(f"  Token IDs: {ids}")
    print()
    
    # ========================================================================
    # Feature 6: Active Learning Interface
    # ========================================================================
    print("="*80)
    print("FEATURE 6: ACTIVE LEARNING INTERFACE")
    print("="*80)
    print()
    
    # Create active learner
    learner = ActiveLearner(model)
    
    # Start with partial symptoms
    partial_symptoms = test_symptoms.clone()
    partial_symptoms[5:] = 0  # Remove half of symptoms
    
    print(f"Starting with partial symptoms: {partial_symptoms[:8].cpu().numpy().tolist()}")
    print()
    
    # Query next symptoms to ask about
    print("Querying most informative symptoms to ask about...")
    print("\nMethod 1: Entropy-based selection")
    recommendations = learner.query_next_symptom(
        partial_symptoms,
        method='entropy',
        top_k=3,
        n_samples=10
    )
    
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. Symptom {rec['symptom_id']}: score={rec['score']:.4f}")
    
    print("\nMethod 2: BALD (Bayesian Active Learning by Disagreement)")
    recommendations_bald = learner.query_next_symptom(
        partial_symptoms,
        method='bald',
        top_k=3,
        n_samples=10
    )
    
    for i, rec in enumerate(recommendations_bald, 1):
        print(f"  {i}. Symptom {rec['symptom_id']}: score={rec['score']:.4f}")
    
    print()
    
    # Interactive diagnosis simulation
    print("Simulating interactive diagnosis...")
    print("-" * 40)
    
    # Create very sparse initial symptoms
    sparse_symptoms = torch.zeros(20, dtype=torch.long, device=device)
    sparse_symptoms[0] = test_symptoms[0]
    sparse_symptoms[1] = test_symptoms[1]
    
    interactive_result = learner.interactive_diagnosis(
        initial_symptoms=sparse_symptoms,
        max_queries=5,
        uncertainty_threshold=0.3,
        confidence_threshold=0.85
    )
    
    print(f"\nCompleted in {interactive_result['num_queries']} queries")
    print("\nQuery history:")
    for query in interactive_result['query_history']:
        if 'queried_symptom' in query:
            print(f"  Query {query['query_num']}: Asked about symptom "
                  f"{query['queried_symptom']['symptom_id']} "
                  f"(confidence: {query['confidence']:.3f}, "
                  f"uncertainty: {query['uncertainty']:.3f})")
    
    print("\nFinal predictions:")
    for i, pred in enumerate(interactive_result['predictions'][:3], 1):
        print(f"  {i}. ICD {pred['icd_code']}: {pred['probability']:.4f}")
    
    print()
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print("✅ Feature 1: Interpretability Tools")
    print("   - Attention visualization shows which symptoms matter most")
    print("   - Integrated gradients provide feature attribution")
    print("   - Counterfactual explanations show impact of removing symptoms")
    print()
    print("✅ Feature 2: Beam Search for Symptom Generation")
    print("   - Generates diverse, high-quality symptom sequences")
    print("   - Keeps track of top-k sequences by cumulative probability")
    print()
    print("✅ Feature 3: Contrastive Learning")
    print("   - Learns better latent space structure")
    print("   - Same-disease symptoms cluster together")
    print()
    print("✅ Feature 4: Ensemble Uncertainty")
    print("   - Decomposes uncertainty into epistemic and aleatoric")
    print("   - Provides more robust predictions with ensemble agreement")
    print()
    print("✅ Feature 5: Symptom Synonym Handling")
    print("   - Maps variants (SOB, dyspnea) to canonical forms")
    print("   - Supports UMLS/SNOMED-CT integration")
    print()
    print("✅ Feature 6: Active Learning Interface")
    print("   - Suggests most informative symptoms to query")
    print("   - Reduces number of questions needed for diagnosis")
    print()
    print("All features demonstrated successfully!")
    print("="*80)


if __name__ == '__main__':
    main()
