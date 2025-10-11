"""
Tests for new features: interpretability, beam search, contrastive learning,
ensemble, symptom normalization, and active learning.
"""

import unittest
import torch
import numpy as np

from brism import (
    BRISM, BRISMConfig,
    BRISMLoss,
    IntegratedGradients,
    AttentionVisualization,
    CounterfactualExplanations,
    AttentionRollout,
    explain_prediction,
    generate_symptoms_beam_search,
    BRISMEnsemble,
    SymptomNormalizer,
    create_default_medical_synonyms,
    ActiveLearner,
)


class TestInterpretability(unittest.TestCase):
    """Test interpretability tools."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BRISMConfig(
            symptom_vocab_size=50,
            icd_vocab_size=20,
            latent_dim=16,
            max_symptom_length=10
        )
        self.model = BRISM(self.config)
        self.symptoms = torch.tensor([1, 5, 12, 15, 0, 0, 0, 0, 0, 0])
    
    def test_attention_visualization(self):
        """Test attention visualization."""
        vis = AttentionVisualization(self.model)
        attention_weights, predictions = vis.get_attention_weights(self.symptoms)
        
        self.assertEqual(attention_weights.shape, (10,))
        self.assertAlmostEqual(attention_weights.sum().item(), 1.0, places=5)
        self.assertEqual(predictions.shape, (self.config.icd_vocab_size,))
    
    def test_integrated_gradients(self):
        """Test integrated gradients."""
        ig = IntegratedGradients(self.model)
        attributions = ig.attribute(self.symptoms, n_steps=10)
        
        self.assertEqual(attributions.shape, (10,))
        # Check that non-zero symptoms have attributions
        self.assertTrue(attributions[:4].abs().sum() > 0)
    
    def test_counterfactual_explanations(self):
        """Test counterfactual explanations."""
        cf = CounterfactualExplanations(self.model)
        explanations = cf.explain_by_removal(self.symptoms)
        
        # Should have explanations for non-padding symptoms
        self.assertGreater(len(explanations), 0)
        self.assertLessEqual(len(explanations), 4)  # 4 non-zero symptoms
        
        # Check explanation format
        for exp in explanations:
            self.assertIn('symptom_id', exp)
            self.assertIn('probability_drop', exp)
            self.assertIn('probability_drop_percentage', exp)
    
    def test_explain_prediction(self):
        """Test comprehensive explanation."""
        explanation = explain_prediction(
            self.model,
            self.symptoms,
            top_k=3,
            method='attention'
        )
        
        self.assertIn('predictions', explanation)
        self.assertEqual(len(explanation['predictions']), 3)
        self.assertIn('attention_weights', explanation)


class TestBeamSearch(unittest.TestCase):
    """Test beam search for symptom generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BRISMConfig(
            symptom_vocab_size=50,
            icd_vocab_size=20,
            latent_dim=16,
            max_symptom_length=10
        )
        self.model = BRISM(self.config)
        self.device = torch.device('cpu')
        self.icd_code = torch.tensor([5])
    
    def test_beam_search_single_beam(self):
        """Test beam search with single output."""
        result = generate_symptoms_beam_search(
            self.model,
            self.icd_code,
            self.device,
            beam_width=3,
            return_all_beams=False
        )
        
        self.assertIn('sequence', result)
        self.assertIn('score', result)
        self.assertEqual(len(result['sequence']), self.config.max_symptom_length)
    
    def test_beam_search_all_beams(self):
        """Test beam search with all beams."""
        result = generate_symptoms_beam_search(
            self.model,
            self.icd_code,
            self.device,
            beam_width=5,
            return_all_beams=True
        )
        
        self.assertIn('beams', result)
        self.assertEqual(len(result['beams']), 5)
        self.assertIn('best_sequence', result)
        
        # Check beam format
        for beam in result['beams']:
            self.assertIn('sequence', beam)
            self.assertIn('score', beam)
            self.assertIn('length', beam)
    
    def test_beam_search_batch(self):
        """Test beam search with batch input."""
        icd_codes = torch.tensor([5, 10])
        results = generate_symptoms_beam_search(
            self.model,
            icd_codes,
            self.device,
            beam_width=3
        )
        
        self.assertEqual(len(results), 2)


class TestContrastiveLearning(unittest.TestCase):
    """Test contrastive learning loss."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BRISMConfig(
            symptom_vocab_size=50,
            icd_vocab_size=20,
            latent_dim=16
        )
        self.model = BRISM(self.config)
        
        # Create loss with contrastive learning
        self.loss_fn = BRISMLoss(
            kl_weight=0.1,
            cycle_weight=1.0,
            contrastive_weight=0.5,
            contrastive_margin=1.0
        )
    
    def test_contrastive_loss_triplet(self):
        """Test triplet loss computation."""
        latents = torch.randn(4, 16)
        labels = torch.tensor([1, 1, 2, 2])
        
        loss = self.loss_fn.contrastive_loss_triplet(latents, labels)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)  # Scalar
        self.assertTrue(loss.item() >= 0)
    
    def test_contrastive_loss_infonce(self):
        """Test InfoNCE loss computation."""
        latents = torch.randn(4, 16)
        labels = torch.tensor([1, 1, 2, 2])
        
        loss = self.loss_fn.contrastive_loss_infonce(latents, labels)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(loss.item() >= 0)
    
    def test_forward_loss_with_contrastive(self):
        """Test forward loss includes contrastive term."""
        symptoms = torch.randint(0, 50, (4, 10))
        icd_codes = torch.tensor([1, 1, 2, 3])
        
        icd_logits, mu, logvar = self.model.forward_path(symptoms)
        loss, loss_dict = self.loss_fn.forward_loss(
            (icd_logits, mu, logvar),
            symptoms,
            icd_codes
        )
        
        # Check that contrastive loss is included
        self.assertIn('forward_contrastive', loss_dict)
        self.assertTrue(loss_dict['forward_contrastive'] >= 0)


class TestEnsemble(unittest.TestCase):
    """Test ensemble uncertainty quantification."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BRISMConfig(
            symptom_vocab_size=50,
            icd_vocab_size=20,
            latent_dim=16,
            max_symptom_length=10
        )
        self.model = BRISM(self.config)
        self.symptoms = torch.tensor([1, 5, 12, 15, 0, 0, 0, 0, 0, 0])
    
    def test_pseudo_ensemble_initialization(self):
        """Test pseudo-ensemble initialization."""
        ensemble = BRISMEnsemble(
            models=[self.model],
            use_pseudo_ensemble=True,
            n_models=5
        )
        
        self.assertTrue(ensemble.use_pseudo_ensemble)
        self.assertEqual(ensemble.n_models, 5)
    
    def test_true_ensemble_initialization(self):
        """Test true ensemble initialization."""
        models = [BRISM(self.config) for _ in range(3)]
        ensemble = BRISMEnsemble(models=models)
        
        self.assertFalse(ensemble.use_pseudo_ensemble)
        self.assertEqual(ensemble.n_models, 3)
    
    def test_ensemble_prediction(self):
        """Test ensemble prediction."""
        ensemble = BRISMEnsemble(
            models=[self.model],
            use_pseudo_ensemble=True,
            n_models=5
        )
        
        mean_probs, std_probs, uncertainty = ensemble.predict_with_uncertainty(
            self.symptoms.unsqueeze(0),
            n_samples=5
        )
        
        self.assertEqual(mean_probs.shape, (1, self.config.icd_vocab_size))
        self.assertEqual(std_probs.shape, (1, self.config.icd_vocab_size))
        self.assertIn('epistemic', uncertainty)
        self.assertIn('aleatoric', uncertainty)
        self.assertIn('total', uncertainty)
    
    def test_ensemble_diagnosis(self):
        """Test ensemble diagnosis."""
        ensemble = BRISMEnsemble(
            models=[self.model],
            use_pseudo_ensemble=True,
            n_models=3
        )
        
        result = ensemble.diagnose_with_ensemble(
            self.symptoms,
            top_k=3,
            n_samples=3
        )
        
        self.assertIn('predictions', result)
        self.assertEqual(len(result['predictions']), 3)
        self.assertIn('uncertainty', result)
        self.assertIn('ensemble_agreement', result)


class TestSymptomNormalization(unittest.TestCase):
    """Test symptom normalization."""
    
    def test_default_abbreviations(self):
        """Test default medical abbreviations."""
        normalizer = SymptomNormalizer()
        
        self.assertEqual(normalizer.normalize('sob'), 'shortness of breath')
        self.assertEqual(normalizer.normalize('cp'), 'chest pain')
        self.assertEqual(normalizer.normalize('ha'), 'headache')
    
    def test_custom_synonyms(self):
        """Test custom synonym mapping."""
        normalizer = SymptomNormalizer()
        normalizer.add_synonym('breathless', 'shortness of breath')
        
        self.assertEqual(normalizer.normalize('breathless'), 'shortness of breath')
    
    def test_canonical_forms(self):
        """Test canonical form mapping."""
        canonical_forms = {
            'shortness of breath': 1,
            'chest pain': 2,
            'headache': 3
        }
        normalizer = SymptomNormalizer(canonical_forms=canonical_forms)
        
        self.assertEqual(normalizer.normalize_to_id('shortness of breath'), 1)
        self.assertEqual(normalizer.normalize_to_id('chest pain'), 2)
    
    def test_umls_mapping(self):
        """Test UMLS synonym mapping."""
        normalizer = SymptomNormalizer()
        synonyms = create_default_medical_synonyms()
        normalizer.build_from_umls(synonyms)
        
        # Test multiple variants map to same canonical form
        sob_variants = ['sob', 'dyspnea', 'breathlessness']
        canonical = normalizer.normalize(sob_variants[0])
        
        for variant in sob_variants[1:]:
            self.assertEqual(normalizer.normalize(variant), canonical)
    
    def test_sequence_normalization(self):
        """Test sequence normalization."""
        normalizer = SymptomNormalizer()
        symptoms = ['SOB', 'CP', 'headache']
        normalized = normalizer.normalize_sequence(symptoms)
        
        self.assertEqual(len(normalized), 3)
        self.assertEqual(normalized[0], 'shortness of breath')
        self.assertEqual(normalized[1], 'chest pain')
        self.assertEqual(normalized[2], 'headache')


class TestActiveLearning(unittest.TestCase):
    """Test active learning interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BRISMConfig(
            symptom_vocab_size=50,
            icd_vocab_size=20,
            latent_dim=16,
            max_symptom_length=10
        )
        self.model = BRISM(self.config)
        self.learner = ActiveLearner(self.model)
        self.symptoms = torch.tensor([1, 5, 0, 0, 0, 0, 0, 0, 0, 0])
    
    def test_query_next_symptom_entropy(self):
        """Test entropy-based query selection."""
        recommendations = self.learner.query_next_symptom(
            self.symptoms,
            method='entropy',
            top_k=3,
            n_samples=5
        )
        
        self.assertEqual(len(recommendations), 3)
        for rec in recommendations:
            self.assertIn('symptom_id', rec)
            self.assertIn('score', rec)
            self.assertIn('method', rec)
            self.assertEqual(rec['method'], 'entropy')
    
    def test_query_next_symptom_bald(self):
        """Test BALD query selection."""
        recommendations = self.learner.query_next_symptom(
            self.symptoms,
            method='bald',
            top_k=3,
            n_samples=5
        )
        
        self.assertEqual(len(recommendations), 3)
        self.assertEqual(recommendations[0]['method'], 'bald')
    
    def test_query_next_symptom_variance(self):
        """Test variance-based query selection."""
        recommendations = self.learner.query_next_symptom(
            self.symptoms,
            method='variance',
            top_k=3,
            n_samples=5
        )
        
        self.assertEqual(len(recommendations), 3)
        self.assertEqual(recommendations[0]['method'], 'variance')
    
    def test_interactive_diagnosis(self):
        """Test interactive diagnosis workflow."""
        result = self.learner.interactive_diagnosis(
            initial_symptoms=self.symptoms,
            max_queries=3,
            uncertainty_threshold=0.3,
            confidence_threshold=0.9
        )
        
        self.assertIn('final_symptoms', result)
        self.assertIn('predictions', result)
        self.assertIn('query_history', result)
        self.assertIn('num_queries', result)
        
        # Should have made some queries
        self.assertGreater(result['num_queries'], 0)
        self.assertLessEqual(result['num_queries'], 3)
    
    def test_add_symptom(self):
        """Test adding symptom to sequence."""
        new_symptoms = self.learner._add_symptom(self.symptoms, 15)
        
        # Should add to first padding position
        self.assertEqual(new_symptoms[2].item(), 15)
        # Original symptoms should remain
        self.assertEqual(new_symptoms[0].item(), 1)
        self.assertEqual(new_symptoms[1].item(), 5)


if __name__ == '__main__':
    unittest.main()
