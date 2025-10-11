"""
Example script demonstrating BRISM model usage.

This script shows:
1. Creating synthetic data
2. Initializing and training the model
3. Making predictions with confidence intervals
4. Generating symptom sequences from ICD codes
"""

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from brism import BRISM, BRISMConfig, train_brism, diagnose_with_confidence
from brism.loss import BRISMLoss
from brism.train import train_epoch_simple
from brism.inference import generate_symptoms_with_uncertainty


class SyntheticMedicalDataset(Dataset):
    """Synthetic dataset for demonstration purposes."""
    
    def __init__(self, num_samples: int = 1000, symptom_vocab_size: int = 1000,
                 icd_vocab_size: int = 500, max_symptom_length: int = 50):
        self.num_samples = num_samples
        self.symptom_vocab_size = symptom_vocab_size
        self.icd_vocab_size = icd_vocab_size
        self.max_symptom_length = max_symptom_length
        
        # Generate synthetic data
        self.data = self._generate_data()
    
    def _generate_data(self):
        """Generate synthetic symptom-ICD pairs."""
        data = []
        
        for _ in range(self.num_samples):
            # Random ICD code
            icd_code = np.random.randint(0, self.icd_vocab_size)
            
            # Generate symptom sequence (5-20 symptoms)
            seq_len = np.random.randint(5, 21)
            # Use ICD code to influence symptoms (simple correlation)
            # First few symptoms are related to ICD code
            symptoms = np.zeros(self.max_symptom_length, dtype=np.int64)
            
            # Add some correlated symptoms
            base_symptom = (icd_code * 2) % self.symptom_vocab_size
            for i in range(min(seq_len, 5)):
                symptoms[i] = (base_symptom + i) % self.symptom_vocab_size
            
            # Add random symptoms
            for i in range(5, seq_len):
                symptoms[i] = np.random.randint(1, self.symptom_vocab_size)
            
            # Padding is 0
            
            data.append({
                'symptoms': symptoms,
                'icd_codes': icd_code,
                'seq_len': seq_len
            })
        
        return data
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'symptoms': torch.tensor(sample['symptoms'], dtype=torch.long),
            'icd_codes': torch.tensor(sample['icd_codes'], dtype=torch.long)
        }


def main():
    """Main example function."""
    
    print("=" * 70)
    print("BRISM: Bayesian Reciprocal ICD-Symptom Model - Example")
    print("=" * 70)
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
    
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = BRISMLoss(kl_weight=0.1, cycle_weight=1.0)
    
    # Train for a few epochs (shortened for demo)
    print("Training model (3 epochs for demo)...")
    print("-" * 70)
    
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
    
    print("-" * 70)
    print("Training completed!")
    print()
    
    # Inference examples
    print("=" * 70)
    print("INFERENCE EXAMPLES")
    print("=" * 70)
    print()
    
    # Example 1: Diagnose from symptoms
    print("Example 1: Diagnose from symptoms with confidence intervals")
    print("-" * 70)
    
    # Get a sample from validation set
    sample = val_dataset[0]
    symptoms = sample['symptoms'].unsqueeze(0)  # Add batch dimension
    true_icd = sample['icd_codes'].item()
    
    print(f"Input symptoms (first 10 non-zero): {symptoms[0][:10].tolist()}")
    print(f"True ICD code: {true_icd}")
    print()
    
    # Diagnose with confidence
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
    print("Uncertainty metrics:")
    print(f"  Entropy: {diagnosis['uncertainty']['entropy']:.4f}")
    print(f"  Average Std: {diagnosis['uncertainty']['average_std']:.4f}")
    print()
    
    # Example 2: Generate symptoms from ICD
    print("Example 2: Generate symptom sequences from ICD code")
    print("-" * 70)
    
    icd_code = torch.tensor(true_icd)
    
    print(f"Input ICD code: {icd_code.item()}")
    print()
    
    # Generate symptoms
    generated = generate_symptoms_with_uncertainty(
        model=model,
        icd_code=icd_code,
        device=device,
        n_samples=20
    )
    
    print(f"Generated symptom sequence (mode):")
    print(f"  First 15 tokens: {generated['mode_sequence'][:15]}")
    print()
    print(f"Token probabilities (first 15):")
    for i, prob in enumerate(generated['token_probabilities'][:15]):
        print(f"  Position {i}: {prob:.2f}")
    
    print()
    print(f"Sequence diversity: {generated['diversity']:.4f}")
    print(f"Unique sequences: {generated['n_unique_sequences']}/{20}")
    print()
    
    # Example 3: Batch diagnosis
    print("Example 3: Batch diagnosis")
    print("-" * 70)
    
    # Get multiple samples
    batch_symptoms = torch.stack([val_dataset[i]['symptoms'] for i in range(3)])
    true_icds = [val_dataset[i]['icd_codes'].item() for i in range(3)]
    
    print("Diagnosing 3 samples...")
    print()
    
    for i in range(3):
        print(f"Sample {i+1}:")
        print(f"  True ICD: {true_icds[i]}")
        
        diagnosis = diagnose_with_confidence(
            model=model,
            symptoms=batch_symptoms[i].unsqueeze(0),
            device=device,
            top_k=3
        )
        
        print(f"  Top prediction: ICD {diagnosis['predictions'][0]['icd_code']} "
              f"(prob: {diagnosis['predictions'][0]['probability']:.4f})")
        print()
    
    print("=" * 70)
    print("Example completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()
