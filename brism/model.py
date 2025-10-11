"""
Core BRISM model architecture with dual encoder-decoder pairs sharing latent space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class BRISMConfig:
    """Configuration for BRISM model."""
    symptom_vocab_size: int = 1000
    icd_vocab_size: int = 500
    symptom_embed_dim: int = 128
    icd_embed_dim: int = 128
    encoder_hidden_dim: int = 256
    latent_dim: int = 64
    decoder_hidden_dim: int = 256
    max_symptom_length: int = 50
    dropout_rate: float = 0.2
    mc_samples: int = 20  # Monte Carlo dropout samples for uncertainty
    use_attention: bool = True  # Use attention for symptom aggregation
    # Temporal encoding settings
    use_temporal_encoding: bool = False  # Enable temporal encoding for symptoms
    temporal_encoding_type: str = 'positional'  # 'positional' or 'timestamp'
    # Temperature scaling for calibration
    temperature: float = 1.0  # Temperature parameter for probability calibration


class Encoder(nn.Module):
    """Generic encoder that maps embeddings to latent distribution."""
    
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """Generic decoder that maps latent to output distribution."""
    
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to output logits.
        
        Args:
            z: Latent representation [batch_size, latent_dim]
            
        Returns:
            Output logits [batch_size, output_dim]
        """
        h = F.relu(self.fc1(z))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        return self.fc_out(h)


class AttentionAggregator(nn.Module):
    """Self-attention mechanism for sequence aggregation."""
    
    def __init__(self, embed_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.attention_linear = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-weighted aggregation of sequence.
        
        Args:
            embeddings: Sequence embeddings [batch_size, seq_len, embed_dim]
            mask: Optional mask for padding [batch_size, seq_len] (1 for valid, 0 for padding)
            
        Returns:
            aggregated: Attention-weighted sum [batch_size, embed_dim]
            attention_weights: Attention weights [batch_size, seq_len]
        """
        # Compute attention scores
        scores = self.attention_linear(embeddings).squeeze(-1)  # [B, L]
        
        # Apply mask if provided (set padded positions to very negative value)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [B, L]
        attention_weights = self.dropout(attention_weights)
        
        # Weighted sum
        aggregated = torch.bmm(attention_weights.unsqueeze(1), embeddings).squeeze(1)  # [B, D]
        
        return aggregated, attention_weights


class SequenceDecoder(nn.Module):
    """Decoder for sequence generation (symptoms)."""
    
    def __init__(self, latent_dim: int, hidden_dim: int, vocab_size: int, 
                 max_length: int, dropout_rate: float = 0.2):
        super().__init__()
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, z: torch.Tensor, target_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode latent to symptom sequence.
        
        Args:
            z: Latent representation [batch_size, latent_dim]
            target_seq: Target sequence for teacher forcing [batch_size, seq_len]
            
        Returns:
            Output logits [batch_size, seq_len, vocab_size]
        """
        batch_size = z.size(0)
        
        # Initialize hidden state from latent
        h0 = self.latent_to_hidden(z).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        
        if target_seq is not None:
            # Teacher forcing
            embedded = self.embedding(target_seq)
            embedded = self.dropout(embedded)
            lstm_out, _ = self.lstm(embedded, (h0, c0))
            lstm_out = self.dropout(lstm_out)
            logits = self.fc_out(lstm_out)
        else:
            # Autoregressive generation
            logits_list = []
            input_token = torch.zeros(batch_size, 1, dtype=torch.long, device=z.device)
            hidden = (h0, c0)
            
            for _ in range(self.max_length):
                embedded = self.embedding(input_token)
                embedded = self.dropout(embedded)
                lstm_out, hidden = self.lstm(embedded, hidden)
                lstm_out = self.dropout(lstm_out)
                step_logits = self.fc_out(lstm_out)
                logits_list.append(step_logits)
                input_token = step_logits.argmax(dim=-1)
            
            logits = torch.cat(logits_list, dim=1)
        
        return logits


class BRISM(nn.Module):
    """
    Bayesian Reciprocal ICD-Symptom Model.
    
    Dual encoder-decoder architecture with shared latent space:
    - Forward: symptoms -> encoder -> latent -> decoder -> ICD probabilities
    - Reverse: ICD codes -> encoder -> latent -> decoder -> symptom sequence
    """
    
    def __init__(self, config: BRISMConfig):
        super().__init__()
        self.config = config
        
        # Temperature parameter for calibration (learnable)
        self.temperature = nn.Parameter(torch.ones(1) * config.temperature)
        
        # Embeddings
        self.symptom_embedding = nn.Embedding(config.symptom_vocab_size, config.symptom_embed_dim)
        self.icd_embedding = nn.Embedding(config.icd_vocab_size, config.icd_embed_dim)
        
        # Temporal encoding (optional)
        self.use_temporal = config.use_temporal_encoding
        if config.use_temporal_encoding:
            from .temporal import TemporalEncoding
            self.temporal_encoding = TemporalEncoding(
                embed_dim=config.symptom_embed_dim,
                max_length=config.max_symptom_length,
                encoding_type=config.temporal_encoding_type,
                dropout=config.dropout_rate
            )
        
        # Symptom aggregation (attention or mean pooling)
        if config.use_attention:
            self.symptom_attention = AttentionAggregator(config.symptom_embed_dim, config.dropout_rate)
        else:
            self.symptom_attention = None
        
        # Forward path: symptoms -> ICD
        self.symptom_encoder = Encoder(
            config.symptom_embed_dim,
            config.encoder_hidden_dim,
            config.latent_dim,
            config.dropout_rate
        )
        self.icd_decoder = Decoder(
            config.latent_dim,
            config.decoder_hidden_dim,
            config.icd_vocab_size,
            config.dropout_rate
        )
        
        # Reverse path: ICD -> symptoms
        self.icd_encoder = Encoder(
            config.icd_embed_dim,
            config.encoder_hidden_dim,
            config.latent_dim,
            config.dropout_rate
        )
        self.symptom_decoder = SequenceDecoder(
            config.latent_dim,
            config.decoder_hidden_dim,
            config.symptom_vocab_size,
            config.max_symptom_length,
            config.dropout_rate
        )
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        
        Args:
            mu: Mean [batch_size, latent_dim]
            logvar: Log variance [batch_size, latent_dim]
            
        Returns:
            Sampled latent [batch_size, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward_path(self, symptoms: torch.Tensor, timestamps: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward path: symptoms -> latent -> ICD probabilities.
        
        Args:
            symptoms: Symptom token IDs [batch_size, seq_len]
            timestamps: Optional timestamps for symptoms [batch_size, seq_len]
            
        Returns:
            icd_logits: ICD prediction logits [batch_size, icd_vocab_size]
            mu: Latent mean [batch_size, latent_dim]
            logvar: Latent log variance [batch_size, latent_dim]
        """
        # Embed symptoms
        symptom_embeds = self.symptom_embedding(symptoms)  # [B, L, D]
        
        # Add temporal encoding if enabled
        if self.use_temporal:
            symptom_embeds = self.temporal_encoding(symptom_embeds, timestamps)
        
        # Aggregate symptoms with attention or mean pooling
        if self.config.use_attention:
            # Create mask for non-zero tokens (assuming 0 is padding)
            mask = (symptoms != 0).float()  # [B, L]
            symptom_repr, _ = self.symptom_attention(symptom_embeds, mask)  # [B, D]
        else:
            # Mean pooling (original behavior)
            symptom_repr = symptom_embeds.mean(dim=1)  # [B, D]
        
        # Encode to latent
        mu, logvar = self.symptom_encoder(symptom_repr)
        z = self.reparameterize(mu, logvar)
        
        # Decode to ICD with temperature scaling
        icd_logits = self.icd_decoder(z) / self.temperature
        
        return icd_logits, mu, logvar
    
    def reverse_path(self, icd_codes: torch.Tensor, target_symptoms: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reverse path: ICD -> latent -> symptom sequence.
        
        Args:
            icd_codes: ICD code token IDs [batch_size]
            target_symptoms: Target symptom sequences for teacher forcing [batch_size, seq_len]
            
        Returns:
            symptom_logits: Symptom sequence logits [batch_size, seq_len, symptom_vocab_size]
            mu: Latent mean [batch_size, latent_dim]
            logvar: Latent log variance [batch_size, latent_dim]
        """
        # Embed ICD
        icd_embeds = self.icd_embedding(icd_codes)  # [B, D]
        
        # Encode to latent
        mu, logvar = self.icd_encoder(icd_embeds)
        z = self.reparameterize(mu, logvar)
        
        # Decode to symptoms
        symptom_logits = self.symptom_decoder(z, target_symptoms)
        
        return symptom_logits, mu, logvar
    
    def cycle_forward(self, symptoms: torch.Tensor, target_symptoms: Optional[torch.Tensor] = None) -> Tuple:
        """
        Cycle: symptoms -> latent -> ICD -> latent' -> symptoms'.
        
        Args:
            symptoms: Input symptom sequence [batch_size, seq_len]
            target_symptoms: Target for reconstruction [batch_size, seq_len]
            
        Returns:
            Tuple of intermediate and final outputs
        """
        # Forward: symptoms -> ICD
        icd_logits, mu1, logvar1 = self.forward_path(symptoms)
        
        # Get predicted ICD (hard decision for cycle)
        icd_pred = icd_logits.argmax(dim=-1)
        
        # Reverse: ICD -> symptoms
        symptom_logits, mu2, logvar2 = self.reverse_path(icd_pred, target_symptoms)
        
        return symptom_logits, icd_logits, mu1, logvar1, mu2, logvar2
    
    def cycle_reverse(self, icd_codes: torch.Tensor, target_symptoms: Optional[torch.Tensor] = None) -> Tuple:
        """
        Cycle: ICD -> latent -> symptoms -> latent' -> ICD'.
        
        Args:
            icd_codes: Input ICD codes [batch_size]
            target_symptoms: Target symptoms for intermediate step [batch_size, seq_len]
            
        Returns:
            Tuple of intermediate and final outputs
        """
        # Reverse: ICD -> symptoms
        symptom_logits, mu1, logvar1 = self.reverse_path(icd_codes, target_symptoms)
        
        # Get predicted symptoms (use argmax for hard decision)
        symptom_pred = symptom_logits.argmax(dim=-1)
        
        # Forward: symptoms -> ICD
        icd_logits, mu2, logvar2 = self.forward_path(symptom_pred)
        
        return icd_logits, symptom_logits, mu1, logvar1, mu2, logvar2
    
    def predict_with_uncertainty(self, symptoms: torch.Tensor, n_samples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict ICD probabilities with uncertainty using Monte Carlo dropout.
        
        Args:
            symptoms: Symptom token IDs [batch_size, seq_len]
            n_samples: Number of MC samples (defaults to config.mc_samples)
            
        Returns:
            mean_probs: Mean predicted probabilities [batch_size, icd_vocab_size]
            std_probs: Standard deviation of predictions [batch_size, icd_vocab_size]
        """
        if n_samples is None:
            n_samples = self.config.mc_samples
        
        self.train()  # Enable dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                icd_logits, _, _ = self.forward_path(symptoms)
                probs = F.softmax(icd_logits, dim=-1)
                predictions.append(probs)
        
        predictions = torch.stack(predictions)  # [n_samples, batch_size, icd_vocab_size]
        mean_probs = predictions.mean(dim=0)
        std_probs = predictions.std(dim=0)
        
        return mean_probs, std_probs
