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
    # Required parameters
    symptom_vocab_size: int = 1000
    icd_vocab_size: int = 500
    
    # Architecture parameters
    symptom_embed_dim: int = 128
    icd_embed_dim: int = 128
    encoder_hidden_dim: int = 256
    latent_dim: int = 64
    decoder_hidden_dim: int = 256
    max_symptom_length: int = 50
    dropout_rate: float = 0.2
    mc_samples: int = 20  # Monte Carlo dropout samples for uncertainty
    
    # Temporal encoding settings (always enabled)
    temporal_encoding_type: str = 'positional'  # 'positional' or 'timestamp'
    
    # Advanced feature hyperparameters
    temperature: float = 1.0  # Temperature parameter for probability calibration
    beam_width: int = 5  # For beam search generation
    n_ensemble_models: int = 5  # For pseudo-ensemble
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate vocabulary sizes
        if self.symptom_vocab_size <= 0:
            raise ValueError(f"symptom_vocab_size must be positive, got {self.symptom_vocab_size}")
        if self.icd_vocab_size <= 0:
            raise ValueError(f"icd_vocab_size must be positive, got {self.icd_vocab_size}")
        
        # Validate dimensions
        if self.symptom_embed_dim <= 0:
            raise ValueError(f"symptom_embed_dim must be positive, got {self.symptom_embed_dim}")
        if self.icd_embed_dim <= 0:
            raise ValueError(f"icd_embed_dim must be positive, got {self.icd_embed_dim}")
        if self.encoder_hidden_dim <= 0:
            raise ValueError(f"encoder_hidden_dim must be positive, got {self.encoder_hidden_dim}")
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        if self.decoder_hidden_dim <= 0:
            raise ValueError(f"decoder_hidden_dim must be positive, got {self.decoder_hidden_dim}")
        if self.max_symptom_length <= 0:
            raise ValueError(f"max_symptom_length must be positive, got {self.max_symptom_length}")
        
        # Validate dropout rate
        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be between 0 and 1, got {self.dropout_rate}")
        
        # Validate mc_samples
        if self.mc_samples < 1:
            raise ValueError(f"mc_samples must be at least 1, got {self.mc_samples}")
        
        # Validate temporal encoding type
        if self.temporal_encoding_type not in ['positional', 'timestamp']:
            raise ValueError(f"temporal_encoding_type must be 'positional' or 'timestamp', got {self.temporal_encoding_type}")
        
        # Validate temperature
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        
        # Validate beam width
        if self.beam_width < 1:
            raise ValueError(f"beam_width must be at least 1, got {self.beam_width}")
        
        # Validate ensemble models
        if self.n_ensemble_models < 1:
            raise ValueError(f"n_ensemble_models must be at least 1, got {self.n_ensemble_models}")


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
        self.vocab_size = vocab_size
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
    
    def beam_search(
        self, 
        z: torch.Tensor, 
        beam_width: int = 5, 
        temperature: float = 1.0,
        length_penalty: float = 1.0,
        max_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate sequences using beam search.
        
        Args:
            z: Latent representation [batch_size, latent_dim]
            beam_width: Number of beams to keep
            temperature: Temperature for softmax (higher = more diverse)
            length_penalty: Penalty for sequence length (>1 favors longer sequences)
            max_length: Maximum sequence length (defaults to self.max_length)
            
        Returns:
            sequences: Top beam_width sequences [batch_size, beam_width, seq_len]
            scores: Log probabilities for each sequence [batch_size, beam_width]
            lengths: Actual lengths of sequences [batch_size, beam_width]
        """
        # Input validation
        assert z.dim() == 2, f"Expected z to be 2D tensor [batch_size, latent_dim], got shape {z.shape}"
        assert beam_width > 0, f"beam_width must be positive, got {beam_width}"
        assert temperature > 0, f"temperature must be positive, got {temperature}"
        
        batch_size = z.size(0)
        device = z.device
        
        # Safety check for maximum length
        search_max_length = min(max_length or self.max_length, self.max_length * 2)  # Cap at 2x model max
        
        # Initialize hidden state from latent
        h0 = self.latent_to_hidden(z).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        
        # Assert hidden state shapes
        assert h0.shape == (1, batch_size, self.hidden_dim), f"Unexpected h0 shape: {h0.shape}"
        
        # Initialize beams: [batch_size, beam_width, seq_len]
        # Start with padding token (0)
        beams = torch.zeros(batch_size, beam_width, search_max_length, dtype=torch.long, device=device)
        beam_scores = torch.zeros(batch_size, beam_width, device=device)
        beam_scores[:, 1:] = float('-inf')  # Only first beam is active initially
        
        # Track which beams are finished (generated EOS or reached max length)
        finished = torch.zeros(batch_size, beam_width, dtype=torch.bool, device=device)
        
        # Expand hidden states for beam search
        # [1, batch_size, hidden_dim] -> [1, batch_size * beam_width, hidden_dim]
        h = h0.repeat(1, beam_width, 1)
        c = c0.repeat(1, beam_width, 1)
        
        for t in range(search_max_length):
            # Early stopping if all beams are finished
            if finished.all():
                break
            
            # Check for beam collapse (all beams identical)
            if t > 0 and beam_width > 1:
                unique_beams = torch.unique(beams[:, :, :t], dim=1)
                if unique_beams.size(1) == 1:
                    # All beams are identical, stop early
                    break
            
            if t == 0:
                # First step: use start token for all beams
                input_token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
                
                # Run LSTM
                embedded = self.embedding(input_token)
                embedded = self.dropout(embedded)
                lstm_out, (h_new, c_new) = self.lstm(embedded, (h0, c0))
                lstm_out = self.dropout(lstm_out)
                step_logits = self.fc_out(lstm_out).squeeze(1)  # [batch_size, vocab_size]
                
                # Assert shape
                assert step_logits.shape == (batch_size, self.vocab_size), \
                    f"Expected step_logits shape {(batch_size, self.vocab_size)}, got {step_logits.shape}"
                
                # Apply temperature
                step_logits = step_logits / temperature
                log_probs = F.log_softmax(step_logits, dim=-1)
                
                # Get top-k tokens for each batch
                topk_log_probs, topk_indices = log_probs.topk(beam_width, dim=-1)
                
                # Boundary check before indexing
                assert topk_indices.min() >= 0 and topk_indices.max() < self.vocab_size, \
                    f"Token indices out of bounds: min={topk_indices.min()}, max={topk_indices.max()}"
                
                # Initialize beams
                beams[:, :, 0] = topk_indices
                beam_scores = topk_log_probs
                
                # Expand hidden states
                h = h_new.repeat(1, beam_width, 1)
                c = c_new.repeat(1, beam_width, 1)
            else:
                # Subsequent steps: expand each beam
                # Boundary check for token index
                assert t > 0 and t < search_max_length, f"Time step {t} out of bounds [0, {search_max_length})"
                
                # Reshape for batch processing
                # [batch_size, beam_width, 1] -> [batch_size * beam_width, 1]
                input_token = beams[:, :, t-1].view(-1, 1)
                
                # Boundary check for input tokens
                assert input_token.min() >= 0 and input_token.max() < self.vocab_size, \
                    f"Input tokens out of bounds: min={input_token.min()}, max={input_token.max()}"
                
                # Run LSTM
                embedded = self.embedding(input_token)
                embedded = self.dropout(embedded)
                lstm_out, (h, c) = self.lstm(embedded, (h, c))
                lstm_out = self.dropout(lstm_out)
                step_logits = self.fc_out(lstm_out).squeeze(1)  # [batch_size * beam_width, vocab_size]
                
                # Apply temperature
                step_logits = step_logits / temperature
                log_probs = F.log_softmax(step_logits, dim=-1)
                
                # Reshape back to [batch_size, beam_width, vocab_size]
                log_probs = log_probs.view(batch_size, beam_width, -1)
                
                # Add previous scores (broadcasting)
                # [batch_size, beam_width, vocab_size] = [batch_size, beam_width, 1] + [batch_size, beam_width, vocab_size]
                candidate_scores = beam_scores.unsqueeze(-1) + log_probs
                
                # For finished beams, force padding token with no additional score
                finished_mask = finished.unsqueeze(-1).expand(-1, -1, self.vocab_size)
                candidate_scores = torch.where(
                    finished_mask,
                    beam_scores.unsqueeze(-1),  # Keep same score
                    candidate_scores
                )
                
                # Flatten beam and vocab dimensions to get top-k across all candidates
                candidate_scores_flat = candidate_scores.view(batch_size, -1)
                
                # Get top beam_width candidates
                topk_scores, topk_indices_flat = candidate_scores_flat.topk(beam_width, dim=-1)
                
                # Convert flat indices back to beam and vocab indices
                topk_beam_indices = topk_indices_flat // self.vocab_size
                topk_token_indices = topk_indices_flat % self.vocab_size
                
                # Boundary checks for computed indices
                assert topk_beam_indices.min() >= 0 and topk_beam_indices.max() < beam_width, \
                    f"Beam indices out of bounds: min={topk_beam_indices.min()}, max={topk_beam_indices.max()}"
                assert topk_token_indices.min() >= 0 and topk_token_indices.max() < self.vocab_size, \
                    f"Token indices out of bounds: min={topk_token_indices.min()}, max={topk_token_indices.max()}"
                
                # Update beams
                new_beams = torch.zeros_like(beams)
                new_finished = torch.zeros_like(finished)
                
                for b in range(batch_size):
                    for k in range(beam_width):
                        beam_idx = topk_beam_indices[b, k]
                        token_idx = topk_token_indices[b, k]
                        
                        # Boundary check before copying
                        assert 0 <= beam_idx < beam_width, f"Invalid beam_idx {beam_idx}"
                        assert 0 <= token_idx < self.vocab_size, f"Invalid token_idx {token_idx}"
                        
                        # Copy previous beam
                        new_beams[b, k, :t] = beams[b, beam_idx, :t]
                        # Add new token
                        new_beams[b, k, t] = token_idx
                        
                        # Update finished status
                        new_finished[b, k] = finished[b, beam_idx] or (token_idx == 0)
                
                beams = new_beams
                beam_scores = topk_scores
                finished = new_finished
                
                # Reorder hidden states according to selected beams
                h_new = torch.zeros_like(h)
                c_new = torch.zeros_like(c)
                for b in range(batch_size):
                    for k in range(beam_width):
                        beam_idx = topk_beam_indices[b, k]
                        src_idx = b * beam_width + beam_idx
                        dst_idx = b * beam_width + k
                        
                        # Boundary checks
                        assert 0 <= src_idx < h.size(1), f"Invalid src_idx {src_idx}"
                        assert 0 <= dst_idx < h.size(1), f"Invalid dst_idx {dst_idx}"
                        
                        h_new[:, dst_idx, :] = h[:, src_idx, :]
                        c_new[:, dst_idx, :] = c[:, src_idx, :]
                h = h_new
                c = c_new
            
            # Early stopping if all beams are finished
            if finished.all():
                break
        
        # Apply length penalty
        lengths = (beams != 0).sum(dim=-1).float()
        beam_scores = beam_scores / (lengths ** length_penalty + 1e-10)
        
        return beams, beam_scores, lengths.long()


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
        
        # Temporal encoding (always enabled)
        from .temporal import TemporalEncoding
        self.temporal_encoding = TemporalEncoding(
            embed_dim=config.symptom_embed_dim,
            max_length=config.max_symptom_length,
            encoding_type=config.temporal_encoding_type,
            dropout=config.dropout_rate
        )
        
        # Symptom aggregation with attention (always enabled)
        self.symptom_attention = AttentionAggregator(config.symptom_embed_dim, config.dropout_rate)
        
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
        
        # Apply temporal encoding (always enabled)
        symptom_embeds = self.temporal_encoding(symptom_embeds, timestamps)
        
        # Aggregate symptoms with attention (always enabled)
        mask = (symptoms != 0).float()  # [B, L]
        symptom_repr, _ = self.symptom_attention(symptom_embeds, mask)  # [B, D]
        
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
