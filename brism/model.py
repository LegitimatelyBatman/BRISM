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
    gumbel_temperature: float = 1.0  # Temperature for Gumbel-softmax in cycle consistency
    use_hard_gumbel: bool = False  # If True, use hard gumbel (one-hot) after softmax

    # Token indices
    pad_token_id: int = 0  # Padding token ID for symptom sequences
    eos_token_id: int = 0  # End-of-sequence token ID (defaults to pad_token_id)
    
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

        # Validate gumbel temperature
        if self.gumbel_temperature <= 0.0:
            raise ValueError(f"gumbel_temperature must be positive, got {self.gumbel_temperature}")

        # Validate token IDs
        if self.pad_token_id < 0 or self.pad_token_id >= self.symptom_vocab_size:
            raise ValueError(f"pad_token_id ({self.pad_token_id}) must be in range [0, {self.symptom_vocab_size})")
        if self.eos_token_id < 0 or self.eos_token_id >= self.symptom_vocab_size:
            raise ValueError(f"eos_token_id ({self.eos_token_id}) must be in range [0, {self.symptom_vocab_size})")


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
            # Start with padding/start token (typically 0, but configurable)
            # Note: config not available in SequenceDecoder, so we use hardcoded 0
            # This should be passed as parameter if custom start token is needed
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
        max_length: Optional[int] = None,
        max_batch_size: Optional[int] = None,
        eos_token_id: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate sequences using beam search.
        
        Args:
            z: Latent representation [batch_size, latent_dim]
            beam_width: Number of beams to keep
            temperature: Temperature for softmax (higher = more diverse)
            length_penalty: Penalty for sequence length (>1 favors longer sequences)
            max_length: Maximum sequence length (defaults to self.max_length)
            max_batch_size: Maximum batch size to process at once (for memory efficiency).
                          If None, processes all batches together. If set, processes in chunks.
            eos_token_id: End-of-sequence token ID (default=0, padding token)
            
        Returns:
            sequences: Top beam_width sequences [batch_size, beam_width, seq_len]
            scores: Log probabilities for each sequence [batch_size, beam_width]
            lengths: Actual lengths of sequences [batch_size, beam_width]
        """
        # If max_batch_size is set and batch_size exceeds it, process in chunks
        batch_size = z.size(0)
        
        if max_batch_size is not None and batch_size > max_batch_size:
            # Process in chunks to reduce memory usage
            all_sequences = []
            all_scores = []
            all_lengths = []
            
            for i in range(0, batch_size, max_batch_size):
                batch_end = min(i + max_batch_size, batch_size)
                z_chunk = z[i:batch_end]
                
                # Recursively call beam_search on chunk (without max_batch_size to avoid infinite recursion)
                chunk_sequences, chunk_scores, chunk_lengths = self.beam_search(
                    z_chunk, beam_width, temperature, length_penalty, max_length,
                    max_batch_size=None, eos_token_id=eos_token_id
                )
                
                all_sequences.append(chunk_sequences)
                all_scores.append(chunk_scores)
                all_lengths.append(chunk_lengths)
            
            # Concatenate results
            sequences = torch.cat(all_sequences, dim=0)
            scores = torch.cat(all_scores, dim=0)
            lengths = torch.cat(all_lengths, dim=0)
            
            return sequences, scores, lengths
        
        # Input validation
        if z.dim() != 2:
            raise ValueError(f"Expected z to be 2D tensor [batch_size, latent_dim], got shape {z.shape}")
        if beam_width <= 0:
            raise ValueError(f"beam_width must be positive, got {beam_width}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        
        batch_size = z.size(0)
        device = z.device
        
        # Safety check for maximum length
        search_max_length = min(max_length or self.max_length, self.max_length * 2)  # Cap at 2x model max
        
        # Validate memory requirements to prevent OOM errors
        # Estimate total memory needed (in number of float32 elements):
        # - Beam storage: batch_size * beam_width * max_length
        # - Hidden states: batch_size * beam_width * hidden_dim * 2 (h and c)
        # - Logits: batch_size * beam_width * vocab_size
        # - Candidate scores: batch_size * beam_width * vocab_size
        beam_storage = batch_size * beam_width * search_max_length
        hidden_states = batch_size * beam_width * self.hidden_dim * 2
        logit_storage = batch_size * beam_width * self.vocab_size
        candidate_scores = batch_size * beam_width * self.vocab_size

        estimated_memory = beam_storage + hidden_states + logit_storage + candidate_scores
        memory_limit = 500_000_000  # 500 million elements (~2GB for float32)

        if estimated_memory > memory_limit:
            raise ValueError(
                f"Beam search memory requirements too large:\n"
                f"  Batch size: {batch_size}\n"
                f"  Beam width: {beam_width}\n"
                f"  Max length: {search_max_length}\n"
                f"  Hidden dim: {self.hidden_dim}\n"
                f"  Vocab size: {self.vocab_size}\n"
                f"  Estimated memory: {estimated_memory:,} elements (~{estimated_memory * 4 / 1e9:.2f}GB)\n"
                f"  Limit: {memory_limit:,} elements (~{memory_limit * 4 / 1e9:.2f}GB)\n"
                f"Reduce batch_size, beam_width, or max_length, or use max_batch_size parameter."
            )
        
        # Initialize hidden state from latent
        h0 = self.latent_to_hidden(z).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        
        # Validate hidden state shapes
        if h0.shape != (1, batch_size, self.hidden_dim):
            raise ValueError(f"Unexpected h0 shape: {h0.shape}, expected {(1, batch_size, self.hidden_dim)}")
        
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
                
                # Validate shape
                if step_logits.shape != (batch_size, self.vocab_size):
                    raise ValueError(f"Expected step_logits shape {(batch_size, self.vocab_size)}, got {step_logits.shape}")
                
                # Apply temperature
                step_logits = step_logits / temperature
                log_probs = F.log_softmax(step_logits, dim=-1)
                
                # Get top-k tokens for each batch
                topk_log_probs, topk_indices = log_probs.topk(beam_width, dim=-1)
                
                # Boundary check before indexing
                if topk_indices.min() < 0 or topk_indices.max() >= self.vocab_size:
                    raise ValueError(f"Token indices out of bounds: min={topk_indices.min()}, max={topk_indices.max()}, vocab_size={self.vocab_size}")
                
                # Initialize beams
                beams[:, :, 0] = topk_indices
                beam_scores = topk_log_probs
                
                # Expand hidden states
                h = h_new.repeat(1, beam_width, 1)
                c = c_new.repeat(1, beam_width, 1)
            else:
                # Subsequent steps: expand each beam
                # Boundary check for token index
                if t <= 0 or t >= search_max_length:
                    raise ValueError(f"Time step {t} out of bounds [0, {search_max_length})")
                
                # Reshape for batch processing
                # [batch_size, beam_width, 1] -> [batch_size * beam_width, 1]
                input_token = beams[:, :, t-1].view(-1, 1)
                
                # Boundary check for input tokens
                if input_token.min() < 0 or input_token.max() >= self.vocab_size:
                    raise ValueError(f"Input tokens out of bounds: min={input_token.min()}, max={input_token.max()}, vocab_size={self.vocab_size}")
                
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
                if topk_beam_indices.min() < 0 or topk_beam_indices.max() >= beam_width:
                    raise ValueError(f"Beam indices out of bounds: min={topk_beam_indices.min()}, max={topk_beam_indices.max()}, beam_width={beam_width}")
                if topk_token_indices.min() < 0 or topk_token_indices.max() >= self.vocab_size:
                    raise ValueError(f"Token indices out of bounds: min={topk_token_indices.min()}, max={topk_token_indices.max()}, vocab_size={self.vocab_size}")
                
                # Update beams
                new_beams = torch.zeros_like(beams)
                new_finished = torch.zeros_like(finished)
                
                for b in range(batch_size):
                    for k in range(beam_width):
                        beam_idx = topk_beam_indices[b, k]
                        token_idx = topk_token_indices[b, k]
                        
                        # Boundary check before copying
                        if not (0 <= beam_idx < beam_width):
                            raise ValueError(f"Invalid beam_idx {beam_idx}, must be in [0, {beam_width})")
                        if not (0 <= token_idx < self.vocab_size):
                            raise ValueError(f"Invalid token_idx {token_idx}, must be in [0, {self.vocab_size})")
                        
                        # Copy previous beam
                        new_beams[b, k, :t] = beams[b, beam_idx, :t]
                        # Add new token
                        new_beams[b, k, t] = token_idx

                        # Update finished status (beam finishes when EOS token is generated)
                        new_finished[b, k] = finished[b, beam_idx] or (token_idx == eos_token_id)
                
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
                        if not (0 <= src_idx < h.size(1)):
                            raise ValueError(f"Invalid src_idx {src_idx}, must be in [0, {h.size(1)})")
                        if not (0 <= dst_idx < h.size(1)):
                            raise ValueError(f"Invalid dst_idx {dst_idx}, must be in [0, {h.size(1)})")
                        
                        h_new[:, dst_idx, :] = h[:, src_idx, :]
                        c_new[:, dst_idx, :] = c[:, src_idx, :]
                h = h_new
                c = c_new
            
            # Early stopping if all beams are finished
            if finished.all():
                break
        
        # Apply length penalty
        # Note: Length calculation assumes padding token is 0
        # If using custom padding tokens, this may need adjustment
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

        # Temperature parameter for calibration (learnable, but constrained)
        self.temperature = nn.Parameter(torch.ones(1) * config.temperature)
        # Temperature bounds to prevent extreme values
        self.register_buffer('temperature_min', torch.tensor(0.01))
        self.register_buffer('temperature_max', torch.tensor(10.0))
        
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
        
    def get_clamped_temperature(self) -> torch.Tensor:
        """
        Get temperature clamped to safe range to prevent numerical instability.

        Returns:
            Clamped temperature value
        """
        return torch.clamp(self.temperature, self.temperature_min, self.temperature_max)

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
        # Create mask: 1 for real tokens, 0 for padding
        mask = (symptoms != self.config.pad_token_id).float()  # [B, L]
        symptom_repr, _ = self.symptom_attention(symptom_embeds, mask)  # [B, D]
        
        # Encode to latent
        mu, logvar = self.symptom_encoder(symptom_repr)
        z = self.reparameterize(mu, logvar)

        # Decode to ICD with temperature scaling (clamped for safety)
        icd_logits = self.icd_decoder(z) / self.get_clamped_temperature()

        return icd_logits, mu, logvar

    def forward(
        self,
        symptoms: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience forward method that matches ``nn.Module`` expectations.

        This enables utilities such as :func:`torch.onnx.export` that expect a
        ``forward`` implementation while keeping :meth:`forward_path` as the
        detailed entry point used throughout the codebase.

        Args:
            symptoms: Symptom token IDs ``[batch_size, seq_len]``.
            timestamps: Optional timestamps ``[batch_size, seq_len]``.

        Returns:
            Tuple containing:
                * ``icd_probs`` – probability distribution over ICD codes
                  ``[batch_size, icd_vocab_size]``.
                * ``latent_mu`` – latent mean from the encoder
                  ``[batch_size, latent_dim]``.
        """
        icd_logits, mu, _ = self.forward_path(symptoms, timestamps)
        icd_probs = F.softmax(icd_logits, dim=-1)
        return icd_probs, mu

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

        Uses Gumbel-softmax for differentiable sampling to maintain gradient flow.

        Args:
            symptoms: Input symptom sequence [batch_size, seq_len]
            target_symptoms: Target for reconstruction [batch_size, seq_len]

        Returns:
            Tuple of intermediate and final outputs
        """
        # Forward: symptoms -> ICD
        icd_logits, mu1, logvar1 = self.forward_path(symptoms)

        # Get predicted ICD using Gumbel-softmax for differentiable sampling
        if self.training:
            # Use Gumbel-softmax during training for gradient flow
            icd_soft = F.gumbel_softmax(
                icd_logits,
                tau=self.config.gumbel_temperature,
                hard=self.config.use_hard_gumbel,
                dim=-1
            )
            # Get soft ICD embedding: weighted sum of all ICD embeddings
            icd_embed_soft = torch.matmul(icd_soft, self.icd_embedding.weight)
        else:
            # Use hard decision during evaluation
            icd_pred = icd_logits.argmax(dim=-1)
            icd_embed_soft = self.icd_embedding(icd_pred)

        # Encode soft ICD embedding to latent
        mu2, logvar2 = self.icd_encoder(icd_embed_soft)
        z = self.reparameterize(mu2, logvar2)

        # Decode to symptoms
        symptom_logits = self.symptom_decoder(z, target_symptoms)

        return symptom_logits, icd_logits, mu1, logvar1, mu2, logvar2
    
    def cycle_reverse(self, icd_codes: torch.Tensor, target_symptoms: Optional[torch.Tensor] = None) -> Tuple:
        """
        Cycle: ICD -> latent -> symptoms -> latent' -> ICD'.

        Uses Gumbel-softmax for differentiable sampling to maintain gradient flow.

        Args:
            icd_codes: Input ICD codes [batch_size]
            target_symptoms: Target symptoms for intermediate step [batch_size, seq_len]

        Returns:
            Tuple of intermediate and final outputs
        """
        # Reverse: ICD -> symptoms
        symptom_logits, mu1, logvar1 = self.reverse_path(icd_codes, target_symptoms)

        # Get predicted symptoms using Gumbel-softmax for differentiable sampling
        # symptom_logits shape: [batch_size, seq_len, vocab_size]
        batch_size, seq_len, vocab_size = symptom_logits.shape

        if self.training:
            # Use Gumbel-softmax during training for gradient flow
            # Reshape to apply gumbel_softmax over vocab dimension
            symptom_soft = F.gumbel_softmax(
                symptom_logits.view(-1, vocab_size),
                tau=self.config.gumbel_temperature,
                hard=self.config.use_hard_gumbel,
                dim=-1
            ).view(batch_size, seq_len, vocab_size)

            # Get soft symptom embeddings: weighted sum of all symptom embeddings
            symptom_embeds_soft = torch.matmul(symptom_soft, self.symptom_embedding.weight)
        else:
            # Use hard decision during evaluation
            symptom_pred = symptom_logits.argmax(dim=-1)
            symptom_embeds_soft = self.symptom_embedding(symptom_pred)

        # Apply temporal encoding
        symptom_embeds_soft = self.temporal_encoding(symptom_embeds_soft, timestamps=None)

        # Aggregate symptoms with attention
        mask = torch.ones(batch_size, seq_len, device=symptom_logits.device)
        symptom_repr, _ = self.symptom_attention(symptom_embeds_soft, mask)

        # Encode to latent
        mu2, logvar2 = self.symptom_encoder(symptom_repr)
        z = self.reparameterize(mu2, logvar2)

        # Decode to ICD with temperature scaling (clamped for safety)
        icd_logits = self.icd_decoder(z) / self.get_clamped_temperature()

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

        if n_samples < 1:
            raise ValueError(f"n_samples must be at least 1, got {n_samples}")

        # Enable dropout while preserving the caller's mode
        was_training = self.training
        
        try:
            self.train()

            predictions = []
            with torch.no_grad():
                for _ in range(n_samples):
                    icd_logits, _, _ = self.forward_path(symptoms)
                    probs = F.softmax(icd_logits, dim=-1)
                    predictions.append(probs)

            predictions = torch.stack(predictions)  # [n_samples, batch_size, icd_vocab_size]
            mean_probs = predictions.mean(dim=0)
            std_probs = predictions.std(dim=0, unbiased=False)

            return mean_probs, std_probs
        finally:
            # Restore previous training mode even if exception occurs
            if not was_training:
                self.eval()
