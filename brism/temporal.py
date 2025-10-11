"""
Temporal encoding for symptom progression modeling.

Handles temporal information about when symptoms appear to capture
disease progression patterns.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class TemporalEncoding(nn.Module):
    """
    Temporal encoding layer that adds positional/temporal information to symptom embeddings.
    
    Supports two modes:
    1. Positional encoding (Transformer-style) - for relative ordering
    2. Timestamp encoding - for absolute time values
    """
    
    def __init__(
        self, 
        embed_dim: int,
        max_length: int = 100,
        encoding_type: str = 'positional',
        dropout: float = 0.1
    ):
        """
        Args:
            embed_dim: Dimension of embeddings
            max_length: Maximum sequence length
            encoding_type: 'positional' or 'timestamp'
            dropout: Dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.encoding_type = encoding_type
        self.dropout = nn.Dropout(dropout)
        
        if encoding_type == 'positional':
            # Create fixed positional encodings (Transformer-style)
            position = torch.arange(max_length).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, embed_dim, 2) * 
                                (-math.log(10000.0) / embed_dim))
            
            pe = torch.zeros(max_length, embed_dim)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            self.register_buffer('pe', pe)
            
        elif encoding_type == 'timestamp':
            # Learnable projection for timestamps
            self.time_projection = nn.Linear(1, embed_dim)
        else:
            raise ValueError(f"Unknown encoding_type: {encoding_type}")
    
    def forward(
        self, 
        embeddings: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Add temporal encoding to embeddings.
        
        Args:
            embeddings: Token embeddings [batch_size, seq_len, embed_dim]
            timestamps: Optional timestamps [batch_size, seq_len] or [batch_size, seq_len, 1]
                       For positional encoding, these are ignored
                       For timestamp encoding, these are the actual time values
        
        Returns:
            Embeddings with temporal encoding [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = embeddings.shape
        
        if self.encoding_type == 'positional':
            # Add fixed positional encodings
            positional_encoding = self.pe[:seq_len, :].unsqueeze(0)
            embeddings = embeddings + positional_encoding
            
        elif self.encoding_type == 'timestamp':
            if timestamps is None:
                # Default to positional if no timestamps provided
                positions = torch.arange(seq_len, device=embeddings.device).float()
                timestamps = positions.unsqueeze(0).expand(batch_size, -1)
            
            # Ensure timestamps have correct shape
            if timestamps.dim() == 2:
                timestamps = timestamps.unsqueeze(-1)  # [B, L, 1]
            
            # Validate timestamp values to prevent overflow or extreme values
            # Check for reasonable range (e.g., timestamps in years should be < 10000)
            max_timestamp = timestamps.abs().max().item()
            if max_timestamp > 1e6:  # Reasonable upper bound
                import warnings
                warnings.warn(
                    f"Timestamp values are very large (max: {max_timestamp:.2e}). "
                    f"This may cause overflow or extreme encoding values. "
                    f"Consider normalizing timestamps to a reasonable range.",
                    UserWarning
                )
            
            # Project timestamps to embedding dimension
            time_encoding = self.time_projection(timestamps)
            embeddings = embeddings + time_encoding
        
        return self.dropout(embeddings)


class TemporalSymptomEncoder(nn.Module):
    """
    Symptom encoder that processes temporal symptom sequences.
    
    Combines symptom embeddings with temporal information before encoding.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        latent_dim: int,
        max_length: int = 50,
        encoding_type: str = 'positional',
        dropout: float = 0.2,
        use_lstm: bool = False
    ):
        """
        Args:
            vocab_size: Size of symptom vocabulary
            embed_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
            latent_dim: Latent space dimension
            max_length: Maximum sequence length
            encoding_type: Type of temporal encoding
            dropout: Dropout rate
            use_lstm: Use LSTM for sequence processing (otherwise attention pooling)
        """
        super().__init__()
        self.use_lstm = use_lstm
        
        # Symptom embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Temporal encoding
        self.temporal_encoding = TemporalEncoding(
            embed_dim=embed_dim,
            max_length=max_length,
            encoding_type=encoding_type,
            dropout=dropout
        )
        
        if use_lstm:
            # LSTM for sequence processing
            self.lstm = nn.LSTM(
                embed_dim, hidden_dim, 
                batch_first=True, 
                bidirectional=True
            )
            lstm_output_dim = hidden_dim * 2
        else:
            # Attention pooling
            self.attention = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
            lstm_output_dim = embed_dim
        
        # Map to latent distribution
        self.fc1 = nn.Linear(lstm_output_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        symptom_ids: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode temporal symptom sequence to latent distribution.
        
        Args:
            symptom_ids: Symptom token IDs [batch_size, seq_len]
            timestamps: Optional timestamps [batch_size, seq_len]
            mask: Optional padding mask [batch_size, seq_len] (1 for valid, 0 for padding)
        
        Returns:
            mu: Latent mean [batch_size, latent_dim]
            logvar: Latent log variance [batch_size, latent_dim]
        """
        # Embed symptoms
        embeddings = self.embedding(symptom_ids)  # [B, L, D]
        
        # Add temporal encoding
        embeddings = self.temporal_encoding(embeddings, timestamps)
        
        # Create mask if not provided
        if mask is None:
            mask = (symptom_ids != 0).float()
        
        # Process sequence
        if self.use_lstm:
            # LSTM processing
            lstm_out, _ = self.lstm(embeddings)  # [B, L, 2*H]
            
            # Masked average pooling
            mask_expanded = mask.unsqueeze(-1).expand_as(lstm_out)
            masked_out = lstm_out * mask_expanded
            sequence_repr = masked_out.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
        else:
            # Attention pooling
            attention_scores = self.attention(embeddings).squeeze(-1)  # [B, L]
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            attention_weights = torch.softmax(attention_scores, dim=-1)  # [B, L]
            
            sequence_repr = torch.bmm(
                attention_weights.unsqueeze(1), 
                embeddings
            ).squeeze(1)  # [B, D]
        
        # Map to latent distribution
        h = torch.relu(self.fc1(sequence_repr))
        h = self.dropout(h)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
