"""
LAS (Listen-Attend-Spell) Decoder for two-pass ASR.
Works with RNN-T encoder output for rescoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LocationAwareAttention(nn.Module):
    """
    Location-aware attention mechanism.
    """
    
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        attention_dim: int = 256,
        num_filters: int = 32,
        kernel_size: int = 31,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        
        # Projections
        self.encoder_proj = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.decoder_proj = nn.Linear(decoder_dim, attention_dim, bias=False)
        
        # Location features
        self.location_conv = nn.Conv1d(
            1, num_filters,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.location_proj = nn.Linear(num_filters, attention_dim, bias=False)
        
        # Energy
        self.energy = nn.Linear(attention_dim, 1, bias=False)
    
    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        attention_weights_prev: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_hidden: Decoder hidden state (batch, decoder_dim)
            encoder_outputs: Encoder outputs (batch, time, encoder_dim)
            attention_weights_prev: Previous attention weights (batch, time)
            mask: Mask for encoder outputs (batch, time)
            
        Returns:
            context: Context vector (batch, encoder_dim)
            attention_weights: Attention weights (batch, time)
        """
        batch_size, time_steps, _ = encoder_outputs.size()
        
        # Project encoder outputs
        encoder_proj = self.encoder_proj(encoder_outputs)  # (B, T, attention_dim)
        
        # Project decoder hidden
        decoder_proj = self.decoder_proj(decoder_hidden).unsqueeze(1)  # (B, 1, attention_dim)
        
        # Location features
        attention_weights_prev = attention_weights_prev.unsqueeze(1)  # (B, 1, T)
        location_features = self.location_conv(attention_weights_prev)  # (B, num_filters, T)
        location_features = location_features.transpose(1, 2)  # (B, T, num_filters)
        location_proj = self.location_proj(location_features)  # (B, T, attention_dim)
        
        # Compute energy
        energy = self.energy(
            torch.tanh(encoder_proj + decoder_proj + location_proj)
        ).squeeze(-1)  # (B, T)
        
        # Apply mask if provided
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(energy, dim=-1)  # (B, T)
        
        # Compute context vector
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (B, 1, T)
            encoder_outputs  # (B, T, encoder_dim)
        ).squeeze(1)  # (B, encoder_dim)
        
        return context, attention_weights


class LASDecoder(nn.Module):
    """
    LAS Decoder for two-pass rescoring.
    """
    
    def __init__(
        self,
        vocab_size: int,
        encoder_dim: int = 512,
        decoder_hidden: int = 512,
        embedding_dim: int = 256,
        attention_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.decoder_hidden = decoder_hidden
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Attention
        self.attention = LocationAwareAttention(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_hidden,
            attention_dim=attention_dim,
        )
        
        # Decoder LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim + encoder_dim,
            hidden_size=decoder_hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        
        # Output projection
        self.output_proj = nn.Linear(
            decoder_hidden + encoder_dim,
            vocab_size,
        )
    
    def forward_step(
        self,
        input_token: torch.Tensor,
        hidden_state: Tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor,
        attention_weights_prev: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Single step of decoder.
        
        Args:
            input_token: Input token (batch, 1)
            hidden_state: LSTM hidden state
            encoder_outputs: Encoder outputs (batch, time, encoder_dim)
            attention_weights_prev: Previous attention weights (batch, time)
            mask: Encoder mask (batch, time)
            
        Returns:
            logits: Output logits (batch, vocab_size)
            hidden_state: Updated hidden state
            attention_weights: Current attention weights (batch, time)
        """
        # Embed input
        embedded = self.embedding(input_token)  # (B, 1, embedding_dim)
        embedded = self.embedding_dropout(embedded)
        
        # Get decoder hidden state for attention
        decoder_hidden = hidden_state[0][-1]  # (B, decoder_hidden)
        
        # Attention
        context, attention_weights = self.attention(
            decoder_hidden, encoder_outputs, attention_weights_prev, mask
        )
        
        # Concatenate embedding and context
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)  # (B, 1, emb+enc)
        
        # LSTM step
        lstm_output, hidden_state = self.lstm(lstm_input, hidden_state)
        lstm_output = lstm_output.squeeze(1)  # (B, decoder_hidden)
        
        # Output projection
        output_input = torch.cat([lstm_output, context], dim=-1)  # (B, decoder_hidden+enc)
        logits = self.output_proj(output_input)  # (B, vocab_size)
        
        return logits, hidden_state, attention_weights
    
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        encoder_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        sos_id: int = 1,
    ) -> torch.Tensor:
        """
        Forward pass with teacher forcing.
        
        Args:
            encoder_outputs: Encoder outputs (batch, time, encoder_dim)
            encoder_lengths: Encoder output lengths (batch,)
            targets: Target tokens (batch, target_len)
            target_lengths: Target lengths (batch,)
            sos_id: Start-of-sequence token ID
            
        Returns:
            logits: Output logits (batch, target_len, vocab_size)
        """
        batch_size = encoder_outputs.size(0)
        max_time = encoder_outputs.size(1)
        max_target_len = targets.size(1)
        device = encoder_outputs.device
        
        # Initialize hidden state
        hidden_state = (
            torch.zeros(self.num_layers, batch_size, self.decoder_hidden, device=device),
            torch.zeros(self.num_layers, batch_size, self.decoder_hidden, device=device),
        )
        
        # Initialize attention weights
        attention_weights = torch.zeros(batch_size, max_time, device=device)
        attention_weights[:, 0] = 1.0  # Focus on first frame initially
        
        # Create encoder mask
        mask = torch.arange(max_time, device=device).unsqueeze(0) < encoder_lengths.unsqueeze(1)
        
        # Prepend SOS token to targets
        sos = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)
        decoder_inputs = torch.cat([sos, targets[:, :-1]], dim=1)  # Shift right
        
        # Collect outputs
        all_logits = []
        
        for t in range(max_target_len):
            input_token = decoder_inputs[:, t:t+1]  # (B, 1)
            
            logits, hidden_state, attention_weights = self.forward_step(
                input_token, hidden_state, encoder_outputs, attention_weights, mask
            )
            
            all_logits.append(logits.unsqueeze(1))
        
        # Concatenate all logits
        all_logits = torch.cat(all_logits, dim=1)  # (B, target_len, vocab_size)
        
        return all_logits
    
    def decode(
        self,
        encoder_outputs: torch.Tensor,
        encoder_lengths: torch.Tensor,
        max_length: int = 100,
        sos_id: int = 1,
        eos_id: int = 2,
        blank_id: int = 0,
    ) -> list:
        """
        Greedy decoding for inference.
        
        Args:
            encoder_outputs: Encoder outputs (batch, time, encoder_dim)
            encoder_lengths: Encoder output lengths (batch,)
            max_length: Maximum decode length
            sos_id: Start token ID
            eos_id: End token ID
            blank_id: Blank token ID
            
        Returns:
            List of decoded sequences
        """
        batch_size = encoder_outputs.size(0)
        max_time = encoder_outputs.size(1)
        device = encoder_outputs.device
        
        # Initialize hidden state
        hidden_state = (
            torch.zeros(self.num_layers, batch_size, self.decoder_hidden, device=device),
            torch.zeros(self.num_layers, batch_size, self.decoder_hidden, device=device),
        )
        
        # Initialize attention weights
        attention_weights = torch.zeros(batch_size, max_time, device=device)
        attention_weights[:, 0] = 1.0
        
        # Create encoder mask
        mask = torch.arange(max_time, device=device).unsqueeze(0) < encoder_lengths.unsqueeze(1)
        
        # Start with SOS token
        input_token = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)
        
        # Decode
        results = [[] for _ in range(batch_size)]
        finished = [False] * batch_size
        
        for _ in range(max_length):
            logits, hidden_state, attention_weights = self.forward_step(
                input_token, hidden_state, encoder_outputs, attention_weights, mask
            )
            
            # Greedy selection
            pred = logits.argmax(dim=-1)  # (B,)
            
            for b in range(batch_size):
                if not finished[b]:
                    token = pred[b].item()
                    if token == eos_id or token == blank_id:
                        finished[b] = True
                    else:
                        results[b].append(token)
            
            if all(finished):
                break
            
            input_token = pred.unsqueeze(1)  # (B, 1)
        
        return results
