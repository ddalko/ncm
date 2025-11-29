"""
RNN-Transducer model implementation.
Includes Encoder, Prediction Network, and Joint Network.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class RNNTEncoder(nn.Module):
    """
    Encoder for RNN-T (shared with LAS).
    Uses stacked LSTM layers.
    """
    
    def __init__(
        self,
        input_dim: int = 80,
        hidden_size: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        
        self.output_dim = hidden_size * 2 if bidirectional else hidden_size
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features (batch, time, input_dim)
            lengths: Actual lengths before padding (batch,)
            
        Returns:
            output: Encoded features (batch, time, hidden_size)
            output_lengths: Output sequence lengths
        """
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM forward
        packed_output, _ = self.lstm(packed)
        
        # Unpack
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )
        
        return output, output_lengths.to(x.device)


class RNNTPredictionNetwork(nn.Module):
    """
    Prediction network for RNN-T.
    Processes previous labels.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_size: int = 512,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
    
    def forward(
        self,
        labels: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            labels: Previous labels (batch, label_len)
            lengths: Actual label lengths (batch,)
            
        Returns:
            output: Prediction network output (batch, label_len, hidden_size)
        """
        # Embed labels
        embedded = self.embedding(labels)
        
        if lengths is not None:
            # Pack padded sequence
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )
        else:
            output, _ = self.lstm(embedded)
        
        return output


class RNNTJointNetwork(nn.Module):
    """
    Joint network for RNN-T.
    Combines encoder and prediction network outputs.
    """
    
    def __init__(
        self,
        encoder_dim: int,
        pred_dim: int,
        joint_dim: int = 512,
        vocab_size: int = 300,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.pred_dim = pred_dim
        self.joint_dim = joint_dim
        self.vocab_size = vocab_size
        
        # Project encoder and prediction outputs
        self.encoder_proj = nn.Linear(encoder_dim, joint_dim)
        self.pred_proj = nn.Linear(pred_dim, joint_dim)
        
        # Final output projection
        self.output = nn.Sequential(
            nn.Tanh(),
            nn.Linear(joint_dim, vocab_size),
        )
    
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        pred_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_outputs: Encoder outputs (batch, time, encoder_dim)
            pred_outputs: Prediction network outputs (batch, label_len, pred_dim)
            
        Returns:
            logits: Joint network logits (batch, time, label_len, vocab_size)
        """
        # Project encoder and prediction outputs
        enc = self.encoder_proj(encoder_outputs)  # (B, T, joint_dim)
        pred = self.pred_proj(pred_outputs)  # (B, U, joint_dim)
        
        # Add time and label dimensions for broadcasting
        enc = enc.unsqueeze(2)  # (B, T, 1, joint_dim)
        pred = pred.unsqueeze(1)  # (B, 1, U, joint_dim)
        
        # Combine
        joint = enc + pred  # (B, T, U, joint_dim)
        
        # Output projection
        logits = self.output(joint)  # (B, T, U, vocab_size)
        
        return logits


class RNNTransducer(nn.Module):
    """
    Complete RNN-Transducer model.
    """
    
    def __init__(
        self,
        input_dim: int = 80,
        vocab_size: int = 300,
        encoder_hidden: int = 512,
        encoder_layers: int = 3,
        pred_hidden: int = 512,
        pred_layers: int = 1,
        joint_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        
        self.encoder = RNNTEncoder(
            input_dim=input_dim,
            hidden_size=encoder_hidden,
            num_layers=encoder_layers,
            dropout=dropout,
            bidirectional=False,
        )
        
        self.prediction = RNNTPredictionNetwork(
            vocab_size=vocab_size,
            hidden_size=pred_hidden,
            num_layers=pred_layers,
            dropout=dropout,
        )
        
        self.joint = RNNTJointNetwork(
            encoder_dim=self.encoder.output_dim,
            pred_dim=pred_hidden,
            joint_dim=joint_dim,
            vocab_size=vocab_size,
        )
    
    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        labels: torch.Tensor,
        label_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: Input features (batch, time, input_dim)
            feature_lengths: Feature lengths (batch,)
            labels: Target labels (batch, label_len)
            label_lengths: Label lengths (batch,)
            
        Returns:
            logits: Joint network logits (batch, time, label_len, vocab_size)
        """
        # Encode
        encoder_outputs, _ = self.encoder(features, feature_lengths)
        
        # Prediction network
        pred_outputs = self.prediction(labels, label_lengths)
        
        # Joint network
        logits = self.joint(encoder_outputs, pred_outputs)
        
        return logits
    
    def recognize(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        blank_id: int = 0,
        max_symbols: int = 100,
    ) -> list:
        """
        Greedy decoding for inference.
        
        Args:
            features: Input features (batch, time, input_dim)
            feature_lengths: Feature lengths (batch,)
            blank_id: Blank token ID
            max_symbols: Maximum number of symbols to emit
            
        Returns:
            List of decoded sequences (one per batch item)
        """
        self.eval()
        with torch.no_grad():
            # Encode
            encoder_outputs, output_lengths = self.encoder(features, feature_lengths)
            
            batch_size = encoder_outputs.size(0)
            results = []
            
            for b in range(batch_size):
                # Greedy decoding for each item in batch
                hyp = []
                pred_input = torch.tensor([[blank_id]], device=features.device)
                
                for t in range(output_lengths[b]):
                    enc_out = encoder_outputs[b:b+1, t:t+1, :]  # (1, 1, encoder_dim)
                    pred_out = self.prediction(pred_input)  # (1, 1, pred_dim)
                    
                    logits = self.joint(enc_out, pred_out)  # (1, 1, 1, vocab_size)
                    logits = logits.squeeze(0).squeeze(0).squeeze(0)  # (vocab_size,)
                    
                    pred = logits.argmax().item()
                    
                    if pred != blank_id:
                        hyp.append(pred)
                        pred_input = torch.tensor([[pred]], device=features.device)
                        
                        if len(hyp) >= max_symbols:
                            break
                
                results.append(hyp)
            
            return results
