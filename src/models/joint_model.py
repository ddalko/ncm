"""
Joint training model combining RNN-T and LAS.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from .rnnt import RNNTEncoder, RNNTPredictionNetwork, RNNTJointNetwork
from .las_decoder import LASDecoder


class JointASRModel(nn.Module):
    """
    Joint RNN-T + LAS model for two-pass ASR.
    Shares encoder between RNN-T and LAS.
    """
    
    def __init__(
        self,
        input_dim: int = 80,
        vocab_size: int = 300,
        # Shared encoder config
        encoder_hidden: int = 512,
        encoder_layers: int = 3,
        # RNN-T config
        pred_hidden: int = 512,
        pred_layers: int = 1,
        joint_dim: int = 512,
        # LAS config
        las_encoder_hidden: int = 512,
        las_decoder_hidden: int = 512,
        las_embedding_dim: int = 256,
        las_attention_dim: int = 256,
        # General
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        
        # Shared encoder (used by both RNN-T and LAS)
        self.shared_encoder = RNNTEncoder(
            input_dim=input_dim,
            hidden_size=encoder_hidden,
            num_layers=encoder_layers,
            dropout=dropout,
            bidirectional=False,
        )
        
        # RNN-T specific components
        self.rnnt_prediction = RNNTPredictionNetwork(
            vocab_size=vocab_size,
            hidden_size=pred_hidden,
            num_layers=pred_layers,
            dropout=dropout,
        )
        
        self.rnnt_joint = RNNTJointNetwork(
            encoder_dim=self.shared_encoder.output_dim,
            pred_dim=pred_hidden,
            joint_dim=joint_dim,
            vocab_size=vocab_size,
        )
        
        # LAS specific components
        # Additional encoder layer for LAS (bidirectional)
        self.las_encoder = nn.LSTM(
            input_size=self.shared_encoder.output_dim,
            hidden_size=las_encoder_hidden,
            num_layers=1,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        
        self.las_decoder = LASDecoder(
            vocab_size=vocab_size,
            encoder_dim=las_encoder_hidden * 2,  # Bidirectional
            decoder_hidden=las_decoder_hidden,
            embedding_dim=las_embedding_dim,
            attention_dim=las_attention_dim,
            dropout=dropout,
        )
    
    def forward_rnnt(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        labels: torch.Tensor,
        label_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for RNN-T.
        
        Returns:
            logits: RNN-T joint network logits (batch, time, label_len, vocab_size)
        """
        # Shared encoder
        encoder_outputs, encoder_lengths = self.shared_encoder(features, feature_lengths)
        
        # Prediction network
        pred_outputs = self.rnnt_prediction(labels, label_lengths)
        
        # Joint network
        logits = self.rnnt_joint(encoder_outputs, pred_outputs)
        
        return logits
    
    def forward_las(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        sos_id: int = 1,
    ) -> torch.Tensor:
        """
        Forward pass for LAS.
        
        Returns:
            logits: LAS decoder logits (batch, target_len, vocab_size)
        """
        # Shared encoder
        shared_enc_outputs, shared_enc_lengths = self.shared_encoder(features, feature_lengths)
        
        # LAS-specific encoder (bidirectional)
        packed = nn.utils.rnn.pack_padded_sequence(
            shared_enc_outputs,
            shared_enc_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        
        packed_output, _ = self.las_encoder(packed)
        
        las_enc_outputs, las_enc_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )
        
        # LAS decoder with teacher forcing
        logits = self.las_decoder(
            encoder_outputs=las_enc_outputs,
            encoder_lengths=las_enc_lengths.to(features.device),
            targets=targets,
            target_lengths=target_lengths,
            sos_id=sos_id,
        )
        
        return logits
    
    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        labels: torch.Tensor,
        label_lengths: torch.Tensor,
        sos_id: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for both RNN-T and LAS.
        
        Returns:
            Dictionary containing:
                - rnnt_logits: RNN-T logits
                - las_logits: LAS logits
        """
        # RNN-T forward
        rnnt_logits = self.forward_rnnt(features, feature_lengths, labels, label_lengths)
        
        # LAS forward
        las_logits = self.forward_las(features, feature_lengths, labels, label_lengths, sos_id)
        
        return {
            'rnnt_logits': rnnt_logits,
            'las_logits': las_logits,
        }
    
    def recognize_rnnt(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        blank_id: int = 0,
        max_symbols: int = 100,
    ) -> list:
        """
        Greedy decoding with RNN-T (first pass).
        """
        self.eval()
        with torch.no_grad():
            # Shared encoder
            encoder_outputs, output_lengths = self.shared_encoder(features, feature_lengths)
            
            batch_size = encoder_outputs.size(0)
            results = []
            
            for b in range(batch_size):
                hyp = []
                pred_input = torch.tensor([[blank_id]], device=features.device)
                
                for t in range(output_lengths[b]):
                    enc_out = encoder_outputs[b:b+1, t:t+1, :]
                    pred_out = self.rnnt_prediction(pred_input)
                    
                    logits = self.rnnt_joint(enc_out, pred_out)
                    logits = logits.squeeze(0).squeeze(0).squeeze(0)
                    
                    pred = logits.argmax().item()
                    
                    if pred != blank_id:
                        hyp.append(pred)
                        pred_input = torch.tensor([[pred]], device=features.device)
                        
                        if len(hyp) >= max_symbols:
                            break
                
                results.append(hyp)
            
            return results
    
    def recognize_two_pass(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        blank_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
        max_symbols: int = 100,
    ) -> Tuple[list, list]:
        """
        Two-pass decoding: RNN-T + LAS rescoring.
        
        Returns:
            rnnt_results: RNN-T first-pass results
            las_results: LAS rescored results
        """
        self.eval()
        with torch.no_grad():
            # First pass: RNN-T
            rnnt_results = self.recognize_rnnt(features, feature_lengths, blank_id, max_symbols)
            
            # Second pass: LAS rescoring
            # Get shared encoder outputs
            shared_enc_outputs, shared_enc_lengths = self.shared_encoder(features, feature_lengths)
            
            # LAS encoder
            packed = nn.utils.rnn.pack_padded_sequence(
                shared_enc_outputs,
                shared_enc_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            
            packed_output, _ = self.las_encoder(packed)
            
            las_enc_outputs, las_enc_lengths = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )
            
            # LAS decode
            las_results = self.las_decoder.decode(
                encoder_outputs=las_enc_outputs,
                encoder_lengths=las_enc_lengths.to(features.device),
                max_length=max_symbols,
                sos_id=sos_id,
                eos_id=eos_id,
                blank_id=blank_id,
            )
            
            return rnnt_results, las_results


if __name__ == '__main__':
    # Test model
    model = JointASRModel(
        input_dim=80,
        vocab_size=300,
        encoder_hidden=512,
        encoder_layers=3,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 2
    time_steps = 100
    label_len = 20
    
    features = torch.randn(batch_size, time_steps, 80)
    feature_lengths = torch.tensor([100, 80])
    labels = torch.randint(0, 300, (batch_size, label_len))
    label_lengths = torch.tensor([20, 15])
    
    outputs = model(features, feature_lengths, labels, label_lengths)
    
    print(f"RNN-T logits shape: {outputs['rnnt_logits'].shape}")
    print(f"LAS logits shape: {outputs['las_logits'].shape}")
