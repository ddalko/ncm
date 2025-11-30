"""
Neural Utterance Confidence Model (NCM)
Based on: Gupta et al., ICASSP 2021
"Neural Utterance Confidence Measure for RNN-Transducers and Two-Pass Models"
"""
import torch
import torch.nn as nn


class NCMModel(nn.Module):
    """
    Neural Confidence Model (NCM) - Binary classifier for ASR hypothesis confidence.
    
    Architecture: 2-layer MLP with 64 hidden units (as described in the paper)
    Input: Concatenated feature vector from RNN-T and LAS
    Output: Sigmoid probability p(accept)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """
        Args:
            input_dim: Dimension of concatenated feature vector
            hidden_dim: Hidden layer dimension (default: 64 as in paper)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 2-layer MLP as described in paper
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: (batch_size, input_dim) - concatenated feature vector
            
        Returns:
            confidence: (batch_size, 1) - probability of acceptance
        """
        return self.mlp(features)
    
    def predict(self, features: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict accept/reject decisions.
        
        Args:
            features: (batch_size, input_dim)
            threshold: Decision threshold (default: 0.5)
            
        Returns:
            decisions: (batch_size,) - 1 for accept, 0 for reject
        """
        with torch.no_grad():
            confidence = self.forward(features)
            decisions = (confidence.squeeze(-1) >= threshold).long()
        return decisions


class NCMFeatureExtractor:
    """
    Feature extractor for NCM from RNN-T and LAS outputs.
    
    Extracts features as described in the paper:
    - RNN-T: Trans (encoder), Pred (prediction net), Joint (joint net logits)
    - LAS: Enc (second-pass encoder), Dec (decoder logits)
    - Beam scores
    """
    
    def __init__(self, top_k: int = 10):
        """
        Args:
            top_k: Number of top logits to extract per timestep
        """
        self.top_k = top_k
    
    def extract_rnnt_features(
        self,
        encoder_outputs: torch.Tensor,
        pred_outputs: torch.Tensor,
        joint_logits: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> dict:
        """
        Extract RNN-T features.
        
        Args:
            encoder_outputs: (B, T, enc_dim) - encoder outputs
            pred_outputs: (B, U, pred_dim) - prediction network outputs
            joint_logits: (B, T, U, vocab) - joint network logits
            feature_lengths: (B,) - actual lengths of encoder outputs
            
        Returns:
            dict with keys:
                'trans': (B, enc_dim) - mean-pooled encoder features
                'pred': (B, pred_dim) - mean-pooled prediction features
                'joint': (B, top_k) - mean of top-K joint logits
        """
        batch_size = encoder_outputs.size(0)
        
        # Trans: Mean-pooled encoder outputs (accounting for actual lengths)
        trans_features = []
        for i in range(batch_size):
            length = feature_lengths[i].item()
            trans_features.append(encoder_outputs[i, :length].mean(dim=0))
        trans_features = torch.stack(trans_features)
        
        # Pred: Mean-pooled prediction network outputs
        pred_features = pred_outputs.mean(dim=1)
        
        # Joint: Top-K logits per timestep, then mean over time
        # Shape: (B, T, U, vocab) -> (B, T, U, top_k)
        top_k_logits, _ = joint_logits.topk(self.top_k, dim=-1)
        # Average over time and label positions: (B, T, U, top_k) -> (B, top_k)
        joint_features = []
        for i in range(batch_size):
            length = feature_lengths[i].item()
            joint_features.append(top_k_logits[i, :length].mean(dim=(0, 1)))
        joint_features = torch.stack(joint_features)
        
        return {
            'trans': trans_features,
            'pred': pred_features,
            'joint': joint_features,
        }
    
    def extract_las_features(
        self,
        las_encoder_outputs: torch.Tensor,
        las_decoder_logits: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> dict:
        """
        Extract LAS features.
        
        Args:
            las_encoder_outputs: (B, T, las_enc_dim) - LAS encoder outputs
            las_decoder_logits: (B, U, vocab) - LAS decoder logits
            feature_lengths: (B,) - actual lengths
            
        Returns:
            dict with keys:
                'enc': (B, las_enc_dim) - mean-pooled LAS encoder features
                'dec': (B, top_k) - mean of top-K decoder logits
        """
        batch_size = las_encoder_outputs.size(0)
        
        # Enc: Mean-pooled LAS encoder outputs
        enc_features = []
        for i in range(batch_size):
            length = feature_lengths[i].item()
            enc_features.append(las_encoder_outputs[i, :length].mean(dim=0))
        enc_features = torch.stack(enc_features)
        
        # Dec: Top-K decoder logits, averaged over sequence
        top_k_logits, _ = las_decoder_logits.topk(self.top_k, dim=-1)
        dec_features = top_k_logits.mean(dim=1)
        
        return {
            'enc': enc_features,
            'dec': dec_features,
        }
    
    def extract_beam_scores(
        self,
        rnnt_beam_scores: torch.Tensor,
        las_beam_scores: torch.Tensor,
    ) -> dict:
        """
        Extract beam search scores.
        
        Args:
            rnnt_beam_scores: (B,) - RNN-T beam log probabilities
            las_beam_scores: (B,) - LAS beam log probabilities
            
        Returns:
            dict with keys:
                'rnnt_score': (B, 1)
                'las_score': (B, 1)
        """
        return {
            'rnnt_score': rnnt_beam_scores.unsqueeze(-1),
            'las_score': las_beam_scores.unsqueeze(-1),
        }
    
    def concatenate_features(
        self,
        rnnt_features: dict,
        las_features: dict,
        beam_scores: dict = None,
    ) -> torch.Tensor:
        """
        Concatenate all features into a single vector.
        
        Args:
            rnnt_features: Dict from extract_rnnt_features
            las_features: Dict from extract_las_features
            beam_scores: Optional dict from extract_beam_scores
            
        Returns:
            features: (B, total_dim) - concatenated feature vector
        """
        feature_list = [
            rnnt_features['trans'],
            rnnt_features['pred'],
            rnnt_features['joint'],
            las_features['enc'],
            las_features['dec'],
        ]
        
        if beam_scores is not None:
            feature_list.extend([
                beam_scores['rnnt_score'],
                beam_scores['las_score'],
            ])
        
        return torch.cat(feature_list, dim=-1)
    
    def get_feature_dim(
        self,
        enc_dim: int,
        pred_dim: int,
        las_enc_dim: int,
        include_beam_scores: bool = True,
    ) -> int:
        """
        Calculate total feature dimension.
        
        Args:
            enc_dim: Encoder output dimension
            pred_dim: Prediction network output dimension
            las_enc_dim: LAS encoder output dimension
            include_beam_scores: Whether to include beam scores
            
        Returns:
            total_dim: Total concatenated feature dimension
        """
        total_dim = (
            enc_dim +           # trans
            pred_dim +          # pred
            self.top_k +        # joint
            las_enc_dim +       # enc
            self.top_k          # dec
        )
        
        if include_beam_scores:
            total_dim += 2  # rnnt_score + las_score
        
        return total_dim
