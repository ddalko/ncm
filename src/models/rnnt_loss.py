"""
RNN-T Loss function.
Uses torchaudio's RNN-T loss implementation.
"""

import torch
import torch.nn as nn
import torchaudio


class RNNTLoss(nn.Module):
    """
    RNN-Transducer loss function.
    """
    
    def __init__(self, blank: int = 0, reduction: str = 'mean'):
        """
        Args:
            blank: Blank token ID
            reduction: Reduction method ('none', 'mean', 'sum')
        """
        super().__init__()
        self.blank = blank
        self.reduction = reduction
        
        # Use torchaudio's RNN-T loss
        self.loss_fn = torchaudio.transforms.RNNTLoss(blank=blank, reduction=reduction)
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        logit_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute RNN-T loss.
        
        Args:
            logits: Joint network outputs (batch, time, target_len+1, vocab_size)
            targets: Target labels (batch, max_target_len)
            logit_lengths: Actual time steps before padding (batch,)
            target_lengths: Actual target lengths before padding (batch,)
            
        Returns:
            loss: Scalar loss value
        """
        # torchaudio expects logits in log_softmax format
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Compute loss
        loss = self.loss_fn(
            logits=log_probs,
            targets=targets.int(),
            logit_lengths=logit_lengths.int(),
            target_lengths=target_lengths.int(),
        )
        
        return loss


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing cross-entropy loss for LAS decoder.
    """
    
    def __init__(
        self,
        vocab_size: int,
        padding_idx: int = 0,
        smoothing: float = 0.1,
        reduction: str = 'mean',
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            padding_idx: Index of padding token (ignored in loss)
            smoothing: Label smoothing factor
            reduction: Reduction method
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            logits: Model predictions (batch, seq_len, vocab_size) or (batch*seq_len, vocab_size)
            targets: Target labels (batch, seq_len) or (batch*seq_len,)
            
        Returns:
            loss: Scalar loss value
        """
        # Flatten if needed
        if logits.dim() == 3:
            batch_size, seq_len, vocab_size = logits.size()
            # Ensure targets match logits sequence length
            if targets.size(1) != seq_len:
                # Truncate or pad targets to match logits
                if targets.size(1) > seq_len:
                    targets = targets[:, :seq_len]
                else:
                    # Pad with padding_idx
                    padding = torch.full(
                        (batch_size, seq_len - targets.size(1)),
                        self.padding_idx,
                        dtype=targets.dtype,
                        device=targets.device
                    )
                    targets = torch.cat([targets, padding], dim=1)
            
            logits = logits.view(-1, vocab_size)
            targets = targets.view(-1)
        
        # Compute log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Create smoothed target distribution
        smooth_targets = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), self.confidence
        )
        smooth_targets += self.smoothing / (self.vocab_size - 1)
        
        # Mask padding tokens
        mask = (targets != self.padding_idx).unsqueeze(1).float()
        smooth_targets = smooth_targets * mask
        
        # Compute loss
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        
        # Apply mask for reduction
        mask = (targets != self.padding_idx).float()
        loss = loss * mask
        
        if self.reduction == 'mean':
            return loss.sum() / (mask.sum() + 1e-8)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class JointLoss(nn.Module):
    """
    Combined loss for joint RNN-T + LAS training.
    """
    
    def __init__(
        self,
        vocab_size: int,
        blank_id: int = 0,
        padding_idx: int = 0,
        rnnt_weight: float = 1.0,
        las_weight: float = 1.0,
        label_smoothing: float = 0.1,
    ):
        """
        Args:
            vocab_size: Vocabulary size
            blank_id: Blank token ID for RNN-T
            padding_idx: Padding token ID for LAS
            rnnt_weight: Weight for RNN-T loss
            las_weight: Weight for LAS loss
            label_smoothing: Label smoothing factor for LAS
        """
        super().__init__()
        self.rnnt_weight = rnnt_weight
        self.las_weight = las_weight
        
        self.rnnt_loss = RNNTLoss(blank=blank_id, reduction='mean')
        self.las_loss = LabelSmoothingLoss(
            vocab_size=vocab_size,
            padding_idx=padding_idx,
            smoothing=label_smoothing,
            reduction='mean',
        )
    
    def forward(
        self,
        rnnt_logits: torch.Tensor,
        las_logits: torch.Tensor,
        targets: torch.Tensor,
        rnnt_logit_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> tuple:
        """
        Compute combined loss.
        
        Args:
            rnnt_logits: RNN-T joint network outputs
            las_logits: LAS decoder outputs
            targets: Target labels
            rnnt_logit_lengths: RNN-T output lengths
            target_lengths: Target sequence lengths
            
        Returns:
            total_loss: Weighted sum of losses
            rnnt_loss: RNN-T loss value
            las_loss: LAS loss value
        """
        # Compute RNN-T loss
        rnnt_loss = self.rnnt_loss(
            rnnt_logits, targets, rnnt_logit_lengths, target_lengths
        )
        
        # Compute LAS loss
        las_loss = self.las_loss(las_logits, targets)
        
        # Combined loss
        total_loss = self.rnnt_weight * rnnt_loss + self.las_weight * las_loss
        
        return total_loss, rnnt_loss, las_loss


if __name__ == '__main__':
    # Test losses
    batch_size = 2
    time_steps = 100
    label_len = 20
    vocab_size = 300
    
    # Test RNN-T loss
    rnnt_loss = RNNTLoss(blank=0)
    rnnt_logits = torch.randn(batch_size, time_steps, label_len + 1, vocab_size)
    targets = torch.randint(1, vocab_size, (batch_size, label_len))
    logit_lengths = torch.tensor([100, 80])
    target_lengths = torch.tensor([20, 15])
    
    loss = rnnt_loss(rnnt_logits, targets, logit_lengths, target_lengths)
    print(f"RNN-T loss: {loss.item():.4f}")
    
    # Test LAS loss
    las_loss = LabelSmoothingLoss(vocab_size=vocab_size, padding_idx=0)
    las_logits = torch.randn(batch_size, label_len, vocab_size)
    loss = las_loss(las_logits, targets)
    print(f"LAS loss: {loss.item():.4f}")
    
    # Test joint loss
    joint_loss = JointLoss(vocab_size=vocab_size)
    total, rnnt, las = joint_loss(
        rnnt_logits,
        las_logits,
        targets,
        logit_lengths,
        target_lengths,
    )
    print(f"Total loss: {total.item():.4f}, RNN-T: {rnnt.item():.4f}, LAS: {las.item():.4f}")
