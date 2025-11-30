"""
Train Neural Utterance Confidence Model (NCM).

Binary classifier training with AUC, EER, and NCE metrics.
"""
import os
import pickle
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from src.models.ncm_model import NCMModel


class NCMDataset(Dataset):
    """Dataset for NCM training."""
    
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            features: (N, feature_dim)
            labels: (N,) - 0 or 1
        """
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32) if isinstance(self.features[idx], np.ndarray) else self.features[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32) if isinstance(self.labels[idx], (int, np.integer)) else self.labels[idx].float()
        return {
            'features': feature,
            'label': label,
        }


def calculate_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Calculate Equal Error Rate (EER).
    
    Args:
        labels: Ground truth (0 or 1)
        scores: Predicted probabilities
        
    Returns:
        eer: Equal Error Rate
    """
    try:
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        fnr = 1 - tpr
        
        # Find threshold where FPR = FNR
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        
        return eer
    except (ValueError, IndexError):
        # Return 0.5 (random guess) if EER cannot be calculated
        return 0.5


def calculate_nce(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Calculate Normalized Cross Entropy (NCE).
    
    NCE = H(p, q) / H(p)
    where H(p, q) is cross entropy and H(p) is entropy of true distribution.
    
    Args:
        labels: Ground truth (0 or 1)
        scores: Predicted probabilities
        
    Returns:
        nce: Normalized Cross Entropy
    """
    epsilon = 1e-10
    
    # Cross entropy
    ce = -np.mean(labels * np.log(scores + epsilon) + (1 - labels) * np.log(1 - scores + epsilon))
    
    # Entropy of true distribution
    p = np.mean(labels)
    if p == 0 or p == 1:
        return 0.0
    entropy = -p * np.log(p) - (1 - p) * np.log(1 - p)
    
    # NCE
    nce = ce / entropy if entropy > 0 else 0.0
    
    return nce


def calculate_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    """Calculate all NCM metrics with error handling."""
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = 0.5
    
    eer = calculate_eer(labels, scores)
    nce = calculate_nce(labels, scores)
    
    return {'auc': auc, 'eer': eer, 'nce': nce}


def train_epoch(
    model: NCMModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    all_labels = []
    all_scores = []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        features = batch['features'].to(device)
        labels = batch['label'].to(device).unsqueeze(-1)  # (B, 1)
        
        # Forward
        scores = model(features)  # (B, 1)
        loss = criterion(scores, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item() * features.size(0)
        all_labels.extend(labels.cpu().numpy().flatten())
        all_scores.extend(scores.detach().cpu().numpy().flatten())
        
        pbar.set_postfix({'loss': loss.item()})
    
    # Calculate metrics
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    avg_loss = total_loss / len(dataloader.dataset)
    metrics = calculate_metrics(all_labels, all_scores)
    
    return {
        'loss': avg_loss,
        **metrics,
    }


@torch.no_grad()
def validate(
    model: NCMModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    
    total_loss = 0.0
    all_labels = []
    all_scores = []
    
    for batch in tqdm(dataloader, desc="Validation"):
        features = batch['features'].to(device)
        labels = batch['label'].to(device).unsqueeze(-1)
        
        # Forward
        scores = model(features)
        loss = criterion(scores, labels)
        
        # Metrics
        total_loss += loss.item() * features.size(0)
        all_labels.extend(labels.cpu().numpy().flatten())
        all_scores.extend(scores.cpu().numpy().flatten())
    
    # Calculate metrics
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    avg_loss = total_loss / len(dataloader.dataset)
    metrics = calculate_metrics(all_labels, all_scores)
    
    return {
        'loss': avg_loss,
        **metrics,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print("=" * 80)
    print("NCM Training")
    print("=" * 80)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get experiment config
    exp_cfg = cfg.get('experiment', cfg)
    
    # WandB
    wandb_cfg = exp_cfg.get('wandb', {})
    if WANDB_AVAILABLE and wandb_cfg.get('enabled', False):
        wandb.init(
            project=wandb_cfg.get('project', 'ssi-ncm'),
            entity=wandb_cfg.get('entity'),
            name=wandb_cfg.get('run_name', 'ncm_training'),
        )
        print(f"WandB run: {wandb.run.name} (ID: {wandb.run.id})")
    
    # Load features
    features_path = Path(exp_cfg.ncm.features_path)
    print(f"Loading features from: {features_path}")
    
    with open(features_path, 'rb') as f:
        data = pickle.load(f)
    
    features = data['features']
    labels = data['labels']
    feature_dim = data['feature_dim']
    
    print(f"Features shape: {features.shape}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Accept: {(labels == 1).sum().item()}, Reject: {(labels == 0).sum().item()}")
    
    # Split into train/val
    full_dataset = NCMDataset(features, labels)
    
    val_size = int(len(full_dataset) * exp_cfg.ncm.get('val_split', 0.2))
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.get('seed', 42))
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=exp_cfg.ncm.batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=exp_cfg.ncm.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    # Model
    model = NCMModel(
        input_dim=feature_dim,
        hidden_dim=exp_cfg.ncm.get('hidden_dim', 64),
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Criterion & Optimizer
    criterion = nn.BCELoss()
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=exp_cfg.ncm.lr,
        weight_decay=exp_cfg.ncm.get('weight_decay', 0.0),
    )
    
    # LR Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
    )
    
    # Output directory
    output_dir = Path(exp_cfg.ncm.get('output_dir', 'exp/ncm_ssi'))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Training loop
    best_auc = 0.0
    best_epoch = 0
    
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    for epoch in range(1, exp_cfg.ncm.num_epochs + 1):
        print(f"\nEpoch {epoch}/{exp_cfg.ncm.num_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"AUC: {train_metrics['auc']:.4f}, "
              f"EER: {train_metrics['eer']:.4f}, "
              f"NCE: {train_metrics['nce']:.4f}")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"AUC: {val_metrics['auc']:.4f}, "
              f"EER: {val_metrics['eer']:.4f}, "
              f"NCE: {val_metrics['nce']:.4f}")
        
        # LR scheduler
        scheduler.step(val_metrics['auc'])
        
        # WandB logging
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                'train/loss': train_metrics['loss'],
                'train/auc': train_metrics['auc'],
                'train/eer': train_metrics['eer'],
                'train/nce': train_metrics['nce'],
                'val/loss': val_metrics['loss'],
                'val/auc': val_metrics['auc'],
                'val/eer': val_metrics['eer'],
                'val/nce': val_metrics['nce'],
                'epoch': epoch,
                'lr': optimizer.param_groups[0]['lr'],
            })
        
        # Save best model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_epoch = epoch
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'config': OmegaConf.to_container(cfg, resolve=False),  # Don't resolve interpolations
            }
            
            torch.save(checkpoint, output_dir / 'best.ckpt', pickle_protocol=4)
            print(f"Saved best model (AUC: {best_auc:.4f})")
        
        # Save last checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_auc': best_auc,
            'config': OmegaConf.to_container(cfg, resolve=False),
        }
        torch.save(checkpoint, output_dir / 'last.ckpt', pickle_protocol=4)
    
    print("\n" + "=" * 80)
    print(f"Training completed!")
    print(f"Best AUC: {best_auc:.4f} at epoch {best_epoch}")
    print("=" * 80)
    
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
