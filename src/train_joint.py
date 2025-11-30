"""
Joint training script for RNN-T + LAS model.
Uses Hydra for configuration and wandb for experiment tracking.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from pathlib import Path

from src.data.ssi_dataset import create_ssi_dataloader
from src.data.featurizer import AudioFeaturizer, collate_fn
from src.models.joint_model import JointASRModel
from src.models.rnnt_loss import JointLoss
from src.models.metrics import SimpleTokenizer, compute_wer, compute_cer


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    tokenizer: SimpleTokenizer,
    device: torch.device,
    epoch: int,
    grad_clip: float = 5.0,
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_rnnt_loss = 0.0
    total_las_loss = 0.0
    num_batches = 0
    
    print(f"\n[Epoch {epoch}] Starting training with {len(dataloader)} batches...")
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", dynamic_ncols=True)
    
    for batch_idx, batch in enumerate(pbar):
        if batch_idx == 0:
            print(f"  → Processing first batch (batch_idx=0)...")
        # Move data to device
        features = batch['features'].transpose(1, 2).to(device)  # (B, T, n_mels)
        feature_lengths = batch['feature_lengths'].to(device)
        texts = batch['texts']
        
        if batch_idx == 0:
            print(f"  → Batch shape: features={features.shape}, lengths={feature_lengths.shape}")
        
        # Encode texts to tokens
        target_tokens = []
        target_lengths = []
        for text in texts:
            tokens = tokenizer.encode(text)
            target_tokens.append(tokens)
            target_lengths.append(len(tokens))
        
        # Pad target tokens
        max_target_len = max(target_lengths)
        padded_targets = torch.zeros(len(texts), max_target_len, dtype=torch.long)
        
        for i, tokens in enumerate(target_tokens):
            padded_targets[i, :len(tokens)] = torch.tensor(tokens)
        
        padded_targets = padded_targets.to(device)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long).to(device)
        
        # For RNN-T: prepend blank to targets for prediction network input
        # This creates targets of length target_len+1
        rnnt_targets = torch.cat([
            torch.full((len(texts), 1), tokenizer.blank_id, dtype=torch.long, device=device),
            padded_targets
        ], dim=1)
        rnnt_target_lengths = target_lengths + 1
        
        # Forward pass
        optimizer.zero_grad()
        
        if batch_idx == 0:
            print(f"  → Starting forward pass...")
        
        outputs = model(
            features=features,
            feature_lengths=feature_lengths,
            labels=rnnt_targets,  # Use RNN-T targets (with prepended blank)
            label_lengths=rnnt_target_lengths,  # Use RNN-T lengths (target_len+1)
            sos_id=tokenizer.sos_id,
        )
        
        # Compute loss - use original targets for loss computation
        loss, rnnt_loss, las_loss = criterion(
            rnnt_logits=outputs['rnnt_logits'],
            las_logits=outputs['las_logits'],
            targets=padded_targets,  # Original targets (without prepended blank)
            rnnt_logit_lengths=feature_lengths,
            target_lengths=target_lengths,  # Original target lengths
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        total_rnnt_loss += rnnt_loss.item()
        total_las_loss += las_loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'rnnt': f"{rnnt_loss.item():.4f}",
            'las': f"{las_loss.item():.4f}",
        })
        
        # Log to wandb every N steps
        if batch_idx % 10 == 0:
            wandb.log({
                'train/loss': loss.item(),
                'train/loss_rnnt': rnnt_loss.item(),
                'train/loss_las': las_loss.item(),
                'train/step': epoch * len(dataloader) + batch_idx,
            })
    
    return {
        'loss': total_loss / num_batches,
        'rnnt_loss': total_rnnt_loss / num_batches,
        'las_loss': total_las_loss / num_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    tokenizer: SimpleTokenizer,
    device: torch.device,
    epoch: int,
) -> dict:
    """Validate the model."""
    model.eval()
    
    total_loss = 0.0
    total_rnnt_loss = 0.0
    total_las_loss = 0.0
    num_batches = 0
    
    all_rnnt_hyps = []
    all_las_hyps = []
    all_refs = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    
    for batch in pbar:
        # Move data to device
        features = batch['features'].transpose(1, 2).to(device)
        feature_lengths = batch['feature_lengths'].to(device)
        texts = batch['texts']
        
        # Encode texts
        target_tokens = []
        target_lengths = []
        for text in texts:
            tokens = tokenizer.encode(text)
            target_tokens.append(tokens)
            target_lengths.append(len(tokens))
        
        max_target_len = max(target_lengths)
        padded_targets = torch.zeros(len(texts), max_target_len, dtype=torch.long)
        
        for i, tokens in enumerate(target_tokens):
            padded_targets[i, :len(tokens)] = torch.tensor(tokens)
        
        padded_targets = padded_targets.to(device)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long).to(device)
        
        # For RNN-T: prepend blank to targets
        rnnt_targets = torch.cat([
            torch.full((len(texts), 1), tokenizer.blank_id, dtype=torch.long, device=device),
            padded_targets
        ], dim=1)
        rnnt_target_lengths = target_lengths + 1
        
        # Forward pass
        outputs = model(
            features=features,
            feature_lengths=feature_lengths,
            labels=rnnt_targets,  # Use RNN-T targets (with prepended blank)
            label_lengths=rnnt_target_lengths,  # Use RNN-T lengths
            sos_id=tokenizer.sos_id,
        )
        
        # Compute loss - use original targets
        loss, rnnt_loss, las_loss = criterion(
            rnnt_logits=outputs['rnnt_logits'],
            las_logits=outputs['las_logits'],
            targets=padded_targets,  # Original targets
            rnnt_logit_lengths=feature_lengths,
            target_lengths=target_lengths,  # Original lengths
        )
        
        total_loss += loss.item()
        total_rnnt_loss += rnnt_loss.item()
        total_las_loss += las_loss.item()
        num_batches += 1
        
        # Decode predictions for metrics
        rnnt_results, las_results = model.recognize_two_pass(
            features, feature_lengths,
            blank_id=tokenizer.blank_id,
            sos_id=tokenizer.sos_id,
            eos_id=tokenizer.eos_id,
        )
        
        # Convert to text
        rnnt_texts = tokenizer.batch_decode(rnnt_results)
        las_texts = tokenizer.batch_decode(las_results)
        
        all_rnnt_hyps.extend(rnnt_texts)
        all_las_hyps.extend(las_texts)
        all_refs.extend(texts)
    
    # Compute metrics
    rnnt_wer = compute_wer(all_rnnt_hyps, all_refs)
    las_wer = compute_wer(all_las_hyps, all_refs)
    rnnt_cer = compute_cer(all_rnnt_hyps, all_refs)
    las_cer = compute_cer(all_las_hyps, all_refs)
    
    return {
        'loss': total_loss / num_batches,
        'rnnt_loss': total_rnnt_loss / num_batches,
        'las_loss': total_las_loss / num_batches,
        'rnnt_wer': rnnt_wer,
        'las_wer': las_wer,
        'rnnt_cer': rnnt_cer,
        'las_cer': las_cer,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.get('entity', None),
        name=cfg.wandb.get('run_name', 'joint_ssi'),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    
    # Create tokenizer
    tokenizer = SimpleTokenizer()
    print(f"Vocabulary size: {len(tokenizer)}")
    
    # Create featurizer
    featurizer = AudioFeaturizer(
        sample_rate=cfg.data.sample_rate,
        n_mels=cfg.data.n_mels,
    )
    
    # Create dataloaders
    def collate_wrapper(batch):
        return collate_fn(batch, featurizer)
    
    train_loader = create_ssi_dataloader(
        split='train',
        manifest_path=cfg.data.get('train_manifest', None),
        batch_size=cfg.training.batch_size,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        shuffle=True,
        collate_fn=collate_wrapper,
    )
    
    val_loader = create_ssi_dataloader(
        split='dev',
        manifest_path=cfg.data.get('dev_manifest', None),
        batch_size=cfg.training.batch_size,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        shuffle=False,
        collate_fn=collate_wrapper,
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    model = JointASRModel(
        input_dim=cfg.data.n_mels,
        vocab_size=len(tokenizer),
        encoder_hidden=cfg.model.encoder_hidden,
        encoder_layers=cfg.model.encoder_layers,
        pred_hidden=cfg.model.pred_hidden,
        pred_layers=cfg.model.pred_layers,
        joint_dim=cfg.model.get('joint_dim', 512),
        las_encoder_hidden=cfg.model.las.get('enc2_hidden', 512),
        las_decoder_hidden=cfg.model.las.dec_hidden,
        las_embedding_dim=cfg.model.las.embedding_dim,
        las_attention_dim=cfg.model.las.get('attention_dim', 256),
        dropout=cfg.model.get('dropout', 0.1),
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss
    criterion = JointLoss(
        vocab_size=len(tokenizer),
        blank_id=tokenizer.blank_id,
        padding_idx=tokenizer.blank_id,
        rnnt_weight=cfg.model.loss_weights.rnnt,
        las_weight=cfg.model.loss_weights.las,
        label_smoothing=cfg.model.get('label_smoothing', 0.1),
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.get('weight_decay', 1e-6),
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
    )
    
    # Training loop
    best_val_loss = float('inf')
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, cfg.training.num_epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.training.num_epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, tokenizer,
            device, epoch, cfg.training.get('grad_clip', 5.0)
        )
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"RNN-T: {train_metrics['rnnt_loss']:.4f}, "
              f"LAS: {train_metrics['las_loss']:.4f}")
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, tokenizer, device, epoch
        )
        
        print(f"Val - Loss: {val_metrics['loss']:.4f}, "
              f"RNN-T WER: {val_metrics['rnnt_wer']:.2f}%, "
              f"LAS WER: {val_metrics['las_wer']:.2f}%")
        
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train/loss_epoch': train_metrics['loss'],
            'train/rnnt_loss_epoch': train_metrics['rnnt_loss'],
            'train/las_loss_epoch': train_metrics['las_loss'],
            'val/loss': val_metrics['loss'],
            'val/rnnt_loss': val_metrics['rnnt_loss'],
            'val/las_loss': val_metrics['las_loss'],
            'val/rnnt_wer': val_metrics['rnnt_wer'],
            'val/las_wer': val_metrics['las_wer'],
            'val/rnnt_cer': val_metrics['rnnt_cer'],
            'val/las_cer': val_metrics['las_cer'],
            'lr': optimizer.param_groups[0]['lr'],
        })
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_metrics['loss'],
            'config': OmegaConf.to_container(cfg, resolve=True),
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, output_dir / 'latest.ckpt')
        
        # Save best checkpoint
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(checkpoint, output_dir / 'best.ckpt')
            print(f"Saved best checkpoint (val_loss: {best_val_loss:.4f})")
    
    print("\nTraining completed!")
    wandb.finish()


if __name__ == '__main__':
    main()
