"""
NCM Evaluation Script with CS@x% RIER metrics
Based on: Neural Utterance Confidence Measure for RNN-Transducers and Two-Pass Models
"""

import os
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from src.models.ncm_model import NCMModel


def calculate_eer(labels, scores):
    """Calculate Equal Error Rate"""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    
    # Find threshold where FPR = FNR
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    
    return eer, eer_threshold


def calculate_nce(labels, scores):
    """Calculate Normalized Cross Entropy"""
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    
    # Binary cross entropy
    bce = F.binary_cross_entropy(scores_tensor, labels_tensor, reduction='mean')
    
    # Entropy of label distribution
    p = labels_tensor.mean()
    if p == 0 or p == 1:
        return 0.0
    h = -p * np.log(p) - (1 - p) * np.log(1 - p)
    
    # NCE = BCE / H
    nce = bce.item() / h
    return nce


def calculate_cs_rier(labels, scores, wer_values, baseline_wer):
    """
    Calculate Cost Saving (CS) vs Relative Increase in Error Rate (RIER)
    
    Distributed-ASR scenario:
    - High-confidence utterances: processed locally (fast, lower quality)
    - Low-confidence utterances: sent to cloud (slow, higher quality)
    
    Args:
        labels: Ground truth (1=Accept, 0=Reject)
        scores: NCM confidence scores
        wer_values: WER for each utterance (from ASR hypothesis)
        baseline_wer: Baseline WER when all utterances go to cloud
    
    Returns:
        Dictionary with CS and RIER at different thresholds
    """
    labels = np.array(labels)
    scores = np.array(scores)
    wer_values = np.array(wer_values)
    
    # Sort by confidence score (descending)
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]
    sorted_wers = wer_values[sorted_indices]
    
    results = {}
    
    # Calculate CS and RIER at different percentages
    percentages = [0, 5, 10, 20, 30, 40, 50]
    
    for pct in percentages:
        # pct% of utterances sent to cloud (lowest confidence)
        n_cloud = int(len(scores) * pct / 100)
        n_local = len(scores) - n_cloud
        
        if n_local == 0:
            # All to cloud
            cs = 0.0
            rier = 0.0
        else:
            # Local: top n_local highest confidence
            # Cloud: bottom n_cloud lowest confidence
            
            # Cost Saving: percentage of utterances processed locally
            cs = 100.0 * n_local / len(scores)
            
            # Calculate WER with this split
            # Assume: cloud has perfect WER (or very low WER)
            # For simplicity, we use the actual WER from reference
            local_wer = sorted_wers[:n_local].mean() if n_local > 0 else 0.0
            cloud_wer = sorted_wers[n_local:].mean() if n_cloud > 0 else 0.0
            
            # Weighted average WER
            mixed_wer = (n_local * local_wer + n_cloud * cloud_wer) / len(scores)
            
            # RIER: Relative increase in error rate
            if baseline_wer > 0:
                rier = 100.0 * (mixed_wer - baseline_wer) / baseline_wer
            else:
                rier = 0.0
        
        results[f'CS@{pct}%'] = cs
        results[f'RIER@{pct}%'] = rier
        results[f'threshold@{pct}%'] = scores[sorted_indices[n_local-1]] if n_local > 0 else 0.0
    
    return results


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print("=" * 80)
    print("NCM Evaluation with CS@x% RIER metrics")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    
    # Get experiment config
    exp_cfg = cfg.get('experiment', cfg)
    
    # Load features
    features_path = Path(exp_cfg.ncm.features_path)
    print(f"\nLoading features from: {features_path}")
    
    with open(features_path, 'rb') as f:
        data = pickle.load(f)
    
    features = data['features']
    labels = data['labels']
    wer_values = data.get('wer', np.zeros(len(labels)))  # WER per utterance
    
    print(f"Total samples: {len(features)}")
    print(f"Feature dim: {features.shape[1]}")
    print(f"Accept (1): {(labels == 1).sum()}, Reject (0): {(labels == 0).sum()}")
    
    # Calculate baseline WER (all utterances)
    baseline_wer = wer_values.mean()
    print(f"\nBaseline WER (all utterances): {baseline_wer:.4f}")
    
    # Load NCM model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    checkpoint_path = exp_cfg.ncm.output_dir + '/best.ckpt'
    print(f"Loading NCM from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    input_dim = features.shape[1]
    
    model = NCMModel(
        input_dim=input_dim,
        hidden_dim=exp_cfg.ncm.hidden_dim
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\nNCM model loaded (epoch {checkpoint['epoch']}, AUC: {checkpoint['best_auc']:.4f})")
    
    # Inference
    print("\n" + "=" * 80)
    print("Running inference...")
    print("=" * 80)
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for i in tqdm(range(len(features)), desc="Inference"):
            feat = torch.tensor(features[i:i+1], dtype=torch.float32).to(device)
            score = model(feat).cpu().item()
            
            all_scores.append(score)
            all_labels.append(labels[i])
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    print("\n" + "=" * 80)
    print("Evaluation Metrics")
    print("=" * 80)
    
    # AUC
    auc = roc_auc_score(all_labels, all_scores)
    print(f"AUC: {auc:.4f}")
    
    # EER
    eer, eer_threshold = calculate_eer(all_labels, all_scores)
    print(f"EER: {eer:.4f} (threshold: {eer_threshold:.4f})")
    
    # NCE
    nce = calculate_nce(all_labels, all_scores)
    print(f"NCE: {nce:.4f}")
    
    # CS @ x% RIER
    print("\n" + "=" * 80)
    print("Cost Saving (CS) vs Relative Increase in Error Rate (RIER)")
    print("=" * 80)
    print("Scenario: High-confidence → Local ASR, Low-confidence → Cloud ASR")
    print()
    
    cs_rier_results = calculate_cs_rier(all_labels, all_scores, wer_values, baseline_wer)
    
    print(f"{'Cloud %':<10} {'CS (%)':<15} {'RIER (%)':<15} {'Threshold':<15}")
    print("-" * 60)
    
    for pct in [0, 5, 10, 20, 30, 40, 50]:
        cs = cs_rier_results[f'CS@{pct}%']
        rier = cs_rier_results[f'RIER@{pct}%']
        thresh = cs_rier_results[f'threshold@{pct}%']
        print(f"{pct:<10} {cs:<15.2f} {rier:<15.2f} {thresh:<15.4f}")
    
    # WandB logging
    if cfg.get('wandb', {}).get('enabled', False):
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"eval_ncm_{exp_cfg.experiment.name}",
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        
        log_dict = {
            'eval/auc': auc,
            'eval/eer': eer,
            'eval/nce': nce,
            'eval/baseline_wer': baseline_wer,
        }
        
        # Add CS@RIER metrics
        for key, value in cs_rier_results.items():
            log_dict[f'eval/{key}'] = value
        
        wandb.log(log_dict)
        wandb.finish()
    
    # Save results
    output_dir = Path(exp_cfg.ncm.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / 'eval_results.pkl'
    
    results = {
        'auc': auc,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'nce': nce,
        'baseline_wer': baseline_wer,
        'cs_rier': cs_rier_results,
        'scores': all_scores,
        'labels': all_labels,
    }
    
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n✓ Results saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print("Evaluation completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
