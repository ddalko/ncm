"""
NCM Lightweight Feature Extraction
Extracts only beam_scores + LAS-decoder features for faster inference
"""

import os
import json
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from src.models.joint_model import JointASRModel
from src.models.ncm_model import NCMFeatureExtractor
from src.data.featurizer import AudioFeaturizer


class SimpleTokenizer:
    """Simple character-level tokenizer"""
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        
    def encode(self, text):
        # Simple character encoding
        return [min(ord(c), self.vocab_size - 1) for c in text.lower()]
        
    def decode(self, tokens):
        return ''.join([chr(t) for t in tokens])


def extract_lightweight_features(
    model,
    audio_features,
    feature_lengths,
    text_tokens,
    extractor,
    cfg,
):
    """Extract only beam_scores + LAS-decoder features"""
    device = next(model.parameters()).device
    batch_size = audio_features.size(0)
    
    # Get encoder outputs (shared between RNN-T and LAS)
    encoder_outputs, enc_lengths = model.shared_encoder(audio_features, feature_lengths)
    
    # === RNN-T beam score (greedy decoding for simplicity) ===
    # Prediction network: start with blank token
    pred_input = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
    pred_outputs = model.rnnt_prediction(pred_input)
    if isinstance(pred_outputs, tuple):
        pred_outputs = pred_outputs[0]
    
    # Joint network: (B, T, U, vocab)
    joint_logits = model.rnnt_joint(
        encoder_outputs.unsqueeze(2),
        pred_outputs.unsqueeze(1)
    )
    
    # Greedy RNN-T score: sum of max logprobs
    rnnt_beam_scores = joint_logits.max(dim=-1)[0].sum(dim=(1, 2))
    
    # === LAS second pass ===
    # LAS encoder (bidirectional LSTM with packing)
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    
    packed = pack_padded_sequence(
        encoder_outputs,
        enc_lengths.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    packed_output, _ = model.las_encoder(packed)
    las_encoder_outputs, _ = pad_packed_sequence(
        packed_output, batch_first=True
    )
    
    # LAS decoder with teacher forcing
    las_decoder_logits = model.forward_las(
        features=audio_features,
        feature_lengths=feature_lengths,
        targets=text_tokens,
        target_lengths=torch.tensor([text_tokens.size(1)], device=device),
        sos_id=0,  # SOS token
    )
    
    # LAS beam score: sum of max logprobs
    las_beam_scores = las_decoder_logits.max(dim=-1)[0].sum(dim=1)
    
    # === Extract LAS-decoder features ===
    las_features = extractor.extract_las_features(
        las_encoder_outputs=las_encoder_outputs,
        las_decoder_logits=las_decoder_logits,
        feature_lengths=enc_lengths,
    )
    
    # === Extract beam scores ===
    beam_scores = extractor.extract_beam_scores(
        rnnt_beam_scores=rnnt_beam_scores,
        las_beam_scores=las_beam_scores,
    )
    
    # === Concatenate features ===
    # Only las_dec + beam_scores
    features = torch.cat([
        las_features['dec'],      # (B, top_k)
        beam_scores['rnnt_score'], # (B, 1)
        beam_scores['las_score'],  # (B, 1)
    ], dim=-1)
    
    return features


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print("=" * 80)
    print("NCM Lightweight Feature Extraction (beam_scores + LAS-decoder only)")
    print("=" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    exp_cfg = cfg.get('experiment', cfg)
    
    output_dir = Path(exp_cfg.ncm.output_dir).parent / 'ncm_features_light'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load model
    print(f"\nLoading model from: {exp_cfg.model.checkpoint}")
    model = JointASRModel(
        input_dim=80,
        vocab_size=exp_cfg.model.vocab_size,
        encoder_hidden=exp_cfg.model.encoder_hidden,
        encoder_layers=exp_cfg.model.encoder_layers,
        pred_hidden=exp_cfg.model.pred_hidden,
        pred_layers=exp_cfg.model.pred_layers,
        las_encoder_hidden=exp_cfg.model.las.enc2_hidden,
        las_decoder_hidden=exp_cfg.model.las.dec_hidden,
        las_embedding_dim=exp_cfg.model.las.embedding_dim,
    ).to(device)
    
    checkpoint = torch.load(exp_cfg.model.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded successfully")
    
    # Feature extractor
    extractor = NCMFeatureExtractor(top_k=exp_cfg.ncm_top_k)
    
    # Feature dimension: top_k (las_dec) + 2 (beam_scores)
    feature_dim = exp_cfg.ncm_top_k + 2
    print(f"Lightweight NCM feature dimension: {feature_dim}")
    print(f"  - LAS decoder top-{exp_cfg.ncm_top_k} logits: {exp_cfg.ncm_top_k}")
    print(f"  - RNN-T beam score: 1")
    print(f"  - LAS beam score: 1")
    
    # Load dataset
    import soundfile as sf
    
    featurizer = AudioFeaturizer(sample_rate=16000, n_mels=80)
    tokenizer = SimpleTokenizer(exp_cfg.model.vocab_size)
    
    manifest_path = cfg.data.dev_manifest
    print(f"\nLoading data from: {manifest_path}")
    
    samples = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    
    print(f"Total samples: {len(samples)}")
    
    # Extract features
    all_features = []
    all_labels = []
    success_count = 0
    error_count = 0
    
    print("\nExtracting lightweight features...")
    with torch.no_grad():
        for i, item in enumerate(tqdm(samples, desc="Processing")):
            try:
                # Load audio
                audio_path = item['audio_path']
                audio, sr = sf.read(audio_path)
                
                if sr != 16000:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                
                # Extract mel features
                audio_tensor = torch.tensor(audio, dtype=torch.float32)
                mel_features = featurizer(audio_tensor.unsqueeze(0))  # (1, n_mels, T)
                mel_features = mel_features.squeeze(0).transpose(0, 1)  # (T, n_mels)
                
                # Skip if too short
                if mel_features.size(0) < 10:
                    continue
                    
                mel_features = mel_features.unsqueeze(0).to(device)  # (1, T, n_mels)
                feature_lengths = torch.tensor([mel_features.size(1)], dtype=torch.long, device=device)
                
                # Tokenize text
                text = item['text']
                tokens = tokenizer.encode(text)
                text_tokens = torch.tensor([tokens], dtype=torch.long, device=device)
                
                # Extract lightweight features
                features = extract_lightweight_features(
                    model=model,
                    audio_features=mel_features,
                    feature_lengths=feature_lengths,
                    text_tokens=text_tokens,
                    extractor=extractor,
                    cfg=exp_cfg,
                )
                
                # Dummy label (will be updated later)
                label = 1  # Accept
                
                all_features.append(features.cpu().squeeze(0).numpy())
                all_labels.append(label)
                success_count += 1
                
            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    print(f"\nError processing sample {i}: {e}")
    
    print(f"\n✓ Successfully extracted features from {success_count} samples")
    print(f"✗ Failed: {error_count} samples")
    
    # Save features
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    
    data = {
        'features': all_features,
        'labels': all_labels,
        'feature_dim': feature_dim,
        'config': {
            'use_rnnt_trans': False,
            'use_rnnt_pred': False,
            'use_rnnt_joint': False,
            'use_las_enc': False,
            'use_las_dec': True,
            'use_beam_scores': True,
            'top_k': exp_cfg.ncm_top_k,
        }
    }
    
    output_path = output_dir / 'features.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\n✓ Saved {len(all_features)} samples to: {output_path}")
    print(f"  Feature shape: {all_features.shape}")
    print(f"  Feature dimension: {feature_dim}")
    
    print("=" * 80)


if __name__ == '__main__':
    main()
