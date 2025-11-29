"""
Audio feature extraction module for ASR.
Converts raw audio waveforms into mel-spectrogram features.
"""

import torch
import torchaudio
import numpy as np
from typing import Tuple


class AudioFeaturizer:
    """
    Extracts mel-spectrogram features from audio waveforms.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        normalize: bool = True,
    ):
        """
        Args:
            sample_rate: Target sampling rate for audio
            n_mels: Number of mel filterbanks
            n_fft: FFT size
            hop_length: Hop length for STFT
            win_length: Window length for STFT
            normalize: Whether to normalize features
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.normalize = normalize
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to mel-spectrogram features.
        
        Args:
            waveform: Audio waveform tensor of shape (1, num_samples) or (num_samples,)
            
        Returns:
            mel_spec: Mel-spectrogram features of shape (n_mels, time_steps)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        # Compute mel-spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Convert to log scale
        mel_spec = torch.log(mel_spec + 1e-9)
        
        # Normalize
        if self.normalize:
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)
        
        # Remove channel dimension: (1, n_mels, time) -> (n_mels, time)
        mel_spec = mel_spec.squeeze(0)
        
        return mel_spec
    
    def compute_length(self, num_samples: int) -> int:
        """
        Compute the output sequence length given input audio length.
        
        Args:
            num_samples: Number of audio samples
            
        Returns:
            Output sequence length in frames
        """
        return (num_samples - self.mel_transform.win_length) // self.mel_transform.hop_length + 1


def collate_fn(batch, featurizer: AudioFeaturizer):
    """
    Collate function for DataLoader.
    Pads sequences to the same length.
    
    Args:
        batch: List of (waveform, text, sample_rate) tuples
        featurizer: AudioFeaturizer instance
        
    Returns:
        features: Padded features tensor (batch, n_mels, max_time)
        feature_lengths: Actual lengths before padding (batch,)
        texts: List of text transcriptions
        text_lengths: Length of each text
    """
    waveforms = []
    texts = []
    sample_rates = []
    
    for item in batch:
        waveforms.append(item['waveform'])
        texts.append(item['text'])
        sample_rates.append(item['sample_rate'])
    
    # Extract features
    features = []
    feature_lengths = []
    
    for waveform, sr in zip(waveforms, sample_rates):
        # Resample if necessary
        if sr != featurizer.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, featurizer.sample_rate)
            waveform = resampler(waveform)
        
        # Extract mel features
        feat = featurizer(waveform)
        features.append(feat)
        feature_lengths.append(feat.shape[1])
    
    # Pad features
    max_len = max(feature_lengths)
    n_mels = features[0].shape[0]
    
    padded_features = torch.zeros(len(features), n_mels, max_len)
    for i, feat in enumerate(features):
        padded_features[i, :, :feat.shape[1]] = feat
    
    feature_lengths = torch.tensor(feature_lengths, dtype=torch.long)
    
    return {
        'features': padded_features,
        'feature_lengths': feature_lengths,
        'texts': texts,
    }
