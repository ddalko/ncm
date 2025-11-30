"""
SSI Speech Emotion Recognition dataset loader for ASR.
Uses HuggingFace datasets library to load stapesai/ssi-speech-emotion-recognition
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torchaudio
from typing import Dict, Optional, List
import json
from pathlib import Path


class SSIDataset(Dataset):
    """
    Dataset class for SSI Speech Emotion Recognition dataset.
    Treats it as a simple ASR corpus.
    """
    
    def __init__(
        self,
        split: str = 'train',
        manifest_path: Optional[str] = None,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
    ):
        """
        Args:
            split: Dataset split ('train', 'validation', 'test')
            manifest_path: Path to manifest file (if using pre-split data)
            max_duration: Maximum audio duration in seconds
            min_duration: Minimum audio duration in seconds
        """
        self.split = split
        self.max_duration = max_duration
        self.min_duration = min_duration
        
        if manifest_path and Path(manifest_path).exists():
            # Load from manifest file
            self.data = self._load_from_manifest(manifest_path)
        else:
            # Load from HuggingFace
            self.data = self._load_from_huggingface()
    
    def _load_from_huggingface(self) -> List[Dict]:
        """Load dataset from HuggingFace hub."""
        print(f"Loading SSI dataset (split: {self.split}) from HuggingFace...")
        
        # Map split names
        split_map = {
            'train': 'train',
            'train_asr': 'train',
            'dev': 'validation',
            'dev_asr': 'validation',
            'validation': 'validation',
            'test': 'test',
        }
        
        hf_split = split_map.get(self.split, 'train')
        
        try:
            dataset = load_dataset('stapesai/ssi-speech-emotion-recognition', split=hf_split)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Using mock data for development...")
            return []
        
        data = []
        for idx, item in enumerate(dataset):
            # Extract relevant fields
            audio_array = item['audio']['array']
            sample_rate = item['audio']['sampling_rate']
            text = item.get('text', '')
            
            # Filter by duration if specified
            duration = len(audio_array) / sample_rate
            if self.max_duration and duration > self.max_duration:
                continue
            if self.min_duration and duration < self.min_duration:
                continue
            
            data.append({
                'id': f"{self.split}_{idx}",
                'audio': audio_array,
                'sample_rate': sample_rate,
                'text': text,
                'duration': duration,
            })
        
        print(f"Loaded {len(data)} samples from {self.split} split")
        return data
    
    def _load_from_manifest(self, manifest_path: str) -> List[Dict]:
        """Load dataset from manifest file (JSONL format)."""
        print(f"Loading data from manifest: {manifest_path}")
        data = []
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                item = json.loads(line.strip())
                # Only store metadata, not the full audio array
                data.append({
                    'id': item['id'],
                    'audio_path': item.get('audio_path', None),  # Path to audio file
                    'audio': item.get('audio', None),  # Legacy: inline audio array
                    'sample_rate': item['sample_rate'],
                    'text': item['text'],
                    'duration': item.get('duration', 0),
                })
                # Show progress every 1000 samples
                if (i + 1) % 1000 == 0:
                    print(f"  → Loaded {i+1} samples...")
        
        print(f"Loaded {len(data)} samples from manifest")
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Returns:
            Dict with keys:
                - waveform: torch.Tensor (1, num_samples)
                - sample_rate: int
                - text: str
                - duration: float
        """
        import soundfile as sf
        
        item = self.data[idx]
        
        # Load audio from file or from inline array
        if item.get('audio_path') and Path(item['audio_path']).exists():
            # Load from audio file (preferred method)
            waveform, sample_rate = sf.read(item['audio_path'])
            waveform = torch.tensor(waveform, dtype=torch.float32)
            sample_rate = item['sample_rate']  # Use stored sample rate
        elif item.get('audio') is not None:
            # Load from inline array (legacy support)
            if isinstance(item['audio'], list):
                waveform = torch.tensor(item['audio'], dtype=torch.float32)
            else:
                waveform = torch.tensor(item['audio'], dtype=torch.float32)
            sample_rate = item['sample_rate']
        else:
            raise ValueError(f"No audio data found for sample {idx}")
        
        # Ensure waveform is 2D: (1, num_samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        return {
            'waveform': waveform,
            'sample_rate': sample_rate,
            'text': item['text'],
            'duration': item.get('duration', len(waveform[0]) / sample_rate),
        }


def create_ssi_dataloader(
    split: str,
    manifest_path: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
    collate_fn=None,
) -> DataLoader:
    """
    Create a DataLoader for SSI dataset.
    
    Args:
        split: Dataset split
        manifest_path: Path to manifest file
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        collate_fn: Custom collate function
        
    Returns:
        DataLoader instance
    """
    dataset = SSIDataset(
        split=split,
        manifest_path=manifest_path,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    return dataloader


if __name__ == '__main__':
    # Test the dataset
    dataset = SSIDataset(split='train')
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Waveform shape: {sample['waveform'].shape}")
        print(f"Sample rate: {sample['sample_rate']}")
        print(f"Text: {sample['text']}")
        print(f"Duration: {sample['duration']:.2f}s")
