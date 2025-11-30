"""
Prepare SSI dataset for ASR training.
Splits the dataset into train/dev/test sets and creates manifest files.
"""

import argparse
from datasets import load_dataset
import json
from pathlib import Path
import random
from tqdm import tqdm


def create_manifest(dataset, output_path: Path):
    """
    Create a manifest file from dataset.
    Saves audio files separately and only stores paths in manifest.
    
    Args:
        dataset: HuggingFace dataset
        output_path: Path to output manifest file
    """
    import numpy as np
    import soundfile as sf
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create directory for audio files
    audio_dir = output_path.parent / f"{output_path.stem}_audio"
    audio_dir.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, item in enumerate(tqdm(dataset, desc=f"Creating {output_path.name}")):
            # Handle AudioDecoder object - decode to get actual audio data
            if 'file_path' in item:
                # Real dataset with AudioDecoder
                audio_decoder = item['file_path']
                
                # Get all audio samples
                audio_samples = audio_decoder.get_all_samples()
                
                # Extract audio array and sample rate
                audio_tensor = audio_samples.data
                sample_rate = audio_samples.sample_rate
                
                # Convert to numpy array and flatten
                audio_array = audio_tensor.squeeze().numpy()
                text = item.get('text', '')
            elif 'audio_array' in item:
                # Dummy data
                audio_array = np.array(item['audio_array'])
                sample_rate = item['sample_rate']
                text = item['text']
            else:
                print(f"Warning: Unknown item format at index {idx}, skipping...")
                continue
            
            # Compute duration
            duration = len(audio_array) / sample_rate
            
            # Save audio to file
            audio_filename = f"{output_path.stem}_{idx:06d}.wav"
            audio_path = audio_dir / audio_filename
            sf.write(audio_path, audio_array, sample_rate)
            
            manifest_item = {
                'id': f"{output_path.stem}_{idx}",
                'audio_path': str(audio_path),  # Store path instead of array
                'sample_rate': sample_rate,
                'text': text,
                'duration': duration,
            }
            
            f.write(json.dumps(manifest_item, ensure_ascii=False) + '\n')
    
    print(f"Created manifest: {output_path} ({len(dataset)} samples)")


def main():
    parser = argparse.ArgumentParser(description='Prepare SSI dataset for ASR')
    parser.add_argument('--output_dir', type=str, default='data/ssi',
                        help='Output directory for manifests')
    parser.add_argument('--train_size', type=int, default=9500,
                        help='Number of training samples')
    parser.add_argument('--dev_size', type=int, default=1500,
                        help='Number of validation samples')
    parser.add_argument('--test_size', type=int, default=0,
                        help='Number of test samples')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    print("Loading SSI dataset from HuggingFace...")
    
    try:
        # Load full dataset
        dataset = load_dataset('stapesai/ssi-speech-emotion-recognition', split='train')
        
        print(f"Loaded {len(dataset)} samples")
        
        # Calculate how many samples we need
        total_needed = args.train_size + args.dev_size + args.test_size
        
        # Directly sample indices instead of converting entire dataset to list
        import numpy as np
        np.random.seed(args.seed)
        indices = np.random.permutation(len(dataset))[:total_needed]
        
        # Split indices
        train_indices = indices[:args.train_size]
        dev_indices = indices[args.train_size:args.train_size + args.dev_size]
        test_indices = indices[args.train_size + args.dev_size:]
        
        print(f"\nDataset split:")
        print(f"  Train: {len(train_indices)} samples")
        print(f"  Dev: {len(dev_indices)} samples")
        print(f"  Test: {len(test_indices)} samples")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create manifests by selecting specific indices
        print("\nCreating manifest files...")
        
        # Wrap indices with dataset
        class IndexedDataset:
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices
            
            def __len__(self):
                return len(self.indices)
            
            def __iter__(self):
                for idx in self.indices:
                    yield self.dataset[int(idx)]
        
        create_manifest(IndexedDataset(dataset, train_indices), output_dir / 'train.jsonl')
        create_manifest(IndexedDataset(dataset, dev_indices), output_dir / 'dev.jsonl')
        if len(test_indices) > 0:
            create_manifest(IndexedDataset(dataset, test_indices), output_dir / 'test.jsonl')
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating dummy data for testing...")
        
        # Create dummy data with proper structure
        class DummyDataset:
            def __init__(self, size):
                self.size = size
                self.items = []
                # Pre-generate all items to avoid issues with iteration
                for i in range(size):
                    self.items.append({
                        'audio_array': [0.0] * 16000,  # 1 second of silence
                        'sample_rate': 16000,
                        'text': f'sample text {i}',
                    })
            
            def __len__(self):
                return self.size
            
            def __iter__(self):
                return iter(self.items)
        
        all_data = DummyDataset(args.train_size + args.dev_size + args.test_size)
        
        # For dummy data, just use simple splits
        train_indices = list(range(args.train_size))
        dev_indices = list(range(args.train_size, args.train_size + args.dev_size))
        test_indices = list(range(args.train_size + args.dev_size, 
                                  args.train_size + args.dev_size + args.test_size))
        
        print(f"\nDataset split:")
        print(f"  Train: {len(train_indices)} samples")
        print(f"  Dev: {len(dev_indices)} samples")
        print(f"  Test: {len(test_indices)} samples")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create manifests
        print("\nCreating manifest files...")
        
        class IndexedDataset:
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices
            
            def __len__(self):
                return len(self.indices)
            
            def __iter__(self):
                items = list(self.dataset)  # Convert to list first
                for idx in self.indices:
                    yield items[int(idx)]
        
        create_manifest(IndexedDataset(all_data, train_indices), output_dir / 'train.jsonl')
        create_manifest(IndexedDataset(all_data, dev_indices), output_dir / 'dev.jsonl')
        if len(test_indices) > 0:
            create_manifest(IndexedDataset(all_data, test_indices), output_dir / 'test.jsonl')
    
    print("\nDataset preparation completed!")
    print(f"Manifest files saved to: {output_dir}")


if __name__ == '__main__':
    main()
