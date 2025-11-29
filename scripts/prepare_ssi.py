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
    
    Args:
        dataset: HuggingFace dataset
        output_path: Path to output manifest file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, item in enumerate(tqdm(dataset, desc=f"Creating {output_path.name}")):
            # Handle AudioDecoder object - decode to get actual audio data
            audio_decoder = item['file_path']  # This is the AudioDecoder object
            
            # Get all audio samples
            audio_samples = audio_decoder.get_all_samples()
            
            # Extract audio array and sample rate
            # audio_samples.data is a torch.Tensor with shape [channels, samples]
            audio_tensor = audio_samples.data
            sample_rate = audio_samples.sample_rate
            
            # Convert to numpy array and flatten (remove channel dimension if mono)
            audio_array = audio_tensor.squeeze().numpy()
            
            text = item.get('text', '')
            
            # Compute duration
            duration = len(audio_array) / sample_rate
            
            manifest_item = {
                'id': f"{output_path.stem}_{idx}",
                'audio': audio_array.tolist(),  # Convert to list for JSON
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
        dataset = load_dataset('stapesai/ssi-speech-emotion-recognition')
        
        # Combine all splits if necessary
        if 'train' in dataset:
            all_data = dataset['train']
        else:
            all_data = dataset
        
        print(f"Loaded {len(all_data)} samples")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating dummy data for testing...")
        
        # Create dummy data
        class DummyDataset:
            def __init__(self, size):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __iter__(self):
                for i in range(self.size):
                    yield {
                        'audio': {
                            'array': [0.0] * 16000,  # 1 second of silence
                            'sampling_rate': 16000,
                        },
                        'text': f'sample text {i}',
                    }
        
        all_data = DummyDataset(args.train_size + args.dev_size + args.test_size)
    
    # Convert to list for shuffling
    all_samples = list(all_data)
    random.shuffle(all_samples)
    
    # Split data
    train_data = all_samples[:args.train_size]
    dev_data = all_samples[args.train_size:args.train_size + args.dev_size]
    test_data = all_samples[args.train_size + args.dev_size:
                           args.train_size + args.dev_size + args.test_size]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Dev: {len(dev_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create manifests
    print("\nCreating manifest files...")
    
    # Wrap data in simple containers
    class SimpleDataset:
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __iter__(self):
            return iter(self.data)
    
    create_manifest(SimpleDataset(train_data), output_dir / 'train.jsonl')
    create_manifest(SimpleDataset(dev_data), output_dir / 'dev.jsonl')
    create_manifest(SimpleDataset(test_data), output_dir / 'test.jsonl')
    
    print("\nDataset preparation completed!")
    print(f"Manifest files saved to: {output_dir}")


if __name__ == '__main__':
    main()
