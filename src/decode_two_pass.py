"""
Decode test set using Two-Pass ASR (RNN-T + LAS rescoring).
"""

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import json
from tqdm import tqdm

from src.data.ssi_dataset import create_ssi_dataloader
from src.data.featurizer import AudioFeaturizer, collate_fn
from src.models.joint_model import JointASRModel
from src.models.metrics import SimpleTokenizer, compute_wer, compute_cer


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main decoding function."""
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create tokenizer
    tokenizer = SimpleTokenizer()
    
    # Create featurizer
    featurizer = AudioFeaturizer(
        sample_rate=cfg.data.sample_rate,
        n_mels=cfg.data.n_mels,
    )
    
    # Create dataloader
    def collate_wrapper(batch):
        return collate_fn(batch, featurizer)
    
    test_loader = create_ssi_dataloader(
        split='test',
        manifest_path=cfg.data.get('test_manifest', None),
        batch_size=cfg.decoding.get('batch_size', 8),
        num_workers=4,
        shuffle=False,
        collate_fn=collate_wrapper,
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Load model
    checkpoint_path = cfg.model.checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = checkpoint.get('config', cfg)
    
    model = JointASRModel(
        input_dim=cfg.data.n_mels,
        vocab_size=len(tokenizer),
        encoder_hidden=model_cfg.model.encoder_hidden,
        encoder_layers=model_cfg.model.encoder_layers,
        pred_hidden=model_cfg.model.pred_hidden,
        pred_layers=model_cfg.model.pred_layers,
        joint_dim=model_cfg.model.get('joint_dim', 512),
        las_encoder_hidden=model_cfg.model.las.get('enc2_hidden', 512),
        las_decoder_hidden=model_cfg.model.las.dec_hidden,
        las_embedding_dim=model_cfg.model.las.embedding_dim,
        las_attention_dim=model_cfg.model.las.get('attention_dim', 256),
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully")
    
    # Decode
    all_results = []
    all_rnnt_hyps = []
    all_las_hyps = []
    all_refs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Decoding (Two-Pass)"):
            features = batch['features'].transpose(1, 2).to(device)
            feature_lengths = batch['feature_lengths'].to(device)
            texts = batch['texts']
            
            # Two-pass decoding: RNN-T + LAS
            rnnt_results, las_results = model.recognize_two_pass(
                features,
                feature_lengths,
                blank_id=tokenizer.blank_id,
                sos_id=tokenizer.sos_id,
                eos_id=tokenizer.eos_id,
                max_symbols=cfg.decoding.get('max_symbols', 200),
            )
            
            # Convert to text
            rnnt_texts = tokenizer.batch_decode(rnnt_results)
            las_texts = tokenizer.batch_decode(las_results)
            
            # Store results
            for i, (rnnt_hyp, las_hyp, ref) in enumerate(zip(rnnt_texts, las_texts, texts)):
                all_results.append({
                    'rnnt_hypothesis': rnnt_hyp,
                    'las_hypothesis': las_hyp,
                    'reference': ref,
                })
                all_rnnt_hyps.append(rnnt_hyp)
                all_las_hyps.append(las_hyp)
                all_refs.append(ref)
    
    # Compute metrics for both passes
    rnnt_wer = compute_wer(all_rnnt_hyps, all_refs)
    rnnt_cer = compute_cer(all_rnnt_hyps, all_refs)
    las_wer = compute_wer(all_las_hyps, all_refs)
    las_cer = compute_cer(all_las_hyps, all_refs)
    
    print(f"\nResults:")
    print(f"RNN-T (First Pass):")
    print(f"  WER: {rnnt_wer:.2f}%")
    print(f"  CER: {rnnt_cer:.2f}%")
    print(f"\nLAS (Second Pass / Rescoring):")
    print(f"  WER: {las_wer:.2f}%")
    print(f"  CER: {las_cer:.2f}%")
    print(f"\nImprovement:")
    print(f"  WER: {rnnt_wer - las_wer:.2f}% (absolute)")
    print(f"  CER: {rnnt_cer - las_cer:.2f}% (absolute)")
    
    # Save results
    output_path = Path(cfg.decoding.get('output_path', 'decode_2pass.jsonl'))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"\nResults saved to {output_path}")
    
    # Save summary
    summary = {
        'rnnt': {
            'wer': rnnt_wer,
            'cer': rnnt_cer,
        },
        'las': {
            'wer': las_wer,
            'cer': las_cer,
        },
        'improvement': {
            'wer': rnnt_wer - las_wer,
            'cer': rnnt_cer - las_cer,
        },
        'num_samples': len(all_results),
    }
    
    summary_path = output_path.parent / 'summary_2pass.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
