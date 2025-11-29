"""
Evaluation metrics for ASR: WER and CER.
"""

import jiwer
from typing import List


def compute_wer(hypotheses: List[str], references: List[str]) -> float:
    """
    Compute Word Error Rate (WER).
    
    Args:
        hypotheses: List of hypothesis strings
        references: List of reference strings
        
    Returns:
        WER as a percentage (0-100)
    """
    if len(hypotheses) == 0 or len(references) == 0:
        return 100.0
    
    try:
        wer = jiwer.wer(references, hypotheses)
        return wer * 100.0
    except Exception as e:
        print(f"Error computing WER: {e}")
        return 100.0


def compute_cer(hypotheses: List[str], references: List[str]) -> float:
    """
    Compute Character Error Rate (CER).
    
    Args:
        hypotheses: List of hypothesis strings
        references: List of reference strings
        
    Returns:
        CER as a percentage (0-100)
    """
    if len(hypotheses) == 0 or len(references) == 0:
        return 100.0
    
    try:
        cer = jiwer.cer(references, hypotheses)
        return cer * 100.0
    except Exception as e:
        print(f"Error computing CER: {e}")
        return 100.0


def batch_decode_tokens(
    token_sequences: List[List[int]],
    tokenizer,
) -> List[str]:
    """
    Decode token sequences to text strings.
    
    Args:
        token_sequences: List of token ID sequences
        tokenizer: Tokenizer with decode method
        
    Returns:
        List of decoded text strings
    """
    texts = []
    for tokens in token_sequences:
        try:
            text = tokenizer.decode(tokens)
            texts.append(text)
        except Exception as e:
            print(f"Error decoding tokens: {e}")
            texts.append("")
    
    return texts


class SimpleTokenizer:
    """
    Simple character-level tokenizer for SSI dataset.
    """
    
    def __init__(self, vocab: List[str] = None):
        """
        Args:
            vocab: List of vocabulary characters
        """
        # Special tokens
        self.blank_token = '<blank>'
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        self.unk_token = '<unk>'
        
        # Build vocabulary
        if vocab is None:
            # Default: English characters + Korean characters + numbers + punctuation
            vocab = [self.blank_token, self.sos_token, self.eos_token, self.unk_token]
            vocab += [' ']  # Space
            vocab += list('abcdefghijklmnopqrstuvwxyz')
            vocab += list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            vocab += list('0123456789')
            vocab += list('.,!?;:\'-"()[]{}')
            
            # Add Korean characters if needed
            # vocab += [chr(i) for i in range(0xAC00, 0xD7A4)]  # Korean syllables
        
        self.vocab = vocab
        self.char2idx = {char: idx for idx, char in enumerate(vocab)}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        
        self.blank_id = self.char2idx[self.blank_token]
        self.sos_id = self.char2idx[self.sos_token]
        self.eos_id = self.char2idx[self.eos_token]
        self.unk_id = self.char2idx[self.unk_token]
    
    def __len__(self):
        return len(self.vocab)
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            
        Returns:
            List of token IDs
        """
        tokens = []
        for char in text:
            if char in self.char2idx:
                tokens.append(self.char2idx[char])
            else:
                tokens.append(self.unk_id)
        return tokens
    
    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            tokens: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text string
        """
        special_ids = {self.blank_id, self.sos_id, self.eos_id}
        
        chars = []
        for token_id in tokens:
            if skip_special_tokens and token_id in special_ids:
                continue
            
            if token_id in self.idx2char:
                chars.append(self.idx2char[token_id])
            else:
                chars.append(self.unk_token)
        
        return ''.join(chars)
    
    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of token ID sequences
        """
        return [self.encode(text) for text in texts]
    
    def batch_decode(
        self,
        token_sequences: List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        Decode a batch of token sequences.
        
        Args:
            token_sequences: List of token ID sequences
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            List of decoded text strings
        """
        return [self.decode(tokens, skip_special_tokens) for tokens in token_sequences]


if __name__ == '__main__':
    # Test tokenizer
    tokenizer = SimpleTokenizer()
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Blank ID: {tokenizer.blank_id}")
    print(f"SOS ID: {tokenizer.sos_id}")
    print(f"EOS ID: {tokenizer.eos_id}")
    
    # Test encoding/decoding
    text = "Hello world!"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    print(f"\nOriginal: {text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")
    
    # Test metrics
    refs = ["hello world", "how are you"]
    hyps = ["hello word", "how r you"]
    
    wer = compute_wer(hyps, refs)
    cer = compute_cer(hyps, refs)
    
    print(f"\nWER: {wer:.2f}%")
    print(f"CER: {cer:.2f}%")
