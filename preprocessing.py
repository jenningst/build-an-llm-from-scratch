import re
from typing import List, Dict


class TokenizerV1:
    def __init__(self, corpus: str):
        self.vocab = self._create_vocabulary(corpus)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize the text into words and punctuation.
        
        Args:
            text: The text to tokenize
            
        Returns:
            List of tokens
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        if not text.strip():
            raise ValueError("Input cannot be empty")
        
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        return preprocessed
    
    def _create_vocabulary(self, corpus: str) -> Dict[str, int]:
        """Create a vocabulary from the corpus.
        
        Args:
            corpus: The corpus to create the vocabulary from
        
        Returns:
            Vocabulary
        """
        tokens = self._tokenize(corpus)
        vocab = {token:i for i, token in enumerate(tokens)}
        return vocab

    def _validate_vocabulary(self, vocab: Dict[str, int]) -> None:
        """Validate the vocabulary.
        
        Args:
            vocab: The vocabulary to validate
            
        Raises:
            ValueError: If the vocabulary is invalid
        """
        if not isinstance(vocab, dict):
            raise ValueError("Vocabulary must be a dictionary")
        if not vocab:
            raise ValueError("Vocabulary cannot be empty")
        if not all(isinstance(token, str) for token in vocab):
            raise ValueError("All vocabulary items must be strings")
        if not all(token.strip() for token in vocab):
            raise ValueError("Vocabulary items cannot be empty strings")

    def _to_ids(self, tokens: List[str]) -> List[int]:
        """Convert the tokens to IDs.
        
        Args:
            tokenizer: The tokenizer to use
            
        Returns:
            List of IDs
        """
        try:
            return [self.vocab[token] for token in tokens]
        except KeyError as e:
            raise KeyError(f"Token not found in vocabulary: {str(e)}")
    
    def encode(self, text: str) -> List[int]:
        """Encode the text into IDs.
        
        Args:
            text: The text to encode
            
        Returns:
            List of IDs
        """
        self._validate_vocabulary(self.vocab)
        tokens = self._tokenize(text)
        return self._to_ids(tokens)
    
    def decode(self, ids: List[int]) -> str:
        """Decode the IDs back to text.
        
        Args:
            ids: The IDs to decode
            
        Returns:
            Decoded text
        """
        self._validate_vocabulary(self.vocab)
        try:
            text = ' '.join(self.vocab[id] for id in ids)
            text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        except KeyError as e:
            raise KeyError(f"ID not found in vocabulary: {str(e)}")
        return text