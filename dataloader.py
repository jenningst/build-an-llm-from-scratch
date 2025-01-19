import torch
import tiktoken
from torch.utils.data import DataLoader, Dataset
from typing import Any, Tuple


class GPTDatasetV1(Dataset):
    def __init__(self, text: str, tokenizer: Any, max_len: int, stride: int):
        """Initialize the dataset.
        
        Args:
            text: The text data to tokenize
            tokenizer: The tokenizer to use
            max_len: The maximum length of the input sequence
            stride: The stride of the sliding window
        """
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_len, stride):
            input_chunk = token_ids[i:i + max_len]
            target_chunk = token_ids[i + 1:i + max_len + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a sample from the dataset."""
        return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader(
        text: str, 
        batch_size: int=4, 
        max_len: int=256, 
        stride: int=128, 
        shuffle: bool=True, 
        drop_last: bool=True, 
        num_workers: int=0
    ) -> DataLoader:
    """Create a DataLoader for the dataset.
    
    Args:
        text: The text data to tokenize
        batch_size: The batch size
        max_len: The maximum length of the input sequence
        stride: The stride of the sliding window
        shuffle: Whether to shuffle the data
        drop_last: Whether to drop the last batch if it's not full
        num_workers: The number of workers to use for loading the data
        
    Returns:
        DataLoader: The DataLoader for the dataset
    """
    tokenizer = tiktoken.encoding_for_model('gpt-2')
    dataset = GPTDatasetV1(text, tokenizer, max_len, stride)
    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last, 
        num_workers=num_workers
    )
    return dataloader