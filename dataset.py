import torch
from torch.utils.data import Dataset
import numpy as np
import config
from data_utils import decode_sequence_to_human_readable, simulate_ca #, plot_ca_simulation
import os

class CATokenDataset(Dataset):
    """PyTorch Dataset for CA token sequences.
    Each sequence is truncated or padded to block_size."""
    def __init__(self, file_path: str, block_size: int):
        """
        Args:
            file_path (str): Path to the .npy file containing token sequences.
            block_size (int): The maximum sequence length for the model. 
                              Sequences will be truncated or padded to this size.
        """
        self.all_sequences = np.load(file_path).astype(np.int64)
        self.block_size = block_size
        
        print(f"Loaded data from {file_path}, shape: {self.all_sequences.shape}")
        print(f"Using block_size: {self.block_size}. Sequences will be truncated/padded.")
        print(f"Number of sequences: {len(self.all_sequences)}")

    def __len__(self):
        return len(self.all_sequences)

    def __getitem__(self, idx):
        token_sequence = self.all_sequences[idx]
        seq_len = len(token_sequence)

        input_ids = np.full(self.block_size, config.PAD_TOKEN_ID, dtype=np.int64)
        attention_mask = np.zeros(self.block_size, dtype=np.int64)

        if seq_len > self.block_size:
            # Truncate
            input_ids[:self.block_size] = token_sequence[:self.block_size]
            attention_mask[:self.block_size] = 1
            # actual_len = self.block_size # not strictly needed for this simplified logic
        else:
            # Pad (or exact fit)
            input_ids[:seq_len] = token_sequence
            attention_mask[:seq_len] = 1
            # actual_len = seq_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(input_ids, dtype=torch.long) # Labels are the same for Causal LM
        }
