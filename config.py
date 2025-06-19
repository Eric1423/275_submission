import numpy as np

# --- Token Definitions ---
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1  # Beginning of Sequence
EOS_TOKEN_ID = 2  # End of Sequence

BIT_0_TOKEN_ID = 3
BIT_1_TOKEN_ID = 4

# Rule tokens: ECA rules 0-255
# We'll map rule number N to token ID RULE_TOKENS_START_ID + N
NUM_RULES = 256
RULE_TOKENS_START_ID = 5
FIRST_RULE_TOKEN_ID = RULE_TOKENS_START_ID
LAST_RULE_TOKEN_ID = RULE_TOKENS_START_ID + NUM_RULES - 1

# Time tokens: Timesteps 0-255
# We'll map timestep T to token ID TIME_TOKENS_START_ID + T
NUM_TIMESTEPS_MAX = 256 # Max value for a timestep
TIME_TOKENS_START_ID = LAST_RULE_TOKEN_ID + 1
FIRST_TIME_TOKEN_ID = TIME_TOKENS_START_ID
LAST_TIME_TOKEN_ID = TIME_TOKENS_START_ID + NUM_TIMESTEPS_MAX - 1

VOCAB_SIZE = LAST_TIME_TOKEN_ID + 1

# --- Data Generation Parameters ---
NUM_CELLS = 256  # Number of cells in the 1D CA
# SAMPLED_TIMESTEPS = 15 # Number of timesteps to sample for each sequence
SAMPLED_TIMESTEPS = 7
MAX_TIMESTEP_VALUE = 255 # Max timestep index that can be sampled (from 0 to 255)

# --- Sequence Length ---
# Max sequence length:
# 1 (RULE_TOKEN)
# + SAMPLED_TIMESTEPS * (1 (TIME_TOKEN) + NUM_CELLS (BIT_TOKENS))
# + 1 (BOS_TOKEN) - Optional, depending on model architecture
# + 1 (EOS_TOKEN) - Optional, depending on model architecture
# For now, let's calculate the core part:
MAX_SEQ_LEN_CORE = 1 + SAMPLED_TIMESTEPS * (1 + NUM_CELLS)
# Example: 1 + 15 * (1 + 256) = 1 + 15 * 257 = 1 + 3855 = 3856
# We might add BOS/EOS tokens later, adjusting this. For now, this is the length of the data payload.

# --- Simulation Parameters ---
DEFAULT_SIMULATION_TIMESTEPS = 256 # Default number of steps to run a full simulation for visualization/ground truth

# --- Training Hyperparameters (placeholders) ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3


# --- File Paths ---
DATA_DIR = "/data1/disrael/ca-transformer/data"
TRAIN_SEQUENCES_FILE = f"{DATA_DIR}/train_sequences.npy"
VAL_SEQUENCES_FILE = f"{DATA_DIR}/val_sequences.npy"
# If we store them as token IDs.

# --- Utility Functions for Tokenization ---
def rule_to_token_id(rule_number: int) -> int:
    """Converts an ECA rule number (0-255) to its corresponding token ID."""
    if not (0 <= rule_number < NUM_RULES):
        raise ValueError(f"Rule number must be between 0 and {NUM_RULES-1}.")
    return RULE_TOKENS_START_ID + rule_number

def token_id_to_rule(token_id: int) -> int:
    """Converts a rule token ID back to the ECA rule number."""
    if not (FIRST_RULE_TOKEN_ID <= token_id <= LAST_RULE_TOKEN_ID):
        raise ValueError(f"Token ID is not a valid rule token ID.")
    return token_id - RULE_TOKENS_START_ID

def time_to_token_id(time_step: int) -> int:
    """Converts a timestep (0-255) to its corresponding token ID."""
    if not (0 <= time_step < NUM_TIMESTEPS_MAX):
        raise ValueError(f"Time step must be between 0 and {NUM_TIMESTEPS_MAX-1}.")
    return TIME_TOKENS_START_ID + time_step

def token_id_to_time(token_id: int) -> int:
    """Converts a time token ID back to the timestep value."""
    if not (FIRST_TIME_TOKEN_ID <= token_id <= LAST_TIME_TOKEN_ID):
        raise ValueError(f"Token ID is not a valid time token ID.")
    return token_id - TIME_TOKENS_START_ID

def cell_state_to_token_id(cell_state: int) -> int:
    """Converts a cell state (0 or 1) to its token ID."""
    if cell_state == 0:
        return BIT_0_TOKEN_ID
    elif cell_state == 1:
        return BIT_1_TOKEN_ID
    else:
        raise ValueError("Cell state must be 0 or 1.")

def token_id_to_cell_state(token_id: int) -> int:
    """Converts a bit token ID back to the cell state (0 or 1)."""
    if token_id == BIT_0_TOKEN_ID:
        return 0
    elif token_id == BIT_1_TOKEN_ID:
        return 1
    else:
        raise ValueError(f"Token ID {token_id} is not a valid bit token ID.")

# Utility functions for token identification (New)
def is_time_token(token_id: int) -> bool:
    return FIRST_TIME_TOKEN_ID <= token_id <= LAST_TIME_TOKEN_ID

def is_cell_token(token_id: int) -> bool:
    return token_id == BIT_0_TOKEN_ID or token_id == BIT_1_TOKEN_ID

def is_rule_token(token_id: int) -> bool:
    return FIRST_RULE_TOKEN_ID <= token_id <= LAST_RULE_TOKEN_ID

if __name__ == '__main__':
    print(f"Vocabulary Size: {VOCAB_SIZE}")
    print(f"Max Core Sequence Length: {MAX_SEQ_LEN_CORE}")
    print(f"Rule 30 token ID: {rule_to_token_id(30)}")
    print(f"Time 100 token ID: {time_to_token_id(100)}")
    print(f"Bit 0 token ID: {BIT_0_TOKEN_ID}")
    print(f"Bit 1 token ID: {BIT_1_TOKEN_ID}")
    # Test round trip
    assert token_id_to_rule(rule_to_token_id(30)) == 30
    assert token_id_to_time(time_to_token_id(100)) == 100
    assert token_id_to_cell_state(BIT_0_TOKEN_ID) == 0
    assert token_id_to_cell_state(BIT_1_TOKEN_ID) == 1
    print("Config checks passed.") 