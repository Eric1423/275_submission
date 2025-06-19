import numpy as np
import random
import os
import argparse

import config
from tqdm import tqdm

def simulate_ca(rule_number: int, initial_state: np.ndarray, num_timesteps: int) -> np.ndarray:
    """Simulates a 1D elementary cellular automaton.

    Args:
        rule_number: The ECA rule number (0-255).
        initial_state: A 1D numpy array representing the initial state of the CA.
                       Should have shape (config.NUM_CELLS,).
        num_timesteps: The number of timesteps to simulate.

    Returns:
        A 2D numpy array of shape (num_timesteps, config.NUM_CELLS) representing 
        the history of the CA states. The first row is the initial_state.
    """
    if not (0 <= rule_number <= 255):
        raise ValueError("Rule number must be between 0 and 255.")
    if initial_state.shape != (config.NUM_CELLS,):
        raise ValueError(f"Initial state must have shape ({config.NUM_CELLS},).")
    if initial_state.dtype != np.uint8:
        initial_state = initial_state.astype(np.uint8)
    if not np.all(np.isin(initial_state, [0, 1])):
        raise ValueError("Initial state must contain only 0s and 1s.")

    rule_binary = np.array([int(bit) for bit in f'{rule_number:08b}'], dtype=np.uint8)
    
    # The history array will store initial_state as the first step (t=0)
    # and then num_timesteps-1 subsequent steps.
    # So, if num_timesteps is 256, we simulate from t=0 to t=255.
    ca_history = np.zeros((num_timesteps, config.NUM_CELLS), dtype=np.uint8)
    ca_history[0, :] = initial_state

    for t in range(1, num_timesteps):
        prev_state = ca_history[t-1, :]
        left_neighbors = np.roll(prev_state, 1)
        right_neighbors = np.roll(prev_state, -1)
        
        # Construct the 3-cell patterns (left, center, right)
        # Convert pattern from binary to integer index (0-7)
        # e.g., (1,1,1) -> 7, (0,0,0) -> 0
        pattern_indices = 4 * left_neighbors + 2 * prev_state + 1 * right_neighbors
        
        # Apply the rule: rule_binary is indexed from right-to-left (LSB is rule for 000)
        # So, for pattern index p, the output is rule_binary[7-p]
        ca_history[t, :] = rule_binary[7 - pattern_indices]
        
    return ca_history

def generate_sequence_for_rule(rule_number: int, sampling_method: str = "random") -> list[int]:
    """Generates a single tokenized sequence for a given CA rule.

    Sequence format: [RULE_ID] [TIME_0] [CELL_0_0] ... [CELL_0_255] ... [TIME_N] [CELL_N_0] ... [CELL_N_255]

    Args:
        rule_number: The CA rule number.
        sampling_method: "random" or "contiguous". Determines how timesteps are selected.

    Returns:
        A list of token IDs representing the sequence.
    """
    sequence = []

    # 1. Add rule token
    sequence.append(config.rule_to_token_id(rule_number))

    # 2. Create initial random state for the CA
    initial_ca_state = np.random.randint(0, 2, size=config.NUM_CELLS, dtype=np.uint8)

    # 3. Simulate CA for MAX_TIMESTEP_VALUE + 1 steps to cover all possible sampled timesteps
    # We need states from t=0 up to t=MAX_TIMESTEP_VALUE
    # The simulation produces timesteps 0 to MAX_TIMESTEP_VALUE (inclusive).
    total_simulated_timesteps = config.MAX_TIMESTEP_VALUE + 1
    ca_history = simulate_ca(rule_number, initial_ca_state, total_simulated_timesteps)

    # 4. Sample timesteps
    # Ensure timesteps are sorted to maintain temporal order in the sequence.
    # config.SAMPLED_TIMESTEPS defines the total number of distinct timesteps in a sequence.
    
    if sampling_method == "random":
        # Existing random sampling method
        sampled_indices = np.sort(np.random.choice(total_simulated_timesteps, 
                                                 size=config.SAMPLED_TIMESTEPS, 
                                                 replace=False))
    elif sampling_method == "contiguous":
        if config.SAMPLED_TIMESTEPS > total_simulated_timesteps:
            raise ValueError("SAMPLED_TIMESTEPS cannot be greater than total_simulated_timesteps for contiguous sampling.")
        
        # Choose a random starting point for the contiguous block of SAMPLED_TIMESTEPS.
        # The block can start at any timestep 's' such that s + SAMPLED_TIMESTEPS -1 < total_simulated_timesteps.
        # So, max_start_index = total_simulated_timesteps - config.SAMPLED_TIMESTEPS
        if total_simulated_timesteps == config.SAMPLED_TIMESTEPS: # Special case: if we sample all, start at 0
             start_index = 0
        else:
             max_start_index = total_simulated_timesteps - config.SAMPLED_TIMESTEPS
             start_index = random.randint(0, max_start_index)
        
        sampled_indices = np.arange(start_index, start_index + config.SAMPLED_TIMESTEPS)
        # np.sort is not strictly needed here as arange produces sorted, but good for consistency.
        sampled_indices = np.sort(sampled_indices) 

    else:
        raise ValueError(f"Unknown sampling_method: {sampling_method}")


    # 5. Construct the rest of the sequence
    for t_idx in sampled_indices:
        # Add time token
        sequence.append(config.time_to_token_id(t_idx))
        # Add cell state tokens
        current_state = ca_history[t_idx, :]
        for cell_val in current_state:
            sequence.append(config.cell_state_to_token_id(cell_val))
            
    return sequence

def generate_and_save_data(num_sequences: int, output_file: str, contiguous_sampling_ratio: float = 0.5, include_bos_eos: bool = False):
    """Generates multiple sequences and saves them to a .npy file.
    Each row in the .npy file will be one sequence (flattened list of token IDs).
    For autoregressive training, input is sequence[:-1] and target is sequence[1:].

    Args:
        num_sequences: Number of sequences to generate.
        output_file: Path to save the .npy file.
        contiguous_sampling_ratio: Proportion of sequences to generate using contiguous timestep sampling.
        include_bos_eos: Whether to add BOS and EOS tokens to each sequence.
    """
    all_sequences = []
    max_len_observed = 0
    random_samples_count = 0
    contiguous_samples_count = 0

    for _ in tqdm(range(num_sequences)):
        rule_number = random.randint(0, config.NUM_RULES - 1)
        
        if random.random() < contiguous_sampling_ratio:
            sampling_method = "contiguous"
            contiguous_samples_count += 1
        else:
            sampling_method = "random"
            random_samples_count += 1
            
        sequence = generate_sequence_for_rule(rule_number, sampling_method=sampling_method)
        
        if include_bos_eos:
            sequence = [config.BOS_TOKEN_ID] + sequence + [config.EOS_TOKEN_ID]

        all_sequences.append(np.array(sequence, dtype=np.int32))
        if len(sequence) > max_len_observed:
            max_len_observed = len(sequence)
    
    print(f"Maximum sequence length observed: {max_len_observed}")
    print(f"Generated {random_samples_count} sequences with random sampling.")
    print(f"Generated {contiguous_samples_count} sequences with contiguous sampling.")
    
    # Ensure DATA_DIR exists
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    # Pad sequences to the max_len_observed if lengths vary (they shouldn't much with fixed SAMPLED_TIMESTEPS)
    # However, if include_bos_eos changes things, or if we plan to vary SAMPLED_TIMESTEPS later.
    # For now, with fixed config, all sequences should have the same length: 
    # core_len = 1 (rule) + SAMPLED_TIMESTEPS * (1 (time) + NUM_CELLS)
    # with bos/eos: core_len + 2
    # Let's verify the expected length:
    expected_len = config.MAX_SEQ_LEN_CORE
    if include_bos_eos:
        expected_len += 2
    
    print(f"Expected sequence length with current config: {expected_len}")
    if max_len_observed != expected_len:
        print(f"Warning: Max observed length {max_len_observed} differs from expected {expected_len}. Padding will be based on max_len_observed.")

    # Pad all sequences to max_len_observed with PAD_TOKEN_ID
    # This is crucial for batching if sequences aren't naturally the same length.
    padded_sequences = np.full((num_sequences, max_len_observed), config.PAD_TOKEN_ID, dtype=np.int32)
    for i, seq in enumerate(all_sequences):
        padded_sequences[i, :len(seq)] = seq
        
    np.save(output_file, padded_sequences)
    print(f"Saved {num_sequences} sequences to {output_file}. Shape: {padded_sequences.shape}")

def decode_sequence_to_human_readable(token_sequence: np.ndarray) -> str:
    """Converts a token sequence into a human-readable string representation.

    Args:
        token_sequence: A 1D numpy array of token IDs.

    Returns:
        A string detailing the decoded sequence.
    """
    output_lines = []
    
    idx = 0
    
    # Handle BOS token
    if token_sequence[idx] == config.BOS_TOKEN_ID:
        output_lines.append("BOS_TOKEN")
        idx += 1
        if idx >= len(token_sequence):
            output_lines.append("Sequence ended after BOS token.")
            return "\n".join(output_lines)

    # Decode Rule ID
    if config.is_rule_token(token_sequence[idx]):
        rule_id = config.token_id_to_rule(token_sequence[idx])
        output_lines.append(f"RULE_ID: {rule_id}")
        idx += 1
    else:
        output_lines.append(f"ERROR: Expected RULE_ID token, got {token_sequence[idx]}")
        return "\n".join(output_lines)

    # Decode Timesteps and Cell States
    while idx < len(token_sequence):
        if token_sequence[idx] == config.EOS_TOKEN_ID:
            output_lines.append("EOS_TOKEN")
            idx += 1
            break 
        if token_sequence[idx] == config.PAD_TOKEN_ID:
            output_lines.append(f"PAD_TOKEN encountered. {len(token_sequence) - idx} padding tokens remaining.")
            break

        # Decode Time ID
        if idx < len(token_sequence) and config.is_time_token(token_sequence[idx]):
            time_val = config.token_id_to_time(token_sequence[idx])
            output_lines.append(f"  TIME_{time_val}:")
            idx += 1
        else:
            output_lines.append(f"ERROR: Expected TIME_ID token, got {token_sequence[idx] if idx < len(token_sequence) else 'EOS'}")
            break 

        # Decode Cell States
        cell_states = []
        for _ in range(config.NUM_CELLS):
            if idx < len(token_sequence) and config.is_cell_token(token_sequence[idx]):
                cell_states.append(str(config.token_id_to_cell_state(token_sequence[idx])))
                idx += 1
            else:
                output_lines.append(f"ERROR: Expected CELL_STATE token, got {token_sequence[idx] if idx < len(token_sequence) else 'EOS'}. Collected {len(cell_states)} cells for TIME_{time_val}.")
                return "\n".join(output_lines) # End processing for this sequence due to error
        
        output_lines.append(f"    CELLS: [{''.join(cell_states[:16])}...{''.join(cell_states[-16:])}] (Total: {len(cell_states)})" )

    if idx < len(token_sequence) and token_sequence[idx] != config.PAD_TOKEN_ID :
         output_lines.append(f"Processing finished, but remaining tokens: {token_sequence[idx:]}")
            
    return "\n".join(output_lines)

def main():
    parser = argparse.ArgumentParser(description="Generate Cellular Automata training and validation data.")
    parser.add_argument(
        "--num_train_samples", 
        type=int, 
        default=1000, 
        help="Number of training sequences to generate."
    )
    parser.add_argument(
        "--num_val_samples", 
        type=int, 
        default=100, 
        help="Number of validation sequences to generate."
    )
    parser.add_argument(
        "--include_bos_eos", 
        action='store_true', 
        help="Include BOS and EOS tokens in the sequences."
    )
    parser.add_argument(
        "--contiguous_ratio_train", 
        type=float, 
        default=0.7,
        help="Proportion of training data using contiguous timestep sampling (0.0 to 1.0)."
    )
    parser.add_argument(
        "--contiguous_ratio_val", 
        type=float, 
        default=0.7,
        help="Proportion of validation data using contiguous timestep sampling (0.0 to 1.0)."
    )

    args = parser.parse_args()

    train_output_file = os.path.join(config.DATA_DIR, f"train_sequences_{args.num_train_samples}s_cr{args.contiguous_ratio_train}.npy")
    val_output_file = os.path.join(config.DATA_DIR, f"val_sequences_{args.num_val_samples}s_cr{args.contiguous_ratio_val}.npy")

    print(f"Generating {args.num_train_samples} training samples with contiguous ratio {args.contiguous_ratio_train}...")
    generate_and_save_data(args.num_train_samples, train_output_file, 
                           contiguous_sampling_ratio=args.contiguous_ratio_train, 
                           include_bos_eos=args.include_bos_eos)

    print(f"\nGenerating {args.num_val_samples} validation samples with contiguous ratio {args.contiguous_ratio_val}...")
    generate_and_save_data(args.num_val_samples, val_output_file, 
                           contiguous_sampling_ratio=args.contiguous_ratio_val,
                           include_bos_eos=args.include_bos_eos)

    print("\nData generation complete.")
    print(f"Training data saved to: {train_output_file}")
    print(f"Validation data saved to: {val_output_file}")

if __name__ == '__main__':
    main()

    print(f"\nPlease check the '{config.DATA_DIR}' directory for output files.") 