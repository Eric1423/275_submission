import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import config
from data_utils import simulate_ca, decode_sequence_to_human_readable

def plot_ca_simulation(automaton_history: np.ndarray, title: str = "Cellular Automaton", save_path: str = None):
    """Plots the evolution of a cellular automaton.

    Args:
        automaton_history: A 2D numpy array (timesteps, cells) from simulate_ca.
        title: The title for the plot.
        save_path: Optional path to save the plot. If None, shows the plot directly.
    """
    plt.figure(figsize=(10, 10 * automaton_history.shape[0] / automaton_history.shape[1]))
    plt.imshow(automaton_history, cmap='binary', interpolation='nearest')
    plt.title(title)
    plt.xlabel("Cell Index")
    plt.ylabel("Time Step")
    
    if save_path:
        # Ensure the directory exists if save_path includes a directory part
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close() # Close the figure to free memory and prevent display
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

def visualize_all_rules(output_dir: str = "rules"):
    """Simulates and saves visualizations for all 256 elementary CA rules."""
    print(f"\\n--- Visualizing all {config.NUM_RULES} ECA rules into '{output_dir}' directory ---")
    initial_state = np.zeros(config.NUM_CELLS, dtype=np.uint8)
    initial_state[config.NUM_CELLS // 2] = 1 # Single 1 in the middle
    num_timesteps = config.DEFAULT_SIMULATION_TIMESTEPS

    # Ensure the main output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for rule_number in tqdm(range(config.NUM_RULES), desc="Visualizing all rules"):
        automaton_history = simulate_ca(rule_number, initial_state.copy(), num_timesteps)
        save_filename = f"rule_{rule_number:03d}.png"
        full_save_path = os.path.join(output_dir, save_filename)
        plot_ca_simulation(automaton_history, 
                           title=f"Rule {rule_number}", 
                           save_path=full_save_path)
    print(f"Finished visualizing all rules. Check the '{output_dir}' directory.")

if __name__ == '__main__':
    # --- Visualize CA from generated data ---
    print("\\nVisualizing CA from generated data (first few samples)...")
    
    # First, ensure data exists (it might be generated by data_utils.py's main)
    if not os.path.exists(config.TRAIN_SEQUENCES_FILE):
        print(f"Training data {config.TRAIN_SEQUENCES_FILE} not found. Cannot visualize.")
        print(f"Please run the main script in data_utils.py first to generate data.")
    else:
        loaded_data = np.load(config.TRAIN_SEQUENCES_FILE)
        num_samples_to_visualize = 2 
        num_samples_to_decode = 2 # This part uses decode_sequence_to_human_readable

        # Call the function to visualize all rules
        visualize_all_rules() # This will save to a directory named "rules"

        print("\\n--- Decoding first few samples (from visualize_utils.py) ---")
        for i in range(min(num_samples_to_decode, len(loaded_data))):
            print(f"\\n--- Decoded Sample {i} ---")
            # decode_sequence_to_human_readable is imported from data_utils
            decoded_str = decode_sequence_to_human_readable(loaded_data[i])
            print(decoded_str)

        for i in range(min(num_samples_to_visualize, len(loaded_data))):
            print(f"\\n--- Visualizing Sample {i} (from visualize_utils.py) ---")
            token_sequence = loaded_data[i]
            
            start_index = 0
            if token_sequence[0] == config.BOS_TOKEN_ID:
                start_index = 1
            
            rule_token_id = token_sequence[start_index]
            if not config.is_rule_token(rule_token_id):
                print(f"Sample {i} does not start with a valid rule token. First token: {rule_token_id}. Skipping visualization.")
                continue
            
            rule_number = config.token_id_to_rule(rule_token_id)
            print(f"Decoded Rule Number: {rule_number}")

            current_token_idx = start_index + 1
            initial_state_for_sim = None
            found_first_state = False

            while current_token_idx < len(token_sequence):
                if token_sequence[current_token_idx] == config.EOS_TOKEN_ID or \
                   token_sequence[current_token_idx] == config.PAD_TOKEN_ID:
                    break

                if config.is_time_token(token_sequence[current_token_idx]):
                    current_token_idx += 1 
                    
                    current_ca_state = []
                    for _ in range(config.NUM_CELLS):
                        if current_token_idx < len(token_sequence) and \
                           config.is_cell_token(token_sequence[current_token_idx]):
                            current_ca_state.append(config.token_id_to_cell_state(token_sequence[current_token_idx]))
                        else:
                            print(f"Warning: Ran out of cell tokens or found non-cell token for sample {i}. State may be incomplete.")
                            break 
                        current_token_idx += 1
                    
                    if len(current_ca_state) == config.NUM_CELLS:
                        initial_state_for_sim = np.array(current_ca_state, dtype=np.uint8)
                        found_first_state = True
                        break 
                    else:
                        print(f"Warning: Incomplete state for sample {i} at first time step. Needed {config.NUM_CELLS}, got {len(current_ca_state)}. Skipping.")
                        break 
                else:
                    current_token_idx += 1 
            
            if found_first_state and initial_state_for_sim is not None:
                num_vis_steps = 64 
                print(f"Simulating Rule {rule_number} for {num_vis_steps} steps from the first state in the sequence.")
                # simulate_ca is imported from data_utils
                simulated_history = simulate_ca(rule_number, initial_state_for_sim, num_timesteps=num_vis_steps)
                # plot_ca_simulation is local to this file
                plot_ca_simulation(simulated_history, title=f"Sample {i} - Rule {rule_number} (Simulated from first state in sequence)")
            elif not found_first_state:
                 print(f"Could not find a complete initial state in sequence for sample {i} to visualize.")
            else:
                print(f"Initial state for simulation was not properly set for sample {i}. Skipping visualization.") 