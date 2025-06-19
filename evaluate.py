import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoConfig
import random
import os
import math
import argparse
from tqdm.auto import tqdm

import config
import data_utils # For simulation and plotting
from dataset import CATokenDataset
from model import get_ca_model

# This function is mostly for the qualitative evaluation later
def generate_ca_sequence_for_qualitative_eval(
    model: AutoModelForCausalLM,
    rule_number: int,
    initial_timesteps: list[int], 
    initial_states: list[np.ndarray], 
    num_predict_timesteps: int,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> tuple[list[int], list[np.ndarray]]:
    """Generates subsequent CA timesteps using the model's .generate() method,
    one timestep at a time by providing the next TIME token.

    Args:
        model: The trained AutoModelForCausalLM.
        rule_number: The CA rule number (0-255).
        initial_timesteps: A list of actual timesteps provided as context.
        initial_states: A list of 1D np.ndarrays (config.NUM_CELLS,) for the initial_timesteps.
        num_predict_timesteps: How many new timesteps (and their states) to predict.
        device: torch device ('cuda' or 'cpu')

    Returns:
        A tuple: (all_timesteps, all_states)
          all_timesteps (list[int]): Original + predicted timesteps.
          all_states (list[np.ndarray]): Original + predicted CA states.
    """
    model.eval()
    model.to(device)

    if len(initial_timesteps) != len(initial_states):
        raise ValueError("initial_timesteps and initial_states must have the same length.")

    base_prompt_tokens = [config.BOS_TOKEN_ID, config.rule_to_token_id(rule_number)]
    current_full_sequence_tokens = list(base_prompt_tokens)

    for t_idx, state_arr in zip(initial_timesteps, initial_states):
        if state_arr.shape != (config.NUM_CELLS,):
            raise ValueError(f"Each initial state must have shape ({config.NUM_CELLS},)")
        current_full_sequence_tokens.append(config.time_to_token_id(t_idx))
        current_full_sequence_tokens.extend(config.cell_state_to_token_id(int(cell_val)) for cell_val in state_arr)
    
    all_timesteps = list(initial_timesteps)
    all_states = [s.copy() for s in initial_states]
    
    last_known_timestep_val = initial_timesteps[-1] if initial_timesteps else -1
    # Fallback if max_position_embeddings is None or 0, though unlikely for pretrained models.
    model_max_len = model.config.max_position_embeddings if model.config.max_position_embeddings else 2048

    with torch.no_grad():
        for i in tqdm(range(num_predict_timesteps), desc="Generating Timesteps"):
            actual_next_timestep_val = last_known_timestep_val + 1 + i 
            
            timestep_val_for_tokenization: int
            if actual_next_timestep_val > config.MAX_TIMESTEP_VALUE:
                timestep_val_for_tokenization = config.MAX_TIMESTEP_VALUE
            else:
                timestep_val_for_tokenization = actual_next_timestep_val
            
            next_time_token_id = config.time_to_token_id(timestep_val_for_tokenization)
            
            # Prepare the history part of the prompt, possibly truncating it
            # Max length for history: model_max_len - (tokens for next step: time_token + NUM_CELLS) - safety_buffer
            max_history_len_for_prompt = model_max_len - config.NUM_CELLS - 1 - 5 # 5 for safety/EOS/PAD
            
            current_history_for_prompt = list(current_full_sequence_tokens) # Make a copy

            if len(current_history_for_prompt) > max_history_len_for_prompt:
                num_dynamic_tokens_to_keep = max_history_len_for_prompt - 2 # -2 for BOS, RULE
                if num_dynamic_tokens_to_keep > 0:
                    dynamic_part = current_history_for_prompt[2:] # Exclude BOS, RULE
                    truncated_dynamic_part = dynamic_part[-num_dynamic_tokens_to_keep:]
                    current_history_for_prompt = current_history_for_prompt[:2] + truncated_dynamic_part
                else: # Only space for BOS, RULE or less (highly unlikely with reasonable model_max_len)
                    current_history_for_prompt = current_history_for_prompt[:max_history_len_for_prompt]
            
            # Construct the actual prompt for generating cells of the current timestep
            prompt_tokens_for_cell_generation = current_history_for_prompt + [next_time_token_id]
            input_ids_for_cells = torch.tensor([prompt_tokens_for_cell_generation], dtype=torch.long).to(device)
            attention_mask_for_cells = torch.ones_like(input_ids_for_cells).to(device)
            current_prompt_len = input_ids_for_cells.shape[1]

            if current_prompt_len + config.NUM_CELLS > model_max_len:
                break
            
            generated_output = model.generate(
                input_ids_for_cells,
                attention_mask=attention_mask_for_cells,
                max_new_tokens=config.NUM_CELLS,
                pad_token_id=config.PAD_TOKEN_ID,
                eos_token_id=config.EOS_TOKEN_ID, 
                bos_token_id=config.BOS_TOKEN_ID, 
                do_sample=False 
            )
            
            generated_cell_tokens = generated_output[0][current_prompt_len:].tolist()

            if len(generated_cell_tokens) < config.NUM_CELLS:
                break

            current_cell_state_ids = []
            valid_state_generated = True
            for k_cell in range(config.NUM_CELLS):
                cell_token_id = generated_cell_tokens[k_cell]
                if not config.is_cell_token(cell_token_id):
                    valid_state_generated = False
                    break
                current_cell_state_ids.append(cell_token_id)
            
            if not valid_state_generated:
                break

            predicted_state_arr = np.array([config.token_id_to_cell_state(t) for t in current_cell_state_ids], dtype=np.uint8)
            
            # Add to our tracking lists
            all_states.append(predicted_state_arr)
            all_timesteps.append(actual_next_timestep_val)
            
            # Update current_full_sequence_tokens with the confirmed step (using potentially frozen time token + generated cells)
            current_full_sequence_tokens.extend([next_time_token_id] + current_cell_state_ids)
            
    return all_timesteps, all_states

def compute_metrics_for_trainer(eval_pred):
    logits_np, labels_np = eval_pred
    logits = torch.from_numpy(logits_np)
    labels = torch.from_numpy(labels_np)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=config.PAD_TOKEN_ID)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    perplexity = math.exp(loss.item())
    return {"perplexity": perplexity, "eval_loss": loss.item()}

def evaluate_checkpoint_quantitatively(args):
    """Evaluates a checkpoint on the validation set for perplexity and loss."""
    print(f"Starting quantitative evaluation for checkpoint: {args.checkpoint_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        # Load config from checkpoint first to get base model name if needed, and block_size
        loaded_model_config = AutoConfig.from_pretrained(args.checkpoint_path, local_files_only=True)
        block_size = loaded_model_config.max_position_embeddings
        # The _name_or_path in the checkpoint's config should ideally be the original base model id, 
        # or the path it was loaded from if it was a local custom model before training.
        # If it refers to the checkpoint itself, provide a sensible default or make it an arg.
        base_model_name_or_path = getattr(loaded_model_config, '_name_or_path', 'EleutherAI/pythia-70m') # Fallback
        print(f"Determined block_size from checkpoint config: {block_size}")
        print(f"Base model name/path from checkpoint config: {base_model_name_or_path}")
    except Exception as e:
        print(f"Error loading model config from checkpoint {args.checkpoint_path}: {e}")
        return

    if not os.path.exists(config.VAL_SEQUENCES_FILE):
        print(f"Validation data file not found: {config.VAL_SEQUENCES_FILE}")
        return
        
    val_dataset = CATokenDataset(file_path=config.VAL_SEQUENCES_FILE, block_size=block_size)
    print(f"Loaded {len(val_dataset)} validation samples from {config.VAL_SEQUENCES_FILE}.")

    try:
        # Use the modified get_ca_model to load and ensure vocab consistency
        model = get_ca_model(model_name_or_path=base_model_name_or_path, checkpoint_path=args.checkpoint_path).to(device)
        print(f"Successfully got model via get_ca_model, using checkpoint {args.checkpoint_path}, now on {device}.")
        # The get_ca_model function handles vocab size consistency checks and logging.
    except Exception as e:
        print(f"Error using get_ca_model with checkpoint {args.checkpoint_path}: {e}")
        return

    eval_output_dir_quant = os.path.join(args.checkpoint_path, "eval_quant_results")
    os.makedirs(eval_output_dir_quant, exist_ok=True)

    training_args_for_eval = TrainingArguments(
        output_dir=eval_output_dir_quant,
        per_device_eval_batch_size=args.eval_batch_size,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args_for_eval,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_for_trainer,
    )

    eval_results = trainer.evaluate()
    print(f"\n--- Quantitative Evaluation Results for: {args.checkpoint_path} ---")
    for key, value in eval_results.items():
        print(f"  {key}: {value}")
    print("----------------------------------------------------")

def evaluate_checkpoint_qualitatively(args):
    """Generates CA, compares to ground truth, calculates bit accuracy, and visualizes."""
    print(f"Starting qualitative evaluation for Rule {args.rule_number} from checkpoint: {args.checkpoint_path}")
    
    # Create a 'figures/qualitative' directory in the CWD if it doesn't exist
    figures_subdir = "qualitative"
    figures_dir = os.path.join("figures", figures_subdir)
    os.makedirs(figures_dir, exist_ok=True)
    print(f"Saving figures to: {os.path.abspath(figures_dir)}")

    # Create a 'logs' directory in the CWD if it doesn't exist
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    print(f"Saving logs to: {os.path.abspath(logs_dir)}")

    # Create directories for ground truth and prediction data
    gt_dir = "logs/results/ground_truth"
    pred_dir = "logs/results/prediction"
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    print(f"Saving ground truth data to: {os.path.abspath(gt_dir)}")
    print(f"Saving prediction data to: {os.path.abspath(pred_dir)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        loaded_model_config = AutoConfig.from_pretrained(args.checkpoint_path, local_files_only=True)
        base_model_name_or_path = getattr(loaded_model_config, '_name_or_path', 'EleutherAI/pythia-70m')
        print(f"Base model name/path from checkpoint config: {base_model_name_or_path}")
        model = get_ca_model(model_name_or_path=base_model_name_or_path, checkpoint_path=args.checkpoint_path).to(device)
        print(f"Successfully got model via get_ca_model, using checkpoint {args.checkpoint_path}, now on {device}.")
    except Exception as e:
        print(f"Error loading model via get_ca_model for qualitative eval: {e}")
        return

    initial_ca_state_t0 = np.zeros(config.NUM_CELLS, dtype=np.uint8)
    initial_ca_state_t0[config.NUM_CELLS // 2] = 1
    print(f"Using center_one initial state for T=0 as the base.")

    if args.start_timestep < 0:
        print("Error: --start_timestep cannot be negative.")
        return
    if args.num_prompt_timesteps <= 0:
        print("Error: --num_prompt_timesteps must be positive.")
        return

    print(f"Rule: {args.rule_number}, Start Timestep: {args.start_timestep}")
    print(f"Prompting with {args.num_prompt_timesteps} timesteps, Predicting {args.num_predict_timesteps} timesteps.")

    max_prompt_timestep_val = args.start_timestep + args.num_prompt_timesteps
    full_simulation_for_prompt_and_gt = data_utils.simulate_ca(args.rule_number, initial_ca_state_t0, num_timesteps=max_prompt_timestep_val + args.num_predict_timesteps)

    if len(full_simulation_for_prompt_and_gt) < max_prompt_timestep_val:
        print(f"Error: Simulation for prompt failed. Needed {max_prompt_timestep_val} timesteps, got {len(full_simulation_for_prompt_and_gt)}.)")
        return

    prompt_timesteps_for_model = list(range(args.start_timestep, max_prompt_timestep_val))
    prompt_states_for_model = [full_simulation_for_prompt_and_gt[t] for t in prompt_timesteps_for_model]

    print(f"Generating {args.num_predict_timesteps} timesteps from the model...")
    model_timesteps, model_states = generate_ca_sequence_for_qualitative_eval(
        model,
        args.rule_number,
        initial_timesteps=prompt_timesteps_for_model,
        initial_states=prompt_states_for_model,
        num_predict_timesteps=args.num_predict_timesteps,
        device=device
    )

    if not model_states:
        print("Model did not generate any states.")
        return

    ground_truth_total_timesteps = args.start_timestep + args.num_prompt_timesteps + args.num_predict_timesteps
    if len(full_simulation_for_prompt_and_gt) < ground_truth_total_timesteps:
        print(f"Warning: Ground truth simulation shorter than requested total. GT Timesteps: {len(full_simulation_for_prompt_and_gt)}, Requested: {ground_truth_total_timesteps}")
        # Adjust ground truth display to what's available
        ground_truth_display_states = full_simulation_for_prompt_and_gt[args.start_timestep:]
    else:
        ground_truth_display_states = full_simulation_for_prompt_and_gt[args.start_timestep:ground_truth_total_timesteps]
    
    ground_truth_combined_states_arr = ground_truth_display_states if ground_truth_display_states.shape[0] > 0 else np.array([]).reshape(0, config.NUM_CELLS)
    model_generated_combined_states_arr = np.array(model_states) # model_states already includes the prompt part from generate_ca_sequence

    # Save the generated data to CSV files
    gt_filename = os.path.join(gt_dir, f"rule{args.rule_number}_ground_truth.csv")
    predicted_filename = os.path.join(pred_dir, f"rule{args.rule_number}_predicted.csv")
    np.savetxt(gt_filename, ground_truth_combined_states_arr, delimiter=",", fmt='%d')
    np.savetxt(predicted_filename, model_generated_combined_states_arr, delimiter=",", fmt='%d')
    print(f"Saved ground truth data to {gt_filename}")
    print(f"Saved predicted data to {predicted_filename}")

    log_text_lines = []
    log_text_lines.append(f"Qualitative Evaluation Log")
    log_text_lines.append(f"Checkpoint: {args.checkpoint_path}")
    log_text_lines.append(f"Rule: {args.rule_number}, Initial State: center_one")
    log_text_lines.append(f"Prompt Timesteps: {args.num_prompt_timesteps} (from T={args.start_timestep} to T={args.start_timestep + args.num_prompt_timesteps -1})")
    log_text_lines.append(f"Predicted Timesteps: {args.num_predict_timesteps}")

    if model_generated_combined_states_arr.shape[0] > args.num_prompt_timesteps and ground_truth_combined_states_arr.shape[0] > args.num_prompt_timesteps:
        num_predicted_by_model = model_generated_combined_states_arr.shape[0] - args.num_prompt_timesteps
        num_comparable_predictions = min(num_predicted_by_model, ground_truth_combined_states_arr.shape[0] - args.num_prompt_timesteps, args.num_predict_timesteps)
        
        if num_comparable_predictions > 0:
            model_predicted_part = model_generated_combined_states_arr[args.num_prompt_timesteps : args.num_prompt_timesteps + num_comparable_predictions]
            gt_predicted_part = ground_truth_combined_states_arr[args.num_prompt_timesteps : args.num_prompt_timesteps + num_comparable_predictions]
            
            correct_bits = np.sum(model_predicted_part == gt_predicted_part)
            total_bits = model_predicted_part.size
            bit_accuracy = correct_bits / total_bits if total_bits > 0 else 0.0
            log_text_lines.append(f"Bit Accuracy over {num_comparable_predictions} predicted steps: {bit_accuracy:.4f} ({correct_bits}/{total_bits} correct bits)")
            print(f"Bit Accuracy over {num_comparable_predictions} predicted steps: {bit_accuracy:.4f}")
        else:
            log_text_lines.append("Not enough common predicted timesteps to compare for bit accuracy.")
            print("Not enough common predicted timesteps to compare for bit accuracy.")
    else:
        log_text_lines.append("Not enough data for bit accuracy calculation (either model or GT missing predicted steps).")
        print("Not enough data for bit accuracy calculation.")

    print("\nPlotting ground truth vs model generation...")
    fig_width_inches = 20.0
    subplot_height_per_timestep_inches = 0.25
    min_subplot_content_height = 2.0 
    max_subplot_content_height = 10.0 
    
    num_rows_plot = model_generated_combined_states_arr.shape[0]
    if num_rows_plot == 0:
        print("No data to plot for model generation (prompt + prediction).")
        # Save log even if plot fails, now to logs_dir
        log_filename_only = os.path.join(logs_dir, f"{filename_base}_accuracy.txt") # Use logs_dir
        with open(log_filename_only, 'w') as f:
            f.write("\n".join(log_text_lines))
        print(f"Saved accuracy log to {log_filename_only}")
        print("----------------------------------------------------")
        return

    content_height = num_rows_plot * subplot_height_per_timestep_inches
    subplot_content_height_clamped = max(min_subplot_content_height, min(max_subplot_content_height, content_height))
    fig_height_inches = (subplot_content_height_clamped + 0.75) * 2
    fig, axes = plt.subplots(2, 1, figsize=(fig_width_inches, fig_height_inches), squeeze=True)

    plot_display_start_timestep = model_timesteps[0] if model_timesteps else args.start_timestep

    setup_plot_axes(axes[0], 
                    f"Ground Truth: Rule {args.rule_number} (Starts at T={plot_display_start_timestep})", 
                    ground_truth_combined_states_arr, 
                    num_prompt_steps=args.num_prompt_timesteps, 
                    draw_red_line_for_prompt_end=False, 
                    plot_start_timestep=plot_display_start_timestep, 
                    config_max_timestep=config.MAX_TIMESTEP_VALUE, 
                    draw_yellow_line_for_training_max=False)

    setup_plot_axes(axes[1], 
                    f"Model Prediction: Rule {args.rule_number} (Starts at T={plot_display_start_timestep})", 
                    model_generated_combined_states_arr, 
                    num_prompt_steps=args.num_prompt_timesteps, 
                    draw_red_line_for_prompt_end=True, 
                    plot_start_timestep=plot_display_start_timestep, 
                    config_max_timestep=config.MAX_TIMESTEP_VALUE, 
                    draw_yellow_line_for_training_max=(model_timesteps[-1] > config.MAX_TIMESTEP_VALUE if model_timesteps else False))

    plt.tight_layout(pad=1.0, h_pad=1.5)
    
    # Construct a descriptive filename base (used for plot and log)
    checkpoint_name_part = os.path.basename(os.path.normpath(args.checkpoint_path))
    # New filename structure: R<rule>_qual_eval_<checkpoint>_S<start>_P<prompt>_N<predict>
    filename_base = f"R{args.rule_number}_qual_eval_{checkpoint_name_part}_S{args.start_timestep}_P{args.num_prompt_timesteps}_N{args.num_predict_timesteps}"

    if model_generated_combined_states_arr.shape[0] == 0:
        print("No data to plot (model generation array is empty). Skipping plot.")
        # Save log even if plot fails, now to logs_dir
        log_filename_only = os.path.join(logs_dir, f"{filename_base}_accuracy.txt") # Use logs_dir
        with open(log_filename_only, 'w') as f:
            f.write("\n".join(log_text_lines))
        print(f"Saved accuracy log to {log_filename_only}")
        print("----------------------------------------------------")
        return

    plot_save_path_png = os.path.join(figures_dir, f"{filename_base}.png")
    # PDF saving removed

    try:
        plt.savefig(plot_save_path_png, dpi=150, bbox_inches='tight')
        print(f"Saved qualitative evaluation plot to: {plot_save_path_png}")
        # PDF saving removed
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close(fig)

    # Save accuracy log to 'logs' directory
    log_filename = os.path.join(logs_dir, f"{filename_base}_accuracy.txt") # Use logs_dir
    with open(log_filename, 'w') as f:
        f.write("\n".join(log_text_lines))
    print(f"Saved accuracy log to {log_filename}")

    print("----------------------------------------------------")

def setup_plot_axes(ax, title_str, data_arr, num_prompt_steps, draw_red_line_for_prompt_end, plot_start_timestep, config_max_timestep, draw_yellow_line_for_training_max):
    ax.imshow(data_arr, cmap='binary', interpolation='nearest', aspect='auto')
    ax.set_title(title_str, fontsize=45)
    ax.set_xticks([]) # Remove x-axis ticks, keeps spine
    ax.set_yticks([]) # Remove y-axis ticks, keeps spine
    # The box (spines) around the plot is visible by default if ticks are removed this way.

    legend_handles = []

    # Add red line for prompt end if requested for this subplot
    if draw_red_line_for_prompt_end and num_prompt_steps > 0 and num_prompt_steps < data_arr.shape[0]:
        # y=0 is top, y=N-1 is bottom row. Line should be after num_prompt_steps-1 row.
        red_line = ax.axhline(y=num_prompt_steps - 0.5, color='red', linestyle='--', linewidth=10, label='End of Prompt') # Further increased linewidth
        legend_handles.append(red_line)
    
    # Add yellow dotted line for exceeding MAX_TIMESTEP_VALUE (generalization regime)
    row_idx_of_config_max_timestep = config_max_timestep - plot_start_timestep

    if draw_yellow_line_for_training_max and plot_start_timestep <= config_max_timestep < plot_start_timestep + data_arr.shape[0]:
        line_y_pos = row_idx_of_config_max_timestep + 0.5 # Draw after the row
        if 0 <= line_y_pos < data_arr.shape[0]:
            yellow_line = ax.axhline(y=line_y_pos, color='yellow', linestyle=':', linewidth=10, label=f'Final Training Timestep') # Further increased linewidth
            legend_handles.append(yellow_line)

    if legend_handles: # Add legend if any lines were drawn
        ax.legend(handles=legend_handles, loc='lower right', fontsize=22) # Further increased legend fontsize

def main():
    parser = argparse.ArgumentParser(description="Evaluate a CA Transformer model checkpoint.")
    subparsers = parser.add_subparsers(dest='mode', help='Evaluation mode', required=True)

    # Subparser for quantitative evaluation
    parser_quant = subparsers.add_parser('quant', help='Quantitative evaluation on validation set (perplexity, loss).')
    parser_quant.add_argument("--checkpoint_path", type=str, default="/data1/disrael/ca-transformer/ckpts/pythia-160m-bs1-data1000000scr0.7/checkpoint-83000", help="Path to the model checkpoint directory.")
    parser_quant.add_argument("--eval_batch_size", type=int, default=8, help="Per-device batch size for evaluation.")
    parser_quant.set_defaults(func=evaluate_checkpoint_quantitatively)

    # Subparser for qualitative evaluation (generation, visualization, accuracy)
    parser_qual = subparsers.add_parser('qual', help='Qualitative CA generation and comparison (uses center_one initial state).')
    parser_qual.add_argument("--checkpoint_path", type=str, default="/data1/disrael/ca-transformer/ckpts/pythia-160m-bs1-data1000000scr0.7/checkpoint-83000", help="Path to the model checkpoint directory.")
    # parser_qual.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint directory.")
    parser_qual.add_argument("--rule_number", type=int, required=True, help="CA rule number to evaluate (0-255), or -1 to evaluate all rules.")
    parser_qual.add_argument("--num_prompt_timesteps", type=int, default=5, help="Number of initial ground truth timesteps to provide as prompt.")
    parser_qual.add_argument("--num_predict_timesteps", type=int, default=512, help="Number of subsequent timesteps to predict with the model.")
    parser_qual.add_argument("--start_timestep", type=int, default=0, help="The initial timestep to start the evaluation from (default: 0).")
    parser_qual.set_defaults(func=evaluate_checkpoint_qualitatively)
    
    args = parser.parse_args()
    if hasattr(args, 'func'):
        if args.mode == 'qual' and args.rule_number == -1:
            import copy
            print("--- Evaluating for all rules (0-255) ---")
            for rule_num in tqdm(range(256, 0, -1), desc="Evaluating all rules"):
                args_copy = copy.deepcopy(args)
                args_copy.rule_number = rule_num
                try:
                    args.func(args_copy)
                except Exception as e:
                    print(f"!!! Evaluation failed for Rule {rule_num} with error: {e} !!!")
            print("--- Finished evaluation for all rules ---")
        else:
            args.func(args)
    else:
      parser.print_help()

if __name__ == "__main__":
    main() 