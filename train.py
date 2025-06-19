import torch
from transformers import TrainingArguments, Trainer, AutoConfig
import numpy as np
import math
import argparse # Added for command-line arguments

import config
from dataset import CATokenDataset
from model import get_ca_model

# Mapping for model sizes to Hugging Face model names
PYTHIA_MODEL_MAP = {
    "14m": "EleutherAI/pythia-14m", # Using non-deduped for smallest sizes, assuming they exist
    "31m": "EleutherAI/pythia-31m", 
    "70m": "EleutherAI/pythia-70m",
    "160m": "EleutherAI/pythia-160m-deduped", # Known to exist
    "410m": "EleutherAI/pythia-410m-deduped", # Known to exist
}

def compute_metrics(eval_pred):
    logits_np, labels_np = eval_pred
    
    # Convert numpy arrays to torch tensors
    logits = torch.from_numpy(logits_np)
    labels = torch.from_numpy(labels_np)
    
    # Shift logits and labels for perplexity calculation
    # (Trainer might do this internally for its own loss, but good to be explicit for metrics)
    # For perplexity, we typically ignore padded tokens.
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=config.PAD_TOKEN_ID)
    
    # Flatten the tokens
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    perplexity = math.exp(loss.item())
    return {"perplexity": perplexity, "eval_loss": loss.item()}

def main():
    # --- Command Line Arguments for Model Size ---
    parser = argparse.ArgumentParser(description="Train a CA Transformer model.")
    parser.add_argument(
        "--model_size", 
        type=str, 
        default="70m",
        choices=list(PYTHIA_MODEL_MAP.keys()),
        help=f"Size of the Pythia model to train. Choices: {list(PYTHIA_MODEL_MAP.keys())}"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1, # Default to 1 as it was before
        help="Per-device batch size for training and evaluation."
    )
    parser.add_argument(
        "--train_data_id",
        type=str,
        required=False, # No longer required if we provide a sensible default
        default="100000s", # Default to match data_utils.py output e.g. train_sequences_100000s.npy
        help="Identifier for the training data file (e.g., '100k', '100000s'). Used to construct filename like 'train_sequences_{id}.npy'."
    )
    parser.add_argument(
        "--train_data_cr",
        type=float,
        default=0.7, # Default to match the default in data_utils.py
        help="Contiguous ratio of the training data file (e.g., 0.0, 0.5, 1.0). Used to construct filename like 'train_sequences_{id}_cr{cr}.npy'."
    )
    args = parser.parse_args()

    # --- Specify Model to Use ---
    model_name_or_path = PYTHIA_MODEL_MAP.get(args.model_size)
    if model_name_or_path is None: # Should not happen due to choices in argparser
        print(f"Error: Invalid model size '{args.model_size}'. Using default 70m.")
        model_name_or_path = PYTHIA_MODEL_MAP["70m"]

    print(f"Using Pythia model size: {args.model_size} ({model_name_or_path})")

    # --- Determine block_size for the dataset ---
    temp_model_config = AutoConfig.from_pretrained(model_name_or_path) 
    block_size = temp_model_config.max_position_embeddings
    
    print(f"Using model: {model_name_or_path}")
    print(f"Using block_size (model context window): {block_size}")
    print(f"Note: Our full data sequence length is {config.MAX_SEQ_LEN_CORE + 2}.")
    print(f"Sequences will be truncated/padded to {block_size} by the Dataset.")

    # --- Construct dynamic training data path ---
    train_file_path = f"{config.DATA_DIR}/train_sequences_{args.train_data_id}_cr{args.train_data_cr}.npy"
    print(f"Using training data file: {train_file_path}")

    val_file_path = f"{config.DATA_DIR}/val_sequences_1000s_cr{args.train_data_cr}.npy"
    print(f"Using validation data file: {val_file_path}")
    
    # --- Load Datasets ---
    train_dataset = CATokenDataset(
        file_path=train_file_path, # Use dynamically constructed path
        block_size=block_size
    )
    val_dataset = CATokenDataset(
        file_path=val_file_path, # Keep val file from config.py
        block_size=block_size
    )

    # Check for GPU ahead of model initialization to use it for model.to(device)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Initialize Model ---
    model = get_ca_model(model_name_or_path=model_name_or_path)

    # --- Training Arguments ---
    batch_size_per_device = args.batch_size # Use command-line argument
    gradient_accumulation_steps = int(32 / batch_size_per_device) # Accumulate gradients to simulate a larger batch size
    # Effective batch size = batch_size_per_device * gradient_accumulation_steps

    if device == "cpu":
        print("Running on CPU, this will be very slow. Consider reducing num_epochs or dataset size for testing.")
        # Further reduce batch size for CPU if needed, or expect long training times.
        # gradient_accumulation_steps = 1 # No real benefit on CPU for accumulation usually

    # --- Dynamic Output Directory & W&B Run Name ---
    run_identifier = f"pythia-{args.model_size}-bs{batch_size_per_device}-data{args.train_data_id}cr{args.train_data_cr}"
    base_output_path = "/data1/disrael/ca-transformer/ckpts"
    training_output_dir = f"{base_output_path}/{run_identifier}"

    training_args = TrainingArguments(
        output_dir=training_output_dir,
        run_name=run_identifier, # Set W&B run name
        num_train_epochs=config.NUM_EPOCHS, # From config.py
        per_device_train_batch_size=batch_size_per_device,
        per_device_eval_batch_size=batch_size_per_device,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_strategy="steps",    # Evaluate every N steps
        eval_steps=500,           # Evaluate every 500 steps
        save_strategy="steps",    # Save checkpoint every N steps
        save_steps=500,           # Save every 500 steps
        save_total_limit=2,             # Keep only the best 2 checkpoints
        logging_dir=f"{training_output_dir}/logs", # Logging dir relative to output_dir
        logging_steps=100,              # Log every 100 steps
        learning_rate=config.LEARNING_RATE, # From config.py
        weight_decay=0.01,
        warmup_steps=500,
        load_best_model_at_end=True,    # Load the best model found during training at the end
        metric_for_best_model="eval_loss", # Use eval_loss to determine the best model
        report_to="wandb", # Changed from "none" to "wandb"
        logging_first_step=True,
    )

    # --- Initialize Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        # Data collator is not strictly needed here because our dataset already returns dicts
        # with input_ids and attention_mask, and padding is handled. 
        # Hugging Face's default data collator for LM will handle labels if not provided.
    )

    # --- Train ---
    print("Starting training...")
    trainer.train()
    print("Training complete.") # Simplified message

    # Save the best model explicitly to a known location based on the run
    best_model_path = f"{training_output_dir}/best_model"
    trainer.save_model(best_model_path)
    print(f"Best model saved to {best_model_path}")

    # --- Evaluate Final Model (which is the best model loaded by Trainer) ---
    print("Evaluating final best model...")
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")

main() 