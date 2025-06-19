from transformers import AutoConfig, AutoModelForCausalLM
import config
import torch

def get_ca_model(model_name_or_path: str, checkpoint_path: str | None = None) -> AutoModelForCausalLM:
    """Loads a Hugging Face Causal LM model, adapts its vocabulary for CA task, 
    and optionally loads weights from a checkpoint.

    Args:
        model_name_or_path (str): The name or path of the base Hugging Face model 
                                  (e.g., 'EleutherAI/pythia-70m').
        checkpoint_path (str | None, optional): Path to a local checkpoint directory 
                                                from which to load model weights after initial setup.
                                                Defaults to None.

    Returns:
        AutoModelForCausalLM: The loaded and adapted model.
    """
    print(f"Loading base model: {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    # Resize token embeddings to match the custom vocabulary size
    current_vocab_size = model.get_input_embeddings().weight.shape[0]
    if current_vocab_size != config.VOCAB_SIZE:
        print(f"Resizing token embeddings from {current_vocab_size} to {config.VOCAB_SIZE}")
        model.resize_token_embeddings(config.VOCAB_SIZE)
        # After resizing, the output LM head (if tied or separate) also needs to be consistent.
        # Most architectures handle this automatically with resize_token_embeddings,
        # but explicitly checking or adjusting model.config.vocab_size might be good.
        model.config.vocab_size = config.VOCAB_SIZE
    else:
        print(f"Vocabulary size already matches config.VOCAB_SIZE ({config.VOCAB_SIZE}). No resize needed.")

    # Update the model config's vocab_size, just in case resize_token_embeddings didn't persist it everywhere
    # This is important for generation config and other internal uses.
    if model.config.vocab_size != config.VOCAB_SIZE:
        model.config.vocab_size = config.VOCAB_SIZE
        print(f"Model config vocab_size explicitly set to {config.VOCAB_SIZE}")

    if checkpoint_path:
        print(f"Loading weights from checkpoint: {checkpoint_path}")
        try:

            print(f"Re-loading model directly from checkpoint: {checkpoint_path} to apply its weights and config.")
            model = AutoModelForCausalLM.from_pretrained(checkpoint_path, local_files_only=True)
            print(f"Model successfully loaded from checkpoint. Its config vocab_size: {model.config.vocab_size}")

            # Now, ensure this loaded model's vocab is consistent with project config.
            if model.get_input_embeddings().weight.shape[0] != config.VOCAB_SIZE:
                print(f"Resizing token embeddings of checkpoint model from {model.get_input_embeddings().weight.shape[0]} to {config.VOCAB_SIZE}")
                model.resize_token_embeddings(config.VOCAB_SIZE)
            
            # Crucially, ensure the model's config object also reflects this vocab size
            if model.config.vocab_size != config.VOCAB_SIZE:
                print(f"Updating model.config.vocab_size from {model.config.vocab_size} to {config.VOCAB_SIZE}")
                model.config.vocab_size = config.VOCAB_SIZE

        except Exception as e:
            print(f"Error loading model weights from checkpoint {checkpoint_path}: {e}")
            print("Ensure the checkpoint path is correct and contains a valid model state.")
            # Optionally, re-raise or handle if critical
            raise
    
    print(f"Final model configuration: Name/Path: {model.name_or_path}, Vocab Size: {model.config.vocab_size}")
    print(f"Model embedding matrix size: {model.get_input_embeddings().weight.shape}")
    if hasattr(model, 'get_output_embeddings') and model.get_output_embeddings() is not None:
        print(f"Model output LM head size: {model.get_output_embeddings().weight.shape}")
    return model
