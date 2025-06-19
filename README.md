
# Probabilistic Sequence Modeling of Cellular Automata

A transformer-based model for learning and generating Elementary Cellular Automata (ECA) patterns. This project trains language models to predict cellular automata evolution by treating CA states as sequences of tokens.

## Overview

This project implements a novel approach to cellular automata modeling by:
- Converting CA rules, timesteps, and cell states into a custom token vocabulary
- Training transformer models (Pythia variants) to predict CA evolution
- Generating both random and contiguous timestep sequences for training
- Evaluating model performance through quantitative metrics and qualitative visualization

## Files

The repository contains seven Python files:

- **`config.py`**: Configuration, tokenization utilities, and global parameters
- **`data_utils.py`**: CA simulation, sequence generation, and data saving
- **`dataset.py`**: PyTorch dataset for loading and batching sequences
- **`model.py`**: Model loading with vocabulary size adaptation
- **`train.py`**: Training script with command-line arguments
- **`evaluate.py`**: Evaluation script with quantitative and qualitative modes
- **`visualize_utils.py`**: Plotting utilities for CA visualization

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd 275_submission
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The project uses a custom tokenization scheme defined in `config.py`:

- **Vocabulary Size**: 515 tokens total
- **Rule Tokens**: 256 tokens (rules 0-255)
- **Time Tokens**: 256 tokens (timesteps 0-255)
- **Cell State Tokens**: 2 tokens (0, 1)
- **Special Tokens**: PAD (0), BOS (1), EOS (2)

Key parameters:
- `NUM_CELLS`: 256 (cells per CA row)
- `SAMPLED_TIMESTEPS`: 7 (timesteps per training sequence)
- `MAX_TIMESTEP_VALUE`: 255 (maximum timestep)

## Data Generation

Generate training and validation data using `data_utils.py`:

```bash
python data_utils.py --num_sequences 100000 --output_file train_sequences_100000s.npy --contiguous_ratio 0.7
python data_utils.py --num_sequences 1000 --output_file val_sequences_1000s.npy --contiguous_ratio 0.7
```

### Data Format

Each sequence follows the format:
```
[RULE_TOKEN] [TIME_0] [CELL_0_0] ... [CELL_0_255] [TIME_1] [CELL_1_0] ... [CELL_1_255] ...
```

### Sampling Methods

- **Random**: Randomly samples timesteps from the CA evolution
- **Contiguous**: Samples consecutive timesteps (better for learning temporal patterns)

## Training

Train a model using different Pythia sizes:

```bash
# Train with 70M parameter model (default)
python train.py --model_size 70m --batch_size 1 --train_data_id 100000s --train_data_cr 0.7

# Train with different model sizes
python train.py --model_size 14m   # 14M parameters
python train.py --model_size 31m   # 31M parameters
python train.py --model_size 160m  # 160M parameters
python train.py --model_size 410m  # 410M parameters
```

### Training Parameters

- **Batch Size**: Per-device batch size (default: 1)
- **Learning Rate**: 1e-4 (from config)
- **Epochs**: 3 (from config)
- **Evaluation**: Every 500 steps
- **Checkpointing**: Every 500 steps, keep best 2

### Output

Training outputs are saved to:
- Checkpoints: `/data1/disrael/ca-transformer/ckpts/{run_identifier}/`
- Logs: Weights & Biases integration
- Best model: `{output_dir}/best_model/`

## Evaluation

### Quantitative Evaluation

Evaluate model perplexity and loss on validation set:

```bash
python evaluate.py quant --checkpoint_path /path/to/checkpoint --eval_batch_size 8
```

### Qualitative Evaluation

Generate and visualize CA predictions for a specific rule:

```bash
# Evaluate single rule
python evaluate.py qual --checkpoint_path /path/to/checkpoint --rule_number 30 --num_prompt_timesteps 2 --num_predict_timesteps 10

# Evaluate all rules (0-255) and save CSV data
python evaluate.py qual --checkpoint_path /path/to/checkpoint --rule_number -1 --num_prompt_timesteps 2 --num_predict_timesteps 10
```

### Output Files

When evaluating with `--rule_number -1`, the script generates:

1. **CSV Data** (in `logs/results/`):
   - `ground_truth/rule{0-255}_ground_truth.csv`: Ground truth CA evolution
   - `prediction/rule{0-255}_predicted.csv`: Model predictions

2. **Visualizations** (in `figures/qualitative/`):
   - PNG plots comparing ground truth vs predictions
   - Filename format: `R{rule}_qual_eval_{checkpoint}_S{start}_P{prompt}_N{predict}.png`

3. **Logs** (in `logs/`):
   - Accuracy logs with bit-level accuracy metrics
   - Filename format: `R{rule}_qual_eval_{checkpoint}_S{start}_P{prompt}_N{predict}_accuracy.txt`

## Model Architecture

The project uses Pythia transformer models with custom tokenization:

- **Base Models**: EleutherAI Pythia variants (14M to 410M parameters)
- **Tokenization**: Custom vocabulary for CA rules, timesteps, and cell states
- **Training**: Autoregressive language modeling (predict next token)
- **Context**: Variable length sequences up to model's maximum position embeddings
- 
### Quick Start

1. Generate training data:
```bash
python data_utils.py --num_sequences 10000 --output_file train_sequences_10k.npy
```

2. Train a small model:
```bash
python train.py --model_size 14m --batch_size 1 --train_data_id 10k
```

3. Evaluate the model:
```bash
python evaluate.py qual --checkpoint_path /path/to/checkpoint --rule_number 30
```

### Batch Evaluation

Evaluate all rules and save results:
```bash
python evaluate.py qual --checkpoint_path /path/to/checkpoint --rule_number -1
```

This will:
- Generate 512 CSV files (256 ground truth + 256 predictions)
- Create visualizations for each rule
- Save accuracy logs
- Show progress with tqdm progress bar
