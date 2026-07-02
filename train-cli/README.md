# Walkthrough - standalone CLI training script `train.py`

This walkthrough details the standalone CLI training script created to replace the Jupyter notebook training workflow.

## Overview
Our `train.py` script is a machine learning training command-line interface that leverages `click` and `uv` for fast, reproducible, and cloud-compatible training.

## Virtual Environment Management
> [!IMPORTANT]
> We use **`uv`** package manager to manage the virtual environment and execute our scripts.
> https://docs.astral.sh/uv/getting-started/installation/

Ensure dependencies are installed and run python commands with `uv run`.

```bash
# Sync dependencies (creates .venv directory)
uv sync
```

## Setup & Credentials (.env)
Before running the training script, configure your environment variables:
1. Copy the `.env.example` file into `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and specify your `WANDB_API_KEY`:
   ```env
   WANDB_API_KEY=your_wandb_api_key_here
   ```
   *Note: If you do not configure it via `.env`, you can also log in to Weights & Biases interactively via the terminal CLI using `uv run wandb login`.*

## Script Capabilities
The script integrates the following core capabilities:
* **Model Architectures**: Managed via a unified `ModelName` enum:
  * `simple_cnn`: A 2D CNN with adaptive average pooling.
  * `cnn_gru`: A dual-branch model combining a 3-layer 2D CNN and a bidirectional GRU.
  * `cnn_lstm`: A dual-branch model combining a 3-layer 2D CNN and a bidirectional LSTM.
  * `transcript_only`: A baseline using a frozen `all-MiniLM-L6-v2` SentenceTransformer and trainable MLP head.
  * `gemma_audio`: An audio-only baseline using a frozen `google/gemma-4-E2B` Conformer audio tower and trainable MLP head.
* **Hyperparameter Configuration**: Configure hyperparameters via CLI options: `--learning-rate`, `--epochs`, `--batch-size`, `--dropout`, `--pad-mode`, `--seed`, etc.
* **Reproducibility**: Automatic random seed configuration via `_set_seed()`.
* **W&B Integration**: Automatic sync to project `Final-UPC-Project`, headless confusion matrix plots, and class-specific validation F1 score tracking (`val_class_f1/<emotion>`).
* **Debugging**: Debug model capabilities using `--overfit-single-batch` to overfit on a single batch.
* **Checkpointing**:
  * Final model checkpoints are saved locally to `last__<model_name>.pt`.
  * Train from a saved checkpoint by using the `--use-last-model` flag.
* **Testing**: Comprehensive post-training evaluation on the test set, outputting a visual F1 per-class bar chart directly to the terminal.

## Remote Training in [Modal](https://modal.com/) (Cloud)
You can run training jobs in the cloud on Modal GPU instances (e.g., an NVIDIA T4) with the `--run-in-modal` flag.

### Detached Runs (Queue and Forget)
If you want to spawn the remote job and close your laptop without losing connection, run the task with the `--detach` flag:
```bash
uv run python train.py --model-name cnn_lstm --run-in-modal --detach
```
This prints a **FunctionCall ID** (e.g., `fc-xxxxxxxxxxxx`).

### Retrieving Results Later
Once the training is complete, retrieve your logs and checkpoints using the `--retrieve` option:
```bash
uv run python train.py --retrieve fc-xxxxxxxxxxxx
```
This fetches the remote run's outputs, retrieves the checkpoint, and writes it locally.

## Command Line Help

`uv run python train.py --help`

```text
Usage: train.py [OPTIONS]

Options:
  --model-name [simple_cnn|cnn_gru|cnn_lstm|transcript_only|gemma_audio]
                                  Name of the model architecture to train.
  --chunks-group-id INTEGER       Chunks group ID for IEMOCAP dataset.
                                  [default: 2]
  --n-mels INTEGER                Number of Mel bands.  [default: 80]
  --n-fft INTEGER                 FFT window size.  [default: 1024]
  --hop-length INTEGER            Hop length.  [default: 256]
  --win-length INTEGER            Window length.  [default: 512]
  --batch-size INTEGER            Batch size for training/validation.
                                  [default: 128]
  --learning-rate FLOAT           Learning rate.  [default: 0.001]
  --epochs INTEGER                Number of training epochs.  [default: 10]
  --dropout FLOAT                 Dropout probability.  [default: 0.0]
  --pad-mode [zero|windowed_repeat]
                                  Padding mode.  [default: windowed_repeat]
  --seed INTEGER                  Random seed for reproducibility.  [default:
                                  42]
  --wandb-project TEXT            W&B Project name.  [default: Final-UPC-
                                  Project]
  --enable-spec-augment / --disable-spec-augment
                                  Enable or disable spectrogram augmentation.
                                  [default: enable-spec-augment]
  --overfit-single-batch / --no-overfit-single-batch
                                  Overfit on a single training batch to verify
                                  model capability.  [default: no-overfit-
                                  single-batch]
  --pitch-shift-prob FLOAT        Probability of applying pitch shift to
                                  training samples. Recommended value: 0.2 to
                                  0.3.  [default: 0.0]
  --use-last-model / --no-use-last-model
                                  Start training using the last run model as
                                  the initial state if it exists.  [default:
                                  no-use-last-model]
  --run-in-modal / --no-run-in-modal
                                  Run training remotely on Modal GPU instead
                                  of locally.  [default: no-run-in-modal]
  --detach / --no-detach          When running in Modal, spawn the task in the
                                  cloud and detach instead of waiting for
                                  results.  [default: no-detach]
  --retrieve TEXT                 Retrieve the results of a detached Modal run
                                  using its FunctionCall ID.
  --help                          Show this message and exit.
```

## End-to-End Training Test Run
Example run of a local 1-epoch overfit on a single batch:
```bash
uv run python train.py --model-name simple_cnn --epochs 1 --overfit-single-batch
```
Terminal logs snippet:
```text
Building datasets...
Using device: mps
Epoch 01 | loss: 1.1304 | train acc: 0.533 | val acc: 0.498 | val F1: 0.379
Saved last model checkpoint to last__simple_cnn.pt

Training complete. Final validation accuracy: 0.498
Evaluating the final model on the test set...

========================================
Test Set Results
Accuracy: 0.476
F1 Score (weighted): 0.360
========================================
F1 per emotion class:
  merged_negative      0.646  ████████████
  merged_sad           0.484  █████████
  merged_positive      0.000  
  merged_neutral       0.063  █
```

## Refreshing [nbdev-upc-aidl-iemocap-datasets](https://github.com/gofordiego/nbdev-upc-aidl-iemocap-datasets) package
```bash
uv pip install --reinstall --no-deps "nbdev-upc-aidl-iemocap-datasets @ git+https://github.com/gofordiego/nbdev-upc-aidl-iemocap-datasets.git"
```
