import itertools
import os
import random
from enum import StrEnum

import click
import matplotlib
import numpy as np


matplotlib.use("Agg")  # Use headless backend
import matplotlib.pyplot as plt
import modal
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv
from nbdev_upc_aidl_iemocap_datasets.core import DatasetsFactory
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

import wandb
from models.cnn_rnn import CNNGRU, CNNLSTM
from models.efficientnet import EfficientNetSER
from models.gemma_audio import GemmaAudioNet, build_gemma_audio_datasets
from models.simple_cnn import SimpleCNN
from models.transcript_only import TranscriptOnlyNet, precompute_transcript_embeddings
from models.wavlm import WavLMNet, build_wavlm_datasets


INWORLD_EMOTIONS = [
    "inworld_emotion_fearful",
    "inworld_emotion_neutral",
    "inworld_emotion_sad",
    "inworld_emotion_calm",
    "inworld_emotion_angry",
    "inworld_emotion_happy",
    "inworld_emotion_surprised",
    "inworld_emotion_disgusted",
    "inworld_emotion_tender",
]

NEW_MERGE_EMOTIONS = [
    "new_merged_positive",
    "new_merged_negative",
    "new_merged_neutral",
    "new_merged_unclear",
]


class ModelName(StrEnum):
    SIMPLE_CNN = "simple_cnn"
    CNN_GRU = "cnn_gru"
    CNN_LSTM = "cnn_lstm"
    TRANSCRIPT_ONLY = "transcript_only"
    GEMMA_AUDIO = "gemma_audio"
    WAVLM_ONLY = "wavlm_only"
    WAVLM_AND_TRANSCRIPT = "wavlm_and_transcript"
    EFFICIENTNET_B0 = "efficientnet_b0"


def _inworld_emotions_merger(row: dict):
    merged_emotions_scores_columns = {}

    merged_emotions_scores_columns["new_merged_positive"] = row["inworld_emotion_happy"]
    merged_emotions_scores_columns["new_merged_negative"] = (
        row["inworld_emotion_angry"] + row["inworld_emotion_fearful"] + row["inworld_emotion_disgusted"]
    )
    merged_emotions_scores_columns["new_merged_neutral"] = row["inworld_emotion_neutral"] + row["inworld_emotion_calm"]
    merged_emotions_scores_columns["new_merged_unclear"] = (
        row["inworld_emotion_sad"] + row["inworld_emotion_tender"] + row["inworld_emotion_surprised"]
    )

    major_emotion_column = max(merged_emotions_scores_columns, key=merged_emotions_scores_columns.get)

    merged_coord_x = (
        merged_emotions_scores_columns["new_merged_negative"] * -1
        + merged_emotions_scores_columns["new_merged_unclear"] * 0
        + merged_emotions_scores_columns["new_merged_neutral"] * 0
        + merged_emotions_scores_columns["new_merged_positive"] * 1
    )
    merged_coord_y = (
        merged_emotions_scores_columns["new_merged_negative"] * 0
        + merged_emotions_scores_columns["new_merged_unclear"] * -1
        + merged_emotions_scores_columns["new_merged_neutral"] * 1
        + merged_emotions_scores_columns["new_merged_positive"] * 0
    )

    return pd.Series(
        {
            "new_merged_major_emotion": major_emotion_column.replace("new_merged_", ""),
            "new_merged_coord_x": merged_coord_x,
            "new_merged_coord_y": merged_coord_y,
        }
        | merged_emotions_scores_columns
    )


def _set_seed(seed: int = 42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _init_modal() -> tuple[modal.App, modal.Function]:
    app = modal.App("upc-iemocap-training")

    image = (
        modal.Image.debian_slim()
        .apt_install("git")
        .pip_install(
            "nbdev-upc-aidl-iemocap-datasets @ git+https://github.com/gofordiego/nbdev-upc-aidl-iemocap-datasets.git#0b37e99e7d4c86c9993c27ce3991d246387fa985"
        )
        .pip_install(
            "click",
            "python-dotenv",
            "wandb",
            "sentence-transformers",
            "transformers",
        )
        .add_local_dir(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
            remote_path="/root/models",
        )
    )

    # Load environment local context
    load_dotenv()
    secrets = []
    secrets.append(modal.Secret.from_dict({"WANDB_API_KEY": os.environ["WANDB_API_KEY"]}))
    stub_fn = app.function(
        image=image,
        gpu="t4",
        cpu=4.0,
        memory=16384,
        timeout=36000,
        secrets=secrets,
    )(
        run_training_process  # already global — wrap it directly
    )

    return app, stub_fn


def get_model(
    model_name: str,
    num_classes: int,
    dropout: float,
    n_mels: int = 80,
) -> nn.Module:
    """Model factory supporting extensible architectures."""
    try:
        model_enum = ModelName(model_name.lower())
    except ValueError:
        raise ValueError(f"Unknown model name: {model_name}. " f"Choose from: {[m.value for m in ModelName]}")

    if model_enum == ModelName.SIMPLE_CNN:
        return SimpleCNN(num_classes=num_classes, dropout=dropout)
    elif model_enum == ModelName.CNN_GRU:
        return CNNGRU(num_classes=num_classes, n_mels=n_mels, dropout=dropout)
    elif model_enum == ModelName.CNN_LSTM:
        return CNNLSTM(num_classes=num_classes, n_mels=n_mels, dropout=dropout)
    elif model_enum == ModelName.TRANSCRIPT_ONLY:
        return TranscriptOnlyNet(num_classes=num_classes, dropout=dropout)
    elif model_enum == ModelName.GEMMA_AUDIO:
        return GemmaAudioNet(num_classes=num_classes, dropout=dropout)
    elif model_enum == ModelName.WAVLM_ONLY:
        return WavLMNet(num_classes=num_classes, mode="audio_only", dropout=dropout)
    elif model_enum == ModelName.WAVLM_AND_TRANSCRIPT:
        return WavLMNet(num_classes=num_classes, mode="multimodal", dropout=dropout)
    elif model_enum == ModelName.EFFICIENTNET_B0:
        return EfficientNetSER(num_classes=num_classes, dropout=dropout, freeze_until=5)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def _plot_confusion_matrix(cm, class_names):
    """Creates a confusion matrix figure for W&B logging."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=8,
        )
    fig.tight_layout()
    return fig


@click.command(context_settings=dict(show_default=True))
@click.option(
    "--model-name",
    required=False,
    type=click.Choice([m.value for m in ModelName], case_sensitive=False),
    help="Name of the model architecture to train.",
)
@click.option(
    "--chunks-group-id",
    default=2,
    type=int,
    help="Chunks group ID for IEMOCAP dataset.",
)
@click.option("--n-mels", default=80, type=int, help="Number of Mel bands.")
@click.option("--n-fft", default=1024, type=int, help="FFT window size.")
@click.option("--hop-length", default=256, type=int, help="Hop length.")
@click.option("--win-length", default=512, type=int, help="Window length.")
@click.option("--batch-size", default=32, type=int, help="Batch size for training/validation.")
@click.option("--learning-rate", default=1e-3, type=float, help="Learning rate.")
@click.option("--epochs", default=10, type=int, help="Number of training epochs.")
@click.option("--dropout", default=0.0, type=float, help="Dropout probability.")
@click.option(
    "--pad-mode",
    default="windowed_repeat",
    type=click.Choice(["zero", "windowed_repeat"]),
    help="Padding mode.",
)
@click.option("--seed", default=42, type=int, help="Random seed for reproducibility.")
@click.option("--wandb-project", default="Final-UPC-Project", help="W&B Project name.")
@click.option(
    "--enable-spec-augment/--disable-spec-augment",
    default=True,
    help="Enable or disable spectrogram augmentation.",
)
@click.option(
    "--overfit-single-batch/--no-overfit-single-batch",
    default=False,
    help="Overfit on a single training batch to verify model capability.",
)
@click.option(
    "--pitch-shift-prob",
    default=0.0,
    type=float,
    help="Probability of applying pitch shift to training samples. Recommended value: 0.2 to 0.3.",
)
@click.option(
    "--use-last-model/--no-use-last-model",
    default=False,
    help="Start training using the last run model as the initial state if it exists.",
)
@click.option(
    "--run-in-modal/--no-run-in-modal",
    default=False,
    help="Run training remotely on Modal GPU instead of locally.",
)
@click.option(
    "--detach/--no-detach",
    default=False,
    help="When running in Modal, spawn the task in the cloud and detach instead of waiting for results.",
)
@click.option(
    "--retrieve",
    default=None,
    type=str,
    help="Retrieve the results of a detached Modal run using its FunctionCall ID.",
)
def main(
    model_name,
    chunks_group_id,
    n_mels,
    n_fft,
    hop_length,
    win_length,
    batch_size,
    learning_rate,
    epochs,
    dropout,
    pad_mode,
    seed,
    wandb_project,
    enable_spec_augment,
    overfit_single_batch,
    pitch_shift_prob,
    use_last_model,
    run_in_modal,
    detach,
    retrieve,
):
    # Load environment variables from .env if present
    load_dotenv()

    if retrieve:
        modal_app, modal_train_remote = _init_modal()
        click.echo(f"Reconnecting to Modal function call {retrieve}...")
        from modal import FunctionCall

        try:
            results = FunctionCall.from_id(retrieve).get()
            click.echo("Results retrieved successfully! Saving checkpoint locally...")
            retrieved_model_name = results["model_name"]
            last_model_file_name = f"last__{retrieved_model_name}.pt"
            torch.save(results["last_state_dict"], last_model_file_name)
            click.echo(f"Saved last model checkpoint to {last_model_file_name}")
        except Exception as e:
            click.echo(f"Error retrieving results: {e}")
        return

    if not model_name:
        raise click.UsageError("--model-name is required unless --retrieve is provided.")

    config = {
        "model": model_name,
        "model_name": model_name,
        "chunks_group_id": chunks_group_id,
        "n_mels": n_mels,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": win_length,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "dropout": dropout,
        "pad_mode": pad_mode,
        "seed": seed,
        "wandb_project": wandb_project,
        "enable_spec_augment": enable_spec_augment,
        "overfit_single_batch": overfit_single_batch,
        "pitch_shift_prob": pitch_shift_prob,
        "use_last_model": use_last_model,
    }

    if run_in_modal:
        modal_app, modal_train_remote = _init_modal()

        # If use_last_model is True, load the state dict locally and include it in config
        if use_last_model:
            last_model_file_name = f"last__{model_name}.pt"
            if os.path.exists(last_model_file_name):
                click.echo(f"Loading local last checkpoint {last_model_file_name} to send to remote worker...")
                config["last_state_dict_to_load"] = torch.load(last_model_file_name, map_location="cpu")
            else:
                click.echo(f"Warning: Local checkpoint {last_model_file_name} not found. Remote will start from scratch.")
                config["last_state_dict_to_load"] = None
        else:
            config["last_state_dict_to_load"] = None

        if detach:
            click.echo("Launching detached remote training on Modal GPU...")
            with modal.enable_output():
                with modal_app.run(detach=True):
                    call = modal_train_remote.spawn(config)
                    click.echo("\n" + "=" * 50)
                    click.echo(f"Task successfully spawned in the cloud! Call ID: {call.object_id}")
                    click.echo("You can now safely close your laptop or disconnect.")
                    click.echo("To retrieve checkpoints and results later, run:")
                    click.echo(f"  uv run python train.py --retrieve {call.object_id}")
                    click.echo("=" * 50 + "\n")
            return
        else:
            click.echo("Launching remote training on Modal GPU...")
            with modal.enable_output():
                with modal_app.run():
                    results = modal_train_remote.remote(config)
    else:
        results = run_training_process(config)

    # Save checkpoint locally (works for both local and remote paths when not detached)
    last_model_file_name = f"last__{model_name}.pt"
    torch.save(results["last_state_dict"], last_model_file_name)
    click.echo(f"Saved last model checkpoint to {last_model_file_name}")


def _maybe_precompute_data_loader(original_train_data, train_data_extractor, batch_size, num_workers, pin_memory):
    """
    This function may precompute audio embeddings for the training data if a
    function to compute them (`train_data_extractor`) is provided.
    """
    return DataLoader(
        original_train_data if train_data_extractor is None else train_data_extractor(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def run_training_process(config: dict):
    model_name = config["model_name"]
    chunks_group_id = config["chunks_group_id"]
    n_mels = config["n_mels"]
    n_fft = config["n_fft"]
    hop_length = config["hop_length"]
    win_length = config["win_length"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    epochs = config["epochs"]
    dropout = config["dropout"]
    pad_mode = config["pad_mode"]
    seed = config["seed"]
    wandb_project = config["wandb_project"]
    enable_spec_augment = config["enable_spec_augment"]
    overfit_single_batch = config["overfit_single_batch"]
    pitch_shift_prob = config["pitch_shift_prob"]
    use_last_model = config["use_last_model"]

    # Set seed
    _set_seed(seed)

    # Initialize DatasetsFactory
    factory = DatasetsFactory(url="https://iemocap-files.plumberslog.com/")

    emotion_columns_override = None
    emotions_merger = None
    if chunks_group_id in (6, 8):
        emotion_columns_override = NEW_MERGE_EMOTIONS
        emotions_merger = _inworld_emotions_merger
    elif chunks_group_id == 7:
        emotion_columns_override = INWORLD_EMOTIONS
        emotions_merger = None

    # Build datasets
    click.echo("Building datasets...")
    train_data = factory.build_dataset(
        id=chunks_group_id,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        pad_mode=pad_mode,
        partition_type="train",
        should_refresh_local_cache=False,
        enable_spec_augment=enable_spec_augment,
        pitch_shift_prob=pitch_shift_prob,
        emotion_columns_override=emotion_columns_override,
        emotions_merger=emotions_merger,
    )
    click.echo(f"{len(train_data)=}")

    val_data = factory.build_dataset(
        id=chunks_group_id,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        pad_mode=pad_mode,
        partition_type="validation",
        should_refresh_local_cache=False,
        enable_spec_augment=False,
        pitch_shift_prob=0.0,
        emotion_columns_override=emotion_columns_override,
        emotions_merger=emotions_merger,
    )
    click.echo(f"{len(val_data)=}")

    test_data = factory.build_dataset(
        id=chunks_group_id,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        pad_mode=pad_mode,
        partition_type="test",
        should_refresh_local_cache=False,
        enable_spec_augment=False,
        pitch_shift_prob=0.0,
        emotion_columns_override=emotion_columns_override,
        emotions_merger=emotions_merger,
    )
    click.echo(f"{len(test_data)=}")

    train_data_extractor = None
    if model_name.lower() == ModelName.TRANSCRIPT_ONLY:
        click.echo("Precomputing text embeddings for TranscriptOnlyNet...")
        train_data = precompute_transcript_embeddings(train_data)
        val_data = precompute_transcript_embeddings(val_data)
        test_data = precompute_transcript_embeddings(test_data)
    elif model_name.lower() == ModelName.GEMMA_AUDIO:
        click.echo("Precomputing Gemma E2B audio embeddings for GemmaAudioNet...")
        # train_data_extractor to precompute audio embeddings for the training data using AUDIO AUGMENTATION on each epoch.
        train_data_extractor, val_data, test_data = build_gemma_audio_datasets(train_data, val_data, test_data)
        # Precompute train_data only once if spec augment is disabled.
        if not enable_spec_augment:
            train_data = train_data_extractor()
            train_data_extractor = None
    elif model_name.lower() in (ModelName.WAVLM_ONLY, ModelName.WAVLM_AND_TRANSCRIPT):
        mode = "multimodal" if model_name.lower() == ModelName.WAVLM_AND_TRANSCRIPT else "audio_only"
        click.echo(f"Building WavLM datasets for mode: {mode}...")
        train_data, val_data, test_data = build_wavlm_datasets(train_data, val_data, test_data, mode=mode)

    num_classes = len(train_data.LABELS_EMOTIONS)
    config["num_classes"] = num_classes

    # Hardware setup
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    click.echo(f"Using device: {device}")

    # This enables much faster, asynchronous page-locked CPU-to-GPU data copies.
    pin_memory = device == "cuda"

    # Setup DataLoaders
    num_workers = 4
    train_loader = _maybe_precompute_data_loader(train_data, train_data_extractor, batch_size, num_workers, pin_memory)
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Setup single batch for overfitting check if enabled
    single_batch = None
    if overfit_single_batch:
        single_batch = next(iter(train_loader))
        click.echo(f"Overfitting on a single batch of size {single_batch[0].size(0)} enabled.")

    # Init W&B run
    wandb.init(
        project=wandb_project,
        config=config,
        name=f"{model_name}_cgid{chunks_group_id}_nc{num_classes}",
    )

    # Build model and send to device
    model = get_model(
        model_name=model_name,
        num_classes=num_classes,
        dropout=dropout,
        n_mels=n_mels,
    ).to(device)

    if use_last_model:
        if config.get("last_state_dict_to_load") is not None:
            click.echo("Initializing model weights from passed state dict.")
            model.load_state_dict(config["last_state_dict_to_load"])
        else:
            last_model_file_name = f"last__{model_name}.pt"
            if os.path.exists(last_model_file_name):
                click.echo(f"Initializing model weights from last checkpoint: {last_model_file_name}")
                model.load_state_dict(torch.load(last_model_file_name, map_location=device))
            else:
                click.echo(f"Warning: Last checkpoint {last_model_file_name} not found. Starting training from scratch.")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Log model architecture
    wandb.watch(model, log="all", log_freq=50)

    best_val_acc = 0.0

    click.echo(f"\nStarting training: {epochs} epochs | batch size: {batch_size} | device: {device}\n")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        click.echo(f"Epoch {epoch+1:02d}/{epochs:02d} | Training...")

        train_batches = [single_batch] * len(train_loader) if overfit_single_batch else train_loader
        total_batches = len(train_batches)
        log_interval = max(1, total_batches // 10)

        for i, (specs, labels) in enumerate(train_batches):
            specs = specs.unsqueeze(1).to(device)
            targets = labels.argmax(dim=1).to(device)

            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

            if (i + 1) % log_interval == 0 or (i + 1) == total_batches:
                click.echo(f"  Step {i+1:04d}/{total_batches:04d} | Batch Loss: {loss.item():.4f}")

        train_acc = train_correct / train_total
        train_loss = train_loss / total_batches

        # Validate
        click.echo(f"Running validation for Epoch {epoch+1:02d}...")
        model.eval()
        val_correct, val_total = 0, 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for specs, labels in val_loader:
                specs = specs.unsqueeze(1).to(device)
                targets = labels.argmax(dim=1).to(device)
                outputs = model(specs)
                preds = outputs.argmax(dim=1)

                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_acc = val_correct / val_total
        val_f1 = f1_score(all_targets, all_preds, average="weighted")
        val_f1_per_class = f1_score(all_targets, all_preds, average=None)

        log_dict = {
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/acc": val_acc,
            "val/f1": val_f1,
            "train/lr": scheduler.get_last_lr()[0],
        }

        # Log individual F1 scores per class
        for emotion, score in zip(val_data.LABELS_EMOTIONS, val_f1_per_class):
            log_dict[f"val_class_f1/{emotion}"] = score

        # Log confusion matrix every 5 epochs or on the last epoch
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            cm = confusion_matrix(all_targets, all_preds)
            fig = _plot_confusion_matrix(cm, train_data.LABELS_EMOTIONS)
            log_dict["val/confusion_matrix"] = wandb.Image(fig)
            plt.close(fig)

        wandb.log(log_dict)

        click.echo(
            f"Epoch {epoch+1:02d} | "
            f"loss: {train_loss:.4f} | "
            f"train acc: {train_acc:.3f} | "
            f"val acc: {val_acc:.3f} | "
            f"val F1: {val_f1:.3f} | "
            f"lr: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Track best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Step learning rate scheduler
        scheduler.step()

        # Update dataloader to precompute new audio embeddings
        if train_data_extractor is not None and (epoch + 1) < epochs:
            train_loader = _maybe_precompute_data_loader(train_data, train_data_extractor, batch_size, num_workers, pin_memory)

    # Save the last run model checkpoint
    last_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    click.echo(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")

    # Evaluate the final model on the test set
    click.echo("Evaluating the final model on the test set...")
    model.eval()

    test_preds, test_targets = [], []
    with torch.no_grad():
        for specs, labels in test_loader:
            specs = specs.unsqueeze(1).to(device)
            targets = labels.argmax(dim=1).to(device)
            outputs = model(specs)
            preds = outputs.argmax(dim=1)

            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())

    test_acc = sum(p == t for p, t in zip(test_preds, test_targets)) / len(test_targets)
    test_f1 = f1_score(test_targets, test_preds, average="weighted")

    click.echo("\n" + "=" * 40)
    click.echo("Test Set Results")
    click.echo(f"Accuracy: {test_acc:.3f}")
    click.echo(f"F1 Score (weighted): {test_f1:.3f}")
    click.echo("=" * 40)

    # Log test results to W&B
    wandb.log(
        {
            "test/acc": test_acc,
            "test/f1": test_f1,
        }
    )

    # Log final test set confusion matrix
    cm_test = confusion_matrix(test_targets, test_preds)
    fig_test = _plot_confusion_matrix(cm_test, test_data.LABELS_EMOTIONS)
    wandb.log({"test/confusion_matrix": wandb.Image(fig_test)})
    plt.close(fig_test)

    # Per-class breakdown
    f1_per_class = f1_score(test_targets, test_preds, average=None)
    click.echo("F1 per emotion class:")
    for emotion, score in zip(test_data.LABELS_EMOTIONS, f1_per_class):
        bar = "█" * int(score * 20)
        click.echo(f"  {emotion:<20} {score:.3f}  {bar}")

    wandb.finish()

    return {
        "model_name": model_name,
        "last_state_dict": last_state_dict,
    }


if __name__ == "__main__":
    main()
