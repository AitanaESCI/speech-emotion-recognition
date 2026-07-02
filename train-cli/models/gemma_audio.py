import functools
from typing import Callable

import click
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models.utils import PrecomputedEmbeddingsDataset


class GemmaAudioNet(nn.Module):
    """
    Gemma-E2B audio-only baseline model using a frozen Gemma-4-E2B audio encoder + projector
    features followed by a trainable MLP head.
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 1536,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # Handle unsqueezed input dimension if input is [Batch, 1, Features]
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.mlp(x)


def _extract_audio_embeddings(wav, sample_rate, device, extractor, model, audio_tower) -> torch.Tensor:
    # Ensure it is float32 mono-channel
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)

    with torch.no_grad():
        # Extract features using extractor
        inputs = extractor(raw_speech=[wav], sampling_rate=sample_rate, return_tensors="pt")
        input_features = inputs["input_features"].to(device, dtype=model.dtype)

        # Forward pass
        projected = audio_tower(input_features)  # Shape: [1, SeqLen, 1536] or a special output wrapper

    # Support tuple/wrapped model outputs cleanly
    if not isinstance(projected, torch.Tensor):
        if hasattr(projected, "last_hidden_state"):
            projected = projected.last_hidden_state
        elif isinstance(projected, (tuple, list)):
            projected = projected[0]

    # Pool across sequence dimension (average pooling) -> [1536]
    return projected.mean(dim=1).squeeze(0).cpu().to(torch.float32)


def _build_precomputed_dataset(split, original_dataset, device, extractor, model, audio_tower) -> PrecomputedEmbeddingsDataset:
    if split == "train":
        click.echo(f"AUDIO AUGMENTATION:")
    click.echo(f"Precomputing Gemma E2B audio embeddings for {split} split...")

    embeddings_list = []
    for idx in tqdm(range(len(original_dataset))):
        wav, sample_rate = original_dataset.get_item_audio_bytes(idx)
        emb = _extract_audio_embeddings(wav, sample_rate, device, extractor, model, audio_tower)
        embeddings_list.append(emb)

    embeddings = torch.stack(embeddings_list)

    # Get soft labels from dataset
    labels = torch.tensor(
        original_dataset.df_dataset_audio_chunks[original_dataset.LABELS_EMOTIONS].values,
        dtype=torch.float32,
    )

    return PrecomputedEmbeddingsDataset(embeddings, labels, original_dataset)


def build_gemma_audio_datasets(
    train_dataset,
    val_dataset,
    test_dataset,
    model_id="google/gemma-4-E2B",
) -> tuple[Callable, PrecomputedEmbeddingsDataset, PrecomputedEmbeddingsDataset]:
    from transformers import AutoModelForMultimodalLM, Gemma4AudioFeatureExtractor

    click.echo(f"Loading Gemma-4-E2B model and feature extractor...")
    extractor = Gemma4AudioFeatureExtractor.from_pretrained(model_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32 if device.type == "cpu" else torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)

    audio_tower = model.model.audio_tower
    audio_tower.eval()

    # Build data generator for train split to allow audio augmentation.
    train_data_extractor = functools.partial(
        _build_precomputed_dataset,
        "train",
        train_dataset,
        device,
        extractor,
        model,
        audio_tower,
    )
    val_data = _build_precomputed_dataset("validation", val_dataset, device, extractor, model, audio_tower)
    test_data = _build_precomputed_dataset("test", test_dataset, device, extractor, model, audio_tower)

    return train_data_extractor, val_data, test_data
