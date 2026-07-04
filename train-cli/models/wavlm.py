import click
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import WavLMModel

# Assuming this import exists in your environment
from models.transcript_only import precompute_transcript_embeddings


def build_wavlm(model_name_or_path="microsoft/wavlm-large", unfreeze_top=6):
    """
    Loads pretrained WavLM model, disables layerdrop, freezes all layers except
    the top N layers and the final layer norm, and enables gradient checkpointing.
    """
    click.echo(f"Loading WavLM model: {model_name_or_path}...")

    model = WavLMModel.from_pretrained(model_name_or_path)
    model.config.layerdrop = 0.0  # disable layer-drop -> stable, deterministic fine-tuning

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze only the top unfreeze_top layers
    if unfreeze_top > 0:
        for layer in model.encoder.layers[-unfreeze_top:]:
            for p in layer.parameters():
                p.requires_grad = True

    # Unfreeze final encoder layer norm
    for p in model.encoder.layer_norm.parameters():
        p.requires_grad = True

    model.gradient_checkpointing_enable()  # trade a little compute for much less GPU memory

    click.echo(f"WavLM model loaded successfully.")

    return model


class WavLMDataset(Dataset):
    """
    Custom wrapper dataset that gets raw audio bytes from the underlying dataset,
    normalizes loudness, and packs raw audio, attention mask, and optionally
    precomputed text embeddings into a single tensor of shape (channels, length).
    """

    def __init__(self, original_dataset, mode="audio_only", text_embeddings=None, max_audio_len=160000):
        self.ds = original_dataset
        self.mode = mode
        self.LABELS_EMOTIONS = original_dataset.LABELS_EMOTIONS
        self.max_audio_len = max_audio_len  # Cap length to safeguard RAM/VRAM (160000 = 10 seconds at 16kHz)

        # Soft labels normalization
        L = torch.tensor(original_dataset.df_dataset_audio_chunks[original_dataset.LABELS_EMOTIONS].values, dtype=torch.float32)
        self.labels = L / L.sum(1, keepdim=True).clamp(min=1e-9)
        self.text_embeddings = text_embeddings

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        # Retrieve the audio and sampling rate
        wav, s = self.ds.get_item_audio_bytes(index)
        wav = np.asarray(wav, dtype=np.float32)

        if wav.ndim > 1:
            wav = wav.mean(axis=1)  # stereo -> mono

        # Resampling if needed
        if s != 16000:
            import librosa

            wav = librosa.resample(wav, orig_sr=s, target_sr=16000)

        # Truncate to avoid exploding shapes if an anomalous file is extremely long
        if len(wav) > self.max_audio_len:
            click.echo(f"Audio too long: {len(wav)} samples, truncating to {self.max_audio_len}")
            wav = wav[: self.max_audio_len]

        # Calculate actual non-zero boundary before packing
        nz = np.nonzero(wav)[0]
        real = int(nz[-1]) + 1 if len(nz) else len(wav)

        # Attention mask
        attn = np.zeros(len(wav), dtype=np.float32)
        attn[:real] = 1.0

        # Pack data into a single tensor to keep dataloaders uniform
        length = len(wav)
        if self.mode == "multimodal":
            # shape (3, length): channel 0 is wav, channel 1 is attn, channel 2 is zero-padded text embedding
            packed = np.zeros((3, length), dtype=np.float32)
            packed[0] = wav
            packed[1] = attn
            if self.text_embeddings is not None:
                txt_emb = self.text_embeddings[index]
                packed[2, : len(txt_emb)] = txt_emb
            return torch.tensor(packed, dtype=torch.float32), self.labels[index]
        else:
            # shape (2, length): channel 0 is wav, channel 1 is attn
            packed = np.zeros((2, length), dtype=np.float32)
            packed[0] = wav
            packed[1] = attn
            return torch.tensor(packed, dtype=torch.float32), self.labels[index]


def build_wavlm_datasets(train_dataset, val_dataset, test_dataset, mode="audio_only"):
    """
    Helper function to build train, validation, and test datasets for WavLM.
    Precomputes text embeddings for multimodal runs.
    """
    text_train = None
    text_val = None
    text_test = None

    click.echo(f"Building WavLM datasets...")

    if mode == "multimodal":
        click.echo("Precomputing transcript embeddings for multimodal model...")
        # Note: Ensure these methods do not store redundant copies in RAM permanently
        train_precomputed = precompute_transcript_embeddings(train_dataset)
        text_train = train_precomputed.embeddings.numpy()

        val_precomputed = precompute_transcript_embeddings(val_dataset)
        text_val = val_precomputed.embeddings.numpy()

        test_precomputed = precompute_transcript_embeddings(test_dataset)
        text_test = test_precomputed.embeddings.numpy()

    train_data = WavLMDataset(train_dataset, mode=mode, text_embeddings=text_train)
    val_data = WavLMDataset(val_dataset, mode=mode, text_embeddings=text_val)
    test_data = WavLMDataset(test_dataset, mode=mode, text_embeddings=text_test)

    return train_data, val_data, test_data


class WavLMNet(nn.Module):
    """
    WavLM network supporting both audio-only and multimodal (audio + transcript text) modes.
    Takes a packed tensor of shape [Batch, 1, Channels, Length] or [Batch, Channels, Length]
    and separates wav, attention mask, and transcript embeddings.
    """

    def __init__(
        self,
        num_classes: int,
        mode: str = "multimodal",
        wavlm_model: str = "microsoft/wavlm-large",
        unfreeze_top: int = 6,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mode = mode
        self.wavlm = build_wavlm(wavlm_model, unfreeze_top)
        H = self.wavlm.config.hidden_size  # 1024 for wavlm-large

        # Input dimension: 2 * H (mean + std pooling) + (384-d text vector if multimodal)
        in_dim = 2 * H + (384 if mode == "multimodal" else 0)

        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def pool(self, hidden, attn):
        # Real sequence lengths after WavLM downsampling CNN steps
        lens = self.wavlm._get_feat_extract_output_lengths(attn.sum(-1)).to(torch.long)
        B, T, H = hidden.shape

        # --- CLEAN VECTORIZED MASKING ---
        # Replaces the explicit for-loop which leaked graph memory.
        steps = torch.arange(T, device=hidden.device).unsqueeze(0)  # Shape: [1, T]
        mask = (steps < lens.unsqueeze(1)).float().unsqueeze(-1)  # Shape: [B, T, 1]
        # --------------------------------

        hidden = hidden.float()
        s = mask.sum(1).clamp(min=1.0)

        mean = (hidden * mask).sum(1) / s
        var = ((hidden - mean.unsqueeze(1)) ** 2 * mask).sum(1) / s
        std = torch.sqrt(var + 1e-6)

        return torch.cat([mean, std], dim=1)  # -> 2048-d voice summary

    def forward(self, x):
        # Handle shape [Batch, 1, Channels, Length]
        if x.dim() == 4:
            x = x.squeeze(1)  # -> [Batch, Channels, Length]

        wav = x[:, 0, :]
        attn = x[:, 1, :]

        # WavLM Model forward pass
        out = self.wavlm(wav, attention_mask=attn.to(torch.long))
        a = self.pool(out.last_hidden_state, attn)

        if self.mode == "multimodal":
            # Extract text embedding from channel 2 (first 384 dimensions)
            txt = x[:, 2, :384]
            features = torch.cat([a, txt], dim=1)
        else:
            features = a

        return self.head(features)
