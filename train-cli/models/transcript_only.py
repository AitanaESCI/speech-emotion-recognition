import click
import torch
import torch.nn as nn

from .utils import PrecomputedEmbeddingsDataset


class TranscriptOnlyNet(nn.Module):
    """
    Text-only emotion classification model using a frozen sentence encoder
    (MiniLM) features followed by a trainable MLP head.
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 384,
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


def precompute_transcript_embeddings(dataset, model_name_or_path="sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer

    click.echo(f"Loading SentenceTransformer: {model_name_or_path}...")
    model = SentenceTransformer(model_name_or_path)

    click.echo("Fetching transcriptions from database...")
    transcriptions = []
    for i in range(len(dataset)):
        if "chunk_transcript" in dataset.df_dataset_audio_chunks.columns:
            # Audio chunk transcription
            t = dataset.df_dataset_audio_chunks.iloc[i]["chunk_transcript"]
        else:
            # Clip transcription
            row = dataset.df_dataset_audio_chunks.iloc[i]
            db_row = dataset._fetch_db_record(row["source_dataset_row_index"])
            t = db_row["transcription"]
        transcriptions.append("" if t is None else str(t).strip())

    click.echo(f"Encoding {len(transcriptions)} transcriptions...")
    embeddings_np = model.encode(transcriptions, show_progress_bar=True, convert_to_numpy=True)
    embeddings = torch.tensor(embeddings_np, dtype=torch.float32)

    # Get soft labels from dataset
    labels = torch.tensor(
        dataset.df_dataset_audio_chunks[dataset.LABELS_EMOTIONS].values,
        dtype=torch.float32,
    )

    return PrecomputedEmbeddingsDataset(embeddings, labels, dataset)
