import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """Applies weighted attention pooling over the time/sequence dimension."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1, bias=False),
        )

    def forward(self, x):
        # x shape: [B, T, D]
        attn_logits = self.attention(x)  # [B, T, 1]
        attn_weights = torch.softmax(attn_logits, dim=1)  # [B, T, 1]
        pooled = torch.sum(x * attn_weights, dim=1)  # [B, D]
        return pooled


class CNNRNNBase(nn.Module):
    """Base class for dual-branch CNN-RNN architectures."""

    def __init__(self, num_classes: int, recurrent_dim: int, dropout: float = 0.0):
        super().__init__()
        # Convolutional branch
        self.conv_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )
        self.conv_attention = AttentionPooling(128)

        # Combined classifier
        combined_dim = 128 + recurrent_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def recurrent_forward(self, rnn_in):
        raise NotImplementedError("Subclasses must implement recurrent_forward")

    def forward(self, x):
        # x shape: [B, 1, n_mels, T]
        B, C, F, T = x.shape

        # 1. Conv branch
        conv_feats = self.conv_branch(x)  # [B, 128, F', T']
        # Pool across frequency dimension to align with time steps
        conv_feats = conv_feats.mean(dim=2)  # [B, 128, T']
        conv_feats = conv_feats.transpose(1, 2)  # [B, T', 128]
        conv_pooled = self.conv_attention(conv_feats)  # [B, 128]

        # 2. Recurrent branch
        # Reshape to [B, T, F]
        rnn_in = x.squeeze(1).transpose(1, 2)  # [B, T, F]
        rnn_pooled = self.recurrent_forward(rnn_in)

        # 3. Combine and classify
        combined = torch.cat([conv_pooled, rnn_pooled], dim=-1)
        return self.classifier(combined)


class CNNGRU(CNNRNNBase):
    """CNN-GRU model with dual-branch architecture and weighted attention pooling."""

    def __init__(
        self,
        num_classes: int,
        n_mels: int = 80,
        gru_hidden_size: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__(
            num_classes=num_classes,
            recurrent_dim=2 * gru_hidden_size,
            dropout=dropout,
        )
        self.gru = nn.GRU(
            input_size=n_mels,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.gru_attention = AttentionPooling(2 * gru_hidden_size)

    def recurrent_forward(self, rnn_in):
        gru_feats, _ = self.gru(rnn_in)  # [B, T, 2 * gru_hidden_size]
        return self.gru_attention(gru_feats)  # [B, 2 * gru_hidden_size]


class CNNLSTM(CNNRNNBase):
    """CNN-LSTM model with dual-branch architecture and weighted attention pooling."""

    def __init__(
        self,
        num_classes: int,
        n_mels: int = 80,
        lstm_hidden_size: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__(
            num_classes=num_classes,
            recurrent_dim=2 * lstm_hidden_size,
            dropout=dropout,
        )
        self.lstm = nn.LSTM(
            input_size=n_mels,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_attention = AttentionPooling(2 * lstm_hidden_size)

    def recurrent_forward(self, rnn_in):
        lstm_feats, _ = self.lstm(rnn_in)  # [B, T, 2 * lstm_hidden_size]
        return self.lstm_attention(lstm_feats)  # [B, 2 * lstm_hidden_size]
