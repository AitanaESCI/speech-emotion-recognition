import torch.nn as nn


class MPSAdaptiveAvgPool2d(nn.Module):
    """Custom AdaptiveAvgPool2d layer to workaround MPS backend limits."""

    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        if x.device.type == "mps":
            return nn.functional.adaptive_avg_pool2d(x.cpu(), self.output_size).to(x.device)
        return nn.functional.adaptive_avg_pool2d(x, self.output_size)


class SimpleCNN(nn.Module):
    """Simple CNN architecture for IEMOCAP emotion classification."""

    def __init__(self, num_classes: int, dropout: float = 0.3, pool_size: int = 4):
        super().__init__()
        self.pool_size = (pool_size, pool_size) if isinstance(pool_size, int) else pool_size
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            MPSAdaptiveAvgPool2d(self.pool_size),
        )

        in_features = 128 * self.pool_size[0] * self.pool_size[1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
