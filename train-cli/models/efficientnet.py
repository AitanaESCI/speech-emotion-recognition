import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetSER(nn.Module):
    """
    EfficientNet-B0 fine-tuned for Speech Emotion Recognition.

    Adapts the pretrained ImageNet backbone to accept single-channel
    log-mel spectrograms (instead of 3-channel RGB images) by averaging
    the pretrained first-conv weights across the channel dimension.

    Args:
        num_classes:  Number of output emotion classes.
        dropout:      Dropout probability before the final linear layer.
        freeze_until: Freeze the first N blocks of EfficientNet features.
                      Higher values = more frozen = less overfitting risk.
                      0 = fully unfrozen (full fine-tuning).
                      7 = only the last block + classifier trained (recommended
                          for small datasets like IEMOCAP).
    """

    def __init__(self, num_classes: int = 4, dropout: float = 0.3, freeze_until: int = 5):
        super().__init__()

        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        # Adapt first conv: RGB (3-channel) → spectrogram (1-channel).
        # We preserve pretrained weights by averaging across the channel dim.
        old_conv = base.features[0][0]
        new_conv = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        base.features[0][0] = new_conv

        # Freeze the first `freeze_until` feature blocks.
        for i, block in enumerate(base.features):
            if i < freeze_until:
                for param in block.parameters():
                    param.requires_grad = False

        self.features = base.features
        self.avgpool = base.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
