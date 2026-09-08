import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SwinConfig, SwinModel


class SwinHABSegmentation(nn.Module):
    """
    Swin Transformer based semantic segmentation model
    for HAB detection.

    Input:
        [batch, 14, 256, 256]

    Output:
        [batch, 2, 256, 256]

    Classes:
        0 = Non-HAB
        1 = HAB
    """

    def __init__(self, num_channels=14, num_classes=2):
        super().__init__()

        self.config = SwinConfig(
            image_size=256,
            num_channels=num_channels,
            num_labels=num_classes,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7
        )

        self.encoder = SwinModel(self.config)

        self.decoder = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, x):
        outputs = self.encoder(
            pixel_values=x
        )

        features = outputs.last_hidden_state

        batch_size, sequence_length, channels = features.shape

        height = width = int(sequence_length ** 0.5)

        features = features.transpose(1, 2).reshape(
            batch_size,
            channels,
            height,
            width
        )

        logits = self.decoder(features)

        logits = F.interpolate(
            logits,
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        )

        return logits