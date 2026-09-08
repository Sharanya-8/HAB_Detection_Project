import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerModel


class SegFormerHABSegmentation(nn.Module):
    """
    SegFormer-based semantic segmentation model for HAB detection.

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

        self.config = SegformerConfig(
            num_channels=num_channels,
            num_labels=num_classes,
            hidden_sizes=[32, 64, 160, 256],
            depths=[2, 2, 2, 2],
            decoder_hidden_size=256,
            image_size=256
        )

        self.encoder = SegformerModel(self.config)

        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        outputs = self.encoder(
            pixel_values=x
        )

        features = outputs.last_hidden_state

        # SegFormer returns features as:
        # [batch, channels, height, width]
        if features.ndim == 4:
            pass

# Some configurations may return:
# [batch, sequence_length, channels]
        elif features.ndim == 3:
            batch_size, sequence_length, channels = features.shape

            height = width = int(sequence_length ** 0.5)

            features = features.transpose(1, 2).reshape(
                batch_size,
                channels,
                height,
                width
    )

        else:
            raise ValueError(
                f"Unexpected SegFormer feature shape: {features.shape}"
            )

        logits = self.decoder(features)

        logits = F.interpolate(
            logits,
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        )

        return logits