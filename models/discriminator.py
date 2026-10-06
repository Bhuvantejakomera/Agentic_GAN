from __future__ import annotations

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """Fully connected discriminator for 28x28 grayscale images (MNIST)."""

    def __init__(self, img_channels: int = 1, img_size: int = 28) -> None:
        super().__init__()
        input_dim = img_channels * img_size * img_size

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.view(x.size(0), -1)
        return self.model(flat)
