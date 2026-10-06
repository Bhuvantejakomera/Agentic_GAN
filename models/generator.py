from __future__ import annotations

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Fully connected generator for 28x28 grayscale images (MNIST)."""

    def __init__(
        self,
        latent_dim: int = 100,
        img_channels: int = 1,
        img_size: int = 28,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size
        output_dim = img_channels * img_size * img_size

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, output_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        generated = self.model(z)
        return generated.view(z.size(0), self.img_channels, self.img_size, self.img_size)
