from __future__ import annotations

import torch
import torch.nn as nn


class DCGANGenerator(nn.Module):
    """DCGAN generator for 28x28 single-channel images (Fashion-MNIST)."""

    def __init__(self, latent_dim: int = 100, img_channels: int = 1) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels

        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, img_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh(),
        )
        self.apply(init_dcgan_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        if z.dim() != 4:
            raise ValueError("Generator input must be [N, latent_dim] or [N, latent_dim, 1, 1]")
        if z.size(1) != self.latent_dim:
            raise ValueError(f"Expected latent dim {self.latent_dim}, got {z.size(1)}")
        return self.net(z)


class DCGANDiscriminator(nn.Module):
    """DCGAN discriminator for 28x28 single-channel images (Fashion-MNIST)."""

    def __init__(self, img_channels: int = 1) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )
        self.apply(init_dcgan_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        return out.view(out.size(0), 1)


def init_dcgan_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)
