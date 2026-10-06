from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torchvision import utils as vutils

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def save_image_grid(
    images: torch.Tensor,
    output_path: str | Path,
    nrow: int = 8,
    normalize: bool = True,
) -> None:
    """Save a batch of images [N, C, H, W] to disk as a grid image."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(images, str(output_path), normalize=normalize, nrow=nrow)


def plot_loss_history(
    g_losses: Sequence[float],
    d_losses: Sequence[float],
    save_path: str | Path,
    title: str = "GAN Training Losses",
) -> None:
    """Plot Generator and Discriminator loss history to a file."""
    if not HAS_MATPLOTLIB:
        print("[Visualizer] matplotlib not installed; skipping plot generation.")
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.plot(g_losses, label="Generator Loss", color="#4C72B0", linewidth=2)
    plt.plot(d_losses, label="Discriminator Loss", color="#DD8452", linewidth=2)
    plt.xlabel("Epoch / Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300)
    plt.close()
