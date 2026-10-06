from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class LatentAgent:
    """Agent that perturbs latent vectors to explore more realistic samples."""

    noise_scale: float = 0.05
    clamp_min: Optional[float] = None
    clamp_max: Optional[float] = None

    def modify(self, z: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(z) * self.noise_scale
        new_z = z + noise
        if self.clamp_min is not None or self.clamp_max is not None:
            min_val = self.clamp_min if self.clamp_min is not None else float("-inf")
            max_val = self.clamp_max if self.clamp_max is not None else float("inf")
            new_z = torch.clamp(new_z, min=min_val, max=max_val)
        return new_z


@torch.no_grad()
def latent_optimization_loop(
    generator: nn.Module,
    agent: LatentAgent,
    latent_dim: int = 100,
    steps: int = 10,
    batch_size: int = 1,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Step 11 latent optimization loop.

    Equivalent pattern:
        z = torch.randn(1, 100)
        for _ in range(10):
            img = generator(z)
            z = agent.modify(z)
    """
    device = torch.device(device)
    z = torch.randn(batch_size, latent_dim, device=device)

    image = generator(z)
    for _ in range(steps):
        image = generator(z)
        z = agent.modify(z)

    return z, image
