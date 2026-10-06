from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_fid import fid_score
from torchvision.models import Inception_V3_Weights, inception_v3


@dataclass
class MetricSnapshot:
    epoch: int
    generator_loss: float
    discriminator_loss: float
    fid_score: Optional[float] = None
    inception_score: Optional[float] = None


@dataclass
class MetricsCollector:
    history: List[MetricSnapshot] = field(default_factory=list)

    def add(
        self,
        epoch: int,
        generator_loss: float,
        discriminator_loss: float,
        fid: Optional[float] = None,
        inception: Optional[float] = None,
    ) -> MetricSnapshot:
        snapshot = MetricSnapshot(
            epoch=epoch,
            generator_loss=float(generator_loss),
            discriminator_loss=float(discriminator_loss),
            fid_score=None if fid is None else float(fid),
            inception_score=None if inception is None else float(inception),
        )
        self.history.append(snapshot)
        return snapshot

    def latest(self) -> Optional[MetricSnapshot]:
        return self.history[-1] if self.history else None

    def to_dict_list(self) -> List[Dict[str, float | int | None]]:
        return [asdict(s) for s in self.history]


def compute_fid(
    real_images_dir: str | Path,
    generated_images_dir: str | Path,
    device: str | torch.device = "cpu",
    batch_size: int = 32,
    dims: int = 2048,
    num_workers: int = 0,
) -> float:
    """Compute FID using pytorch-fid between two image directories."""
    paths = [str(Path(real_images_dir)), str(Path(generated_images_dir))]
    value = fid_score.calculate_fid_given_paths(
        paths=paths,
        batch_size=batch_size,
        device=str(device),
        dims=dims,
        num_workers=num_workers,
    )
    return float(value)


@torch.no_grad()
def compute_inception_score(
    images: torch.Tensor,
    splits: int = 10,
    device: str | torch.device = "cpu",
) -> float:
    """
    Compute Inception Score from a batch of images.

    Input images should be [N, 3, H, W] in range [-1, 1] or [0, 1].
    """
    if images.dim() != 4 or images.size(1) != 3:
        raise ValueError("images must be a 4D tensor with shape [N, 3, H, W]")

    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(device)
    model.eval()

    x = images.detach().to(device)
    x = (x + 1.0) / 2.0 if x.min() < 0 else x
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

    logits = model(x)
    probs = F.softmax(logits, dim=1).cpu().numpy()

    n = probs.shape[0]
    if n < splits:
        splits = max(1, n)

    split_scores: List[float] = []
    for part in np.array_split(probs, splits):
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-12) - np.log(py + 1e-12))
        split_scores.append(float(np.exp(np.mean(np.sum(kl, axis=1)))))

    return float(np.mean(split_scores))


def summarize_losses(generator_losses: Sequence[float], discriminator_losses: Sequence[float]) -> Dict[str, float]:
    if not generator_losses or not discriminator_losses:
        raise ValueError("Both generator_losses and discriminator_losses must be non-empty")

    g = np.asarray(generator_losses, dtype=np.float64)
    d = np.asarray(discriminator_losses, dtype=np.float64)
    return {
        "generator_loss_mean": float(g.mean()),
        "generator_loss_min": float(g.min()),
        "generator_loss_max": float(g.max()),
        "discriminator_loss_mean": float(d.mean()),
        "discriminator_loss_min": float(d.min()),
        "discriminator_loss_max": float(d.max()),
    }
