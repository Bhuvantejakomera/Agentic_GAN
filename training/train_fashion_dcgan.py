from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.hyperparameter_agent import HyperparameterAgent
from models.dcgan import DCGANDiscriminator, DCGANGenerator


@dataclass
class FashionDataConfig:
    data_dir: Path
    batch_size: int = 128
    num_workers: int = 2
    image_size: int = 28
    shuffle: bool = True


@dataclass
class TrainConfig:
    epochs: int = 20
    latent_dim: int = 100
    learning_rate: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    sample_interval: int = 1
    seed: int = 42
    use_cpu: bool = False
    output_dir: Path = PROJECT_ROOT / "outputs" / "fashion_dcgan"
    log_interval: int = 200


def get_fashion_dataloader(config: FashionDataConfig) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    dataset = datasets.FashionMNIST(
        root=str(config.data_dir),
        train=True,
        transform=transform,
        download=True,
    )

    use_cuda = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=use_cuda,
        drop_last=True,
    )


def get_device(use_cpu: bool = False) -> torch.device:
    if use_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def save_samples(
    generator: DCGANGenerator,
    latent_dim: int,
    device: torch.device,
    save_dir: Path,
    epoch: int,
    num_images: int = 64,
) -> None:
    generator.eval()
    z = torch.randn(num_images, latent_dim, 1, 1, device=device)
    fake_images = generator(z)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"epoch_{epoch:03d}.png"
    vutils.save_image(fake_images, str(out_path), normalize=True, nrow=8)
    generator.train()


def train_fashion_dcgan(train_cfg: TrainConfig, data_cfg: FashionDataConfig) -> None:
    set_seed(train_cfg.seed)
    device = get_device(train_cfg.use_cpu)
    print(f"Using device: {device}")

    generator = DCGANGenerator(latent_dim=train_cfg.latent_dim, img_channels=1).to(device)
    discriminator = DCGANDiscriminator(img_channels=1).to(device)

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=train_cfg.learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2),
    )
    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=train_cfg.learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2),
    )

    agent = HyperparameterAgent(
        current_lr=train_cfg.learning_rate,
        current_batch_size=data_cfg.batch_size,
    )
    current_batch_size = data_cfg.batch_size

    for epoch in range(1, train_cfg.epochs + 1):
        epoch_data_cfg = FashionDataConfig(
            data_dir=data_cfg.data_dir,
            batch_size=current_batch_size,
            num_workers=data_cfg.num_workers,
            image_size=data_cfg.image_size,
            shuffle=data_cfg.shuffle,
        )
        train_loader = get_fashion_dataloader(epoch_data_cfg)

        g_running = 0.0
        d_running = 0.0
        num_steps = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{train_cfg.epochs}", leave=False)
        for step, (real_images, _) in enumerate(progress, start=1):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            optimizer_d.zero_grad()
            d_real = discriminator(real_images)
            d_real_loss = criterion(d_real, real_labels)

            z = torch.randn(batch_size, train_cfg.latent_dim, 1, 1, device=device)
            fake_images = generator(z)
            d_fake = discriminator(fake_images.detach())
            d_fake_loss = criterion(d_fake, fake_labels)

            d_loss = 0.5 * (d_real_loss + d_fake_loss)
            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()
            z = torch.randn(batch_size, train_cfg.latent_dim, 1, 1, device=device)
            generated_images = generator(z)
            g_pred = discriminator(generated_images)
            g_loss = criterion(g_pred, real_labels)
            g_loss.backward()
            optimizer_g.step()

            g_running += g_loss.item()
            d_running += d_loss.item()
            num_steps += 1

            if step % train_cfg.log_interval == 0:
                progress.set_postfix(
                    g_loss=f"{g_loss.item():.4f}",
                    d_loss=f"{d_loss.item():.4f}",
                    lr=f"{agent.current_lr:.6f}",
                    bs=current_batch_size,
                )

        epoch_g_loss = g_running / max(num_steps, 1)
        epoch_d_loss = d_running / max(num_steps, 1)

        action = agent.update(epoch_g_loss, epoch_d_loss)
        if action.changed:
            agent.apply_learning_rate(optimizer_g, optimizer_d)
        current_batch_size = action.batch_size

        print(
            f"Epoch {epoch:03d} | "
            f"G Loss: {epoch_g_loss:.4f} | "
            f"D Loss: {epoch_d_loss:.4f} | "
            f"LR: {action.learning_rate:.6f} | "
            f"Next Batch Size: {action.batch_size} | "
            f"Agent: {action.reason}"
        )

        if epoch % train_cfg.sample_interval == 0:
            sample_dir = train_cfg.output_dir / "samples"
            save_samples(generator, train_cfg.latent_dim, device, sample_dir, epoch)

    model_dir = train_cfg.output_dir / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(generator.state_dict(), model_dir / "generator_fashion_dcgan.pt")
    torch.save(discriminator.state_dict(), model_dir / "discriminator_fashion_dcgan.pt")
    print(f"Saved checkpoints to: {model_dir}")


def parse_args() -> argparse.Namespace:
    default_data_dir = PROJECT_ROOT / "data"
    default_output_dir = PROJECT_ROOT / "outputs" / "fashion_dcgan"
    parser = argparse.ArgumentParser(description="Train DCGAN on Fashion-MNIST")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--sample-interval", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_cfg = FashionDataConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    train_cfg = TrainConfig(
        epochs=args.epochs,
        latent_dim=args.latent_dim,
        learning_rate=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        sample_interval=args.sample_interval,
        seed=args.seed,
        use_cpu=args.use_cpu,
        output_dir=args.output_dir,
        log_interval=args.log_interval,
    )
    train_fashion_dcgan(train_cfg, data_cfg)
