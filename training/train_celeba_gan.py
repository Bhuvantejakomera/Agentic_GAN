from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import List

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils as vutils
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.hyperparameter_agent import HyperparameterAgent

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class CelebADataConfig:
    data_dir: Path
    batch_size: int = 128
    num_workers: int = 2
    image_size: int = 64
    shuffle: bool = True


@dataclass
class TrainConfig:
    epochs: int = 30
    latent_dim: int = 100
    learning_rate: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    sample_interval: int = 1
    seed: int = 42
    use_cpu: bool = False
    output_dir: Path = PROJECT_ROOT / "outputs" / "celeba_gan"
    log_interval: int = 200
    ngf: int = 64
    ndf: int = 64


class CelebAImageDataset(Dataset):
    def __init__(self, root: Path, transform: transforms.Compose) -> None:
        self.root = root
        self.transform = transform
        self.image_paths = self._collect_image_paths(root)
        if not self.image_paths:
            raise FileNotFoundError(
                f"No images found under {root}. "
                f"Place CelebA images under this folder (flat or nested)."
            )

    @staticmethod
    def _collect_image_paths(root: Path) -> List[Path]:
        if not root.exists():
            raise FileNotFoundError(f"Data directory does not exist: {root}")
        image_paths: List[Path] = []
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                image_paths.append(path)
        image_paths.sort()
        return image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.image_paths[index]
        with Image.open(image_path) as img:
            rgb_image = img.convert("RGB")
        return self.transform(rgb_image)


class CelebAGenerator(nn.Module):
    def __init__(self, latent_dim: int = 100, ngf: int = 64, channels: int = 3) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        self.apply(init_dcgan_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        if z.size(1) != self.latent_dim:
            raise ValueError(f"Expected latent dim {self.latent_dim}, got {z.size(1)}")
        return self.net(z)


class CelebADiscriminator(nn.Module):
    def __init__(self, ndf: int = 64, channels: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )
        self.apply(init_dcgan_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.view(out.size(0), 1)


def init_dcgan_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


def resolve_data_dir(data_dir: Path) -> Path:
    """Resolve dataset directory from common relative path variants."""
    candidates: List[Path] = []
    if data_dir.is_absolute():
        candidates.append(data_dir)
    else:
        candidates.append((Path.cwd() / data_dir).resolve())
        candidates.append((PROJECT_ROOT / data_dir).resolve())
        candidates.append((PROJECT_ROOT.parent / data_dir).resolve())

        parts = data_dir.parts
        if parts and parts[0] == "gan_agent_project":
            stripped = Path(*parts[1:]) if len(parts) > 1 else Path(".")
            candidates.append((PROJECT_ROOT / stripped).resolve())
            candidates.append((PROJECT_ROOT.parent / stripped).resolve())

    # Preserve order, drop duplicates.
    unique_candidates: List[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    candidate_text = "\n".join(f"  - {p}" for p in unique_candidates)
    raise FileNotFoundError(
        "Data directory does not exist. Tried:\n"
        f"{candidate_text}\n"
        "Provide a valid CelebA image folder path, e.g.:\n"
        "  --data-dir data/celeba\n"
        "or\n"
        "  --data-dir /absolute/path/to/celeba"
    )


def get_celeba_dataloader(config: CelebADataConfig) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize(config.image_size),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = CelebAImageDataset(config.data_dir, transform=transform)
    use_cuda = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=use_cuda,
        drop_last=True,
    )


def check_celeba_dataset(data_cfg: CelebADataConfig) -> None:
    resolved_data_dir = resolve_data_dir(data_cfg.data_dir)
    loader_cfg = CelebADataConfig(
        data_dir=resolved_data_dir,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        image_size=data_cfg.image_size,
        shuffle=False,
    )
    loader = get_celeba_dataloader(loader_cfg)
    images = next(iter(loader))

    print("CelebA dataset loaded successfully.")
    print(f"Data dir: {resolved_data_dir}")
    print(f"Total images: {len(loader.dataset)}")
    print(f"Batches per epoch (drop_last=True): {len(loader)}")
    print(f"Batch shape: {tuple(images.shape)}")  # [B, 3, H, W]
    print(f"Pixel range after normalization: [{images.min():.3f}, {images.max():.3f}]")


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
    generator: CelebAGenerator,
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


def train_celeba_gan(train_cfg: TrainConfig, data_cfg: CelebADataConfig) -> None:
    set_seed(train_cfg.seed)
    device = get_device(train_cfg.use_cpu)
    resolved_data_dir = resolve_data_dir(data_cfg.data_dir)
    print(f"Using device: {device}")
    print(f"Using data dir: {resolved_data_dir}")

    generator = CelebAGenerator(
        latent_dim=train_cfg.latent_dim,
        ngf=train_cfg.ngf,
        channels=3,
    ).to(device)
    discriminator = CelebADiscriminator(ndf=train_cfg.ndf, channels=3).to(device)

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
        epoch_data_cfg = CelebADataConfig(
            data_dir=resolved_data_dir,
            batch_size=current_batch_size,
            num_workers=data_cfg.num_workers,
            image_size=data_cfg.image_size,
            shuffle=data_cfg.shuffle,
        )
        loader = get_celeba_dataloader(epoch_data_cfg)

        g_running = 0.0
        d_running = 0.0
        num_steps = 0

        progress = tqdm(loader, desc=f"Epoch {epoch}/{train_cfg.epochs}", leave=False)
        for step, real_images in enumerate(progress, start=1):
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
            save_samples(
                generator=generator,
                latent_dim=train_cfg.latent_dim,
                device=device,
                save_dir=train_cfg.output_dir / "samples",
                epoch=epoch,
            )

    ckpt_dir = train_cfg.output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(generator.state_dict(), ckpt_dir / "generator_celeba.pt")
    torch.save(discriminator.state_dict(), ckpt_dir / "discriminator_celeba.pt")
    print(f"Saved checkpoints to: {ckpt_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Face Generation GAN on CelebA")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "celeba")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "celeba_gan")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--sample-interval", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument(
        "--check-data-only",
        action="store_true",
        help="Only load and validate CelebA dataset, then exit.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_cfg = CelebADataConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    if args.check_data_only:
        check_celeba_dataset(data_cfg)
        raise SystemExit(0)

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
        ngf=args.ngf,
        ndf=args.ndf,
    )
    train_celeba_gan(train_cfg, data_cfg)
