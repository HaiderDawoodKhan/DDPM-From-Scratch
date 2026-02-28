from __future__ import annotations
"""Training script for DDPM on FashionMNIST.

Objective (L_simple):
min E_{x0, t, epsilon}[ ||epsilon - epsilon_theta(x_t, t)||_2^2 ]

where x_t is formed from x_0 with the forward process.
"""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm

from diffusion.ddpm import ancestral_sample
from diffusion.forward import draw_training_triplet
from diffusion.schedule import compute_diffusion_buffers, make_beta_schedule
from models.unet import SmallUNet


def build_fashionmnist_loader(data_dir: Path, batch_size: int, num_workers: int = 4) -> DataLoader:
    """Create train dataloader with pixel scaling from [0,1] to [-1,1]."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.FashionMNIST(root=str(data_dir), train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)


def grad_norm(model: torch.nn.Module) -> float:
    """Compute global L2 norm of all gradients (for explosion monitoring)."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


def param_norm(model: torch.nn.Module) -> float:
    """Compute global L2 norm of parameters (for scale monitoring)."""
    total = 0.0
    for p in model.parameters():
        total += p.detach().pow(2).sum().item()
    return total ** 0.5


def save_image_grid(tensor: torch.Tensor, out_path: Path, nrow: int = 8) -> None:
    """Convert generated tensors back to [0,1] and save a tiled grid."""
    images = ((tensor.clamp(-1.0, 1.0) + 1.0) / 2.0).cpu()
    grid = utils.make_grid(images, nrow=nrow)
    utils.save_image(grid, str(out_path))


def parse_args() -> argparse.Namespace:
    """CLI arguments for schedule/model/training and logging cadence."""
    parser = argparse.ArgumentParser(description="Train DDPM on FashionMNIST")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw"], default="adam")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--schedule-type", type=str, choices=["linear", "cosine"], default="linear")
    parser.add_argument("--beta-min", type=float, default=1e-4)
    parser.add_argument("--beta-max", type=float, default=2e-2)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--sample-every", type=int, default=5000)
    parser.add_argument("--ckpt-every", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps")
    return parser.parse_args()


def main() -> None:
    """Run optimization loop and periodically save samples/checkpoints."""
    args = parse_args()
    torch.manual_seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    loader = build_fashionmnist_loader(args.data_dir, args.batch_size, args.num_workers)
    model = SmallUNet(in_channels=1, out_channels=1, channel_multipliers=(32, 64, 128), time_emb_dim=128).to(device)

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    betas = make_beta_schedule(
        L=args.timesteps,
        schedule_type=args.schedule_type,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        device=device,
    )
    buffers = compute_diffusion_buffers(betas)

    log_path = args.out_dir / "train_log.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "grad_norm", "param_norm"])

    step = 0
    data_iter = iter(loader)
    pbar = tqdm(total=args.steps, desc="Training")

    while step < args.steps:
        try:
            x0, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x0, _ = next(data_iter)

        x0 = x0.to(device)

        # Sample a random timestep and corresponding noisy image x_t.
        t, eps, xt = draw_training_triplet(x0, buffers)
        eps_hat = model(xt, t)
        # DDPM noise-prediction loss.
        loss = F.mse_loss(eps_hat, eps)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = grad_norm(model)
        optimizer.step()
        pnorm = param_norm(model)

        step += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.4f}", gnorm=f"{gnorm:.2f}")

        with log_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step, float(loss.item()), gnorm, pnorm])

        # Periodically run full ancestral sampling (all L reverse steps).
        if step % args.sample_every == 0 or step == 1:
            samples, _ = ancestral_sample(
                model=model,
                buffers=buffers,
                shape=(64, 1, 28, 28),
                device=device,
                capture_every=None,
            )
            save_image_grid(samples, args.out_dir / "samples" / f"step_{step:07d}.png", nrow=8)

        # Save model/optimizer state so training can be resumed/evaluated.
        if step % args.ckpt_every == 0 or step == args.steps:
            state = {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "timesteps": args.timesteps,
                "schedule_type": args.schedule_type,
                "beta_min": args.beta_min,
                "beta_max": args.beta_max,
            }
            torch.save(state, args.out_dir / "checkpoints" / f"ddpm_step_{step:07d}.pt")

    pbar.close()


if __name__ == "__main__":
    main()
