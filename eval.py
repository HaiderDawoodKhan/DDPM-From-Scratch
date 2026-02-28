from __future__ import annotations
"""Evaluation and ablation utilities for trained DDPM checkpoints.

Task 6 (Ablations):
- Schedule ablation support (linear vs cosine): alpha_bar and SNR plots,
  optional training-curve overlay, optional matched-checkpoint sample grids.
- Sampling-step ablation: reduced reverse-step sampling (DDIM-style skipping)
  and sample-quality diagnostics.

Task 7 (Final outputs + diagnostics):
- Final 64-sample grid.
- Fixed-seed denoising trajectory at selected timesteps.
- Overfitting diagnostic via nearest-neighbor search in pixel space.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, utils

from diffusion.ddpm import ancestral_sample, ddim_sample_deterministic
from diffusion.schedule import compute_diffusion_buffers, make_beta_schedule
from models.unet import SmallUNet


def save_image_grid(tensor: torch.Tensor, out_path: Path, nrow: int = 8) -> None:
    """Map tensors from [-1,1] back to [0,1] and save as image grid."""
    images = ((tensor.clamp(-1.0, 1.0) + 1.0) / 2.0).cpu()
    grid = utils.make_grid(images, nrow=nrow)
    utils.save_image(grid, str(out_path))


def summarize_csv(csv_path: Path, out_txt: Path) -> None:
    """Write compact scalar diagnostics from train_log.csv."""
    if not csv_path.exists():
        return
    losses = []
    grad_norms = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            losses.append(float(row["loss"]))
            grad_norms.append(float(row["grad_norm"]))
    if not losses:
        return
    msg = (
        f"steps={len(losses)}\n"
        f"loss_start={losses[0]:.6f}\n"
        f"loss_end={losses[-1]:.6f}\n"
        f"loss_min={min(losses):.6f}\n"
        f"grad_norm_max={max(grad_norms):.6f}\n"
    )
    out_txt.write_text(msg)


def parse_step_list(step_csv: str) -> list[int]:
    vals = []
    for token in step_csv.split(","):
        token = token.strip()
        if token:
            vals.append(int(token))
    return vals


def load_training_loss(csv_path: Path) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    losses: list[float] = []
    if not csv_path.exists():
        return steps, losses
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
    return steps, losses


def plot_schedule_ablation(L: int, beta_min: float, beta_max: float, out_path: Path) -> None:
    """Plot alpha_bar and SNR curves for linear vs cosine schedules."""
    betas_linear = make_beta_schedule(L=L, schedule_type="linear", beta_min=beta_min, beta_max=beta_max)
    betas_cosine = make_beta_schedule(L=L, schedule_type="cosine", beta_min=beta_min, beta_max=beta_max)

    buf_lin = compute_diffusion_buffers(betas_linear)
    buf_cos = compute_diffusion_buffers(betas_cosine)

    ab_lin = buf_lin.alpha_bars.cpu().numpy()
    ab_cos = buf_cos.alpha_bars.cpu().numpy()
    snr_lin = (buf_lin.alpha_bars / (1.0 - buf_lin.alpha_bars)).cpu().numpy()
    snr_cos = (buf_cos.alpha_bars / (1.0 - buf_cos.alpha_bars)).cpu().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ab_lin, label="linear")
    ax[0].plot(ab_cos, label="cosine")
    ax[0].set_title("alpha_bar(t)")
    ax[0].set_xlabel("t")
    ax[0].legend()

    ax[1].plot(snr_lin, label="linear")
    ax[1].plot(snr_cos, label="cosine")
    ax[1].set_title("SNR(t) = alpha_bar/(1-alpha_bar)")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("t")
    ax[1].legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_curve_ablation(main_log: Path, alt_log: Path, out_path: Path, alt_label: str) -> None:
    """Overlay two training curves for matched-compute comparison."""
    s1, l1 = load_training_loss(main_log)
    s2, l2 = load_training_loss(alt_log)
    if not l1 or not l2:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s1, l1, label="main")
    ax.plot(s2, l2, label=alt_label)
    ax.set_title("Schedule ablation: training loss curves")
    ax.set_xlabel("step")
    ax.set_ylabel("MSE loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def default_trajectory_labels(L: int) -> list[int]:
    """Return labels in 1-based DDPM notation: {L, 3L/4, L/2, L/4, 1}."""
    labels = [L, max(1, (3 * L) // 4), max(1, L // 2), max(1, L // 4), 1]
    out: list[int] = []
    for x in labels:
        if x not in out:
            out.append(x)
    return out


def labels_to_capture_steps(L: int, labels: list[int]) -> list[int]:
    """Convert 1-based labels to internal 0-based steps; keep L as x_L (initial noise)."""
    steps: list[int] = []
    for lab in labels:
        if lab >= L:
            steps.append(L)
        elif lab <= 1:
            steps.append(0)
        else:
            steps.append(lab - 1)
    return steps


def save_trajectory_figure(snapshots: list[torch.Tensor], labels: list[int], out_path: Path) -> None:
    """Save first-sample denoising trajectory with timestep labels."""
    if not snapshots:
        return

    k = min(len(snapshots), len(labels))
    fig, axes = plt.subplots(1, k, figsize=(3 * k, 3))
    if k == 1:
        axes = [axes]

    for idx in range(k):
        img = ((snapshots[idx][0:1] + 1.0) / 2.0).clamp(0.0, 1.0)
        axes[idx].imshow(img[0].permute(1, 2, 0).squeeze(), cmap="gray")
        axes[idx].set_title(f"i={labels[idx]}")
        axes[idx].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def pairwise_diversity(samples: torch.Tensor, max_items: int = 16) -> float:
    """Compute average pairwise L2 distance among a subset of generated samples."""
    x = samples[:max_items].reshape(min(max_items, samples.shape[0]), -1).float().cpu()
    if x.shape[0] < 2:
        return 0.0
    d = torch.cdist(x, x, p=2)
    n = x.shape[0]
    return float(d.sum().item() / max(n * (n - 1), 1))


def build_train_tensor(data_dir: Path) -> torch.Tensor:
    """Load FashionMNIST train split in [-1,1] for nearest-neighbor checks."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ])
    ds = datasets.FashionMNIST(root=str(data_dir), train=True, download=True, transform=transform)
    all_x = torch.stack([ds[i][0] for i in range(len(ds))], dim=0)
    return all_x


def nearest_neighbors_pixel(generated: torch.Tensor, train_x: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
    """Find nearest train image per generated sample in raw pixel L2 space."""
    gen_flat = generated.detach().cpu().reshape(generated.shape[0], -1).float()
    train_flat = train_x.reshape(train_x.shape[0], -1).float()

    best_dist = torch.full((gen_flat.shape[0],), float("inf"))
    best_idx = torch.zeros((gen_flat.shape[0],), dtype=torch.long)

    for start in range(0, train_flat.shape[0], chunk_size):
        end = min(start + chunk_size, train_flat.shape[0])
        chunk = train_flat[start:end]
        d = torch.cdist(gen_flat, chunk, p=2)
        vals, idx = torch.min(d, dim=1)
        improved = vals < best_dist
        best_dist[improved] = vals[improved]
        best_idx[improved] = idx[improved] + start

    return best_idx


def save_nn_pair_grid(generated: torch.Tensor, train_x: torch.Tensor, nn_idx: torch.Tensor, out_path: Path) -> None:
    """Save a 2-column grid: [generated, nearest-train-neighbor]."""
    k = min(generated.shape[0], nn_idx.shape[0])
    paired = []
    for i in range(k):
        paired.append(generated[i].detach().cpu())
        paired.append(train_x[nn_idx[i]].detach().cpu())
    pair_tensor = torch.stack(paired, dim=0)
    save_image_grid(pair_tensor, out_path, nrow=2)


def parse_args() -> argparse.Namespace:
    """CLI arguments for checkpoint loading, ablations, and diagnostics."""
    parser = argparse.ArgumentParser(description="Evaluate DDPM checkpoints and run Task 6/7 diagnostics")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--compare-checkpoint", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--capture-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps")
    parser.add_argument("--train-log", type=Path, default=Path("outputs/train_log.csv"))
    parser.add_argument("--train-log-alt", type=Path, default=None)
    parser.add_argument("--train-log-alt-label", type=str, default="alt")
    parser.add_argument("--ddim-steps", type=int, default=100)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--nn-pairs", type=int, default=16)
    parser.add_argument("--run-nn-check", action="store_true")
    return parser.parse_args()


def load_model_and_buffers(checkpoint: Path, device: torch.device) -> tuple[dict, SmallUNet, object]:
    ckpt = torch.load(checkpoint, map_location=device)
    model = SmallUNet(in_channels=1, out_channels=1, channel_multipliers=(32, 64, 128), time_emb_dim=128).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    timesteps = int(ckpt.get("timesteps", 1000))
    beta_min = float(ckpt.get("beta_min", 1e-4))
    beta_max = float(ckpt.get("beta_max", 2e-2))
    schedule_type = str(ckpt.get("schedule_type", "linear"))

    betas = make_beta_schedule(
        L=timesteps,
        schedule_type=schedule_type,
        beta_min=beta_min,
        beta_max=beta_max,
        device=device,
    )
    buffers = compute_diffusion_buffers(betas)
    return ckpt, model, buffers


def main() -> None:
    """Run baseline sampling, ablations, trajectory visualization, and NN diagnostics."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "intermediates").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "ablation_steps").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    ckpt, model, buffers = load_model_and_buffers(args.checkpoint, device=device)
    L = int(ckpt.get("timesteps", 1000))
    beta_min = float(ckpt.get("beta_min", 1e-4))
    beta_max = float(ckpt.get("beta_max", 2e-2))

    # Task 7: baseline final grid with fixed shape (default 64 = 8x8).
    final, snapshots = ancestral_sample(
        model=model,
        buffers=buffers,
        shape=(args.samples, 1, 28, 28),
        device=device,
        capture_every=args.capture_every,
        seed=args.seed,
    )
    save_image_grid(final, args.out_dir / "sample_grid_final.png", nrow=8)

    for idx, snap in enumerate(snapshots):
        save_image_grid(snap, args.out_dir / "intermediates" / f"snapshot_{idx:03d}.png", nrow=8)

    summarize_csv(args.train_log, args.out_dir / "diagnostics.txt")

    # Task 6: schedule ablation plots (linear vs cosine curves, independent of checkpoint).
    plot_schedule_ablation(
        L=L,
        beta_min=beta_min,
        beta_max=beta_max,
        out_path=args.out_dir / "schedule_ablation_alpha_snr.png",
    )

    if args.train_log_alt is not None:
        plot_training_curve_ablation(
            main_log=args.train_log,
            alt_log=args.train_log_alt,
            out_path=args.out_dir / "schedule_ablation_training_curves.png",
            alt_label=args.train_log_alt_label,
        )

    if args.compare_checkpoint is not None and args.compare_checkpoint.exists():
        _, model_alt, buffers_alt = load_model_and_buffers(args.compare_checkpoint, device=device)
        alt_final, _ = ancestral_sample(
            model=model_alt,
            buffers=buffers_alt,
            shape=(args.samples, 1, 28, 28),
            device=device,
            seed=args.seed,
        )
        save_image_grid(alt_final, args.out_dir / "schedule_ablation_alt_sample_grid.png", nrow=8)

    # Task 6: deterministic DDIM sampling (step-reduced, no stochastic noise).
    ddim_steps = max(2, min(args.ddim_steps, L))
    ddim_final, _, ddim_ts = ddim_sample_deterministic(
        model=model,
        buffers=buffers,
        shape=(args.samples, 1, 28, 28),
        device=device,
        num_steps=ddim_steps,
        seed=args.seed,
    )
    save_image_grid(ddim_final, args.out_dir / f"sample_grid_ddim_deterministic_steps_{ddim_steps}.png", nrow=8)

    ddim_contrast = float(ddim_final.detach().cpu().std().item())
    ddim_diversity = pairwise_diversity(ddim_final)
    ddim_metrics_csv = args.out_dir / "ablation_steps" / "ddim_deterministic_metrics.csv"
    with ddim_metrics_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ddim_steps", "pixel_std", "pairwise_l2_diversity"])
        writer.writerow([ddim_steps, ddim_contrast, ddim_diversity])

    ddim_note = [
        "Deterministic DDIM summary:",
        f"- steps={ddim_steps}",
        "- eta=0.0 (deterministic reverse update, no per-step random noise)",
        f"- pixel_std={ddim_contrast:.6f}",
        f"- pairwise_l2_diversity={ddim_diversity:.6f}",
        f"- internal timestep count={len(ddim_ts)}",
    ]
    (args.out_dir / "ablation_steps" / "ddim_deterministic_notes.txt").write_text("\n".join(ddim_note))

    # Task 7: fixed-seed denoising trajectory at i in {L, 3L/4, L/2, L/4, 1}.
    traj_labels = default_trajectory_labels(L)
    traj_capture_steps = labels_to_capture_steps(L, traj_labels)
    _, traj_snaps = ancestral_sample(
        model=model,
        buffers=buffers,
        shape=(1, 1, 28, 28),
        device=device,
        capture_steps=traj_capture_steps,
        seed=args.seed,
    )
    save_trajectory_figure(
        snapshots=traj_snaps,
        labels=traj_labels,
        out_path=args.out_dir / "fixed_seed_denoising_trajectory.png",
    )

    # Task 7: overfitting/memorization diagnostic via nearest-neighbor in pixel space.
    if args.run_nn_check:
        nn_k = min(args.nn_pairs, args.samples)
        generated_batch = final[:nn_k]
        train_x = build_train_tensor(args.data_dir)
        nn_idx = nearest_neighbors_pixel(generated_batch, train_x)
        save_nn_pair_grid(
            generated=generated_batch,
            train_x=train_x,
            nn_idx=nn_idx,
            out_path=args.out_dir / "nearest_neighbor_pairs_pixel_l2.png",
        )


if __name__ == "__main__":
    main()
