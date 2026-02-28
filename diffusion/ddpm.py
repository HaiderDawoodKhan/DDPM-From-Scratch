from __future__ import annotations
"""DDPM ancestral sampler.

Starts from Gaussian noise x_T and iteratively applies reverse transitions
from timestep T-1 down to 0.
"""

import torch

from diffusion.posterior import p_sample_step
from diffusion.posterior import predict_x0_from_eps
from diffusion.schedule import DiffusionBuffers, extract


@torch.no_grad()
def ancestral_sample(
    model: torch.nn.Module,
    buffers: DiffusionBuffers,
    shape: tuple[int, int, int, int],
    device: torch.device,
    capture_every: int | None = None,
    capture_steps: list[int] | None = None,
    seed: int | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run full ancestral sampling with optional intermediate snapshots.

    Args:
    - model: epsilon predictor epsilon_theta(x_t, t)
    - buffers: precomputed diffusion scalars
    - shape: output tensor shape (B, C, H, W)
    - capture_every: if set, store x_t every N reverse steps
    """
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)
    B, C, H, W = shape
    xt = torch.randn(B, C, H, W, device=device)
    snapshots: list[torch.Tensor] = []
    capture_set = set(capture_steps) if capture_steps is not None else None
    L = buffers.betas.shape[0]

    if capture_set is not None and L in capture_set:
        snapshots.append(xt.detach().cpu())

    # Iterate reverse chain: t = L-1, ..., 0.
    for step in reversed(range(L)):
        if capture_every is not None and step % capture_every == 0:
            snapshots.append(xt.detach().cpu())
        if capture_set is not None and step in capture_set:
            snapshots.append(xt.detach().cpu())

        t = torch.full((B,), step, device=device, dtype=torch.long)
        eps_hat = model(xt, t)
        xt, _ = p_sample_step(xt=xt, t=t, eps_hat=eps_hat, buffers=buffers, clip_x0=True)

    return xt, snapshots


def make_subsampled_timesteps(L: int, num_steps: int) -> torch.Tensor:
    """Create a descending timestep list for reduced-step sampling."""
    if num_steps <= 1:
        raise ValueError("num_steps must be >= 2.")
    steps = torch.linspace(L - 1, 0, steps=num_steps)
    steps = torch.round(steps).long()
    steps = torch.unique_consecutive(steps)
    if steps[-1].item() != 0:
        steps = torch.cat([steps, torch.zeros(1, dtype=torch.long)], dim=0)
    return steps


@torch.no_grad()
def ddim_sample(
    model: torch.nn.Module,
    buffers: DiffusionBuffers,
    shape: tuple[int, int, int, int],
    device: torch.device,
    num_steps: int,
    eta: float = 0.0,
    capture_steps: list[int] | None = None,
    seed: int | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Reduced-step DDIM-style sampler using timestep skipping.

    This provides a compute-matched way to evaluate fewer reverse steps.
    When eta=0 it is deterministic conditioned on the random start x_T.
    """
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)

    B, C, H, W = shape
    xt = torch.randn(B, C, H, W, device=device)
    snapshots: list[torch.Tensor] = []
    capture_set = set(capture_steps) if capture_steps is not None else None
    ts = make_subsampled_timesteps(L=buffers.betas.shape[0], num_steps=num_steps).to(device)

    if capture_set is not None and buffers.betas.shape[0] in capture_set:
        snapshots.append(xt.detach().cpu())

    for i, step_t in enumerate(ts):
        step = int(step_t.item())
        if capture_set is not None and step in capture_set:
            snapshots.append(xt.detach().cpu())

        t = torch.full((B,), step, device=device, dtype=torch.long)
        eps_hat = model(xt, t)
        x0_hat = predict_x0_from_eps(xt=xt, t=t, eps_hat=eps_hat, buffers=buffers, clip_x0=True)

        if i + 1 < len(ts):
            next_step = int(ts[i + 1].item())
            a_next = extract(buffers.alpha_bars, torch.full((B,), next_step, device=device, dtype=torch.long), xt.shape)
        else:
            a_next = torch.ones((B, 1, 1, 1), device=device, dtype=xt.dtype)

        a_t = extract(buffers.alpha_bars, t, xt.shape)
        sigma = eta * torch.sqrt((1.0 - a_next) / (1.0 - a_t)) * torch.sqrt(1.0 - (a_t / a_next))
        noise = torch.randn_like(xt)
        direction = torch.sqrt(torch.clamp(1.0 - a_next - sigma**2, min=0.0)) * eps_hat
        xt = torch.sqrt(a_next) * x0_hat + direction + sigma * noise

    return xt, snapshots, ts.detach().cpu()
