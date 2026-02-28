from __future__ import annotations
"""Forward diffusion helpers.

Implements sampling from q(x_t | x_0) and drawing training tuples for L_simple.
"""

import torch

from diffusion.schedule import DiffusionBuffers, extract


def sample_timesteps(batch_size: int, L: int, device: torch.device | None = None) -> torch.Tensor:
    """Sample t uniformly from {0, ..., L-1} for each item in a batch."""
    return torch.randint(low=0, high=L, size=(batch_size,), device=device, dtype=torch.long)


def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    buffers: DiffusionBuffers,
    eps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample x_t from q(x_t|x_0) using the reparameterization form.

    Formula:
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon,
    epsilon ~ N(0, I).
    """
    if eps is None:
        eps = torch.randn_like(x0)
    sqrt_alpha_bar_t = extract(buffers.sqrt_alpha_bars, t, x0.shape)
    sqrt_one_minus_alpha_bar_t = extract(buffers.sqrt_one_minus_alpha_bars, t, x0.shape)
    return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * eps


def draw_training_triplet(
    x0: torch.Tensor,
    buffers: DiffusionBuffers,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (t, epsilon, x_t) used in DDPM noise-prediction training.

    The model is trained to predict epsilon from (x_t, t).
    """
    b = x0.shape[0]
    t = sample_timesteps(batch_size=b, L=buffers.betas.shape[0], device=x0.device)
    eps = torch.randn_like(x0)
    xt = q_sample(x0=x0, t=t, buffers=buffers, eps=eps)
    return t, eps, xt
