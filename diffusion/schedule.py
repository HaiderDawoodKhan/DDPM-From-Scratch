from __future__ import annotations
"""Utilities for DDPM schedule construction and cached scalar buffers.

Notation (matching the DDPM paper):
- beta_t: forward-process noise variance at step t
- alpha_t = 1 - beta_t
- alpha_bar_t = prod_{s=1..t} alpha_s

These buffers are precomputed once and then indexed per-batch timestep.
"""

from dataclasses import dataclass
import math

import torch


@dataclass
class DiffusionBuffers:
    """Precomputed 1D tensors used by forward and reverse diffusion formulas."""
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    alpha_bars_prev: torch.Tensor
    sqrt_alphas: torch.Tensor
    sqrt_alpha_bars: torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor
    one_over_sqrt_alphas: torch.Tensor
    sqrt_recip_alpha_bars: torch.Tensor
    sqrt_recipm1_alpha_bars: torch.Tensor
    posterior_variance: torch.Tensor
    posterior_log_variance_clipped: torch.Tensor
    posterior_mean_coef1: torch.Tensor
    posterior_mean_coef2: torch.Tensor


def make_beta_schedule(
    L: int,
    schedule_type: str = "linear",
    beta_min: float = 1e-4,
    beta_max: float = 2e-2,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create beta_t schedule of length L.

    Supported schedules:
    - linear: beta_t = linspace(beta_min, beta_max, L)
    - cosine: derived from cosine alpha_bar(t) (Nichol & Dhariwal)
    """
    if L <= 0:
        raise ValueError("L must be positive.")

    if schedule_type == "linear":
        return torch.linspace(beta_min, beta_max, L, dtype=torch.float32, device=device)

    if schedule_type == "cosine":
        s = 0.008

        def alpha_bar_fn(t: float) -> float:
            angle = ((t + s) / (1.0 + s)) * (math.pi / 2.0)
            return math.cos(angle) ** 2

        betas = []
        for i in range(L):
            t1 = i / L
            t2 = (i + 1) / L
            beta = min(1.0 - alpha_bar_fn(t2) / alpha_bar_fn(t1), 0.999)
            betas.append(beta)
        return torch.tensor(betas, dtype=torch.float32, device=device)

    raise ValueError(f"Unsupported schedule_type={schedule_type}; use 'linear' or 'cosine'.")


def compute_diffusion_buffers(betas: torch.Tensor) -> DiffusionBuffers:
    """Precompute all scalars required by q(x_t|x_0) and q(x_{t-1}|x_t,x_0).

    Forward process:
    q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)

    Posterior terms (closed form):
    q(x_{t-1} | x_t, x_0) = N(mu_tilde_t, beta_tilde_t I)
    with coefficients cached as posterior_mean_coef{1,2} and posterior_variance.
    """
    if betas.ndim != 1:
        raise ValueError("betas must be a 1D tensor of shape [L].")

    # alpha_t = 1 - beta_t
    alphas = 1.0 - betas
    # alpha_bar_t = product_{s<=t} alpha_s
    alpha_bars = torch.cumprod(alphas, dim=0)
    # alpha_bar_{t-1}, with alpha_bar_{-1} defined as 1 for convenience.
    alpha_bars_prev = torch.cat([torch.ones(1, device=betas.device, dtype=betas.dtype), alpha_bars[:-1]], dim=0)

    sqrt_alphas = torch.sqrt(alphas)
    sqrt_alpha_bars = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
    one_over_sqrt_alphas = 1.0 / sqrt_alphas

    sqrt_recip_alpha_bars = torch.sqrt(1.0 / alpha_bars)
    sqrt_recipm1_alpha_bars = torch.sqrt((1.0 / alpha_bars) - 1.0)

    # beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
    posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
    posterior_variance = torch.clamp(posterior_variance, min=1e-20)
    posterior_log_variance_clipped = torch.log(posterior_variance)

    # mu_tilde_t = coef1_t * x_0 + coef2_t * x_t
    posterior_mean_coef1 = betas * torch.sqrt(alpha_bars_prev) / (1.0 - alpha_bars)
    posterior_mean_coef2 = (1.0 - alpha_bars_prev) * torch.sqrt(alphas) / (1.0 - alpha_bars)

    return DiffusionBuffers(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        alpha_bars_prev=alpha_bars_prev,
        sqrt_alphas=sqrt_alphas,
        sqrt_alpha_bars=sqrt_alpha_bars,
        sqrt_one_minus_alpha_bars=sqrt_one_minus_alpha_bars,
        one_over_sqrt_alphas=one_over_sqrt_alphas,
        sqrt_recip_alpha_bars=sqrt_recip_alpha_bars,
        sqrt_recipm1_alpha_bars=sqrt_recipm1_alpha_bars,
        posterior_variance=posterior_variance,
        posterior_log_variance_clipped=posterior_log_variance_clipped,
        posterior_mean_coef1=posterior_mean_coef1,
        posterior_mean_coef2=posterior_mean_coef2,
    )


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Gather timestep-indexed values and reshape to broadcast over image tensors.

    Input:
    - a: [L]
    - t: [B]
    Returns shape [B, 1, 1, 1] for image-like x_shape.
    """
    if t.dtype != torch.long:
        t = t.long()
    out = a.gather(0, t)
    return out.view(-1, *([1] * (len(x_shape) - 1)))
