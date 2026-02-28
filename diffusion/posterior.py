from __future__ import annotations
"""Closed-form posterior and reverse-step utilities for DDPM.

Given x_t and model-predicted epsilon_hat, this file computes:
- x0_hat estimate
- posterior mean/variance terms
- one reverse sampling step x_{t-1} ~ p_theta(x_{t-1}|x_t)
"""

import torch

from diffusion.schedule import DiffusionBuffers, extract


def predict_x0_from_eps(
    xt: torch.Tensor,
    t: torch.Tensor,
    eps_hat: torch.Tensor,
    buffers: DiffusionBuffers,
    clip_x0: bool = True,
) -> torch.Tensor:
    """Recover x0 estimate from x_t and predicted noise epsilon_hat.

    From x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) epsilon,
    we derive:
    x0_hat = (1/sqrt(alpha_bar_t)) x_t - sqrt(1/alpha_bar_t - 1) epsilon_hat.
    """
    sqrt_recip_alpha_bar_t = extract(buffers.sqrt_recip_alpha_bars, t, xt.shape)
    sqrt_recipm1_alpha_bar_t = extract(buffers.sqrt_recipm1_alpha_bars, t, xt.shape)
    x0_hat = sqrt_recip_alpha_bar_t * xt - sqrt_recipm1_alpha_bar_t * eps_hat
    if clip_x0:
        x0_hat = torch.clamp(x0_hat, -1.0, 1.0)
    return x0_hat


def q_posterior_mean_var(
    x0: torch.Tensor,
    xt: torch.Tensor,
    t: torch.Tensor,
    buffers: DiffusionBuffers,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute q(x_{t-1}|x_t,x_0) Gaussian parameters.

    Uses cached coefficients so that:
    mu_tilde_t = c1_t * x_0 + c2_t * x_t,
    beta_tilde_t = posterior_variance_t.
    """
    coef1 = extract(buffers.posterior_mean_coef1, t, xt.shape)
    coef2 = extract(buffers.posterior_mean_coef2, t, xt.shape)
    mean = coef1 * x0 + coef2 * xt
    var = extract(buffers.posterior_variance, t, xt.shape)
    return mean, var


def p_mean_from_eps(
    xt: torch.Tensor,
    t: torch.Tensor,
    eps_hat: torch.Tensor,
    buffers: DiffusionBuffers,
    clip_x0: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build p_theta mean/variance by plugging x0_hat into posterior formula.

    DDPM parameterization predicts epsilon, then converts to x0_hat,
    then reuses q posterior closed-form with x0 replaced by x0_hat.
    """
    x0_hat = predict_x0_from_eps(xt=xt, t=t, eps_hat=eps_hat, buffers=buffers, clip_x0=clip_x0)
    mean, var = q_posterior_mean_var(x0=x0_hat, xt=xt, t=t, buffers=buffers)
    return mean, var, x0_hat


@torch.no_grad()
def p_sample_step(
    xt: torch.Tensor,
    t: torch.Tensor,
    eps_hat: torch.Tensor,
    buffers: DiffusionBuffers,
    clip_x0: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one reverse step x_{t-1}.

    For t>0: x_{t-1} = mean + sqrt(var) * z, z~N(0,I)
    For t=0: no additional noise is added.
    """
    mean, var, x0_hat = p_mean_from_eps(xt=xt, t=t, eps_hat=eps_hat, buffers=buffers, clip_x0=clip_x0)
    noise = torch.randn_like(xt)
    nonzero_mask = (t > 0).float().view(-1, 1, 1, 1)
    xt_prev = mean + nonzero_mask * torch.sqrt(var) * noise
    return xt_prev, x0_hat
