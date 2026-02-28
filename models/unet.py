from __future__ import annotations
"""Small DDPM U-Net epsilon predictor.

Design choices (per assignment):
- Channel progression: 32 -> 64 -> 128
- GroupNorm + SiLU activations
- No attention blocks
- Timestep conditioning via sinusoidal embedding + MLP
- Embedding is injected as a bias term inside residual blocks
"""

import math

import torch
import torch.nn as nn


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Create sinusoidal timestep embeddings (Transformer-style frequencies)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResidualBlock(nn.Module):
    """Residual block with FiLM-like timestep bias injection.

The projected time embedding is added after the first convolution:
h = conv1(norm+act(x)) + time_proj(t), then conv2(norm+act(h)).
"""
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(groups, in_ch), num_channels=in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch),
        )

        self.norm2 = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    """Downsample by factor 2 using strided convolution."""
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    """Upsample by factor 2 using transposed convolution."""
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class SmallUNet(nn.Module):
    """Three-level U-Net with skip connections at matching resolutions.

    Input: x_t and timestep t
    Output: epsilon_hat with same spatial shape as x_t
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channel_multipliers: tuple[int, int, int] = (32, 64, 128),
        time_emb_dim: int = 128,
    ):
        super().__init__()
        c1, c2, c3 = channel_multipliers

        self.in_conv = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.down1 = ResidualBlock(c1, c1, time_emb_dim)
        self.downsample1 = Downsample(c1)

        self.down2 = ResidualBlock(c1, c2, time_emb_dim)
        self.downsample2 = Downsample(c2)

        self.bottleneck = ResidualBlock(c2, c3, time_emb_dim)

        self.up2 = Upsample(c3)
        self.dec2 = ResidualBlock(c3 + c2, c2, time_emb_dim)

        self.up1 = Upsample(c2)
        self.dec1 = ResidualBlock(c2 + c1, c1, time_emb_dim)

        self.out_norm = nn.GroupNorm(num_groups=8, num_channels=c1)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=3, padding=1)

        self.time_emb_dim = time_emb_dim

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise epsilon_hat given noised image x_t and timestep t."""
        t_emb = timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        x0 = self.in_conv(x)

        h1 = self.down1(x0, t_emb)
        d1 = self.downsample1(h1)

        h2 = self.down2(d1, t_emb)
        d2 = self.downsample2(h2)

        b = self.bottleneck(d2, t_emb)

        # Decoder skip-connections fuse encoder features at same resolution.
        u2 = self.up2(b)
        u2 = torch.cat([u2, h2], dim=1)
        u2 = self.dec2(u2, t_emb)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, h1], dim=1)
        u1 = self.dec1(u1, t_emb)

        out = self.out_conv(self.out_act(self.out_norm(u1)))
        return out
