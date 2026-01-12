from einops.layers.torch import Rearrange
import torch.nn as nn
from typing import List, Tuple
import torch

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class SpatioTemporalMixerBlock(nn.Module):
    """Mixer block with time, token and channel mixing."""
    def __init__(
        self,
        dim,
        num_patch,
        num_frames,
        token_dim,
        time_dim,
        channel_dim,
        dropout=0.,
    ):
        super().__init__()

        self.time_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b t s c -> b s c t'),
            FeedForward(num_frames, time_dim, dropout),
            Rearrange('b s c t -> b t s c'),
        )

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b t s c -> b t c s'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b t c s -> b t s c'),
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.time_mix(x)
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x
