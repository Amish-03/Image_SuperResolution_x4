"""
RCAN – Residual Channel Attention Network for ×4 Super-Resolution.

Architecture highlights:
  • Deep residual network with Channel Attention (CA) blocks.
  • PixelShuffle (×4) upsampling – no transposed convolutions.
  • Fully deterministic at inference (no Dropout, no noise injection).
  • RGB mean subtraction / addition for DIV2K.
"""

import torch
import torch.nn as nn
import math


# ─── Building blocks ────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""

    def __init__(self, n_feats, reduction=16):
        super().__init__()
        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feats, n_feats // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats // reduction, n_feats, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.body(x)


class RCAB(nn.Module):
    """Residual Channel Attention Block."""

    def __init__(self, n_feats, reduction=16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True),
            ChannelAttention(n_feats, reduction),
        )

    def forward(self, x):
        return x + self.body(x)


class ResidualGroup(nn.Module):
    """Stack of RCAB blocks with a skip connection."""

    def __init__(self, n_feats, n_resblocks, reduction=16):
        super().__init__()
        body = [RCAB(n_feats, reduction) for _ in range(n_resblocks)]
        body.append(nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return x + self.body(x)


# ─── Upsampler utility ──────────────────────────────────────────────────

def _make_upsample(n_feats, scale):
    """Build a PixelShuffle upsampler for the given scale (2 or 4)."""
    layers = []
    if scale in (2, 4):
        for _ in range(int(math.log2(scale))):
            layers.append(nn.Conv2d(n_feats, n_feats * 4, 3, padding=1, bias=True))
            layers.append(nn.PixelShuffle(2))
            layers.append(nn.ReLU(inplace=True))
    elif scale == 3:
        layers.append(nn.Conv2d(n_feats, n_feats * 9, 3, padding=1, bias=True))
        layers.append(nn.PixelShuffle(3))
        layers.append(nn.ReLU(inplace=True))
    else:
        raise ValueError(f"Unsupported scale: {scale}")
    return nn.Sequential(*layers)


# ─── Full Network ───────────────────────────────────────────────────────

class RCAN(nn.Module):
    """
    Residual Channel Attention Network (RCAN).

    Default hyper-parameters give ~15.6 M params which comfortably fits
    on a single A6000 GPU for ×4 SR training with 64-px LR patches.

    Args:
        n_resgroups : number of Residual Groups
        n_resblocks : number of RCAB blocks per group
        n_feats     : base feature channels
        reduction   : channel attention reduction ratio
        scale       : upsampling factor (default 4)
    """

    def __init__(self, n_resgroups=10, n_resblocks=20, n_feats=64,
                 reduction=16, scale=4):
        super().__init__()

        # DIV2K RGB mean (computed over training split)
        rgb_mean = torch.FloatTensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1)
        self.register_buffer("rgb_mean", rgb_mean)

        # Head
        self.head = nn.Conv2d(3, n_feats, 3, padding=1, bias=True)

        # Body — stacked Residual Groups + tail conv
        body = [ResidualGroup(n_feats, n_resblocks, reduction)
                for _ in range(n_resgroups)]
        body.append(nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True))
        self.body = nn.Sequential(*body)

        # Tail — PixelShuffle ×scale → final 3-ch image
        self.upsample = _make_upsample(n_feats, scale)
        self.tail = nn.Conv2d(n_feats, 3, 3, padding=1, bias=True)

    def forward(self, x):
        # Subtract mean
        x = x - self.rgb_mean

        # Feature extraction
        head = self.head(x)
        body = self.body(head) + head          # long skip

        # Reconstruction
        up = self.upsample(body)
        out = self.tail(up)

        # Add mean
        out = out + self.rgb_mean
        return out


def make_model(**kwargs):
    """Factory helper."""
    return RCAN(**kwargs)
