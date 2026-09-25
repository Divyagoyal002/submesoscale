"""Super-resolution CNNs.

Both networks use the *pre-upsampling* design: the LR ADT has already been
interpolated onto the target grid, and fine-scale SST is available there too, so
the network works at a single resolution and predicts a residual correction.

* :class:`SRCNN` - Dong et al. (2015), the original 3-layer SR network (baseline).
* :class:`ResidualSR` - EDSR-style residual blocks (no batch-norm, residual scaling)
  with optional dilated convolutions to enlarge the receptive field cheaply, in the
  spirit of the dilated adaptive residual network used by the reference study.
"""

from __future__ import annotations

import torch
from torch import nn

from submeso.config import ModelConfig
from submeso.data.dataset import N_INPUT_CHANNELS


class SRCNN(nn.Module):
    def __init__(self, in_channels: int = N_INPUT_CHANNELS, channels: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, channels, 9, padding=4, padding_mode="replicate"),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels // 2, 5, padding=2, padding_mode="replicate"),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 5, padding=2, padding_mode="replicate"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class ResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1, res_scale: float = 0.1):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv2d(
            channels, channels, 3, padding=pad, dilation=dilation, padding_mode="replicate"
        )
        self.conv2 = nn.Conv2d(
            channels, channels, 3, padding=pad, dilation=dilation, padding_mode="replicate"
        )
        self.act = nn.GELU()
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_scale * self.conv2(self.act(self.conv1(x)))


class ResidualSR(nn.Module):
    def __init__(
        self,
        in_channels: int = N_INPUT_CHANNELS,
        channels: int = 64,
        n_blocks: int = 8,
        dilated: bool = True,
        res_scale: float = 0.1,
    ):
        super().__init__()
        dilations = [1, 2, 4, 2] if dilated else [1]
        self.head = nn.Conv2d(in_channels, channels, 3, padding=1, padding_mode="replicate")
        self.blocks = nn.Sequential(
            *[ResBlock(channels, dilations[i % len(dilations)], res_scale) for i in range(n_blocks)]
        )
        self.fuse = nn.Conv2d(channels, channels, 3, padding=1, padding_mode="replicate")
        self.tail = nn.Conv2d(channels, 1, 3, padding=1, padding_mode="replicate")
        nn.init.zeros_(self.tail.weight)  # start as identity (residual = 0 -> LR input)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.head(x)
        h = h + self.fuse(self.blocks(h))
        return self.tail(h)


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.name == "srcnn":
        return SRCNN(channels=cfg.channels)
    if cfg.name == "resnet":
        return ResidualSR(
            channels=cfg.channels,
            n_blocks=cfg.n_blocks,
            dilated=cfg.dilated,
            res_scale=cfg.res_scale,
        )
    raise ValueError(f"Unknown model '{cfg.name}' (expected 'resnet' or 'srcnn')")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
