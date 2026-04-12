"""
airbench_net.py

Minimal Airbench94-style network architecture (from the paper) with MaxPool2d
replaced by LSEPool2d (log-sum-exp pooling) for analyticity.
"""

from __future__ import annotations

from torch import nn

from polytensor import LSEPool2d


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Mul(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)

    def forward(self, x):
        return x * self.scale


def conv(ch_in: int, ch_out: int) -> nn.Module:
    return nn.Conv2d(ch_in, ch_out, kernel_size=3, padding="same", bias=False)


def make_net() -> nn.Module:
    act = lambda: nn.GELU()  # exact erf-based GELU
    bn = lambda ch: nn.BatchNorm2d(ch)

    return nn.Sequential(
        nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=2, padding=0, bias=True),
            act(),
        ),
        nn.Sequential(
            conv(24, 64),
            LSEPool2d(2),
            bn(64), act(),
            conv(64, 64),
            bn(64), act(),
        ),
        nn.Sequential(
            conv(64, 256),
            LSEPool2d(2),
            bn(256), act(),
            conv(256, 256),
            bn(256), act(),
        ),
        nn.Sequential(
            conv(256, 256),
            LSEPool2d(2),
            bn(256), act(),
            conv(256, 256),
            bn(256), act(),
        ),
        LSEPool2d(3),
        Flatten(),
        nn.Linear(256, 10, bias=False),
        Mul(1 / 9),
    )
