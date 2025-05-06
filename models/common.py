#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：simple_yolo 
@File    ：common.py
@Author  ：kg5s55
@Description: 
"""
import torch
import torch.nn as nn


def auto_pad(k, padding=None, d=1):
    if d > 1:
        # 等价卷积
        k = (k - 1) * d if isinstance(k, int) else [(x - 1) * d for x in k]
    if padding is None:
        padding = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return padding


class Conv(nn.Module):
    def __init__(self, in_c, out_c, k, s, padding=None, g=1, d=1, act=nn.SiLU()):
        super.__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, padding=auto_pad(k, padding, d), groups=g, dilation=d, bias=False)
        self.act = act if act is not None else nn.Identity()
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, in_c, out_c, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(out_c * e)  # hidden channels
        self.cv1 = Conv(in_c, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, out_c, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
