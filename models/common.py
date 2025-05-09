#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：simple_yolo 
@File    ：common.py
@Author  ：kg5s55
@Description: 
"""
from modulefinder import Module

import torch
import torch.nn as nn
import math

def auto_pad(k,padding=None,d=1):
    """
    只考虑一边假定输入特征长度为x，卷积核为k，步长为s，自动padding的结果为k//2 最终的输出尺寸为\frac{x-k+2(k//2)}{s}+1
    如果k为偶数则上述的式子为\frac{x}{s}+1
    如果k为奇数则上述的式子为\frac{x}{s}
    等效卷积核的计算也是 应该是只这对奇数卷积核设计的
    """
    if d > 1:
        # 等效卷积核尺寸
        k = d * (k-1) if isinstance(k,int) else [d*(k-1) +1 for x in k ]
    if padding is None:
        padding = k // 2 if isinstance(k,int) else [x // 2 for x in k]
    return padding

# 普通2为卷积+bn层+激活函数的组合模块Conv
class Conv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, in_c, out_c, k=1, s=1, padding=None, g=1, d=1,act=True ):
        super().__init__()
        self.act = self.default_act if act is True   else act  if isinstance(act, nn.Module) else nn.Identity()
        self.conv = nn.Conv2d(in_c,out_c,k,s,padding,groups=g,dilation=d,bias=False)
        self.bn = nn.BatchNorm2d(out_c)
    def forward(self,x):
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

class SPPF(nn.Module):
    def __init__(self,in_c,out_c,k=5):
        """
        改进SPP 原先得最大池化尺寸（5，9，13）可通过3个5*5最大池化串联代替，例如9*9最大池化 可用两个5*5最大池化串联替代，
        13*13可用3个5*5最大池化替代
        """
        super().__init__()
        c_ = in_c // 2
        self.cv1 = Conv(in_c,c_,1,1) # 1*1卷积
        self.cv2 = Conv(c_*4,out_c,1,1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    def forward(self,x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y,1))
# 分组卷积
class DWConv(Conv):
    def __init__(self, in_c, out_c, k=1, s=1, padding=None, d=1,act=True ):
        super().__init__( in_c, out_c, k, s,padding=padding,g=math.gcd(in_c,out_c),d=d,act=act)



class Detect(nn.Module):
    def __init__(self,nc=80,ch=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor

        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels

        # 回归分支
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )

        # 类别分支
        self.cv3 = (
            nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
    def forward(self,x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        # if self.training:
        return x
        # y = self._inference(x)
        # return y if self.export else (y,x)


# concat
class Concat(nn.Module):
    def __init__(self,dim=1):
        super().__init__()
        self.dim = dim
    def forward(self,x):
        return torch.cat(x,self.dim)

class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)