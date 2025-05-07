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
    def __init__(self, in_c, out_c, k, s, padding=None, g=1, d=1,act=None ):
        super.__init__()
        self.act = act if act is not None else nn.Identity()
        self.conv = nn.Conv2d(in_c,out_c,k,s,padding,groups=g,dilation=d,bias=False)
        self.bn = nn.BatchNorm2d(out_c)
    def forward(self,x):
        return self.act(self.bn(self.conv(x)))

# class C2f(nn.Module):


