#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：simple_yolo 
@File    ：build_yolo.py
@Author  ：kg5s55
@Description: 
"""
import os
import yaml
import torch
import ast
import math
import torch.nn as nn
from models.common import (
    Conv,
    DWConv,
    C2f,
    Detect,
    SPPF
)
def parse_cfg(cfg):
    if isinstance(cfg,str):
        with open(cfg,"r") as f:
            cfg = yaml.safe_load(f)
    # print(cfg)
    return cfg
def make_divisible(x, divisor):
    """
    Returns the nearest number that is divisible by the given divisor.

    Args:
        x (int): The number to make divisible.
        divisor (int | torch.Tensor): The divisor.

    Returns:
        (int): The nearest number divisible by the divisor.
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor
def build_model(cfg):
    cfg = parse_cfg(cfg)
    ch = cfg['ch']
    # 解析
    # scales是 s,n,m,l,x
    # act是激活函数
    # nc是类别数
    nc, act, scales = (cfg.get(x) for x in ("nc", "activation", "scales"))
    # nsmlx选择
    if scales:
        scale = cfg.get("scale")
        if not scale:
            scale  = tuple(scales.keys())[0]
            depth, width, max_channels = scales[scale]

    # 激活函数选择
    if act:
        Conv.default_act = eval(act)
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(cfg["backbone"] + cfg["head"]):  # from, number, module, args
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                try:
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
                except ValueError:
                    pass
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in {
            Conv , SPPF,DWConv,C2f
        }:
            c1, c2 = ch[f], args[0] #输入输出通道
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m in {
                C2f
            }:
                args.insert(2, n)  # number of repeats
                n = 1
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
