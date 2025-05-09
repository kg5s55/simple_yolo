#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：simple_yolo 
@File    ：build_yolo.py
@Author  ：kg5s55
@Description: 
"""
# import os
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
    SPPF,
    Concat
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
    param = []
    for i, (f, n, m, args) in enumerate(cfg["backbone"] + cfg["head"]):  # from, number, module, args
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                try:
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a) # 'None'-->
                    # print(args[j],"ok",args,i,j)
                except ValueError:
                    pass
        # 参数n的更新
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
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
        else:
            c2 = ch[f]
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        param.append({m:m_.np})
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    print(param)
    return nn.Sequential(*layers), sorted(save)

if __name__ == "__main__":
    # print(globals())
    # print(locals())
    model,_ = build_model("yolov8.yaml")

