import math
from dataclasses import dataclass
from typing import Literal
from types import FunctionType

import torch
import torch.nn as nn
import torch.nn.functional as F


def g_func(z, t):
    return 2. * ((z + 1.) / 2.).pow(t) - 1.


def cosface_additive_margin(cos_theta, b, one_hot, t, m, r, **kwargs):

    g_cos_theta_m = g_func(cos_theta, t) - m * (2. * one_hot - 1.)

    logits = r * g_cos_theta_m
    if b:
        logits += b
    return logits


def arcface_additive_margin(cos_theta, b, y, index_y, t, m, r, **kwargs):

    cos_theta_m = torch.acos(cos_theta)
    cos_theta_m[index_y, y] += m
    cos_theta_m[index_y, y].clamp_max_(math.pi)
    cos_theta_m = torch.cos(cos_theta_m)

    g_cos_theta_m = g_func(cos_theta_m, t)

    logits = r * g_cos_theta_m
    if b:
        logits += b
    return logits


def sphereface2_multiplicative_additive_margin(cos_theta, b, y, index_y, t, m, r, **kwargs):

    cos_theta_m = torch.acos(cos_theta)
    cos_theta_m[index_y, y] *= m
    cos_theta_m[index_y, y].clamp_max_(math.pi)
    cos_theta_m = torch.cos(cos_theta_m)

    g_cos_theta_m = g_func(cos_theta_m, t)

    logits = r * g_cos_theta_m
    if b:
        logits += b
    return logits


@dataclass
class MarginFunc:
    C:FunctionType = cosface_additive_margin
    A:FunctionType = arcface_additive_margin
    M:FunctionType = sphereface2_multiplicative_additive_margin


class SphereLoss(nn.Module):

    def __init__(self, scale:float=1, margin:float=0.4, lamb:float=0.7, t:int=3,
                margin_type:Literal['C', 'A', 'M']='C'):
        super(SphereLoss, self).__init__()
        self.r = scale
        self.m = margin
        self.t = t
        self.lamb = lamb
        self.margin_func = getattr(MarginFunc, margin_type)
        self.g_func = g_func

    def forward(self, x, y, b:torch.Tensor=None):
        index_y = torch.arange(len(y))

        cos_theta = x

        one_hot = torch.zeros_like(cos_theta)
        one_hot[index_y, y] = 1

        logits = self.margin_func(cos_theta=cos_theta, b=b, y=y, index_y=index_y, one_hot=one_hot, **self.__dict__)

        weight = self.lamb * one_hot + (1. - self.lamb) * (1. - one_hot)
        weight = weight / self.r

        # return logits, weight
        loss = F.binary_cross_entropy_with_logits(
                logits, one_hot, weight=weight)

        return loss, logits, weight