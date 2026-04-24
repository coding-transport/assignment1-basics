# _*_ coding : UTF-8 _*_
# @Time : 2026/4/24 18:12
# @Author : Yif Wang
# @file : model
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class Linear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_out, d_in))

    def forward(self, data):
        result = data @ self.weight.T
        return result