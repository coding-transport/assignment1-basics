# _*_ coding : UTF-8 _*_
# @Time : 2026/4/24 18:12
# @Author : Yif Wang
# @file : model.py
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Dict


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        # 标准命名：weight
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight)


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        # 标准命名：weight
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        nn.init.normal_(self.weight)


    def forward(self, token_ids: Tensor) -> Tensor:
        return F.embedding(token_ids, self.weight)


class Swiglu(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # 命名与主流模型（如 Llama）和你的测试用例对齐
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model))

        for p in [self.w1, self.w2, self.w3]:
            nn.init.kaiming_uniform_(p, a=5 ** 0.5)


    def forward(self, x: Tensor) -> Tensor:
        # SwiGLU: (SiLU(x @ w1.T) * (x @ w3.T)) @ w2.T
        gate = F.linear(x, self.w1)
        up = F.linear(x, self.w3)
        return F.linear(F.silu(gate) * up, self.w2)

class Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d = d_model
        self.num_heads = num_heads
        self.k_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.q_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.v_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.o_proj = nn.Parameter(torch.empty(d_model, d_model))

    def scaled_dot_product_attention(self, Q, K, V, mask):
        d_k = Q.size(-1)
        scores = Q @ K.transpose(-2, -1)/math.sqrt(d_k)
        if mask is not None:
            # 确保 mask 为 False 的地方在 softmax 后变为 0
            scores = scores.masked_fill(mask == False, -1e9)
        weights = torch.softmax(scores, dim=-1)
        return weights @ V