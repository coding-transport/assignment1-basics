# _*_ coding : UTF-8 _*_
# @Time : 2026/4/24 18:12
# @Author : Yif Wang
# @file : model.py
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict
from cs336_basics.utils import *


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        # 标准命名：weight
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        # 标准命名：weight
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        nn.init.normal_(self.weight)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


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
        gate = x @ self.w1.T
        up = x @ self.w3.T
        return silu(gate) * up @ self.w2.T


class Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d = d_model
        self.num_heads = num_heads
        self.q_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.k_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.v_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.o_proj = nn.Parameter(torch.empty(d_model, d_model))

    def scaled_dot_product_attention(self, Q, K, V, mask):
        d_k = K.size(-1)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            # 确保 mask 为 False 的地方在 softmax 后变为 0
            scores = scores.masked_fill(mask == False, -1e9)
        weights = torch.softmax(scores, dim=-1)
        return weights @ V

    def forward(self, data):
        queries = data @ self.q_proj.T
        keys = data @ self.k_proj.T
        values = data @ self.v_proj.T
        n = values.size(-1) // self.num_heads
        results = Tensor([])
        for i in range(self.num_heads):
            Q = queries[:, :, i * n:(i + 1) * n]
            K = keys[:, :, i * n:(i + 1) * n]
            V = values[:, :, i * n:(i + 1) * n]
            mask = self.get_causal_mask(data.size(1))
            temp = self.scaled_dot_product_attention(Q, K, V, mask)
            results = torch.cat((results, temp), dim=-1)
        return results @ self.o_proj.T

    def get_causal_mask(self, seq_len):
        """
        生成一个上三角矩阵掩码
        返回 shape: (seq_len, seq_len)
        """
        # torch.triu 生成上三角，diagonal=1 表示不包含主对角线
        mask = (1 - torch.triu(torch.ones(seq_len, seq_len), diagonal=1)).bool()
        return mask  # True 表示需要被屏蔽的位置


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps):
        super().__init__()
        # 标准命名：weight
        self.weights = nn.Parameter(torch.empty(d_model, ))
        self.eps = eps
        nn.init.kaiming_uniform_(self.weights, a=5 ** 0.5)

    def forward(self, x: Tensor) -> Tensor:
        x_bar = x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x_bar @ self.weights.T
