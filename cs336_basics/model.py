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
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        # SwiGLU: (SiLU(x @ w1.T) * (x @ w3.T)) @ w2.T
        gate = self.w1(x)
        up = self.w3(x)
        return self.w2(silu(gate) * up)


class Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d = d_model
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask):
        d_k = K.size(-1)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            # 确保 mask 为 False 的地方在 softmax 后变为 0
            scores = scores.masked_fill(mask == False, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        return weights @ V

    def forward(self, data):
        queries = self.q_proj(data)
        keys = self.k_proj(data)
        values = self.v_proj(data)
        n = values.size(-1) // self.num_heads
        results = Tensor([])
        for i in range(self.num_heads):
            Q = queries[:, :, i * n:(i + 1) * n]
            K = keys[:, :, i * n:(i + 1) * n]
            V = values[:, :, i * n:(i + 1) * n]
            mask = self.get_causal_mask(data.size(1))
            temp = self.scaled_dot_product_attention(Q, K, V, mask)
            results = torch.cat((results, temp), dim=-1)
        return self.output_proj(results)

    def get_causal_mask(self, seq_len):
        """
        生成一个上三角矩阵掩码
        返回 shape: (seq_len, seq_len)
        """
        # torch.triu 生成上三角，diagonal=1 表示不包含主对角线
        mask = (1 - torch.triu(torch.ones(seq_len, seq_len), diagonal=1)).bool()
        return mask  # True 表示需要被屏蔽的位置


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        # 标准命名：weight
        self.weight = nn.Parameter(torch.empty(d_model, ))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        x_bar = x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x_bar * self.weight


class RoPE(nn.Module):
    def __init__(self, d_k: int, theta: float,
                 max_seq_len: int, ):
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2).float() / self.d_k))
        t = torch.arange(self.max_seq_len)
        freqs = torch.einsum("i,j->ij", t, inv_freq)

        # 4. 最后计算 sin 和 cos
        self.sin_cached = freqs.sin()  # 每一行对应一个位置，每一列对应一对维度
        self.cos_cached = freqs.cos()

    def forward(self, in_query_or_key, token_positions):
        sin = self.sin_cached[token_positions]
        cos = self.cos_cached[token_positions]
        sin = torch.repeat_interleave(sin, 2, dim=-1)
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        x = in_query_or_key
        x_left = x[..., 0::2]  # 偶数下标
        x_right = x[..., 1::2]  # 奇数下标

        x_neg_swapped = torch.stack([-x_right, x_left], dim=-1).flatten(-2)

        # 4. 应用 RoPE 公式
        return (x * cos) + (x_neg_swapped * sin)


class Attention_with_PoPE(nn.Module):
    def __init__(self, d_model, num_heads, rope, token_positions):
        super().__init__()
        self.d = d_model
        self.num_heads = num_heads
        self.d = d_model
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.rope = rope
        self.token_positions = token_positions

    def forward(self, data):
        queries = self.q_proj(data)
        keys = self.k_proj(data)
        values = self.v_proj(data)
        n = values.size(-1) // self.num_heads
        results = Tensor([])
        for i in range(self.num_heads):
            Q = queries[:, :, i * n:(i + 1) * n]
            K = keys[:, :, i * n:(i + 1) * n]
            V = values[:, :, i * n:(i + 1) * n]
            Q = self.rope(Q, self.token_positions)
            K = self.rope(K, self.token_positions)
            mask = self.get_causal_mask(data.size(1))
            temp = self.scaled_dot_product_attention(Q, K, V, mask)
            results = torch.cat((results, temp), dim=-1)
        return self.output_proj(results)

    def get_causal_mask(self, seq_len):
        """
        生成一个上三角矩阵掩码
        返回 shape: (seq_len, seq_len)
        """
        # torch.triu 生成上三角，diagonal=1 表示不包含主对角线
        mask = (1 - torch.triu(torch.ones(seq_len, seq_len), diagonal=1)).bool()
        return mask  # True 表示需要被屏蔽的位置

    def scaled_dot_product_attention(self, Q, K, V, mask):
        d_k = K.size(-1)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            # 确保 mask 为 False 的地方在 softmax 后变为 0
            scores = scores.masked_fill(mask == False, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        return weights @ V


class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, seq_len):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        # 对应参数名: ln1.weight
        self.ln1 = RMSNorm(d_model)
        # 对应参数名: attn.xxx
        rope = RoPE(self.d_model // self.num_heads, theta, max_seq_len)
        token_positions = torch.arange(seq_len).unsqueeze(0)  # [1, seq_len]
        self.attn = Attention_with_PoPE(d_model, num_heads, rope, token_positions)
        # 对应参数名: ln2.weight
        self.ln2 = RMSNorm(d_model)
        # 对应参数名: ffn.xxx
        self.ffn = Swiglu(d_model, d_ff)

    def forward(self, in_features):
        x = self.attn(self.ln1(in_features)) + in_features
        x = self.ffn(self.ln2(x)) + x
        return x


class MyLLM(nn.Module):
    def __init__(self, vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float,
                 seq_len):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            Transformer(d_model, num_heads, d_ff, context_length, rope_theta, seq_len) for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        # 最后的收尾
        x = self.ln_final(x)
        return self.lm_head(x)