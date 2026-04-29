# _*_ coding : UTF-8 _*_
# @Time : 2026/4/24 18:13
# @Author : Yif Wang
# @file : optimizer
import math
from typing import Optional, Callable, overload

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(self, params, lr=0.99, betas=(0.9, 0.9), weight_decay=0.1, eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # 指数移动平均：一阶矩（动量）
                    state['exp_avg'] = torch.zeros_like(p)
                    # 指数移动平均：二阶矩（梯度平方）
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                t = state['step']
                lr = group['lr']

                exp_avg = group['betas'][0] * exp_avg + (1 - group['betas'][0]) * p.grad
                exp_avg_sq = group['betas'][1] * exp_avg_sq + (1 - group['betas'][1]) * p.grad ** 2
                exp_avg_hat = exp_avg / (1 - group['betas'][0] ** t)
                exp_avg_sq_hat = exp_avg_sq / (1 - group['betas'][1] ** t)
                p.add_(p * group['weight_decay'], alpha=-lr)
                update = exp_avg_hat / (torch.sqrt(exp_avg_sq_hat) + group['eps'])
                p.add_(update, alpha=-lr)
                state['exp_avg'] = exp_avg
                state['exp_avg_sq'] = exp_avg_sq
        return loss


def get_lr_cosine_schedule(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    elif warmup_iters <= it < cosine_cycle_iters:
        coeff = 0.5 * (1 + math.cos(torch.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)))
        lr_res = min_learning_rate + coeff * (max_learning_rate - min_learning_rate)
        return lr_res
    else:
        return min_learning_rate
