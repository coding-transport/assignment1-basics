# _*_ coding : UTF-8 _*_
# @Time : 2026/4/28 17:03
# @Author : Yif Wang
# @file : utils
import torch


def silu(data):
    return data * torch.sigmoid(data)


def softmax(data, dim=-1):
    max_val = torch.max(data, dim=dim, keepdim=True)[0]
    soft = torch.exp(data - max_val)
    return soft / soft.sum(dim=dim, keepdim=True)


def cross_entropy(data, target, dim=-1):
    batch_size = data.shape[0]
    max_val = torch.max(data, dim=dim, keepdim=True)[0]
    data = data - max_val
    prob = data - torch.log(torch.exp(data).sum(dim=dim, keepdim=True))
    log_p_target = -prob[torch.arange(batch_size), target]
    return torch.mean(log_p_target)

def gradient_clip(parameters, max_l2_norm):
    total_norm_sq = 0
    for tensor in parameters:
        if tensor.grad is not None:
            total_norm_sq += torch.sum(tensor.grad ** 2)

    total_norm = torch.sqrt(total_norm_sq)

    if total_norm > max_l2_norm:
        clip_coeff = max_l2_norm / (total_norm + 1e-6)
        for tensor in parameters:
            if tensor.grad is not None:
                # 使用统一的系数进行缩放
                tensor.grad.detach().mul_(clip_coeff)

def save_checkpoint(model, optimizer, iteration, out):
    pass


def load_checkpoint(src, model, optimizer):
    pass
