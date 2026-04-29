# _*_ coding : UTF-8 _*_
# @Time : 2026/4/28 17:05
# @Author : Yif Wang
# @file : data
import torch


def get_batch(data, batch_size, context_length, device):
    randrange = len(data) - context_length
    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data)
    ix = torch.randint(0, randrange, (batch_size,))
    x = torch.stack([data[i: i + context_length] for i in ix])
    # y 形状: (batch_size, context_length)，即 x 往后偏移一位的目标
    y = torch.stack([data[i + 1: i + context_length + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
