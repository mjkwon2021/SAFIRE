"""
Myung-Joon Kwon
2024-01-22
"""
import warnings

import torch
from torch import Tensor
from typing import Callable, Optional
import torch.nn.functional as F
from torch.nn import Module


class ILW_BCEWithLogitsLoss(Module):
    def __init__(self, reduction: str = 'mean', ignore_label=-1) -> None:
        super().__init__()
        assert reduction == 'mean'
        self.reduction = reduction
        self.ignore_label = ignore_label

    def forward(self, input: Tensor, target: Tensor,) -> Tensor:
        num_of_zeros = (target == 0).view((target.shape[0], -1)).sum(dim=-1).clamp(min=1)
        num_of_ones = (target == 1).view((target.shape[0], -1)).sum(dim=-1).clamp(min=1)
        weight_zeros = torch.clamp(0.5 * (num_of_ones + num_of_zeros) / num_of_zeros, min=0.1, max=10)
        weight_zero_elements = weight_zeros[:,None,None,None].repeat(1,1,target.shape[2],target.shape[3])
        weight_ones = torch.clamp(0.5 * (num_of_ones + num_of_zeros) / num_of_ones, min=0.1, max=10)
        weight_one_elements = weight_ones[:,None,None,None].repeat(1,1,target.shape[2],target.shape[3])
        weights_elements = torch.where(target == 0, weight_zero_elements, weight_one_elements)
        weights_elements = torch.where(target == self.ignore_label, 0, weights_elements)
        bce = F.binary_cross_entropy_with_logits(input, target.float(), reduction='none')
        ilw_bce = bce * weights_elements
        b = ilw_bce.sum() / (target != self.ignore_label).sum()
        return b


class Xent_NoIgnore(Module):
    def __init__(self, reduction: str = 'mean', ignore_label=-1) -> None:
        super().__init__()
        assert reduction == 'mean'
        self.reduction = reduction
        self.ignore_label = ignore_label

    def forward(self, input: Tensor, target: Tensor,) -> Tensor:
        target[target == self.ignore_label] = 0
        bce = F.binary_cross_entropy_with_logits(input, target.float(), reduction='mean')
        return bce


class PixelAccWithIgnoreLabel(Module):
    def __init__(self, ignore_label=-1) -> None:
        super().__init__()
        self.ignore_label = ignore_label

    def forward(self, input: Tensor, target: Tensor,) -> Tensor:
        correct = (input>=0) == target
        correct_without_ignore_label = torch.logical_and(correct, target != self.ignore_label)
        correct_count = torch.sum(correct_without_ignore_label.view(correct_without_ignore_label.shape[0],-1), dim=1)
        acc = correct_count / (target != self.ignore_label).view(target.shape[0],-1).sum(dim=1)
        return acc

