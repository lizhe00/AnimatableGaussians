"""
Losses for meshes
Borrowed from: https://github.com/ShichenLiu/SoftRas
Note that I changed the implementation of laplacian matrices from dense tensor to COO sparse tensor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

import network.styleunet.conv2d_gradfix as conv2d_gradfix


class SecondOrderSmoothnessLossForSequence(nn.Module):
    def __init__(self):
        super(SecondOrderSmoothnessLossForSequence, self).__init__()

    def forward(self, x, dim=0):
        assert x.shape[dim] > 3
        a = x.shape[dim]
        a0 = torch.arange(0, a-2).long().to(x.device)
        a1 = torch.arange(1, a-1).long().to(x.device)
        a2 = torch.arange(2, a).long().to(x.device)
        x0 = torch.index_select(x, dim, index=a0)
        x1 = torch.index_select(x, dim, index=a1)
        x2 = torch.index_select(x, dim, index=a2)

        l = (2*x1 - x2 - x0).pow(2)
        return torch.mean(l)


class WeightedMSELoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super(WeightedMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target, weight):
        return F.mse_loss(pred * weight, target * weight, reduction=self.reduction)


class CosineSimilarityLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super(CosineSimilarityLoss, self).__init__()
        self.reduction = reduction
        if reduction not in ['mean', 'none', 'sum']:
            raise RuntimeError('Unknown reduction type! It should be in ["mean", "none", "sum"]')

    def forward(self, pred, target, weight=None, dim=-1, normalized=True):
        if normalized:      # assumes both ```pred``` and ```target``` have been normalized
            cs = 1 - torch.sum(pred*target, dim=dim)
        else:
            cs = 1 - F.cosine_similarity(pred, target, dim=dim)

        if weight is not None:
            cs = weight * cs
        if self.reduction == 'mean':
            return torch.mean(cs)
        else:
            return torch.sum(cs)


class LeastMagnitudeLoss(nn.Module):
    def __init__(self, average=False):
        super(LeastMagnitudeLoss, self).__init__()
        self.average = average

    def forward(self, x):
        batch_size = x.size(0)
        dims = tuple(range(x.ndimension())[1:])
        x = x.pow(2).sum(dims)
        if self.average:
            return x.sum() / batch_size
        else:
            return x.sum()


class NegIOULoss(nn.Module):
    def __init__(self, average=False):
        super(NegIOULoss, self).__init__()
        self.average = average

    def forward(self, predict, target):
        dims = tuple(range(predict.ndimension())[1:])
        intersect = (predict * target).sum(dims)
        union = (predict + target - predict * target).sum(dims) + 1e-6
        return 1. - (intersect / union).sum() / intersect.nelement()


class KLDLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(KLDLoss, self).__init__()
        self.reduction = reduction

    def forward(self, mu, logvar):
        d = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        if self.reduction == 'mean':
            return d / mu.shape[0]
        return d


class PhaseTransitionsPotential(nn.Module):
    """
    Refer to: Phase Transitions, Distance Functions, and Implicit Neural Representations
    """
    def __init__(self, reduction='mean'):
        super(PhaseTransitionsPotential, self).__init__()
        self.reduction = reduction

    def forward(self, x):
        assert torch.all(x >= 0) and torch.all(x <= 1)
        s = 2 * x - 1
        l = s ** 2 - 2 * torch.abs(s) +1
        if self.reduction == 'mean':
            return torch.mean(l)
        return l


class TotalVariationLoss(nn.Module):
    """
    https://discuss.pytorch.org/t/implement-total-variation-loss-in-pytorch/55574
    """
    def __init__(self, scale_factor=None):
        super(TotalVariationLoss, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        if self.scale_factor is not None:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')

        assert len(x.shape) == 4
        tv_h = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
        l = (tv_h+tv_w) / np.prod(x.shape)
        return l


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

