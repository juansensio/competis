from collections.abc import Mapping


def deep_update(source, overrides):
    for key, value in overrides.items():
        if isinstance(value, Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source

import torch
import torch.nn.functional as F

def log_cosh_dice(pr, gt, eps=1.): # no se si está bien :(
    pr = pr.view(pr.shape[0], 1, -1)
    gt = gt.view(gt.shape[0], 1, -1)
    pr = torch.sigmoid(pr)
    tp = torch.sum(gt * pr, axis=(-2,-1))
    fp = torch.sum(pr, axis=(-2,-1)) - tp
    fn = torch.sum(gt, axis=(-2,-1)) - tp
    loss = torch.log(torch.cosh(1. - (2.*tp + eps) / (2.*tp + fn + fp + eps)))
    return torch.mean(loss)

def my_dice(pr, gt, eps=1.): # no se si está bien :(
    pr = pr.view(pr.shape[0], 1, -1)
    gt = gt.view(gt.shape[0], 1, -1)
    pr = torch.sigmoid(pr)
    tp = torch.sum(gt * pr, axis=(-2,-1))
    fp = torch.sum(pr, axis=(-2,-1)) - tp
    fn = torch.sum(gt, axis=(-2,-1)) - tp
    loss = 1. - (2.*tp + eps) / (2.*tp + fn + fp + eps)
    return torch.mean(loss)