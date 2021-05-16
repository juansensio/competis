import torch
import torch.nn.functional as F

def focal(pr, gt, alpha = .25, gamma = 2.0, eps=1e-8):
    bce = F.binary_cross_entropy_with_logits(pr, gt, reduction='none')
    pt = torch.exp(- bce)
    loss = alpha*(1. - pt)**gamma*bce
    return torch.mean(loss)