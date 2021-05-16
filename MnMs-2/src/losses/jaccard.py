import torch
import torch.nn.functional as F

def jaccard(pr, gt, eps=1.):
    pr = torch.sigmoid(pr)
    intersection = torch.sum(gt * pr, axis=(-2,-1))
    union = torch.sum(gt, axis=(-2,-1)) + torch.sum(pr, axis=(-2,-1)) - intersection + eps
    iou = (intersection + eps) / union
    return torch.mean(1. - iou)
