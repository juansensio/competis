import torch
import torch.nn.functional as F

def bce(pr, gt):
    #input (torch.Tensor): input data tensor with shape :math:`(B, *)`.
    #target (torch.Tensor): the target tensor with shape :math:`(B, *)`.
    return F.binary_cross_entropy_with_logits(pr, gt)