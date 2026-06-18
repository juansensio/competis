import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Optional, Sequence, List
from scipy.interpolate import PchipInterpolator
from src.util.kde_helpers import Y_REG_MAG_BINS, weighted_gaussian_kde
from src.dataloader.dataset_stats import Y_REG_FREQ_BINS
from src.dataloader.challenge_one_dataloader import get_y_reg_bin_counts_step


class MSE(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x:torch.Tensor, y:torch.Tensor):
        return F.mse_loss(x, y)
    

class L1(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x:torch.Tensor, y:torch.Tensor):
        return F.l1_loss(x, y)
    

class FreqWeightedL1(nn.Module):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        # emperical cmf
        self.cmf = torch.tensor(Y_REG_FREQ_BINS) / torch.tensor(Y_REG_FREQ_BINS).sum()
        
        # sum(cmf) ~= 1.0
        assert (self.cmf.sum() < 1.01) and (self.cmf.sum() > 0.99)
        
        self.bins = torch.tensor(Y_REG_MAG_BINS)

        self.eps  = 1 / torch.tensor(Y_REG_FREQ_BINS).sum().item()

        # HACK: hyperparameter
        self.bandwidth  = 0.05

    def forward(self, preds:torch.Tensor, target:torch.Tensor) -> torch.Tensor:

        # [B, 1]; approximate density of target labels
        pdf_values   = weighted_gaussian_kde(target, self.bins.to(preds.device), self.cmf.to(preds.device), self.bandwidth).sum(dim=1, keepdim=True) + self.eps
        loss_weights = 1 / pdf_values
        errors       = abs(preds - target)
        w_errors     = errors * loss_weights
        
        return w_errors.mean()
    

class MagWeightedL1(nn.Module):
    """
    Simple l1 variant where we weight by magnitude y
    """

    def __init__(self, alpha=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha=1.0

    def forward(self, preds:torch.Tensor, target:torch.Tensor):
        
        # simply weight by the approx max of the target 
        return F.l1_loss(preds, target) * (self.alpha * torch.logsumexp(target, (0, 1)))
    

class CategoricalCrossEntropy(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, logits:torch.Tensor, target:torch.Tensor):
        pred = F.softmax(logits, dim=1)
        return F.cross_entropy(pred, target)


class ClassWeightedCategoricalCrossEntropy(nn.Module):

    def __init__(self, n_classes: int=128, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        freqs        = ClassWeightedCategoricalCrossEntropy.calculate_freqs(n_classes)
        freqs       += (1) # no zero weights

        # HACK: log weights
        rel_freqs    =  -torch.log((freqs / freqs.sum()))
        # rel_freqs    = (rel_freqs - rel_freqs.min()) / (rel_freqs.max() - rel_freqs.min())
        # rel_freqs    = rel_freqs.clip(.1, 1.)
        
        self.weights = rel_freqs

    @staticmethod
    def calculate_freqs(n_classes):
        _, _, freqs = get_y_reg_bin_counts_step(n_classes)
        return freqs

    def forward(self, logits:torch.Tensor, target:torch.Tensor):

        self.weights = self.weights.to(target.device)
        
        # pred = F.softmax(logits, dim=1)
        return F.cross_entropy(logits, target, weight=self.weights, reduction="sum")


if __name__ == "__main__":

    loss = ClassWeightedCategoricalCrossEntropy(n_classes=256)
    y     = torch.rand(2, 256)
    y_hat = torch.rand(2, 265)
    loss(y_hat, y)
    breakpoint()