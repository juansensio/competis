import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pprint import pprint
from typing import Callable, List, Union
from sklearn.metrics import top_k_accuracy_score
from torchmetrics.regression import CriticalSuccessIndex, ContinuousRankedProbabilityScore
from torchmetrics.classification import F1Score, MulticlassF1Score

from src.dataloader.dataset_stats import Y_REG_NORM_BINS


@torch.no_grad()
def mean_csi(logits:torch.Tensor, target:torch.Tensor, reduce="mean") -> float:
    """
    args
    ---
    :logits: unnormalized model class preds
    :target: one-hot categorical label with identical shape to `logits`
    """

    # [B, N]
    logits = logits.detach().clone()
    target = target.detach().clone()
    
    max_i        = torch.argmax(logits, dim=1, keepdim=True)
    pred         = torch.zeros(logits.shape).to(target.device)
    pred         = pred.scatter(1, max_i, 1.0)

    if reduce == "none":
        from sklearn.metrics import jaccard_score
        
        # logits/labels: torch.Tensor
        # logits: [B, C]
        # * raw model outputs for C classes
        # labels: [B, C]
        # * one-hot labels

        # 1. Convert raw logits to a binary mask of the top-k highest scores
        K = 5
        preds        = torch.zeros_like(logits).scatter_(1, logits.topk(K).indices, 1.0)
        intersection = (preds * target).sum(dim=1)
        union        = (preds + target).clamp(max=1.0).sum(dim=1)
        return (intersection / (union + 1e-7))

        # csi = CriticalSuccessIndex(0.5, keep_sequence_dim=0).to(target.device)
        # return csi(pred, target)
    else:
        csi = CriticalSuccessIndex(0.5).to(target.device)

    return csi(pred, target).item()
    

@torch.no_grad()
def mean_f1(logits:torch.Tensor, target:torch.Tensor, reduce="mean") -> float:
    """
    args
    ---
    :logits: unnormalized model class preds
    :target: one-hot categorical label with identical shape to `logits`
    """

    logits = logits.detach().clone()
    target = target.detach().clone()

    max_i        = torch.argmax(logits, dim=1, keepdim=True)
   
    pred         = torch.zeros(logits.shape).to(target.device)
    pred         = pred.scatter(1, max_i, 1.0)

    if reduce == "none":
        
        # # HACK: return top@5 scores
        # _target = target.argmax(dim=-1) # convert one-hot vectors -> top1 labels
        # _logits = logits

        # # arange cls label idxs
        # labels=np.arange(0, logits.shape[-1])
        
        # out = None

        # for B in range(_logits.shape[0]):

        #     score = top_k_accuracy_score(_target[B, ...].unsqueeze(0).cpu().numpy(), _logits[B, ...].unsqueeze(0).cpu().numpy(), k=5, labels=labels)
        #     if out == None: out = torch.tensor([score])
        #     else: out = torch.cat([out, torch.tensor([score])])

        # Compare top 5 prediction indices against the true label index
        out = (logits.topk(3, dim=-1).indices == target.argmax(dim=-1, keepdim=True)).any(dim=-1).float()
        
        return out
    else:
        f1_metric    = MulticlassF1Score(num_classes=logits.shape[-1]).to(target.device)
    
    return f1_metric(pred, target).item()


def mean_crps(logits:torch.Tensor, target:torch.Tensor, reduce="mean") -> float:
    """
    args
    ---
    :logits: unnormalized model class preds
    - [B, N]
    :target: one-hot categorical label with identical shape to `logits`
    - [B, N]
    """

    assert logits.shape == target.shape, f"Shape of logits: {logits.shape} must match target shape: {target.shape}"

    logits = logits.detach().clone()
    target = target.detach().clone()
    
    B, N   = logits.shape

    label  = torch.zeros(logits.shape).to(logits.device)
    max_i  = torch.argmax(target, dim=1)

    # TODO: replace with something efficient/vectorized
    for b, idx in enumerate(max_i): label[b, idx] = 1

    # ground truth degenerate target
    label_cdf = label.cumsum(dim=1)

    # predict cdf from logits
    pred_cdf  = F.softmax(logits, dim=1).cumsum(dim=1)

    mask      = (pred_cdf >= label_cdf) * 1
    _not_mask = (pred_cdf < label_cdf ) * 1

    crps = ((((pred_cdf - label_cdf) * mask) + ((label_cdf - pred_cdf) * _not_mask)) ** 2).sum(dim=1)

    if reduce == "mean":
        return crps.mean()
    elif reduce == "none":
        return crps
    else:
        raise Exception()


class BinnedEvalMetric(nn.Module):

    def __init__(self, f: Callable, bins: Union[torch.Tensor, None]=None):
        """
        args
        ---
        :bins: optional 1D array of class-coefficients to reweight metrics
        """
        
        super().__init__()

        if bins != None:
            assert type(bins) is torch.Tensor, f"Error: bins must be a 1D tensor."
            assert len(bins.shape) == 1, f"Error: bins must be a 1D tensor."
            # assert (bins.sum().item() < 1.01 and bins.sum().item() > 0.99), \
            #     f"Error: bins must be a probability density over outputs;   \
            #     expected bins.sum() == 1.0, got: {bins.sum().item()}"

        self.f    = f
        self.bins = bins

    @torch.no_grad()
    def forward(self, logits:torch.Tensor, target:torch.Tensor) -> float:
        """
        args
        ---
        :logits: unnormalized model class preds
        :target: one-hot categorical label with identical shape to `logits`
        """

        self.bins = self.bins.to(logits.device)
        
        assert self.bins.shape[0] == logits.shape[-1] and self.bins.shape[0] == target.shape[-1], \
            f"Error: expected len self.bins to equal len logits/target.shape[-1]"
        
        unnorm: float = self.f(logits, target, reduce="none")
        
        # 1. get class indices
        max_i         = torch.argmax(logits, dim=1, keepdim=True).to(logits.device)
        scale         = self.bins[max_i].squeeze().to(logits.device)
        
        # [B] * [B]
        return  (scale * unnorm).mean().item()
    

class Evaluator(nn.Module):
    """
    A reusable evalutor used to wrap models and track bin-normalized statistics.
    * Forward pass calculates per-bin  metrics/stores as an instance variable.
    """

    def __init__(
            self, 
            model: nn.Module,
        ):

        super().__init__()
        self.model = model
        self.crps  = {}
        self.csi   = {}
        self.f1    = {}

    def __repr__(self):

        mcrps = torch.tensor([v.mean() for (k, v) in self.crps.items()]).mean()
        mcsi  = torch.tensor([v.mean() for (k, v) in self.csi.items()]).mean()
        mtop3 = torch.tensor([v.mean() for (k, v) in self.f1.items()]).mean()

        # NOTE: this is actually top@5
        # all_preds = None
        # for (k, v) in self.f1.items(): all_preds = v if all_preds is None else torch.cat([all_preds, v])
        # mf1   = torch.tensor([v.mean() for (k, v) in self.f1.items()]).mean()
        # acc = all_preds.sum() / len(all_preds)

        pprint(f"CRPS: {self.crps}\nCSI: {self.csi}\nmTop@3: {self.f1}\n")
        pprint("----------------------------------------------------")
        print(f"mCRPS: {mcrps}\nmCSI: {mcsi}\nmTop@3: {mtop3}\n")

        return ""

    @torch.no_grad()
    def forward(self, X: torch.Tensor, y: torch.Tensor) -> None:
        
        y_hat     = self.model(X)

        pred_idxs = torch.argmax(y_hat, dim=-1)
        labels    = torch.argmax(y, dim=-1)

        csi       = mean_csi( y_hat, y, reduce="none")
        crps      = mean_crps(y_hat, y, reduce="none")
        f1        = mean_f1(  y_hat, y, reduce="none")

        metrics = [
        mean_crps(y_hat, y, reduce="none"),
        mean_csi( y_hat, y, reduce="none"),
        mean_f1(  y_hat, y, reduce="none")
        ]

        # Iterate only over unique classes present (O(Classes) instead of O(Batch))
        for label in labels.unique():
            mask = (labels == label)
            l_idx = label.item()

            for d, metric in zip([self.crps, self.csi, self.f1], metrics):
                vals = metric[mask].cpu() # Extract batch subset
                d[l_idx] = torch.cat([d[l_idx], vals]) if l_idx in d else vals  
        
        # for d, metric in zip([self.crps, self.csi, self.f1], [crps, csi, f1]):

        #     for i, l in enumerate([t.item() for t in labels]):
            
        #         if l not in d: d[l] = metric[i].cpu().unsqueeze(0)
        #         else:
        #             d[l] = torch.cat([d[l], metric[i].cpu().unsqueeze(0)])
                
                # try:

                #     if l not in d: d[l] = metric[i].cpu().unsqueeze(0)
                #     else:
                #         d[l] = torch.cat([d[l], metric[i].cpu().unsqueeze(0)])

                # except:

                #     print(f"Error processing val sample")


if __name__ == "__main__":

    # logits = torch.rand(10, 10)
    # labels = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 10)    
    # print(mean_crps(logits, labels, reduce="none"))
    # print(mean_f1(logits, labels, reduce="none"))
    # print(mean_csi(logits, labels, reduce="none"))
    # breakpoint()
    
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from src.model.SaTformer.SaTformer import SaTformer
    from src.dataloader.challenge_one_dataloader import Sat2RadDataset

    import os
    from torch.distributed import init_process_group, destroy_process_group
    import random

    torch.cuda.set_device(0)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(random.randint(10000, 20000))
    torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
    
    BASELINE_FP = "/playpen-ssd/levi/w4c/w4c-25/__exps__/2025-11-12_22-24-31_SaTformer-cat-loss-CCE-lr=1e-5-BS=128-N=64-ATTN=ST^2-baseline/best.pth"
    ATTN_ST2_FP = "/playpen-ssd/levi/w4c/w4c-25/__exps__/2025-11-12_22-20-32_SaTformer-cat-loss-weightedCCE-lr=1e-5-BS=128-N=64-ATTN=ST^2/best.pth"
    ATTN_S2T_FP = "/playpen-ssd/levi/w4c/w4c-25/__exps__/2025-11-12_22-02-09_SaTformer-cat-loss-weightedCCE-lr=1e-5-BS=128-N=64-ATTN=S->T/best.pth"
    ATTN_T2S_FP = "/playpen-ssd/levi/w4c/w4c-25/__exps__/2025-11-12_22-00-53_SaTformer-cat-loss-weightedCCE-lr=1e-5-BS=128-N=64-ATTN=T->S/best.pth"

    ds = Sat2RadDataset(split="val", n_classes=64)
    dl = DataLoader(ds, batch_size=128, num_workers=16)

    import random
    random.seed = 42
    ds.steps_per_epoch = 500
    
    for fp, exp_name in zip([BASELINE_FP, ATTN_ST2_FP, ATTN_T2S_FP, ATTN_ST2_FP], ["baseline", "S->T", "T->S", "ST**2"]):

        
        wrapper = torch.load(fp, map_location='cpu', weights_only=False)
        model   = wrapper.module.cpu().eval()
        e       = Evaluator(model).cuda()
        
        with torch.no_grad():
            for item in tqdm(dl):
                X, y = item["X_norm"].cuda(), item["y_reg_norm_oh"].cuda()
                e(X, y)

        print(e)
        print(exp_name)

        # breakpoint()