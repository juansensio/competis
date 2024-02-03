import torch
import torch.nn as nn
import torch.nn.functional as F


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, logits, labels):
        """
        Binary Lovasz hinge loss
        logits: [B, 1, H, W] Variable, logits at each pixel (before Softmax).
        labels: [B, 1, H, W] Tensor, binary ground truth masks (0 or 1).
        """
        # logits = logits.unsqueeze(1)  # [B, 1, H, W]
        # labels = labels.unsqueeze(1)  # [B, 1, H, W]
        logits_flat = logits.view(-1)
        labels_flat = labels.view(-1)
        errors = (logits_flat - labels_flat).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        gt_sorted = labels_flat[perm]
        grad = lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovász extension w.r.t sorted errors
    See Alg. 1 in the Lovász-Softmax paper.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 0/0 division
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class ModifiedLovaszLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(ModifiedLovaszLoss, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        # Flatten the logits and labels
        logits = logits.view(-1)
        labels = labels.view(-1)

        # Calculate the surrogate for the Dice coefficient
        dice_surrogate = self.dice_surrogate(logits, labels)

        # Symmetric Lovasz-Softmax loss
        loss = (
            self.lovasz_hinge(logits, labels) + self.lovasz_hinge(-logits, 1 - labels)
        ) / 2

        return loss

    def lovasz_hinge(self, logits, labels):
        errors = 1 + F.elu(logits * (2 * labels - 1), self.alpha)
        errors_sorted, indices = torch.sort(errors, descending=True)
        labels_sorted = labels[indices.data]
        dice = self.dice_surrogate(logits, labels)
        grad = dice.derivative()
        # losses = grad[indices] * errors_sorted
        losses = grad * errors_sorted
        return losses.mean()

    class dice_surrogate:
        def __init__(self, logits, labels):
            self.tp = (logits * labels).sum()
            self.fp = (logits * (1 - labels)).sum()
            self.fn = ((1 - logits) * labels).sum()
            self.numerator = 2 * self.tp
            self.denominator = 2 * self.tp + self.fp + self.fn

        def __call__(self):
            return self.numerator / (self.denominator + 1e-6)

        def derivative(self):
            # Derivative of the surrogate with respect to true positive rate
            d_numerator = 2
            d_denominator = 2 + (self.fp - self.fn) / (self.tp + 1e-6)
            return (
                d_numerator / (self.denominator + 1e-6)
                - (self.numerator / (self.denominator + 1e-6) ** 2) * d_denominator
            )
