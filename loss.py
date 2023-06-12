import warnings
import torch
import torch.nn as nn


class Shrinkage_loss(nn.Module):
    def __init__(self, a, c):
        super(Shrinkage_loss, self).__init__()
        self._a = a
        self._c = c

    def forward(self, pred, gt):
        l1 = torch.mean((pred - gt))
        l2 = l1 ** 2
        shrinkage_loss = l2 / (1 + torch.exp(self._a * (self._c - l2)))
        return shrinkage_loss


class MSELoss_(nn.Module):
    def __init__(self, pred, gt) -> None:
        super().__init__(MSELoss_, self)
        self._pred = pred
        self._gt = gt

        if not (pred.size() == gt.size()):
            warnings.warn(
                "Using a target size ({}) that is different to the input size ({}). "
                "This will likely lead to incorrect results due to broadcasting. "
                "Please ensure they have the same size.".format(gt.size(), pred.size()),
                stacklevel=2,
            )
    
    def forward(self, pred, gt):
        loss = torch.mean(((pred - gt) ** 2))
        return loss