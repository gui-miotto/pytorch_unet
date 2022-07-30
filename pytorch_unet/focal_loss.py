from typing import Optional

import torch
from torch.nn import functional as F


class FocalLoss(torch.nn.Module):
    def __init__(self,
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.,
                ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return FunctionalFocalLoss(
            input=input,
            target=target,
            gamma=self.gamma,
            alpha=self.alpha,
        )

def FunctionalFocalLoss(
    input: torch.Tensor,
    target: torch.Tensor,
    gamma: Optional[float] = 2.,
    alpha: Optional[torch.Tensor] = None,
    ):
    # input size: (batch_size, n_classes, d1, d2, ..., dn) = (b, c, *d)
    # target size: (b, *d)

    # For each pixel, get the predicted likelihood corresponding to the true class
    log_p = F.log_softmax(input=input, dim=1)  # size = (b, c, *d)
    log_pt = torch.gather(input=log_p, dim=1, index=target.unsqueeze(1))  # size = (b, 1, *d)
    pt = log_pt.squeeze().exp()  # (batch_size, *d)

    # Compute the cross-entropy weighted by the alphas: -alpha * log(pt)
    ce = F.nll_loss(input=log_p, target=target, weight=alpha, reduction="none")  # size = (b, *d)

    # compute focal term: (1. - pt) ^ gamma
    focal_term = (1. - pt) ** gamma  # size = (b, *d)

    # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
    loss = focal_term * ce  # size = (b, *d)
    loss = loss.mean()  # size = scalar
    return loss
