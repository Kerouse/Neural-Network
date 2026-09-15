import torch
from torch import nn


class CrossEntropyLoss(nn.Module):
    """自定义交叉熵损失（Cross Entropy Loss）。

    对 logits 做稳定 Softmax，再取真实类别概率的负对数均值。
    不调用 nn.CrossEntropyLoss / F.cross_entropy。
    """

    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        shifted = logits - logits.max(dim=1, keepdim=True).values
        exp_z = torch.exp(shifted)
        probs = exp_z / exp_z.sum(dim=1, keepdim=True)
        batch = logits.size(0)
        picked = probs[torch.arange(batch, device=logits.device), targets]
        return -torch.log(picked.clamp_min(self.eps)).mean()
