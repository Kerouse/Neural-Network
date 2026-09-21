import torch
from torch import nn

from .backbone import BackBone
from .head import Head


class LeNet5(nn.Module):
    """LeNet-5：BackBone（卷积）+ Head（全连接）。

    32×32 灰度图 -> 10 类 logits。
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = BackBone()
        self.head = Head(num_classes=num_classes)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return self.head(self.backbone(x))

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            x = x.unsqueeze(1)
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        digit = torch.argmax(probs, dim=1)
        one_hot = nn.functional.one_hot(digit, num_classes=logits.size(1)).float()
        return {
            "digit": digit,
            "one_hot": one_hot,
            "probs": probs,
            "logits": logits,
        }
