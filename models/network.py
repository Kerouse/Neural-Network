import torch
from torch import nn


class FullyConnectedNN(nn.Module):
    """自定义全连接网络（Fully Connected Neural Network, FCNN）。

    结构与课件一致：
        输入 1024（32×32 单通道图像展平）
        -> 隐层1 256 + Sigmoid
        -> 隐层2 128 + Sigmoid
        -> 隐层3 64  + Sigmoid
        -> 输出 10 个类别 logits（损失里再做 Softmax）
    """

    def __init__(self, layer_sizes=(1024, 256, 128, 64, 10)):
        super().__init__()
        self.layer_sizes = list(layer_sizes)
        modules = []
        last = len(self.layer_sizes) - 2
        for i, (in_dim, out_dim) in enumerate(
            zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        ):
            modules.append(nn.Linear(in_dim, out_dim))
            if i < last:
                modules.append(nn.Sigmoid())
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        expected = self.layer_sizes[0]
        if x.size(-1) != expected:
            raise ValueError(f"输入最后一维应为 {expected}，实际为 {x.size(-1)}")
        return self.net(x)

    @torch.no_grad()
    def predict(self, x):
        """图片或特征 -> 预测数字、one-hot、概率。"""
        self.eval()
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim == 2 and tuple(x.shape) == (32, 32):
            x = x.reshape(1, -1)
        elif x.ndim == 3:
            x = x.reshape(x.size(0), -1)
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        digit = torch.argmax(probs, dim=1)
        n_class = self.layer_sizes[-1]
        one_hot = torch.nn.functional.one_hot(digit, num_classes=n_class).float()
        return {
            "digit": digit,
            "one_hot": one_hot,
            "probs": probs,
            "logits": logits,
        }
