from torch import nn


class BackBone(nn.Module):
    """LeNet-5 卷积骨干：两层卷积 + 两次下采样。

    输入 1@32×32
        Conv-1  5×5, stride=1, pad=0, 6 核  ->  6@28×28
        Pool-1  2×2, stride=2, pad=0        ->  6@14×14
        Conv-2  5×5, stride=1, pad=0, 16 核 -> 16@10×10
        Pool-2  2×2, stride=2, pad=0        -> 16@5×5
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.pool1(x)
        x = self.act(self.conv2(x))
        x = self.pool2(x)
        return x
