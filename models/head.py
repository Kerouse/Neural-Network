from torch import nn


class Head(nn.Module):
    """LeNet-5 全连接分类头。

    输入 16@5×5 展平为 400
        C5  Linear 400 -> 120
        F6  Linear 120 -> 84
        OUT Linear  84 -> 10  （logits）
    """

    def __init__(self, in_channels=16, feat_h=5, feat_w=5, num_classes=10):
        super().__init__()
        in_features = in_channels * feat_h * feat_w
        self.fc1 = nn.Linear(in_features, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.act = nn.Tanh()

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)
