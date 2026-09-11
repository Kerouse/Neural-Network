from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """所有网络的共同接口：call 对外，forward 做计算。

    激活写在基类上，全连接 / 卷积 / 循环都用 self.sigmoid 等，不必另开模板文件。
    """

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def softmax(z, axis=1):
        z = z - np.max(z, axis=axis, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=axis, keepdims=True)

    @staticmethod
    def relu(z):
        return np.maximum(0.0, z)

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    def __call__(self, x):
        return self.call(x)

    def prepare_input(self, x):
        return np.asarray(x, dtype=np.float64)

    def call(self, x):
        return self.forward(self.prepare_input(x))

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError


class FullyConnectedNN(BaseModel):
    """全连接神经网络（Fully Connected Neural Network, FCNN）前向传播。

    结构：
        输入 1024（32×32 单通道图像展平）
        -> 隐层1 256 + Sigmoid
        -> 隐层2 128 + Sigmoid
        -> 隐层3 64  + Sigmoid
        -> 输出 10  + Softmax
    """

    def __init__(self, layer_sizes=(1024, 256, 128, 64, 10), seed=42):
        self.layer_sizes = list(layer_sizes)
        self.rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        self._init_params()

    def _init_params(self):
        """按层随机初始化 W、b。权重用输入维度缩放，减轻 Sigmoid 饱和。"""
        for in_dim, out_dim in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            scale = np.sqrt(1.0 / in_dim)
            W = self.rng.normal(0.0, scale, size=(in_dim, out_dim))
            b = np.zeros((1, out_dim))
            self.weights.append(W)
            self.biases.append(b)

    def prepare_input(self, x):
        """把 32×32 图或特征向量整理成 (N, 1024)。"""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        elif x.ndim == 2 and x.shape == (32, 32):
            x = x.reshape(1, -1)
        elif x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        expected = self.layer_sizes[0]
        if x.shape[-1] != expected:
            raise ValueError(
                f"输入最后一维应为 {expected}，实际为 {x.shape[-1]}"
            )
        return x

    def forward(self, x):
        """逐层前向：z = aW + b，隐层 Sigmoid，输出 Softmax。"""
        a = np.asarray(x, dtype=np.float64)
        if a.ndim == 1:
            a = a.reshape(1, -1)

        acts = [a]
        last = len(self.weights) - 1
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            a = self.softmax(z) if i == last else self.sigmoid(z)
            acts.append(a)
        return a, acts

    def call(self, x):
        """对外接口 f(x)：图片 -> 预测数字、one-hot、概率。"""
        x = self.prepare_input(x)
        probs, acts = self.forward(x)
        n_class = self.layer_sizes[-1]
        digit = np.argmax(probs, axis=1)
        one_hot = np.eye(n_class, dtype=np.float64)[digit]
        return {
            "digit": digit,
            "one_hot": one_hot,
            "probs": probs,
            "activations": acts,
        }
