from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class DigitImageDataset(Dataset):
    """自定义手写数字数据集。

    从目录中读取 PNG，用文件名第一段作为标签：`{digit}_{index}.png`。
    处理流程：灰度 -> 缩放到 32×32 -> [0, 1] 归一化 -> 展平为 1024 维。
    """

    def __init__(self, image_dir, image_size=32):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"数据目录不存在: {self.image_dir}")

        self.samples = []
        for path in sorted(self.image_dir.glob("*.png")):
            label_text = path.stem.split("_")[0]
            if not label_text.isdigit():
                continue
            self.samples.append((path, int(label_text)))

        if not self.samples:
            raise FileNotFoundError(f"目录中没有可用 PNG: {self.image_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as img:
            img = img.convert("L")
            if img.size != (self.image_size, self.image_size):
                img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            pixels = np.asarray(img, dtype=np.float32) / 255.0

        x = torch.from_numpy(pixels.reshape(-1))
        y = torch.tensor(label, dtype=torch.long)
        return x, y
