from pathlib import Path

import torch

from models import FullyConnectedNN
from train import OUTPUT_DIR, inspect_weights

WEIGHT_PATH = OUTPUT_DIR / "best.pth"


def main():
    if not WEIGHT_PATH.is_file():
        raise FileNotFoundError(f"找不到权重: {WEIGHT_PATH}，请先运行 train.py")

    # 1. 用 torch.load 打开二进制权重，看成「层名 -> 张量」字典
    state = inspect_weights(WEIGHT_PATH)

    # 2. 装回网络，确认能对上结构
    model = FullyConnectedNN()
    model.load_state_dict(state)
    model.eval()
    print()
    print("已成功 load_state_dict，网络可以用来预测。")
    print("示例：随便送一张全 0 图，看 10 类概率。")
    dummy = torch.zeros(1, 1024)
    with torch.no_grad():
        probs = torch.softmax(model(dummy), dim=1)[0]
    for digit, p in enumerate(probs.tolist()):
        print(f"  数字 {digit}: {p:.4f}")


if __name__ == "__main__":
    main()
