import torch

from models import LeNet5
from train import OUTPUT_DIR, inspect_weights

WEIGHT_PATH = OUTPUT_DIR / "best.pt"


def main():
    if not WEIGHT_PATH.is_file():
        raise FileNotFoundError(f"找不到权重: {WEIGHT_PATH}，请先运行 train.py")

    state = inspect_weights(WEIGHT_PATH)

    model = LeNet5()
    model.load_state_dict(state)
    model.eval()
    print()
    print("已成功 load_state_dict，网络可以用来预测。")
    print("示例：全 0 的 1×32×32 图，看 10 类概率。")
    dummy = torch.zeros(1, 1, 32, 32)
    with torch.no_grad():
        probs = torch.softmax(model(dummy), dim=1)[0]
    for digit, p in enumerate(probs.tolist()):
        print(f"  数字 {digit}: {p:.4f}")


if __name__ == "__main__":
    main()
