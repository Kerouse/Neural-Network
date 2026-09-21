"""独立评估脚本：加载 best.pt，在测试集上算 loss / acc，并画混淆矩阵。"""

import torch

from train import OUTPUT_DIR, build_trainer, inspect_weights


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = build_trainer(device)
    best_path = OUTPUT_DIR / "best.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"找不到 {best_path}，请先运行 train.py")

    inspect_weights(best_path)
    trainer.save_confusion_matrix(best_path)
    test_loss, test_acc = trainer.evaluate()
    print(f"测试集  loss={test_loss:.4f}  acc={test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
