from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import DigitImageDataset
from losses import CrossEntropyLoss
from models import FullyConnectedNN

RAW_DATA_DIR = Path(r"C:\Users\kelouse\PycharmProjects\raw date\Neural Network")
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
N_CLASS = 10


def collect_predictions(model, loader, device):
    """在数据集上收集真实标签和预测标签。"""
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            y_true.append(labels.cpu())
            y_pred.append(torch.argmax(logits, dim=1).cpu())
    return torch.cat(y_true), torch.cat(y_pred)


def confusion_matrix(y_true, y_pred, n_class=N_CLASS):
    """手动统计混淆矩阵：行=真实类别，列=预测类别。"""
    cm = torch.zeros(n_class, n_class, dtype=torch.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm, save_path, acc):
    """画出并保存混淆矩阵图。"""
    cm_np = cm.numpy()
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_np, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(cm_np.shape[0])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)
    ax.set_xlabel("预测数字")
    ax.set_ylabel("真实数字")
    ax.set_title(f"测试集混淆矩阵  准确率 {acc * 100:.2f}%")

    thresh = cm_np.max() / 2.0 if cm_np.max() else 0
    for i in range(cm_np.shape[0]):
        for j in range(cm_np.shape[1]):
            ax.text(
                j,
                i,
                str(cm_np[i, j]),
                ha="center",
                va="center",
                color="white" if cm_np[i, j] > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def inspect_weights(weight_path):
    """打开 .pth 权重：打印每层参数名、形状、数值范围。"""
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    print("=" * 56)
    print(f"权重文件: {weight_path}")
    print(f"包含 {len(state)} 组参数")
    print("=" * 56)
    for name, tensor in state.items():
        arr = tensor.detach().float()
        print(
            f"  {name:20s}  shape={tuple(arr.shape)}  "
            f"min={arr.min().item():8.4f}  max={arr.max().item():8.4f}  "
            f"mean={arr.mean().item():8.4f}"
        )
    n_param = sum(v.numel() for v in state.values())
    print(f"参数总量: {n_param}")
    return state


class Trainer:
    """训练器：前向、反向、评估。每 10 个 epoch 打印训练集/测试集准确率。"""

    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        train_loader,
        test_loader,
        device,
        log_every=10,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.log_every = log_every

    def _run_epoch(self, loader, train):
        self.model.train(train)
        total_loss = 0.0
        correct = 0
        total = 0
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                if train:
                    self.optimizer.zero_grad()
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)
                if train:
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item() * labels.size(0)
                pred = torch.argmax(logits, dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        return total_loss / total, correct / total

    def fit(self, epochs):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        best_acc = -1.0
        best_path = OUTPUT_DIR / "best.pth"

        print("=" * 56)
        print("手写数字识别训练（PyTorch CPU）")
        print("=" * 56)
        print(f"device     : {self.device}")
        print(f"train size : {len(self.train_loader.dataset)}")
        print(f"test size  : {len(self.test_loader.dataset)}")
        print(f"epochs     : {epochs}  (每 {self.log_every} 个 epoch 输出准确率)")
        print()

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._run_epoch(self.train_loader, train=True)
            if epoch % self.log_every == 0 or epoch == epochs:
                test_loss, test_acc = self._run_epoch(self.test_loader, train=False)
                print(
                    f"Epoch {epoch:3d}/{epochs}  "
                    f"train_loss={train_loss:.4f}  train_acc={train_acc * 100:6.2f}%  "
                    f"test_loss={test_loss:.4f}  test_acc={test_acc * 100:6.2f}%"
                )
                if test_acc > best_acc:
                    best_acc = test_acc
                    torch.save(self.model.state_dict(), best_path)

        print()
        print(f"最优测试准确率: {best_acc * 100:.2f}%")
        print(f"权重已保存    : {best_path}")
        return best_path

    def save_confusion_matrix(self, weight_path=None):
        """加载最优权重，在测试集上画混淆矩阵。"""
        path = Path(weight_path) if weight_path else OUTPUT_DIR / "best.pth"
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        y_true, y_pred = collect_predictions(self.model, self.test_loader, self.device)
        acc = (y_true == y_pred).float().mean().item()
        cm = confusion_matrix(y_true, y_pred)
        fig_path = OUTPUT_DIR / "confusion_matrix.png"
        plot_confusion_matrix(cm, fig_path, acc)
        print(f"混淆矩阵已保存: {fig_path}")
        print(f"测试准确率    : {acc * 100:.2f}%")
        return fig_path, cm, acc


def build_loader(image_dir, batch_size, shuffle):
    dataset = DigitImageDataset(image_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def build_trainer(device):
    train_loader = build_loader(RAW_DATA_DIR / "training_img", batch_size=64, shuffle=True)
    test_loader = build_loader(RAW_DATA_DIR / "test_img", batch_size=64, shuffle=False)
    model = FullyConnectedNN()
    return Trainer(
        model=model,
        loss_fn=CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        log_every=10,
    )


def main(eval_only=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = build_trainer(device)
    best_path = OUTPUT_DIR / "best.pth"
    if eval_only:
        if not best_path.is_file():
            raise FileNotFoundError(f"找不到权重文件: {best_path}，请先训练。")
        print("跳过训练，直接用已有权重评估。")
    else:
        best_path = trainer.fit(epochs=50)

    inspect_weights(best_path)
    trainer.save_confusion_matrix(best_path)


if __name__ == "__main__":
    import sys

    main(eval_only="--eval" in sys.argv)
