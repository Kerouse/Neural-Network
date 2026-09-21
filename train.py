from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import DigitImageDataset
from losses import CrossEntropyLoss
from models import LeNet5

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


def per_class_precision_recall(cm):
    """由混淆矩阵计算每个数字的精确率、召回率、准确率、F1。

    精确率 = TP / (TP+FP)
    召回率 = TP / (TP+FN)
    准确率 = (TP+TN) / N，TN 为「不是 k 且没被预测成 k」
    分母为 0 时记为 0。
    """
    cm_f = cm.to(dtype=torch.float64)
    n_total = cm_f.sum()
    tp = torch.diag(cm_f)
    pred_as_k = cm_f.sum(dim=0)
    true_k = cm_f.sum(dim=1)
    fp = pred_as_k - tp
    fn = true_k - tp
    tn = n_total - tp - fp - fn
    precision = torch.where(pred_as_k > 0, tp / pred_as_k, torch.zeros_like(tp))
    recall = torch.where(true_k > 0, tp / true_k, torch.zeros_like(tp))
    accuracy = torch.where(n_total > 0, (tp + tn) / n_total, torch.zeros_like(tp))
    f1_den = precision + recall
    f1 = torch.where(f1_den > 0, 2 * precision * recall / f1_den, torch.zeros_like(tp))
    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
        "support": true_k,
        "predicted": pred_as_k,
        "tp": tp,
    }


def print_precision_recall(scores):
    """打印每个数字及宏平均的精确率 / 召回率 / 准确率 / F1。"""
    p = scores["precision"]
    r = scores["recall"]
    a = scores["accuracy"]
    f1 = scores["f1"]
    support = scores["support"]
    print("=" * 64)
    print("每个数字的精确率 / 召回率 / 准确率 / F1")
    print("=" * 64)
    print(
        f"{'数字':>4}  {'精确率':>8}  {'召回率':>8}  {'准确率':>8}  "
        f"{'F1':>8}  {'真实张数':>8}"
    )
    for k in range(N_CLASS):
        print(
            f"{k:>4}  {p[k] * 100:7.2f}%  {r[k] * 100:7.2f}%  "
            f"{a[k] * 100:7.2f}%  {f1[k] * 100:7.2f}%  {int(support[k]):>8}"
        )
    print(
        f"{'宏平均':>4}  {p.mean() * 100:7.2f}%  {r.mean() * 100:7.2f}%  "
        f"{a.mean() * 100:7.2f}%  {f1.mean() * 100:7.2f}%"
    )


def plot_precision_recall(scores, save_path):
    """精确率、召回率、准确率各一张梯状图，上下合并成一张。

    每个数字占一格台阶，黑边勾出左右边界，避免台阶连在一起看不清。
    """
    _use_chinese_font()
    digits = np.arange(N_CLASS)
    edges = np.arange(N_CLASS + 1) - 0.5
    panels = [
        (scores["precision"].numpy() * 100, "精确率梯状图", "#1f77b4"),
        (scores["recall"].numpy() * 100, "召回率梯状图", "#ff7f0e"),
        (scores["accuracy"].numpy() * 100, "准确率梯状图", "#2ca02c"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for ax, (values, title, color) in zip(axes, panels):
        ax.stairs(
            values,
            edges,
            fill=True,
            color=color,
            alpha=0.38,
            baseline=0,
        )
        ax.stairs(
            values,
            edges,
            fill=False,
            color="#222222",
            linewidth=1.8,
            baseline=0,
        )
        for boundary in edges:
            ax.axvline(boundary, color="#444444", linewidth=0.9, alpha=0.85, zorder=3)
        ax.plot(digits, values, "o", color=color, markersize=7, zorder=4, markeredgecolor="#222")
        for x, y in zip(digits, values):
            ax.annotate(
                f"{y:.1f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
            )
        ax.set_xlim(-0.5, N_CLASS - 0.5)
        ymin = max(0.0, float(values.min()) - 8.0)
        ax.set_ylim(ymin, 108)
        ax.set_ylabel("百分比（%）")
        ax.set_title(title)
        ax.set_xticks(digits)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    axes[-1].set_xlabel("数字")
    fig.suptitle("每个数字的精确率 / 召回率 / 准确率（梯状图）", fontsize=14)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def save_precision_recall_table(scores, save_path):
    """把每类指标写成 UTF-8 文本。"""
    p = scores["precision"]
    r = scores["recall"]
    f1 = scores["f1"]
    support = scores["support"]
    lines = [
        "digit\tprecision\trecall\taccuracy\tf1\tsupport",
    ]
    a = scores["accuracy"]
    for k in range(N_CLASS):
        lines.append(
            f"{k}\t{p[k].item():.6f}\t{r[k].item():.6f}\t"
            f"{a[k].item():.6f}\t{f1[k].item():.6f}\t{int(support[k])}"
        )
    lines.append(
        f"macro\t{p.mean().item():.6f}\t{r.mean().item():.6f}\t"
        f"{a.mean().item():.6f}\t{f1.mean().item():.6f}\t"
    )
    save_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return save_path


def _use_chinese_font():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_train_curves(history, save_path):
    """准确率曲线和损失下降曲线画在同一张图（左右两栏）。"""
    _use_chinese_font()
    epochs = history["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    ax.plot(epochs, [a * 100 for a in history["train_acc"]], label="训练集", color="#1f77b4")
    ax.plot(epochs, [a * 100 for a in history["test_acc"]], label="测试集", color="#ff7f0e")
    ax.set_xlabel("迭代次数（epoch）")
    ax.set_ylabel("准确率（%）")
    ax.set_title("准确率随迭代次数变化")
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    ax = axes[1]
    ax.plot(epochs, history["train_loss"], label="训练集", color="#1f77b4")
    ax.plot(epochs, history["test_loss"], label="测试集", color="#ff7f0e")
    ax.set_xlabel("迭代次数（epoch）")
    ax.set_ylabel("损失（Cross Entropy）")
    ax.set_title("损失函数下降曲线")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_confusion_matrix(cm, save_path, acc):
    """画出并保存混淆矩阵图。"""
    cm_np = cm.numpy()
    _use_chinese_font()

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


def collect_errors(model, loader, device):
    """收集预测 ≠ 真实的样本，按预测标签 0–9 排序。"""
    model.eval()
    errors = []
    with torch.no_grad():
        for images, labels in loader:
            preds = torch.argmax(model(images.to(device)), dim=1).cpu()
            labels = labels.cpu()
            images = images.cpu()
            for img, true_y, pred_y in zip(images, labels, preds):
                if int(true_y) != int(pred_y):
                    errors.append(
                        (img.squeeze(0).numpy(), int(true_y), int(pred_y))
                    )
    errors.sort(key=lambda item: (item[2], item[1]))
    return errors


def _save_nine_grid(items, save_path, title):
    """一张图放 9 张误分类样本（3×3）。不足 9 张的格子留空。"""
    _use_chinese_font()
    fig, axes = plt.subplots(3, 3, figsize=(8.2, 8.6))
    for i, ax in enumerate(axes.flat):
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= len(items):
            ax.axis("off")
            continue
        img, true_y, pred_y = items[i]
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"预 {pred_y}  ←  真 {true_y}", color="#c0392b", fontsize=11)
        for spine in ax.spines.values():
            spine.set_color("#c0392b")
            spine.set_linewidth(1.4)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    return save_path


def save_misclassified_grids(errors, out_dir):
    """按预测值 0–9 罗列错分图片：整体分页每张 9 图，再按预测类别各存一份。"""
    out_dir = Path(out_dir)
    if out_dir.exists():
        for old in out_dir.glob("*.png"):
            old.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    per_page = 9
    if not errors:
        print("没有预测错误的样本。")
        return saved

    total_pages = (len(errors) + per_page - 1) // per_page
    for page, start in enumerate(range(0, len(errors), per_page), start=1):
        chunk = errors[start : start + per_page]
        path = out_dir / f"all_p{page:02d}.png"
        _save_nine_grid(
            chunk,
            path,
            f"预测错误（按预测值 0–9）  第 {page}/{total_pages} 页  共 {len(errors)} 张",
        )
        saved.append(path)

    by_pred = {d: [] for d in range(N_CLASS)}
    for item in errors:
        by_pred[item[2]].append(item)
    for pred_y in range(N_CLASS):
        group = by_pred[pred_y]
        if not group:
            continue
        pages = (len(group) + per_page - 1) // per_page
        for page, start in enumerate(range(0, len(group), per_page), start=1):
            suffix = f"_p{page}" if pages > 1 else ""
            path = out_dir / f"pred_{pred_y}{suffix}.png"
            _save_nine_grid(
                group[start : start + per_page],
                path,
                f"预测值 = {pred_y}  的错误样本  {len(group)} 张",
            )
            saved.append(path)

    return saved


def inspect_weights(weight_path):
    """打开 .pt 权重：打印每层参数名、形状、数值范围。"""
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    print("=" * 56)
    print(f"权重文件: {weight_path}")
    print(f"包含 {len(state)} 组参数")
    print("=" * 56)
    for name, tensor in state.items():
        arr = tensor.detach().float()
        print(
            f"  {name:32s}  shape={tuple(arr.shape)}  "
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

    def evaluate(self, loader=None):
        """在指定集上评估：返回平均损失、准确率。默认用测试集。"""
        loader = self.test_loader if loader is None else loader
        return self._run_epoch(loader, train=False)

    def fit(self, epochs):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        best_acc = -1.0
        best_path = OUTPUT_DIR / "best.pt"
        last_path = OUTPUT_DIR / "last.pt"

        print("=" * 56)
        print("手写数字识别训练（LeNet-5 / PyTorch CPU）")
        print("=" * 56)
        print(f"device     : {self.device}")
        print(f"train size : {len(self.train_loader.dataset)}")
        print(f"test size  : {len(self.test_loader.dataset)}")
        print(f"epochs     : {epochs}  (每 {self.log_every} 个 epoch 输出准确率)")
        print()

        history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
        }

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._run_epoch(self.train_loader, train=True)
            test_loss, test_acc = self._run_epoch(self.test_loader, train=False)

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)

            if epoch % self.log_every == 0 or epoch == epochs:
                print(
                    f"Epoch {epoch:3d}/{epochs}  "
                    f"train_loss={train_loss:.4f}  train_acc={train_acc * 100:6.2f}%  "
                    f"test_loss={test_loss:.4f}  test_acc={test_acc * 100:6.2f}%"
                )
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(self.model.state_dict(), best_path)
            torch.save(self.model.state_dict(), last_path)

        curve_path = OUTPUT_DIR / "train_curves.png"
        plot_train_curves(history, curve_path)

        print()
        print(f"最优测试准确率: {best_acc * 100:.2f}%")
        print(f"best.pt 已保存: {best_path}")
        print(f"last.pt 已保存: {last_path}")
        print(f"训练曲线已保存: {curve_path}")
        return best_path, last_path

    def save_confusion_matrix(self, weight_path=None):
        """加载最优权重，在测试集上画混淆矩阵。"""
        path = Path(weight_path) if weight_path else OUTPUT_DIR / "best.pt"
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        y_true, y_pred = collect_predictions(self.model, self.test_loader, self.device)
        acc = (y_true == y_pred).float().mean().item()
        cm = confusion_matrix(y_true, y_pred)
        fig_path = OUTPUT_DIR / "confusion_matrix.png"
        plot_confusion_matrix(cm, fig_path, acc)
        print(f"混淆矩阵已保存: {fig_path}")
        print(f"测试准确率    : {acc * 100:.2f}%")

        scores = per_class_precision_recall(cm)
        print_precision_recall(scores)
        pr_fig = plot_precision_recall(scores, OUTPUT_DIR / "precision_recall_accuracy.png")
        pr_txt = save_precision_recall_table(scores, OUTPUT_DIR / "precision_recall.txt")
        print(f"精确率/召回率图: {pr_fig}")
        print(f"精确率/召回率表: {pr_txt}")

        errors = collect_errors(self.model, self.test_loader, self.device)
        error_dir = OUTPUT_DIR / "misclassified"
        error_paths = save_misclassified_grids(errors, error_dir)
        print(f"错分样本 {len(errors)} 张，已按预测值 0–9 存入: {error_dir}")
        for p in error_paths:
            print(f"    {p.name}")
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
    model = LeNet5()
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
    best_path = OUTPUT_DIR / "best.pt"
    last_path = OUTPUT_DIR / "last.pt"

    if eval_only:
        if not best_path.is_file():
            raise FileNotFoundError(f"找不到权重文件: {best_path}，请先训练。")
        print("跳过训练，直接用已有权重评估。")
    else:
        best_path, last_path = trainer.fit(epochs=50)

    inspect_weights(best_path)
    if last_path.is_file():
        print(f"last.pt 路径  : {last_path}")

    trainer.save_confusion_matrix(best_path)
    test_loss, test_acc = trainer.evaluate()
    print(f"评估结果      test_loss={test_loss:.4f}  test_acc={test_acc * 100:.2f}%")


if __name__ == "__main__":
    import sys

    main(eval_only="--eval" in sys.argv)
