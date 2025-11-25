from pathlib import Path
from source.cnn_retrieve_images import YOLODataset
from collections import Counter
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassConfusionMatrix
)

def validate_dataset(dataset, dataloader):
    batch = next(iter(dataloader))
    print("Batch images:", batch["images"].shape)
    print("Class names:", batch["class_names"])
    print("First sample path:", batch["paths"][0])
    print("Transform pipeline:", dataset.transform)
    sample = dataset[0]
    img_tensor = sample["image"]
    print("Image tensor shape:", img_tensor.shape)
    print("Pixel stats after normalization:")
    print("Min:", img_tensor.min().item(), "Max:", img_tensor.max().item())


def label_histogram(dataset):
    hist = Counter(int(dataset[i]["label"]) for i in range(len(dataset)))
    names = dataset.subset_class_names
    print("Per-class counts:")
    for k, name in enumerate(names):
        print(f"{k:2d} {name:>15}: {hist.get(k, 0)}")
    return hist




def evaluate_classification(
    model, dataloader, num_classes, figures_path, timestamp,
    device='cpu', class_names=None, save_figure=True, show_plot=False
):
    print("[evaluate_classification] starting...", flush=True)

    model.eval()
    model.to(device)

    f1_metric        = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    precision_metric = MulticlassPrecision(num_classes=num_classes, average=None).to(device)
    recall_metric    = MulticlassRecall(num_classes=num_classes, average=None).to(device)
    confmat_metric   = MulticlassConfusionMatrix(num_classes=num_classes).to(device)

    correct = 0
    total = 0

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if isinstance(batch, dict):
                    x_batch, y_batch = batch['images'], batch['labels']
                else:
                    x_batch, y_batch = batch[0], batch[1]

                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = model(x_batch)
                pred_classes = preds.argmax(dim=1)

                correct += (pred_classes == y_batch).sum().item()
                total   += y_batch.size(0)

                # Defensive checks
                if pred_classes.numel() == 0 or y_batch.numel() == 0:
                    print(f"[evaluate_classification] empty batch at idx={batch_idx}", flush=True)

                assert pred_classes.max().item() < num_classes, "Predicted class index out of range"
                assert y_batch.max().item() < num_classes, "True label index out of range"

                # Update metrics (class indices)
                f1_metric.update(pred_classes, y_batch)
                precision_metric.update(pred_classes, y_batch)
                recall_metric.update(pred_classes, y_batch)
                confmat_metric.update(pred_classes, y_batch)

        if total == 0:
            print("[evaluate_classification] WARNING: total==0 (no samples).", flush=True)
            accuracy = float('nan')
        else:
            accuracy = correct / total

        macro_f1            = float(f1_metric.compute().item())
        precision_per_class = precision_metric.compute().detach().cpu().numpy()
        recall_per_class    = recall_metric.compute().detach().cpu().numpy()
        confmat             = confmat_metric.compute().detach().cpu().numpy()

        # Print metrics
        print(f"Accuracy: {accuracy:.4f}", flush=True)
        print(f"Macro F1-score: {macro_f1:.4f}", flush=True)
        print("Precision per class:", precision_per_class, flush=True)
        print("Recall per class:", recall_per_class, flush=True)

        # Plot confusion matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            confmat, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names if class_names else range(num_classes),
            yticklabels=class_names if class_names else range(num_classes)
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        if save_figure:
            fig_path = figures_path / f"{timestamp}__Confusion matrix.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"[evaluate_classification] saved figure to {fig_path}", flush=True)
        if show_plot:
            plt.show()
        else:
            plt.close()

        results = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "precision_per_class": precision_per_class,
            "recall_per_class": recall_per_class,
            "confusion_matrix": confmat
        }
        print("[evaluate_classification] done.", flush=True)
        return results

    except Exception as e:
        # Ensure errors are visible
        print("[evaluate_classification] ERROR:", repr(e), flush=True)
        raise



