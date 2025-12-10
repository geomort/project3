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
    """
    Docstrings made with Copilot and edited
    Inspect a dataset and its dataloader by printing sample batch details.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset to validate.
    dataloader : torch.utils.data.DataLoader
        Dataloader for the dataset.

    Prints
    ------
    - Batch image tensor shape
    - Class names in batch
    - First sample path
    - Transform pipeline
    - Single sample image tensor shape and pixel stats (min/max after normalization)
    """

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
    """
    Docstrings made with Copilot and edited
    Compute and display a histogram of class labels in a dataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset with 'label' field and `subset_class_names` attribute.

    Returns
    -------
    collections.Counter
        Counts of samples per class index.

    Prints
    ------
    Per-class counts with class names.
    """
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
    """
    Docstrings made with Copilot and edited
    Evaluate a classification model and summarize performance metrics.

    Parameters
    ----------
    model : torch.nn.Module
        Trained classification model to evaluate.
    dataloader : torch.utils.data.DataLoader
        Iterable providing (images, labels) batches or dicts with 'images' and 'labels'.
    num_classes : int
        Number of target classes.
    figures_path : pathlib.Path
        Directory path where the confusion matrix figure will be saved.
    timestamp : str
        Timestamp string used in the saved figure filename.
    device : str, optional
        Evaluation device ('cpu' or 'cuda'). Default is 'cpu'.
    class_names : list[str] or None, optional
        Optional class names for axis tick labels in the confusion matrix.
    save_figure : bool, optional
        If True, save the confusion matrix plot to `figures_path`. Default is True.
    show_plot : bool, optional
        If True, display the plot interactively. Default is False.

    Returns
    -------
    dict
        A dictionary with:
        - 'accuracy' : float
            Overall accuracy across all samples.
        - 'macro_f1' : float
            Macro-averaged F1 score.
        - 'precision_per_class' : numpy.ndarray
            Precision for each class (shape: [num_classes]).
        - 'recall_per_class' : numpy.ndarray
            Recall for each class (shape: [num_classes]).
        - 'confusion_matrix' : numpy.ndarray
            Confusion matrix (shape: [num_classes, num_classes]).

    Prints
    ------
    Summary of accuracy, macro F1-score, per-class precision and recall,
    and logging messages (start, saved figure path, completion, warnings)
    """
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



