


import matplotlib.pyplot as plt
import torch
import random
import torch.nn as nn


import matplotlib.pyplot as plt


def plot_filter_weights(model, rows=2, cols=4, channel=0):
    """
    Plots random filters from the first convolutional layer of a model.

    Args:
        model: PyTorch model with a 'features' attribute.
        rows (int): Number of rows in the plot grid.
        cols (int): Number of columns in the plot grid.
        channel (int): Which input channel to visualize (default: 0).
    """
    # Get weights from first conv layer
    weights = model.features[0].weight.data.cpu()  # shape: [out_channels, in_channels, kH, kW]
    
    num_filters = rows * cols
    total_filters = weights.shape[0]

    # Pick random filter indices
    random_indices = random.sample(range(total_filters), min(num_filters, total_filters))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for i, idx in enumerate(random_indices):
        axes[i].imshow(weights[idx, channel], cmap='gray')
        axes[i].axis('off')

    # Hide unused axes if num_filters < rows*cols
    for j in range(len(random_indices), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random



def plot_image(image, title=None):
    image_np = image.permute(1, 2, 0).numpy()
    plt.imshow(image_np)
    plt.axis('off')  
    plt.title('CHANGE TITLE')
    plt.show()

def plot_feature_maps(model, image, layers_to_show=None, num_maps=8, rows=None, cols=None, cmap='gray'):
    """
    Visualize randomly selected feature maps for selected layers of a CNN with user-defined rows/cols and minimal whitespace.
    """
    model.eval()
    with torch.no_grad():
        x = image.unsqueeze(0)  # Add batch dimension
        for i, layer in enumerate(model.features):
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                if layers_to_show is None or i in layers_to_show:
                    total_maps = x.shape[1]
                    maps_to_show = min(num_maps, total_maps)

                    # Randomly select indices
                    selected_indices = random.sample(range(total_maps), maps_to_show)

                    # Use user-defined layout or compute dynamically
                    if rows is None or cols is None:
                        cols = min(maps_to_show, 6)
                        rows = (maps_to_show + cols - 1) // cols

                    # Dynamic figure size based on rows/cols
                    fig_width = cols * 1.8
                    fig_height = rows * 1.8

                    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), constrained_layout=True)
                    axes = axes.flatten()

                    for j, idx in enumerate(selected_indices):
                        axes[j].imshow(x[0, idx].cpu(), cmap=cmap)
                        axes[j].axis('off')

                    # Hide unused axes
                    for k in range(maps_to_show, len(axes)):
                        axes[k].axis('off')

                    # Title close to top without extra padding
                    fig.suptitle(f'Random feature maps after layer {i} ({layer.__class__.__name__})')#, y=0.99)
                    plt.show()


def plot_training_history(history):

    plt.figure(figsize=(10,5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["valid_loss"], label="Valid Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["valid_acc"], label="Valid Accuracy")
    plt.legend()
    plt.show()



def plot_random_predictions(model, dataset, num_samples=12, rows=None, cols=None,
                             class_names=None, device="cpu", show_prob=True):
    """
    Plots random predictions from a dataset using a trained model.

    Args:
        model: Trained PyTorch model.
        dataset: Dataset (e.g., ImageFolder or custom Dataset).
        num_samples: Number of random samples to display.
        rows: Number of rows in subplot grid (optional).
        cols: Number of columns in subplot grid (optional).
        class_names: List of class names. If None, tries dataset.classes.
        device: "cpu" or "cuda".
        show_prob: Whether to display prediction confidence.
    """
    model.eval()
    model.to(device)

    # Get class names
    if class_names is None and hasattr(dataset, "classes"):
        class_names = dataset.classes
    elif class_names is None:
        raise ValueError("Class names must be provided if dataset has no 'classes' attribute.")

    # Randomly select indices
    indices = random.sample(range(len(dataset)), num_samples)

    # Extract images and labels
    images = torch.stack([dataset[i][0] for i in indices]).to(device)
    labels = torch.tensor([dataset[i][1] for i in indices]).to(device)

    # Get predictions
    with torch.no_grad():
        predictions = model(images)
        probs = torch.softmax(predictions, dim=1)

    # Determine subplot layout
    if rows is None or cols is None:
        cols = min(num_samples, 6)
        rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for i in range(num_samples):
        ax = axes[i]
        ax.set_xticks([]); ax.set_yticks([])

        img = images[i].cpu().permute(1, 2, 0)  # [C, H, W] → [H, W, C]
        y_pred_idx = torch.argmax(predictions[i]).item()
        true_idx = labels[i].item()

        y_pred_name = class_names[y_pred_idx]
        true_name = class_names[true_idx]
        confidence = probs[i][y_pred_idx].item() * 100

        color = "green" if y_pred_idx == true_idx else "red"

        ax.imshow(img)
        title = f"Pred: {y_pred_name} ({confidence:.1f}%)\nTrue: {true_name}" if show_prob else f"Pred: {y_pred_name}\nTrue: {true_name}"
        ax.set_title(title, color=color, fontsize=8)

    # Hide unused axes
    for j in range(num_samples, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()