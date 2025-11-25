import random
import matplotlib.pyplot as plt
import torch
import math
import numpy as np
import torch.nn as nn




def find_normalize(dataset):
    """
    Try to extract the Normalize(mean,std) from dataset.transform (torchvision).
    Returns (mean, std) as tensors on CPU, or None if not found.
    """
    if not hasattr(dataset, "transform"):
        return None
    if hasattr(dataset.transform, "transforms"):
        for t in dataset.transform.transforms:
            # torchvision.transforms.Normalize
            if t.__class__.__name__ == "Normalize":
                mean = torch.tensor(t.mean, dtype=torch.float32)
                std  = torch.tensor(t.std, dtype=torch.float32)
                return mean, std
    return None


def denormalize_img(img, mean=None, std=None):
    """
    Converts image (Tensor, NumPy, or PIL) to NumPy [H,W,C] in [0,1].
    """
    import torch
    import numpy as np
    from PIL import Image
    import torchvision.transforms as T

    # Convert PIL to tensor
    if isinstance(img, Image.Image):
        img = T.ToTensor()(img)

    # Convert NumPy to tensor
    if isinstance(img, np.ndarray):
        img = torch.tensor(img)

    # Ensure tensor
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"Unsupported image type: {type(img)}")

    x = img.detach().cpu()

    # Handle mean/std
    if mean is not None and std is not None:
        if isinstance(mean, list):
            mean = torch.tensor(mean)
        if isinstance(std, list):
            std = torch.tensor(std)
        x = x * std.view(-1, 1, 1) + mean.view(-1, 1, 1)

    # Clamp and convert to NumPy float
    x = x.clamp(0, 1)
    x = x.permute(1, 2, 0).numpy().astype(np.float32)
    return x





def _infer_mean_std_from_transforms(transforms):
    """Try to detect torchvision.transforms.Normalize(mean, std) in a Compose."""
    if hasattr(transforms, "transforms"):
        for t in transforms.transforms:
            # torchvision.transforms.Normalize has mean/std attributes
            if hasattr(t, "mean") and hasattr(t, "std"):
                # t.mean/t.std may be tensors; convert to list of floats
                mean = [float(x) for x in (t.mean.tolist() if hasattr(t.mean, "tolist") else t.mean)]
                std  = [float(x) for x in (t.std.tolist()  if hasattr(t.std,  "tolist") else t.std)]
                return mean, std
    return [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

def plot_random_image_per_class(dataset, mean=None, std=None, max_cols=4, verbose=False,
                                save_figure=False, show_plot=False, figures_path=None, timestamp=None):
    import math
    import numpy as np
    import matplotlib.pyplot as plt

    class_names = dataset.subset_class_names
    num_classes = len(class_names)

    # Infer normalization stats if not provided
    if mean is None or std is None:
        mean, std = _infer_mean_std_from_transforms(dataset.transform)

    # Build indices per class
    indices_per_class = [[] for _ in range(num_classes)]
    for idx in range(len(dataset)):
        sample = dataset[idx]
        cls_id = int(sample["label"])
        if 0 <= cls_id < num_classes:
            indices_per_class[cls_id].append(idx)

    # Choose one random index per class
    chosen = []
    for cls_id in range(num_classes):
        candidates = indices_per_class[cls_id]
        if candidates:
            sel_idx = random.choice(candidates)
            chosen.append(sel_idx)
            if verbose:
                print(f"[RANDOM PICK] {class_names[cls_id]} -> idx {sel_idx}")
        else:
            chosen.append(None)
            if verbose:
                print(f"[SKIP] No samples for class '{class_names[cls_id]}'")

    # Plot
    cols = min(max_cols, num_classes)
    rows = math.ceil(num_classes / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = axes.ravel() if isinstance(axes, np.ndarray) else [axes]

    for cls_idx in range(num_classes):
        ax = axes[cls_idx]
        sel_idx = chosen[cls_idx]
        ax.set_title(class_names[cls_idx])
        if sel_idx is not None:
            sample = dataset[sel_idx]
            img = sample.get("image", None)
            if img is None:
                ax.set_title(f"{class_names[cls_idx]}\n(Image missing)")
                ax.axis("off")
                continue
            img_disp = denormalize_img(img, mean, std)
            ax.imshow(img_disp)
        ax.axis("off")

    for k in range(num_classes, len(axes)):
        axes[k].axis("off")

    plt.suptitle("Random image per class", fontsize=16)
    plt.tight_layout()
    if save_figure:
        plt.savefig(figures_path / f"{timestamp}_random_image_per_class.png", dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()





def plot_loss_accuracy(history, figures_path, timestamp, save_figure, show_plot):
    # --- Loss Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["valid_loss"], label="Validation Loss")
    if "test_loss" in history:
        plt.plot(history["test_loss"], label="Test Loss", linestyle="--")
    plt.legend()
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    if save_figure:
        plt.savefig(figures_path / f"{timestamp}_Loss.png", dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()

    # --- Accuracy Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["valid_acc"], label="Validation Accuracy")
    if "test_acc" in history:
        plt.plot(history["test_acc"], label="Test Accuracy", linestyle="--")
    plt.legend()
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    if save_figure:
        plt.savefig(figures_path / f"{timestamp}_Accuracy.png", dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_random_predictions(model, dataset, num_samples, path, time, number_images,
                            rows=None, cols=None, class_names=None, selected_classes=None,
                            device="cpu", show_prob=True, save_figures=True, show_plot=True,
                            max_cols=2):
    """
    Displays random predictions from a trained model on a given dataset.
    Robust layout: minimal whitespace, dynamic sizing.
    """

    import math
    import random
    import torch
    import matplotlib.pyplot as plt

    # Normalization info
    norm = find_normalize(dataset)
    mean, std = norm if norm is not None else (None, None)

    model.eval()
    model.to(device)

    # Resolve class names
    if class_names is None:
        if hasattr(dataset, "subset_class_names") and dataset.subset_class_names:
            class_names = dataset.subset_class_names
        elif hasattr(dataset, "class_names") and dataset.class_names:
            class_names = dataset.class_names
        else:
            raise ValueError("class_names must be provided if dataset lacks names.")

    # Handle selected classes
    selected_indices = None
    if selected_classes is not None:
        selected_indices = [class_names.index(name) for name in selected_classes]

    # Sample indices
    num_samples = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)

    # Extract images and labels
    imgs, lbls = [], []
    for i in indices:
        item = dataset[i]
        if isinstance(item, dict):
            img, label = item["image"], item["label"]
        else:
            img, label = item[0], item[1]
        imgs.append(img)
        lbls.append(label)

    images = torch.stack(imgs).to(device)
    labels = torch.tensor(lbls, dtype=torch.long).to(device)

    # Predictions
    with torch.no_grad():
        logits = model(images)
        probs = torch.softmax(logits, dim=1)

    # Dynamic grid
    if rows is None or cols is None:
        cols = min(max_cols, num_samples)
        rows = math.ceil(num_samples / cols)

    # Dynamic figure size: smaller for small grids
    base_size = 3
    fig_width = max(6, cols * base_size)
    fig_height = max(4, rows * base_size)

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)

    for i in range(num_samples):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        ax.set_xticks([]); ax.set_yticks([])

        img_disp = denormalize_img(images[i], mean, std)
        ax.imshow(img_disp)

        # Prediction logic
        if selected_indices is not None and logits.shape[1] != len(selected_indices):
            sel_logits = logits[i][selected_indices]
            pred_idx_in_selected = torch.argmax(sel_logits).item()
            y_pred_idx = selected_indices[pred_idx_in_selected]
            y_pred_name = class_names[y_pred_idx]
            confidence = torch.softmax(sel_logits, dim=0)[pred_idx_in_selected].item() * 100
        else:
            y_pred_idx = torch.argmax(logits[i]).item()
            y_pred_name = class_names[y_pred_idx]
            confidence = probs[i][y_pred_idx].item() * 100

        true_idx = labels[i].item()
        true_name = class_names[true_idx]

        color = "green" if y_pred_name == true_name else "red"
        title = f"Pred: {y_pred_name} ({confidence:.1f}%)\nTrue: {true_name}" if show_prob \
                else f"Pred: {y_pred_name}\nTrue: {true_name}"
        ax.set_title(title, color=color, fontsize=9)

    # Hide unused panels
    for k in range(num_samples, rows * cols):
        r, c = divmod(k, cols)
        axes[r][c].axis("off")

    plt.suptitle("Predictions and True Labels", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reserve space for title

    if save_figures:
        plt.savefig(path / f'{time}_Predictions_{number_images}_samples{num_samples}.png',
                    dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()




import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path

def find_conv_layers(model: nn.Module):
    """
    Returns an ordered list of (name, layer) for all nn.Conv2d layers in the model.
    The order reflects the forward traversal order of modules.
    """
    convs = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            convs.append((name, module))
    return convs

def plot_filter_weights(
    model: nn.Module,
    path: Path,
    time: str,
    number_images: int,
    rows: int = 2,
    cols: int = 4,
    channel: int = 0,
    title: str = 'Filter weights',
    save_figures: bool = True,
    show_plot: bool = True,
    layer_index: int = 0,
    layer_name: str | None = None,
    normalize_each_filter: bool = True,
    cmap: str = 'viridis',
    seed: int | None = None,
):
    """
    Plot random filters from a chosen Conv2d layer in a PyTorch model.

    Args:
        model (nn.Module): The model containing convolutional layers.
        path (Path | str): Directory to save the figure if `save_figures=True`.
        time (str): Identifier for filename (e.g., timestamp).
        number_images (int): Included in filename for context.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        channel (int): Input channel index to visualize (0 for R in RGB, 0 for grayscale).
        title (str): Title of the plot.
        save_figures (bool): Whether to save the figure as PNG.
        show_plot (bool): Whether to show the figure.
        layer_index (int): Index of conv layer to visualize (0 = first conv layer).
        layer_name (str | None): If provided, selects the conv layer by its name.
        normalize_each_filter (bool): If True, min-max normalize each filter for display.
        cmap (str): Matplotlib colormap.
        seed (int | None): Random seed for reproducible filter selection.

    Behavior:
        - Auto-discovers Conv2d layers (works for ResNet, custom CNNs, Sequential nets).
        - Selects filters randomly from chosen conv layer.
        - Visualizes weights for a single input channel per filter.
        - Saves and/or shows the plot.

    Notes:
        - Conv weights shape: [out_channels, in_channels, kH, kW].
        - `channel` must be < in_channels of the chosen layer.
        - For depthwise convs, in_channels may equal groups; still works the same.

    Returns:
        dict with metadata about the plotted layer.
    """
    # Resolve path
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Discover conv layers
    conv_layers = find_conv_layers(model)
    if len(conv_layers) == 0:
        raise ValueError("No nn.Conv2d layers found in the model.")

    # Select layer by name or index
    if layer_name is not None:
        matches = [(n, l) for (n, l) in conv_layers if n == layer_name]
        if not matches:
            available = ", ".join(n for n, _ in conv_layers)
            raise ValueError(f"Layer name '{layer_name}' not found. Available conv layers: {available}")
        chosen_name, chosen_layer = matches[0]
    else:
        if not (0 <= layer_index < len(conv_layers)):
            raise IndexError(f"layer_index {layer_index} out of range. Found {len(conv_layers)} conv layers.")
        chosen_name, chosen_layer = conv_layers[layer_index]

    # Get weights
    W = chosen_layer.weight.detach().cpu()  # [out_channels, in_channels, kH, kW]
    out_channels, in_channels, kH, kW = W.shape

    if channel < 0 or channel >= in_channels:
        raise IndexError(f"Requested channel={channel} but layer '{chosen_name}' has in_channels={in_channels}.")

    # Select filters
    num_filters = rows * cols
    total_filters = out_channels

    if seed is not None:
        random.seed(seed)

    indices = random.sample(range(total_filters), k=min(num_filters, total_filters))

    # Prepare figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.array(axes).reshape(-1)

    for i, idx in enumerate(indices):
        # Slice single-channel kernel
        kernel = W[idx, channel].numpy()  # [kH, kW]

        # Optional normalization per filter for better contrast
        if normalize_each_filter:
            k_min, k_max = kernel.min(), kernel.max()
            if k_max > k_min:
                kernel = (kernel - k_min) / (k_max - k_min)  # scale to [0,1]

        axes[i].imshow(kernel, cmap=cmap)
        axes[i].set_title(f"{chosen_name}\nfilter={idx}, ch={channel}", fontsize=8)
        axes[i].axis('off')

    # Hide unused axes
    for j in range(len(indices), len(axes)):
        axes[j].axis('off')

    # Layout & title
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.suptitle(title, fontsize=12)

    # Save/show
    fname = f"{time}_filter_weights_{chosen_name}_nimgs{number_images}_layer{layer_index}.png"
    if save_figures:
        out_path = path / fname
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"[plot_filter_weights] Saved: {out_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return {
        "chosen_layer_name": chosen_name,
        "out_channels": out_channels,
        "in_channels": in_channels,
        "kernel_size": (kH, kW),
        "num_plotted": len(indices),
        "indices": indices,
        "filename": fname if save_figures else None,
    }




def plot_image(image, path, time, number_images, title='CHANGE TITLE',
               save_figures=False, show_plot=False, dataset=None):
    """
    Displays a single image (tensor or numpy) using Matplotlib and optionally saves it.
    Automatically denormalizes if Normalize(mean,std) is found in dataset.transform.

    Args:
        image: torch.Tensor [C,H,W] or numpy.ndarray [H,W,C]
        path: Path-like for saving
        time: string identifier (e.g., timestamp)
        number_images: int for filename context
        title: str
        save_figures: bool
        show_plot: bool
        dataset: optional dataset to auto-detect Normalize(mean,std)
    """
    # --- Detect normalization from dataset ---
    mean, std = None, None
    if dataset is not None:
        norm = find_normalize(dataset)
        if norm is not None:
            mean, std = norm

    # --- Prepare image for display ---
    image_np = denormalize_img(image, mean, std)  # returns [H,W,C] numpy in [0,1]

    # --- Plot with extra title space ---
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image_np)
    ax.axis('off')
    ax.set_title(title, fontsize=14, pad=15)

    fig.tight_layout()
    fig.subplots_adjust(top=0.87)  # more space for title

    # --- Save/Show ---
    if save_figures:
        fig.savefig(path / f'{time}_sample_image_number_images{number_images}.png',
                    dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close(fig)




import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union

def find_conv_layers(model: nn.Module) -> List[Tuple[str, nn.Conv2d]]:
    """
    Return (name, layer) tuples for all nn.Conv2d modules in forward-traversal order.
    Works for ResNet, VGG, custom Sequential models, etc.
    """
    convs = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            convs.append((name, module))
    return convs

def _sanitize(name: str) -> str:
    return name.replace(".", "_").replace("/", "_")



def plot_feature_maps(
    model: nn.Module,
    image: Union[torch.Tensor, "PIL.Image.Image", np.ndarray],
    path: Union[Path, str],
    time: str,
    number_images: int,
    layers_to_show: Optional[List[int]] = None,      # indices in conv-layer order
    layer_names: Optional[List[str]] = None,         # exact module names (e.g., "conv1", "layer1.0.conv1")
    num_maps: int = 8,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    cmap: str = "gray",
    save_figures: bool = True,
    show_plot: bool = True,
    device: Union[str, torch.device] = "cpu",
    input_is_tensor: bool = True,                    # set False if passing a PIL image or np.ndarray
    preprocess: Optional[callable] = None,           # e.g., weights.transforms() or custom Compose
    seed: Optional[int] = None
) -> Dict[str, Dict]:
    """
    Visualize randomly selected feature maps from chosen Conv2d layers using forward hooks.
    Works with ResNet and custom CNNs.

    - If `layer_names` is given, those modules are used (preferred).
    - Else if `layers_to_show` is given, indices into the discovered conv layers are used.
    - Else, plots all Conv2d layers (be careful: ResNet has many).

    Returns:
        A dict mapping layer_name -> metadata dict (num_maps_plotted, shape, indices, filename).
    """
    # Reproducibility for random map selection
    if seed is not None:
        random.seed(seed)

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Prep model/device
    model.eval()
    device = torch.device(device)
    model.to(device)

    # Prepare input tensor [1, C, H, W]
    if not input_is_tensor:
        if preprocess is None:
            raise ValueError("If input_is_tensor=False, you must provide a preprocess callable (e.g., weights.transforms()).")
        x = preprocess(image).unsqueeze(0).to(device)
    else:
        # Expect CHW float tensor (already normalized if needed)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if image.ndim == 3:  # CHW
            x = image.unsqueeze(0).to(device)
        elif image.ndim == 4:  # BCHW
            x = image.to(device)
        else:
            raise ValueError(f"Unsupported input tensor shape: {image.shape}")

    # Discover conv layers
    conv_layers = find_conv_layers(model)
    if len(conv_layers) == 0:
        raise ValueError("No nn.Conv2d layers found in the model.")

    # Resolve which layers to visualize
    selected: List[Tuple[str, nn.Conv2d]] = []
    if layer_names is not None and len(layer_names) > 0:
        names_set = set(layer_names)
        name_to_layer = {n: l for (n, l) in conv_layers}
        missing = [n for n in names_set if n not in name_to_layer]
        if missing:
            available = ", ".join(n for n, _ in conv_layers)
            raise ValueError(f"Layer names not found: {missing}\nAvailable conv layers: {available}")
        selected = [(n, name_to_layer[n]) for n in layer_names]
    elif layers_to_show is not None and len(layers_to_show) > 0:
        for idx in layers_to_show:
            if not (0 <= idx < len(conv_layers)):
                raise IndexError(f"layers_to_show index {idx} out of range (found {len(conv_layers)} conv layers).")
            selected.append(conv_layers[idx])
    else:
        # Default: all conv layers (can be many on ResNet)
        selected = conv_layers

    # Register hooks and run forward pass
    captured: Dict[str, torch.Tensor] = {}
    hooks = []

    def _make_hook(name: str):
        def hook(module, input, output):
            # Capture the feature maps: output shape [B, C, H, W]
            captured[name] = output.detach().to("cpu")
        return hook

    try:
        for name, layer in selected:
            hooks.append(layer.register_forward_hook(_make_hook(name)))

        with torch.no_grad():
            _ = model(x)

    finally:
        # Clean up hooks
        for h in hooks:
            h.remove()

    # Plot
    results: Dict[str, Dict] = {}
    for name, _layer in selected:
        if name not in captured:
            # Layer may be skipped due to conditional flow; warn and continue.
            print(f"[plot_feature_maps] Warning: no output captured for layer '{name}'.")
            continue

        feat = captured[name]  # [B, C, H, W]
        B, C, H, W = feat.shape
        if B == 0 or C == 0:
            print(f"[plot_feature_maps] Warning: empty output at layer '{name}'.")
            continue

        maps_to_show = min(num_maps, C)
        indices = random.sample(range(C), maps_to_show)

        # Dynamic grid
        if rows is None or cols is None:
            cols_eff = min(maps_to_show, 6)
            rows_eff = (maps_to_show + cols_eff - 1) // cols_eff
        else:
            rows_eff, cols_eff = rows, cols

        fig_w = cols_eff * 1.8
        fig_h = rows_eff * 1.8

        fig, axes = plt.subplots(rows_eff, cols_eff, figsize=(fig_w, fig_h), constrained_layout=True)
        axes = np.array(axes).reshape(-1)

        for j, idx in enumerate(indices):
            # Use the first sample in batch for visualization
            img = feat[0, idx].numpy()
            # Optional min-max normalization for display (improves contrast)
            vmin, vmax = float(img.min()), float(img.max())
            if vmax > vmin:
                img = (img - vmin) / (vmax - vmin)

            axes[j].imshow(img, cmap=cmap)
            axes[j].axis("off")
            axes[j].set_title(f"ch={idx}", fontsize=7)

        # Hide extra axes
        for k in range(maps_to_show, len(axes)):
            axes[k].axis("off")

        fig.suptitle(f"Feature maps: {_sanitize(name)}", fontsize=11)

        # Save
        fname = f"{time}_FeatureMaps_{_sanitize(name)}_nimgs{number_images}.png"
        if save_figures:
            out_path = path / fname
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"[plot_feature_maps] Saved: {out_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        results[name] = {
            "shape": (B, C, H, W),
            "num_maps_plotted": maps_to_show,
            "indices": indices,
            "filename": fname if save_figures else None,
        }

    return results







"""
def plot_feature_maps(model, model_layer_number, image, path, time, number_images, layers_to_show=None, num_maps=8, rows=None, cols=None, cmap='gray', save_figures=None, show_plot=None):
    
    Docstring created by Copilot:

    Visualizes randomly selected feature maps from specified convolutional layers of a CNN.

    This function performs a forward pass of the input image through the model and plots
    feature maps from selected layers. It supports dynamic grid layout and minimal whitespace
    for better visualization.

    Args:
        model (torch.nn.Module): A PyTorch CNN model with a `features` attribute.
        model_layer_number (int): Identifier for the layer (used in the saved filename and title).
        image (torch.Tensor): Input image tensor with shape [C, H, W].
        path (Path or str): Directory path where the figure will be saved if `save_figures` is True.
        time (str): A string identifier (e.g., timestamp) used in the saved filename.
        number_images (int): Number of images used in training or context, included in filename.
        layers_to_show (list[int], optional): List of layer indices to visualize. If None, all Conv2d layers are considered.
        num_maps (int, optional): Number of feature maps to display per layer. Default is 8.
        rows (int, optional): Number of rows in the plot grid. If None, computed dynamically.
        cols (int, optional): Number of columns in the plot grid. If None, computed dynamically.
        cmap (str, optional): Colormap for displaying feature maps. Default is 'gray'.
        save_figures (bool, optional): Whether to save the figure to disk. Default is None (treated as False).
        show_plot (bool, optional): Whether to display the plot interactively. Default is None (treated as False).

    Behavior:
        - Performs a forward pass through all layers in `model.features`.
        - For each Conv2d layer in `layers_to_show` (or all if None), randomly selects `num_maps` feature maps.
        - Dynamically computes grid layout if `rows` or `cols` are not provided.
        - Displays feature maps using Matplotlib with minimal whitespace.
        - Saves the figure as a PNG file if `save_figures` is True.
        - Shows or closes the plot based on `show_plot`.

    Notes:
        - Feature maps are normalized automatically by Matplotlib's default behavior.
        - The figure title includes the layer number for clarity.

    Example:
        plot_feature_maps(model, 3, image_tensor, Path('./plots'), '2025-11-18', 100,
                          layers_to_show=[2, 4], num_maps=6, cmap='viridis', save_figures=True, show_plot=True)
    

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
                    fig.suptitle(f'Feature map after layer {model_layer_number}')
                    #fig.suptitle(f'Random feature maps after layer {i} ({layer.__class__.__name__})')#, y=0.99)
                    if save_figures: plt.savefig(path / f'{time}_Feature_map_model_layer{model_layer_number}_number_images{number_images}.png', dpi=300, bbox_inches='tight')
                    if show_plot:
                        plt.show()
                    else:
                        plt.close()
"""