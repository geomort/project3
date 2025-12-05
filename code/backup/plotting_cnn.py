


import matplotlib.pyplot as plt
import torch
import random
import torch.nn as nn


import matplotlib.pyplot as plt
from datetime import datetime



def plot_filter_weights(model, path, time, number_images, rows=2, cols=4, channel=0, title='Filter weights', save_figures=True, show_plot=True):

    """
    Docstring created by Copilot:

    Plots random filters from the first convolutional layer of a PyTorch model.

    This function selects a random subset of filters from the first convolutional layer
    of the given model and visualizes them in a grid layout using Matplotlib.

    Args:
        model (torch.nn.Module): A PyTorch model with a 'features' attribute containing layers.
        path (Path or str): Directory path where the figure will be saved if `save_figures` is True.
        time (str): A string identifier (e.g., timestamp) used in the saved filename.
        number_images (int): Number of images used in training or context, included in filename.
        rows (int, optional): Number of rows in the plot grid. Default is 2.
        cols (int, optional): Number of columns in the plot grid. Default is 4.
        channel (int, optional): Input channel index to visualize. Default is 0.
        title (str, optional): Title for the plot. Default is 'Filter weights'.
        save_figures (bool, optional): Whether to save the figure to disk. Default is True.
        show_plot (bool, optional): Whether to display the plot interactively. Default is True.

    Behavior:
        - Randomly selects `rows * cols` filters (or fewer if the layer has fewer filters).
        - Displays each filter using the viridis colormap.
        - Saves the figure as a PNG file if `save_figures` is True.
        - Shows or closes the plot based on `show_plot`.

    Notes:
        - Assumes the first layer in `model.features` is a convolutional layer.
        - The weights are expected to have shape [out_channels, in_channels, kH, kW].

    Example:
        plot_filter_weights(model, Path('./plots'), '2025-11-18', 100, rows=3, cols=3)
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
        axes[i].imshow(weights[idx, channel], cmap='viridis')
        axes[i].axis('off')

    # Hide unused axes if num_filters < rows*cols
    for j in range(len(random_indices), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    fig.subplots_adjust(top=0.9)  # Reserve space for suptitle
    plt.suptitle(title)
    if save_figures: plt.savefig(path / f'{time}_filter_weights_number_images{number_images}.png', dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()




def plot_image(image, path, time, number_images, title='CHANGE TITLE', save_figures=None, show_plot=None):
    """
    Docstring created by Copilot:

    Displays a single image tensor using Matplotlib and optionally saves it.

    Converts a PyTorch image tensor from shape [C, H, W] to [H, W, C] for visualization.
    The image is shown without axes and can be saved to disk if requested.

    Args:
        image (torch.Tensor): Image tensor with shape [C, H, W].
        path (Path or str): Directory path where the figure will be saved if `save_figures` is True.
        time (str): A string identifier (e.g., timestamp) used in the saved filename.
        number_images (int): Number of images used in training or context, included in filename.
        title (str, optional): Title for the plot. Default is 'CHANGE TITLE'.
        save_figures (bool, optional): Whether to save the figure to disk. Default is None (treated as False).
        show_plot (bool, optional): Whether to display the plot interactively. Default is None (treated as False).

    Behavior:
        - Converts the tensor to NumPy and displays it using `imshow`.
        - Removes axes for a cleaner look.
        - Saves the figure as a PNG file if `save_figures` is True.
        - Shows or closes the plot based on `show_plot`.

    Example:
        plot_image(image_tensor, Path('./plots'), '2025-11-18', 100, title='Sample Image', save_figures=True, show_plot=True)
    """

    image_np = image.permute(1, 2, 0).numpy()
    plt.imshow(image_np)
    plt.axis('off')  
    plt.title(title)
    if save_figures: plt.savefig(path / f'{time}_sample_image__number_images{number_images}.png', dpi=300, bbox_inches='tight')
        
    if show_plot:
        plt.show()
    else:
        plt.close()




def plot_feature_maps(model, model_layer_number, image, path, time, number_images, layers_to_show=None, num_maps=8, rows=None, cols=None, cmap='gray', save_figures=None, show_plot=None):
    """
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
                    fig.suptitle(f'Feature map after layer {model_layer_number}')
                    #fig.suptitle(f'Random feature maps after layer {i} ({layer.__class__.__name__})')#, y=0.99)
                    if save_figures: plt.savefig(path / f'{time}_Feature_map_model_layer{model_layer_number}_number_images{number_images}.png', dpi=300, bbox_inches='tight')
                    if show_plot:
                        plt.show()
                    else:
                        plt.close()


def plot_training_history(history, images_for_train_validate_test, path, save_figures=True, show_plot=True):
    """
    Docstring created by Copilot:

    Plots the training and validation loss curves over epochs.

    This function visualizes the loss progression during model training and validation,
    helping to assess convergence and potential overfitting.

    Args:
        history (dict): Dictionary containing loss values per epoch. Must include:
            - "train_loss" (list or array): Training loss values.
            - "valid_loss" (list or array): Validation loss values.
        images_for_train_validate_test (int): Number of images used for training/validation/testing,
            included in the saved filename.
        path (Path or str): Directory path where the figure will be saved if `save_figures` is True.
        save_figures (bool, optional): Whether to save the figure to disk. Default is True.
        show_plot (bool, optional): Whether to display the plot interactively. Default is True.

    Behavior:
        - Creates a line plot of training and validation loss vs. epochs.
        - Adds labels, legend, and title for clarity.
        - Saves the figure as a PNG file if `save_figures` is True.
        - Shows or closes the plot based on `show_plot`.

    Notes:
        - The filename includes a timestamp and the number of images for traceability.
        - Ensure `history` contains keys "train_loss" and "valid_loss".


    Example:
        plot_training_history(history, 100, Path('./plots'), save_figures=True, show_plot=True)
    """


    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

    plt.figure(figsize=(10,5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["valid_loss"], label="Valid Loss")
    plt.legend()
    plt.title("Training and validation loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if save_figures: plt.savefig(path / f'{timestamp}_training_loss_number_images{images_for_train_validate_test}.png', dpi=300, bbox_inches='tight')
    if show_plot:
            plt.show()
    else:
        plt.close()


    plt.figure(figsize=(10,5))
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["valid_acc"], label="Valid Accuracy")
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title("Training and validation accuracy")
    if save_figures: plt.savefig(path / f'{timestamp}_training_accuracy_number_images{images_for_train_validate_test}.png', dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_random_predictions(model, dataset, num_samples, path, time, number_images,
                            rows=None, cols=None, class_names=None, selected_classes=None,
                            device="cpu", show_prob=True, save_figures=True, show_plot=True):
    """
    Displays random predictions from a trained model on a given dataset.
    Supports filtering predictions to selected classes by names and correct true label mapping.
    """

    model.eval()
    model.to(device)

    # Validate class names
    if class_names is None:
        if hasattr(dataset, "classes"):
            class_names = dataset.classes
        else:
            raise ValueError("class_names must be provided if dataset has no 'classes' attribute.")

    # Map selected class names to indices
    selected_indices = None
    if selected_classes is not None:
        selected_indices = [class_names.index(name) for name in selected_classes]
        print(f"[INFO] Selected classes: {selected_classes}")
        print(f"[INFO] Selected indices: {selected_indices}")

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

        # Handle prediction logic
        if selected_indices is not None:
            if predictions.shape[1] == len(selected_indices):
                # Model trained on selected classes only
                pred_idx_in_selected = torch.argmax(predictions[i]).item()
                y_pred_name = selected_classes[pred_idx_in_selected]
                confidence = probs[i][pred_idx_in_selected].item() * 100
            else:
                # Model trained on all classes
                selected_logits = predictions[i][selected_indices]
                pred_idx_in_selected = torch.argmax(selected_logits).item()
                y_pred_idx = selected_indices[pred_idx_in_selected]
                y_pred_name = class_names[y_pred_idx]
                confidence = torch.softmax(selected_logits, dim=0)[pred_idx_in_selected].item() * 100

            # ✅ True label mapping uses selected_classes
            true_idx = labels[i].item()
            true_name = selected_classes[true_idx]
        else:
            y_pred_idx = torch.argmax(predictions[i]).item()
            y_pred_name = class_names[y_pred_idx]
            confidence = probs[i][y_pred_idx].item() * 100
            true_idx = labels[i].item()
            true_name = class_names[true_idx]

        color = "green" if y_pred_name == true_name else "red"

        ax.imshow(img)
        title = f"Pred: {y_pred_name} ({confidence:.1f}%)\nTrue: {true_name}" if show_prob else f"Pred: {y_pred_name}\nTrue: {true_name}"
        ax.set_title(title, color=color, fontsize=8)

    # Hide unused axes
    for j in range(num_samples, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    plt.suptitle('Predictions and true labels')

    if save_figures:
        plt.savefig(path / f'{time}_Predictions_and_true_number_images{number_images}_samples{num_samples}.png',
                    dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()